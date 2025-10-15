"""Chat backends behind one `complete()` call, plus a disk cache and a spend cap.

The cache key is sha256(model, messages, temperature, seed, max_tokens). OpenAI's `seed` is best
effort, so the cache (and the committed responses) is what makes a real-model run reproducible.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

Messages = list[dict[str, str]]

# USD per 1M tokens (input, output), list prices used for the spend estimate.
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
}


@dataclass
class Completion:
    text: str
    model: str
    usage: dict = field(default_factory=dict)
    cached: bool = False


class LLM(Protocol):
    model_id: str

    def complete(
        self, messages: Messages, *, temperature: float, seed: int, max_tokens: int
    ) -> Completion: ...


class BudgetExceeded(RuntimeError):
    pass


def _retryable(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    return status in (408, 409, 429, 500, 502, 503, 504) or isinstance(
        exc, httpx.TransportError | TimeoutError
    )


class OpenAIChat:
    def __init__(
        self, model: str = "gpt-4o-mini", base_url: str | None = None, timeout: float = 60
    ) -> None:
        from openai import OpenAI

        if not os.environ.get("OPENAI_API_KEY") and base_url is None:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.model_id = model
        self._client = OpenAI(base_url=base_url, timeout=timeout, max_retries=0)

    @retry(
        retry=retry_if_exception(_retryable),
        wait=wait_exponential(min=2, max=60),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def complete(
        self, messages: Messages, *, temperature: float, seed: int, max_tokens: int
    ) -> Completion:
        r = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
        )
        u = r.usage
        usage = {
            "prompt_tokens": u.prompt_tokens if u else 0,
            "completion_tokens": u.completion_tokens if u else 0,
            "system_fingerprint": r.system_fingerprint,
            "finish_reason": r.choices[0].finish_reason,
        }
        return Completion(r.choices[0].message.content or "", r.model, usage)


class OllamaChat:
    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct",
        base_url: str = "http://localhost:11434",
        timeout: float = 600,
    ) -> None:
        self.model_id = model
        self._url = base_url.rstrip("/") + "/api/chat"
        self._http = httpx.Client(timeout=timeout)

    @retry(
        retry=retry_if_exception(_retryable),
        wait=wait_exponential(min=1, max=20),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def complete(
        self, messages: Messages, *, temperature: float, seed: int, max_tokens: int
    ) -> Completion:
        # The old client used `temperature or default`, which silently turned 0.0 into 0.1.
        payload = {
            "model": self.model_id,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "seed": seed, "num_predict": max_tokens},
        }
        r = self._http.post(self._url, json=payload)
        r.raise_for_status()
        body = r.json()
        usage = {
            "prompt_tokens": body.get("prompt_eval_count", 0),
            "completion_tokens": body.get("eval_count", 0),
        }
        return Completion(body["message"]["content"], self.model_id, usage)


def cost_usd(model: str, usage: dict) -> float:
    base = next((k for k in PRICES if model.startswith(k)), None)
    if base is None:
        return 0.0
    pin, pout = PRICES[base]
    return (usage.get("prompt_tokens", 0) * pin + usage.get("completion_tokens", 0) * pout) / 1e6


class CachedLLM:
    """Disk cache + spend tracking around any backend."""

    def __init__(self, inner: LLM, cache_dir: Path | None, max_cost_usd: float | None = None):
        self.inner = inner
        self.model_id = inner.model_id
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_cost_usd = max_cost_usd
        self.spent_usd = 0.0
        self.calls = 0
        self.hits = 0
        self._lock = threading.Lock()

    def _key(self, messages: Messages, temperature: float, seed: int, max_tokens: int) -> str:
        blob = json.dumps([self.model_id, messages, temperature, seed, max_tokens], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        assert self.cache_dir is not None
        safe = self.model_id.replace("/", "_").replace(":", "_")
        return self.cache_dir / safe / key[:2] / f"{key}.json"

    def complete(
        self, messages: Messages, *, temperature: float, seed: int, max_tokens: int
    ) -> Completion:
        key = self._key(messages, temperature, seed, max_tokens)
        if self.cache_dir is not None:
            p = self._path(key)
            if p.exists():
                d = json.loads(p.read_text())
                with self._lock:
                    self.hits += 1
                return Completion(d["text"], d["model"], d["usage"], cached=True)
        with self._lock:
            if self.max_cost_usd is not None and self.spent_usd >= self.max_cost_usd:
                raise BudgetExceeded(f"spent ${self.spent_usd:.3f} of ${self.max_cost_usd:.2f}")
        t0 = time.perf_counter()
        c = self.inner.complete(messages, temperature=temperature, seed=seed, max_tokens=max_tokens)
        c.usage["latency_s"] = round(time.perf_counter() - t0, 3)
        c.usage["cost_usd"] = cost_usd(c.model, c.usage)
        with self._lock:
            self.calls += 1
            self.spent_usd += c.usage["cost_usd"]
        if self.cache_dir is not None:
            p = self._path(key)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps({"text": c.text, "model": c.model, "usage": c.usage}))
            os.replace(tmp, p)
        return c

    def stats(self) -> dict:
        return {"calls": self.calls, "cache_hits": self.hits, "spent_usd": round(self.spent_usd, 4)}
