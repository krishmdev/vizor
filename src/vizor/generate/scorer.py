"""Teacher-forced citation scoring: how likely the model is to write each source index at a
citation site of a fixed answer, given the prompt.

A `Scorer` takes chat messages, a continuation (the assistant's answer text) and citation sites
(character offsets in the continuation where an index digit starts, with the candidate index
strings). It returns, per site, the log-probability of each candidate as the next text at that
point and the same probabilities renormalized over the candidates. No sampling is involved, so
the result is a pure function of the weights, the prompt tokens and the continuation tokens.

Backends:
- `LocalhostScorer`: Localhost AI's `POST /v1/score` (the production path).
- `MLXScorer`: mlx-lm in process, for checks on real weights (optional dependency).
- `FakeScorer`: deterministic and content-sensitive, for offline tests and CI.

`CachedScorer` keys every result on sha256([scorer_id, model pin, messages, continuation, sites,
"score-v1"]), so a different server build, weights revision or tokenizer never reuses another's
scores.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import httpx

from vizor.generate.llm import Messages

SCORE_SCHEMA = "score-v1"


@dataclass(frozen=True)
class Site:
    """A citation site: `char_offset` in the continuation where the index text starts."""

    char_offset: int
    candidates: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"char_offset": self.char_offset, "candidates": list(self.candidates)}


@dataclass
class SiteScore:
    char_offset: int
    token_index: int
    logprobs: dict[str, float]
    renorm: dict[str, float]

    @classmethod
    def from_dict(cls, d: dict) -> SiteScore:
        return cls(
            int(d["char_offset"]),
            int(d.get("token_index", -1)),
            {str(k): float(v) for k, v in d["logprobs"].items()},
            {str(k): float(v) for k, v in d["renorm"].items()},
        )


@dataclass
class ScoreResult:
    sites: list[SiteScore]
    meta: dict = field(default_factory=dict)
    cached: bool = False

    def to_dict(self) -> dict:
        return {"sites": [asdict(s) for s in self.sites], "meta": self.meta}


class Scorer(Protocol):
    scorer_id: str

    def pin(self) -> dict:
        """What the scores depend on besides the inputs: model, revision, server commit, weights
        and tokenizer hashes. Part of the cache key."""
        ...

    def score(self, messages: Messages, continuation: str, sites: list[Site]) -> ScoreResult: ...


def renormalize(logprobs: dict[str, float]) -> dict[str, float]:
    """Probabilities over the candidates only (softmax of their log-probabilities)."""
    if not logprobs:
        return {}
    m = max(logprobs.values())
    ex = {k: math.exp(v - m) for k, v in logprobs.items()}
    z = sum(ex.values())
    return {k: v / z for k, v in ex.items()}


# ---------------------------------------------------------------------------------- Localhost AI
class LocalhostScorer:
    """Client for Localhost AI's `POST /v1/score`.

    Request: {model, messages, continuation, sites: [{char_offset, candidates}],
    chat_template_kwargs}. Response: {model, revision, commit, tokenizer_sha, weights_sha?,
    sites: [{token_index, candidates: {str: logprob}, renorm: {str: p}}]}, one entry per
    requested site, in order."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:8431",
        timeout: float = 600,
        chat_template_kwargs: dict | None = None,
        server_meta: dict | None = None,
    ) -> None:
        self.model = model
        self.scorer_id = f"localhost-ai/{model}"
        root = base_url.rstrip("/")
        root = root[: -len("/v1")] if root.endswith("/v1") else root
        self.url = root + "/v1/score"
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.server_meta = dict(server_meta or {})
        self._http = httpx.Client(timeout=timeout)
        self._pin: dict | None = None
        self._lock = threading.Lock()

    def _post(self, messages: Messages, continuation: str, sites: list[Site]) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "continuation": continuation,
            "sites": [s.to_dict() for s in sites],
            "chat_template_kwargs": self.chat_template_kwargs,
        }
        r = self._http.post(self.url, json=payload)
        r.raise_for_status()
        body = r.json()
        if len(body.get("sites", [])) != len(sites):
            raise ValueError(
                f"/v1/score returned {len(body.get('sites', []))} sites for {len(sites)} requested"
            )
        return body

    @staticmethod
    def _meta(body: dict) -> dict:
        keys = ("model", "revision", "commit", "tokenizer_sha", "weights_sha")
        return {k: body.get(k) for k in keys if body.get(k) is not None}

    def pin(self) -> dict:
        """Probe the server once for its model, revision, commit and hashes. A commit given in
        the config (`--server-commit`) must match what the server reports."""
        with self._lock:
            if self._pin is None:
                body = self._post(
                    [{"role": "user", "content": "Cite [1]."}], "[1]", [Site(1, ("1",))]
                )
                meta = self._meta(body)
                want = self.server_meta.get("commit")
                if want and meta.get("commit") and not str(meta["commit"]).startswith(want[:7]):
                    raise RuntimeError(
                        f"server reports commit {meta['commit']}, config says {want}"
                    )
                self._pin = {"backend": "localhost-ai", "url": self.url, **self.server_meta, **meta}
            return dict(self._pin)

    def score(self, messages: Messages, continuation: str, sites: list[Site]) -> ScoreResult:
        if not sites:
            return ScoreResult([], self.pin())
        body = self._post(messages, continuation, sites)
        out = []
        for s, r in zip(sites, body["sites"], strict=True):
            lp = {str(k): float(v) for k, v in r["candidates"].items()}
            ren = r.get("renorm") or renormalize(lp)
            out.append(SiteScore(s.char_offset, int(r.get("token_index", -1)), lp, dict(ren)))
        return ScoreResult(out, self._meta(body))


# ------------------------------------------------------------------------------------------ MLX
class MLXScorer:
    """mlx-lm in process. One forward per site over the prompt plus the continuation up to the
    site; a candidate's log-probability is that of its tokens after the common token prefix, so
    a tokenizer that merges "[1" into one token is still scored from the site. Slower than the
    server's single forward per answer; meant for equivalence checks on real weights."""

    def __init__(
        self, repo: str, revision: str | None = None, chat_template_kwargs: dict | None = None
    ) -> None:
        import mlx.core as mx  # noqa: F401  (fail early when mlx is missing)
        from mlx_lm import load

        self.repo, self.revision = repo, revision
        self.scorer_id = f"mlx/{repo}"
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.model, self.tokenizer = load(repo, revision=revision) if revision else load(repo)

    def pin(self) -> dict:
        tok_sha = None
        path = getattr(self.tokenizer, "name_or_path", None)
        if path and Path(path, "tokenizer.json").exists():
            tok_sha = hashlib.sha256(Path(path, "tokenizer.json").read_bytes()).hexdigest()
        return {
            "backend": "mlx",
            "repo": self.repo,
            "revision": self.revision,
            "tokenizer_sha": tok_sha,
        }

    def _logprobs(self, ids: list[int]):
        import mlx.core as mx

        logits = self.model(mx.array([ids]))[0].astype(mx.float32)
        return logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    def _seq(self, ids: list[int], start: int) -> float:
        if start >= len(ids):
            return 0.0
        lp = self._logprobs(ids)
        return float(sum(lp[i - 1, ids[i]].item() for i in range(start, len(ids))))

    def score(self, messages: Messages, continuation: str, sites: list[Site]) -> ScoreResult:
        head = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **self.chat_template_kwargs
        )
        enc = self.tokenizer.encode
        out = []
        for s in sites:
            prefix = head + continuation[: s.char_offset]
            p_ids = enc(prefix, add_special_tokens=False)
            lp = {}
            for c in s.candidates:
                c_ids = enc(prefix + c, add_special_tokens=False)
                k = 0
                while k < min(len(p_ids), len(c_ids)) and p_ids[k] == c_ids[k]:
                    k += 1
                lp[c] = self._seq(c_ids, k) - self._seq(p_ids, k)
            out.append(SiteScore(s.char_offset, len(p_ids), lp, renormalize(lp)))
        return ScoreResult(out, self.pin())


# ----------------------------------------------------------------------------------------- Fake
_TOK = re.compile(r"[a-z0-9]+")
_SENT_END = re.compile(r"[.!?](?=\s)|\n")


class FakeScorer:
    """Deterministic stand-in: the logit of source i at a site is 0.6 x the number of words the
    sentence before the site shares with source i's rendered text, minus a small position prior
    (0.15 per slot). So it rewards sources whose shown text matches the claim, and editing or
    dropping a passage moves the score the way it should, with no model at all."""

    scorer_id = "fake-scorer/v1"

    def __init__(self, overlap_weight: float = 0.6, position_prior: float = 0.15) -> None:
        self.w, self.prior = overlap_weight, position_prior

    def pin(self) -> dict:
        return {"backend": "fake", "w": self.w, "prior": self.prior}

    def score(self, messages: Messages, continuation: str, sites: list[Site]) -> ScoreResult:
        from vizor.generate.prompt import parse_prompt

        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        _, sources = parse_prompt(user)
        words = {
            str(s.index): set(_TOK.findall(f"{s.title} {s.description} {s.content}".lower()))
            for s in sources
        }
        out = []
        for s in sites:
            before = continuation[: s.char_offset]
            cut = [m.end() for m in _SENT_END.finditer(before.rstrip("[ (").rstrip())]
            sentence = set(_TOK.findall(before[cut[-1] if cut else 0 :].lower()))
            logits = {
                c: self.w * len(sentence & words.get(c, set()))
                - self.prior * (int(c) - 1 if c.isdigit() else 0)
                for c in s.candidates
            }
            m = max(logits.values())
            z = math.log(sum(math.exp(v - m) for v in logits.values())) + m
            lp = {c: v - z for c, v in logits.items()}
            out.append(SiteScore(s.char_offset, s.char_offset, lp, renormalize(lp)))
        return ScoreResult(out, {"model": self.scorer_id})


# ---------------------------------------------------------------------------------------- Cache
class CachedScorer:
    """Disk cache around any scorer, mirroring `CachedLLM`."""

    def __init__(self, inner: Scorer, cache_dir: Path | None) -> None:
        self.inner = inner
        self.scorer_id = inner.scorer_id
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.calls = 0
        self.hits = 0
        self._lock = threading.Lock()

    def pin(self) -> dict:
        return self.inner.pin()

    def key(self, messages: Messages, continuation: str, sites: list[Site]) -> str:
        parts = [
            self.scorer_id,
            self.pin(),
            messages,
            continuation,
            [s.to_dict() for s in sites],
            SCORE_SCHEMA,
        ]
        return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()

    def _path(self, key: str) -> Path:
        assert self.cache_dir is not None
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", self.scorer_id)
        return self.cache_dir / safe / key[:2] / f"{key}.json"

    def score(self, messages: Messages, continuation: str, sites: list[Site]) -> ScoreResult:
        key = self.key(messages, continuation, sites)
        if self.cache_dir is not None:
            p = self._path(key)
            if p.exists():
                d = json.loads(p.read_text())
                with self._lock:
                    self.hits += 1
                return ScoreResult(
                    [SiteScore.from_dict(s) for s in d["sites"]], d["meta"], cached=True
                )
        res = self.inner.score(messages, continuation, sites)
        with self._lock:
            self.calls += 1
        if self.cache_dir is not None:
            p = self._path(key)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(res.to_dict()))
            os.replace(tmp, p)
        return res

    def stats(self) -> dict:
        return {"calls": self.calls, "cache_hits": self.hits}
