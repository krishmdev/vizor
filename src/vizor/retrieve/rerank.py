from __future__ import annotations

import hashlib
from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

import numpy as np

from vizor.models import PINS, limit_torch_threads, local_path


class Reranker(Protocol):
    reranker_id: str

    def score(self, query: str, texts: Sequence[str]) -> np.ndarray: ...


class NoopReranker:
    """Keeps retrieval order; score() returns NaN so the cascade falls back to cosine."""

    reranker_id = "none"

    def score(self, query: str, texts: Sequence[str]) -> np.ndarray:
        return np.full(len(texts), np.nan)


class CrossEncoderReranker:
    def __init__(self, batch_size: int = 32, max_length: int = 512) -> None:
        from sentence_transformers import CrossEncoder

        pin = PINS["reranker"]
        self.reranker_id = f"cross-encoder/{pin.repo}/{pin.revision[:12]}"
        limit_torch_threads()
        self._model = CrossEncoder(local_path("reranker"), max_length=max_length, device="cpu")
        self._bs = batch_size
        self._cache: dict[str, float] = {}

    def score(self, query: str, texts: Sequence[str]) -> np.ndarray:
        keys = [hashlib.sha1(f"{query}\x1f{t}".encode()).hexdigest() for t in texts]
        todo = {k: t for k, t in zip(keys, texts, strict=True) if k not in self._cache}
        if todo:
            pairs = [(query, t) for t in todo.values()]
            logits = self._model.predict(pairs, batch_size=self._bs, show_progress_bar=False)
            for k, v in zip(todo, np.asarray(logits).ravel(), strict=True):
                self._cache[k] = float(v)
        return np.array([self._cache[k] for k in keys])


@lru_cache(maxsize=2)
def make_reranker(name: str = "cross-encoder") -> Reranker:
    if name in ("none", "noop"):
        return NoopReranker()
    if name == "cross-encoder":
        return CrossEncoderReranker()
    raise ValueError(f"unknown reranker {name!r}")
