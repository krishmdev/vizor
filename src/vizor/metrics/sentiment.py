"""Sentiment over the sentences that cite a domain and over its retrieved passages.

The default model is cardiffnlp/twitter-roberta-base-sentiment-latest (score = P(pos) - P(neg)).
It was trained on tweets, so product-copy scores carry some domain shift. VADER is the fallback
when the model files are not available (CI, or before `make models`).
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Sequence
from typing import Protocol

from vizor.models import PINS, limit_torch_threads, local_path, model_available


class Sentiment(Protocol):
    backend_id: str

    def __call__(self, texts: Sequence[str]) -> list[float]: ...


class _Cached:
    backend_id = "base"

    def __init__(self) -> None:
        self._cache: dict[str, float] = {}
        self._lock = threading.Lock()

    def _score(self, texts: list[str]) -> list[float]:
        raise NotImplementedError

    def __call__(self, texts: Sequence[str]) -> list[float]:
        with self._lock:
            return self._call_locked(texts)

    def _call_locked(self, texts: Sequence[str]) -> list[float]:
        keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
        todo = [(k, t) for k, t in zip(keys, texts, strict=True) if k not in self._cache]
        if todo:
            uniq = dict(todo)
            for k, v in zip(uniq, self._score(list(uniq.values())), strict=True):
                self._cache[k] = v
        return [self._cache[k] for k in keys]


class VaderSentiment(_Cached):
    backend_id = "vader/vaderSentiment-3.3.2/compound"

    def __init__(self) -> None:
        super().__init__()
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self._sia = SentimentIntensityAnalyzer()

    def _score(self, texts: list[str]) -> list[float]:
        return [self._sia.polarity_scores(t)["compound"] for t in texts]


class RobertaSentiment(_Cached):
    def __init__(self, batch_size: int = 32) -> None:
        super().__init__()
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        pin = PINS["sentiment"]
        self.backend_id = f"hf/{pin.repo}/{pin.revision[:12]}/p_pos-p_neg"
        self._torch = torch
        limit_torch_threads()
        path = local_path("sentiment")
        self._tok = AutoTokenizer.from_pretrained(path)
        self._model = AutoModelForSequenceClassification.from_pretrained(path).eval()
        labels = {v.lower(): k for k, v in self._model.config.id2label.items()}
        self._pos, self._neg = labels["positive"], labels["negative"]
        self._bs = batch_size

    def _score(self, texts: list[str]) -> list[float]:
        out: list[float] = []
        for i in range(0, len(texts), self._bs):
            batch = self._tok(
                texts[i : i + self._bs],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            with self._torch.no_grad():
                p = self._model(**batch).logits.softmax(-1)
            out.extend((p[:, self._pos] - p[:, self._neg]).tolist())
        return out


def make_sentiment(name: str = "auto") -> Sentiment:
    if name == "vader":
        return VaderSentiment()
    if name == "roberta":
        return RobertaSentiment()
    if name == "auto":
        return RobertaSentiment() if model_available("sentiment") else VaderSentiment()
    raise ValueError(f"unknown sentiment backend {name!r}")
