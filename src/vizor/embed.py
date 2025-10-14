"""Text embedders. Each carries an `embedder_id` (provider/model/revision/dim/preprocessing-hash)
that tags every cache entry and index built from it, so vectors from different models never mix.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from functools import lru_cache
from typing import Literal, Protocol

import numpy as np

from vizor.models import PINS, limit_torch_threads, local_path

Kind = Literal["query", "passage"]
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    embedder_id: str
    dim: int

    def encode(self, texts: Sequence[str], kind: Kind = "passage") -> np.ndarray: ...


def _prep_hash(*parts: object) -> str:
    return hashlib.sha1(repr(parts).encode()).hexdigest()[:8]


class _CachedEmbedder:
    embedder_id = "base"
    dim = 0

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], np.ndarray] = {}

    def _encode(self, texts: list[str], kind: Kind) -> np.ndarray:
        raise NotImplementedError

    def encode(self, texts: Sequence[str], kind: Kind = "passage") -> np.ndarray:
        keys = [(kind, hashlib.sha1(t.encode()).hexdigest()) for t in texts]
        missing = {k: t for k, t in zip(keys, texts, strict=True) if k not in self._cache}
        if missing:
            vecs = self._encode(list(missing.values()), kind)
            for k, v in zip(missing, vecs, strict=True):
                self._cache[k] = v
        if not keys:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._cache[k] for k in keys])


class HashingEmbedder(_CachedEmbedder):
    """Feature-hashed unigrams + bigrams, sublinear tf, L2-normalized. For tests and CI only."""

    _TOKEN = re.compile(r"[a-z0-9]+")

    def __init__(self, dim: int = 384) -> None:
        super().__init__()
        self.dim = dim
        self.embedder_id = f"hashing/blake2b-uni-bigram/v1/{dim}/{_prep_hash('lower', 'sublinear')}"

    def _vec(self, text: str) -> np.ndarray:
        toks = self._TOKEN.findall(text.lower())
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]
        v = np.zeros(self.dim, dtype=np.float32)
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        v = np.sign(v) * np.log1p(np.abs(v))
        n = np.linalg.norm(v)
        return v / n if n else v

    def _encode(self, texts: list[str], kind: Kind) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])


class SentenceTransformerEmbedder(_CachedEmbedder):
    def __init__(self, batch_size: int = 64) -> None:
        super().__init__()
        from sentence_transformers import SentenceTransformer

        pin = PINS["embedder"]
        # CPU keeps runs repeatable across machines; MPS gives slightly different floats.
        limit_torch_threads()
        self._model = SentenceTransformer(local_path("embedder"), device="cpu")
        self.dim = int(self._model.get_embedding_dimension())
        self._bs = batch_size
        prep = _prep_hash(BGE_QUERY_PREFIX, "normalize", self._model.max_seq_length)
        self.embedder_id = f"sentence-transformers/{pin.repo}/{pin.revision[:12]}/{self.dim}/{prep}"

    def _encode(self, texts: list[str], kind: Kind) -> np.ndarray:
        if kind == "query":
            texts = [BGE_QUERY_PREFIX + t for t in texts]
        return self._model.encode(
            texts,
            batch_size=self._bs,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)


@lru_cache(maxsize=4)
def make_embedder(name: str = "bge") -> Embedder:
    if name == "hashing":
        return HashingEmbedder()
    if name == "bge":
        return SentenceTransformerEmbedder()
    raise ValueError(f"unknown embedder {name!r}")
