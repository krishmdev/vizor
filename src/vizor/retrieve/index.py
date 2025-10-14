"""Inner-product vector index over L2-normalized passage embeddings.

Two backends with identical results for exact search:
- `numpy`: normalized float32 matrix, argpartition top-k. The default on macOS, where faiss-cpu,
  torch and scikit-learn each bundle their own libomp and loading them together in one process
  segfaults. The demo corpus is a few hundred passages, so exact NumPy search costs nothing.
- `faiss`: IndexFlatIP (IVF past ~20K passages). The default on Linux; imported lazily.

Indexes are tagged with the embedder_id that produced them and refuse queries or loads from any
other embedder. `IndexStore` builds each new generation in a temp directory and swaps a pointer
file atomically, so a failure mid-build leaves the previous generation serving.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vizor.embed import Embedder

IVF_THRESHOLD = 20_000


class EmbedderMismatch(RuntimeError):
    pass


def default_backend() -> str:
    return os.environ.get("VIZOR_INDEX") or ("numpy" if sys.platform == "darwin" else "faiss")


def _faiss():
    if sys.platform == "darwin" and "torch" in sys.modules:
        raise RuntimeError(
            "refusing to load faiss in a process that already loaded torch on macOS "
            "(duplicate libomp crashes); use the numpy index backend"
        )
    import faiss

    return faiss


@dataclass
class VectorIndex:
    embedder_id: str
    dim: int
    ids: list[str]
    vectors: np.ndarray
    backend: str = "auto"

    def __post_init__(self) -> None:
        self.vectors = np.ascontiguousarray(self.vectors, dtype=np.float32)
        if self.backend == "auto":
            self.backend = default_backend()
        if self.backend not in ("numpy", "faiss"):
            raise ValueError(f"unknown index backend {self.backend!r}")
        self._faiss_index = None
        if self.backend == "faiss":
            faiss = _faiss()
            if len(self.ids) > IVF_THRESHOLD:
                nlist = int(np.sqrt(len(self.ids)))
                quant = faiss.IndexFlatIP(self.dim)
                idx = faiss.IndexIVFFlat(quant, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
                idx.train(self.vectors)
                idx.nprobe = max(1, nlist // 10)
            else:
                idx = faiss.IndexFlatIP(self.dim)
            if len(self.ids):
                idx.add(self.vectors)
            self._faiss_index = idx

    @classmethod
    def build(
        cls, embedder: Embedder, ids: Sequence[str], texts: Sequence[str], backend: str = "auto"
    ) -> VectorIndex:
        vecs = embedder.encode(list(texts), kind="passage")
        return cls(embedder.embedder_id, embedder.dim, list(ids), vecs, backend)

    def search(self, qvec: np.ndarray, k: int, embedder_id: str) -> list[tuple[str, float]]:
        if embedder_id != self.embedder_id:
            raise EmbedderMismatch(f"index built with {self.embedder_id}, query from {embedder_id}")
        if not self.ids:
            return []
        k = min(k, len(self.ids))
        q = np.ascontiguousarray(qvec.reshape(1, -1), dtype=np.float32)
        if self._faiss_index is not None:
            scores, idx = self._faiss_index.search(q, k)
            pairs = zip(idx[0], scores[0], strict=True)
        else:
            s = self.vectors @ q[0]
            part = np.argpartition(-s, k - 1)[:k] if k < len(s) else np.arange(len(s))
            top = part[np.lexsort((part, -s[part]))]
            pairs = zip(top, s[top], strict=True)
        return [(self.ids[i], float(sc)) for i, sc in pairs if i >= 0]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)
        meta = {"embedder_id": self.embedder_id, "dim": self.dim, "ids": self.ids}
        (path / "meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: Path, embedder_id: str, backend: str = "auto") -> VectorIndex:
        meta = json.loads((path / "meta.json").read_text())
        if meta["embedder_id"] != embedder_id:
            raise EmbedderMismatch(f"{path} holds {meta['embedder_id']}, wanted {embedder_id}")
        return cls(
            meta["embedder_id"], meta["dim"], meta["ids"], np.load(path / "vectors.npy"), backend
        )


class IndexStore:
    """Generations of an index under one directory, with an atomically swapped CURRENT pointer."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def current(self) -> Path | None:
        ptr = self.root / "CURRENT"
        return self.root / ptr.read_text().strip() if ptr.exists() else None

    def load_current(self, embedder_id: str, backend: str = "auto") -> VectorIndex | None:
        cur = self.current()
        if cur is None:
            return None
        return VectorIndex.load(cur, embedder_id, backend)

    def build(
        self, embedder: Embedder, ids: Sequence[str], texts: Sequence[str], backend: str = "auto"
    ) -> VectorIndex:
        gen = f"gen-{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}-{len(os.listdir(self.root))}"
        tmp = self.root / f".{gen}.tmp"
        try:
            index = VectorIndex.build(embedder, ids, texts, backend)
            index.save(tmp)
            tmp.rename(self.root / gen)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        ptr_tmp = self.root / ".CURRENT.tmp"
        ptr_tmp.write_text(gen)
        os.replace(ptr_tmp, self.root / "CURRENT")
        return index
