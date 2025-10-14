"""The retrieval half of the MCA cascade: bi-encoder recall -> MaxP doc grouping -> cross-encoder
rerank -> top-n selection under a RetrievalPolicy.

1. Vector index top-k passages (k=50) for the query (NumPy on macOS, FAISS elsewhere).
2. Group by document, scoring each by its best passage (MaxP), keep the top 15 documents.
3. Cross-encoder scores (query, best passage); final score = sigmoid(logit), or the cosine when
   reranking is off.
4. Apply the policy (boost, ordering) and keep the top n=5 sources.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from vizor.embed import Embedder
from vizor.optimize.retrieval_policy import RELEVANCE, RetrievalPolicy
from vizor.retrieve.chunk import Passage, passages
from vizor.retrieve.index import VectorIndex
from vizor.retrieve.rerank import Reranker
from vizor.types import SourceDoc, stable_seed


@dataclass(frozen=True)
class Candidate:
    doc: SourceDoc
    best_passage: Passage
    retrieval_score: float
    rerank_score: float
    final_score: float


@dataclass(frozen=True)
class Selection:
    sources: list[Candidate]
    candidates: list[Candidate]
    forced: bool | None = None  # for target_at: whether a target doc was available to place


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


class Cascade:
    def __init__(
        self,
        docs: Iterable[SourceDoc],
        embedder: Embedder,
        reranker: Reranker,
        k_passages: int = 50,
        k_docs: int = 15,
        n_sources: int = 5,
        backend: str = "auto",
    ) -> None:
        self.docs: dict[str, SourceDoc] = {d.doc_id: d for d in docs}
        self.embedder = embedder
        self.reranker = reranker
        self.k_passages = k_passages
        self.k_docs = k_docs
        self.n_sources = n_sources
        self.backend = backend
        self.passages: dict[str, Passage] = {}
        self.by_doc: dict[str, list[Passage]] = {}
        for d in self.docs.values():
            ps = passages(d)
            self.by_doc[d.doc_id] = ps
            self.passages.update({p.passage_id: p for p in ps})
        ids = list(self.passages)
        self.index = VectorIndex.build(
            embedder, ids, [self.passages[i].text for i in ids], backend=backend
        )
        self._vec = dict(zip(ids, self.index.vectors, strict=True))

    def with_docs(self, changed: Mapping[str, SourceDoc]) -> Cascade:
        """A new cascade over the corpus with some documents replaced. Unchanged passages hit the
        embedder's cache, so only edited text is re-embedded."""
        docs = [changed.get(i, d) for i, d in self.docs.items()]
        return Cascade(
            docs,
            self.embedder,
            self.reranker,
            self.k_passages,
            self.k_docs,
            self.n_sources,
            self.backend,
        )

    @property
    def embedder_id(self) -> str:
        return self.embedder.embedder_id

    def query_vector(self, text: str) -> np.ndarray:
        return self.embedder.encode([text], kind="query")[0]

    def passage_scores(self, qvec: np.ndarray, doc_id: str) -> list[tuple[Passage, float]]:
        return [(p, float(self._vec[p.passage_id] @ qvec)) for p in self.by_doc[doc_id]]

    def candidates(self, query: str, policy: RetrievalPolicy = RELEVANCE) -> list[Candidate]:
        qvec = self.query_vector(query)
        hits = self.index.search(qvec, self.k_passages, self.embedder_id)
        best: dict[str, tuple[str, float]] = {}
        for pid, score in hits:
            doc_id = self.passages[pid].doc_id
            pre = policy.target_boost and policy.boost_stage == "pre"
            if pre and self.docs[doc_id].role == "target":
                score += policy.target_boost
            if doc_id not in best or score > best[doc_id][1]:
                best[doc_id] = (pid, score)
        top = sorted(best.items(), key=lambda kv: -kv[1][1])[: self.k_docs]
        texts = [self.passages[pid].text for _, (pid, _) in top]
        logits = self.reranker.score(query, texts)
        out = []
        for (doc_id, (pid, rscore)), logit in zip(top, logits, strict=True):
            final = rscore if np.isnan(logit) else _sigmoid(logit)
            doc = self.docs[doc_id]
            if policy.target_boost and policy.boost_stage == "post" and doc.role == "target":
                final += policy.target_boost
            out.append(Candidate(doc, self.passages[pid], rscore, float(logit), final))
        out.sort(key=lambda c: (-c.final_score, c.doc.doc_id))
        return out

    def select(self, query: str, policy: RetrievalPolicy = RELEVANCE) -> Selection:
        cands = self.candidates(query, policy)
        n = self.n_sources
        if policy.order == "target_at":
            target = next((c for c in cands if c.doc.role == "target"), None)
            if target is None:
                return Selection(cands[:n], cands, forced=False)
            # Exactly one target source, placed in the requested slot; the rest keep their order.
            others = [c for c in cands if c.doc.role != "target"][: n - 1]
            slot = min(max(1, policy.target_position or 1), len(others) + 1)
            return Selection([*others[: slot - 1], target, *others[slot - 1 :]], cands, forced=True)
        top = cands[:n]
        if policy.order == "reverse":
            top = top[::-1]
        elif policy.order == "random":
            rng = np.random.default_rng(stable_seed("order", query))
            top = [top[i] for i in rng.permutation(len(top))]
        return Selection(top, cands)
