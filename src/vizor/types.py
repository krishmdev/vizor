from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

Role = Literal["target", "competitor"]


def stable_id(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def stable_seed(*parts: Any) -> int:
    """64-bit seed from arbitrary parts. Python's hash() is salted per process, so never use it."""
    h = hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little")


@dataclass(frozen=True)
class SourceDoc:
    doc_id: str
    url: str
    domain: str
    role: Role
    title: str
    meta_description: str
    headings: tuple[str, ...]
    body: str
    faq: tuple[tuple[str, str], ...] = ()
    jsonld: tuple[dict, ...] = ()
    links: tuple[tuple[str, str], ...] = ()
    provenance: dict = field(default_factory=dict)
    version: int = 0
    applied_transforms: tuple[str, ...] = ()

    @property
    def paragraphs(self) -> list[str]:
        return [p.strip() for p in self.body.split("\n\n") if p.strip()]

    @property
    def n_words(self) -> int:
        return len(self.body.split())

    def evolve(self, transform: str, **changes: Any) -> SourceDoc:
        return replace(
            self,
            version=self.version + 1,
            applied_transforms=(*self.applied_transforms, transform),
            **changes,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["headings"] = list(self.headings)
        d["faq"] = [list(x) for x in self.faq]
        d["jsonld"] = list(self.jsonld)
        d["links"] = [list(x) for x in self.links]
        d["applied_transforms"] = list(self.applied_transforms)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> SourceDoc:
        return cls(
            doc_id=d["doc_id"],
            url=d["url"],
            domain=d["domain"],
            role=d["role"],
            title=d.get("title", ""),
            meta_description=d.get("meta_description", ""),
            headings=tuple(d.get("headings", ())),
            body=d.get("body", ""),
            faq=tuple(tuple(x) for x in d.get("faq", ())),
            jsonld=tuple(d.get("jsonld", ())),
            links=tuple(tuple(x) for x in d.get("links", ())),
            provenance=d.get("provenance", {}),
            version=d.get("version", 0),
            applied_transforms=tuple(d.get("applied_transforms", ())),
        )


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    intent: str
    cluster_id: int = -1


@dataclass(frozen=True)
class SourceRef:
    position: int  # 1-based, as shown to the model
    doc_id: str
    domain: str
    url: str
    retrieval_score: float
    rerank_score: float
    final_score: float


@dataclass(frozen=True)
class CitedSentence:
    pos: int
    text: str
    n_words: int
    citations: tuple[int, ...]


@dataclass
class Answer:
    query_id: str
    sample: int
    seed: int
    model: str
    prompt_hash: str
    sources: list[SourceRef]
    text: str
    sentences: list[CitedSentence]
    hallucinated_citations: list[int]
    usage: dict = field(default_factory=dict)
    # citation-like text the parser could not map to a source index (e.g. "[Source A]")
    unparsed_markers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "sample": self.sample,
            "seed": self.seed,
            "model": self.model,
            "prompt_hash": self.prompt_hash,
            "sources": [asdict(s) for s in self.sources],
            "text": self.text,
            "sentences": [{**asdict(s), "citations": list(s.citations)} for s in self.sentences],
            "hallucinated_citations": self.hallucinated_citations,
            "usage": self.usage,
            "unparsed_markers": self.unparsed_markers,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Answer:
        return cls(
            query_id=d["query_id"],
            sample=d["sample"],
            seed=d["seed"],
            model=d["model"],
            prompt_hash=d["prompt_hash"],
            sources=[SourceRef(**s) for s in d["sources"]],
            text=d["text"],
            sentences=[
                CitedSentence(s["pos"], s["text"], s["n_words"], tuple(s["citations"]))
                for s in d["sentences"]
            ],
            hallucinated_citations=d.get("hallucinated_citations", []),
            usage=d.get("usage", {}),
            unparsed_markers=d.get("unparsed_markers", []),
        )
