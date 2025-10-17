from __future__ import annotations

from pydantic import BaseModel, Field


class AnswerRequest(BaseModel):
    query: str = Field(min_length=3, max_length=300)
    sample: int = 0
    policy: str = "relevance"
    arms: list[str] = Field(default_factory=list, description="page transforms to apply first")


class SourceOut(BaseModel):
    position: int
    domain: str
    role: str
    url: str
    title: str
    retrieval_score: float
    rerank_score: float
    final_score: float
    label: str
    pwc_share: float
    citations: int


class SentenceOut(BaseModel):
    pos: int
    text: str
    n_words: int
    decay: float
    citations: list[int]


class AnswerOut(BaseModel):
    query: str
    model: str
    is_fake_llm: bool
    seed: int
    text: str
    sources: list[SourceOut]
    sentences: list[SentenceOut]
    hallucinated_citations: list[int]


class JobRequest(BaseModel):
    queries: int | None = Field(default=8, ge=1, le=200)
    samples: int | None = Field(default=2, ge=1, le=10)
    arms: list[str] | None = None


class JobOut(BaseModel):
    job_id: str
    kind: str
    status: str
    out_dir: str
    error: str | None = None
