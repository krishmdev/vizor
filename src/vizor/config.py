"""Experiment configuration (YAML) and the factories that turn it into an engine."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from vizor.models import project_root


class LLMConfig(BaseModel):
    backend: Literal["fake", "openai", "ollama"] = "fake"
    model: str = "gpt-4o-mini"
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int = 450
    workers: int = 1
    max_cost_usd: float | None = None


class SandboxConfig(BaseModel):
    arms: list[str] = Field(
        default_factory=lambda: [
            "noop",
            "aa_resample",
            "metadata",
            "faq_rewrite",
            "jsonld_insert",
            "internal_links",
            "stats_surface",
            "quote_surface",
            "keyword_stuffing",
            "engine:reverse",
        ]
    )
    llm_rewrites: bool = False
    allow_fabrication: bool = False
    position_sweep: list[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5])
    boost_sweep: list[float] = Field(default_factory=lambda: [0.0, 0.02, 0.05, 0.1])
    bootstrap: int = 5000


class BanditConfig(BaseModel):
    rounds: int = 2000
    runs: int = 20
    greedy_steps: int = 4


class Config(BaseModel):
    project: str = "data/demo/project.yaml"
    embedder: Literal["bge", "hashing"] = "bge"
    reranker: Literal["cross-encoder", "none"] = "cross-encoder"
    index: Literal["auto", "numpy", "faiss"] = "auto"
    sentiment: Literal["auto", "roberta", "vader"] = "auto"
    decay: Literal["paper", "reference"] = "paper"
    samples: int = 5
    queries: int | None = None
    seed: int = 0
    llm: LLMConfig = Field(default_factory=LLMConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    bandit: BanditConfig = Field(default_factory=BanditConfig)

    @classmethod
    def load(cls, path: str | Path | None) -> Config:
        if path is None:
            return cls()
        return cls.model_validate(yaml.safe_load(Path(path).read_text()) or {})

    def project_path(self) -> Path:
        p = Path(self.project)
        return p if p.is_absolute() else project_root() / p


def cache_dir() -> Path:
    return project_root() / ".cache"


def make_llm(cfg: Config, embedder=None):
    from vizor.generate.llm import CachedLLM, OllamaChat, OpenAIChat

    if cfg.llm.backend == "fake":
        from vizor.generate.fake_llm import FakeLLM

        assert embedder is not None
        return FakeLLM(embedder)
    if cfg.llm.backend == "openai":
        inner = OpenAIChat(cfg.llm.model, base_url=cfg.llm.base_url)
    else:
        inner = OllamaChat(cfg.llm.model, base_url=cfg.llm.base_url or "http://localhost:11434")
    return CachedLLM(inner, cache_dir() / "llm", cfg.llm.max_cost_usd)


def build(cfg: Config):
    """Project, docs, queries and a ready Engine for this config."""
    from vizor.embed import make_embedder
    from vizor.generate.engine import Engine
    from vizor.ingest.corpus import load_docs, load_project, load_queries
    from vizor.retrieve.cascade import Cascade
    from vizor.retrieve.rerank import make_reranker

    project = load_project(cfg.project_path())
    docs = load_docs(project)
    queries = load_queries(project.queries_path) if project.queries_path else []
    if cfg.queries:
        queries = _balanced(queries, cfg.queries)
    embedder = make_embedder(cfg.embedder)
    reranker = make_reranker(cfg.reranker)
    cascade = Cascade(docs, embedder, reranker, backend=cfg.index)
    llm = make_llm(cfg, embedder)
    engine = Engine(
        cascade,
        llm,
        temperature=cfg.llm.temperature,
        max_tokens=cfg.llm.max_tokens,
        base_seed=cfg.seed,
        workers=cfg.llm.workers,
    )
    return project, docs, queries, engine


def _balanced(queries, n: int):
    """First n queries taken round-robin across intents, so small runs still cover every intent."""
    by: dict[str, list] = {}
    for q in queries:
        by.setdefault(q.intent, []).append(q)
    out = []
    while len(out) < min(n, len(queries)):
        for group in by.values():
            if group and len(out) < n:
                out.append(group.pop(0))
    return out
