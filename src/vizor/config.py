"""Experiment configuration (YAML) and the factories that turn it into an engine."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from vizor.models import project_root


class LLMConfig(BaseModel):
    # "openai_compat": a local OpenAI-compatible server (no key, no pricing, no spend cap)
    backend: Literal["fake", "openai", "ollama", "openai_compat"] = "fake"
    model: str = "gpt-4o-mini"
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int = 450
    workers: int = 1
    max_cost_usd: float | None = None
    # Optional system message sent before the GEO user prompt (e.g. to get a small local model to
    # use [n] markers at all). Recorded in the manifest; empty for the OpenAI run.
    system_prompt: str = ""
    # Free-form facts about the serving stack (server name, commit, preset), copied into the
    # manifest with the rest of the config. For openai_compat they are part of the cache key,
    # and `vizor experiment --server-commit` fills in "commit" (the run refuses to start without).
    server_meta: dict[str, str] = Field(default_factory=dict)


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
    # Run the position and boost sweeps on a balanced subset of this many queries (None = all).
    sweep_queries: int | None = None
    # The metric the Holm verdict is computed on: PAWC share, citation share (share of the
    # answer's valid markers), or whether the answer names the target brand.
    primary_metric: Literal["imp_pwc", "c_share", "mentioned"] = "imp_pwc"
    # "query": estimates weight queries equally and the test ranks page means (Study 1).
    # "page": estimate, interval and test all use the unweighted mean of page means (Study 2).
    weighting: Literal["query", "page"] = "query"
    # For guarded LLM rewrites: fewer accepted pages than this makes the arm inconclusive (its
    # comparisons are reported but can support no claim). 0 disables the rule.
    rewrite_min_accepted: int = 0


class RenderConfig(BaseModel):
    # Which passages each source shows in the prompt (see vizor.generate.prompt.choose_passages).
    passage_policy: Literal["query-top3", "body-top3", "top2+faq1"] = "query-top3"


class Comparison(BaseModel):
    """One AP comparison: prompt set `set` against prompt set `ref`, Holm-adjusted within
    `family`. Only the `primary` family can support a claim."""

    name: str
    set: str
    ref: str = "baseline"
    family: str = "secondary"


class ScoreConfig(BaseModel):
    """Teacher-forced attribution-propensity scoring (`vizor score`)."""

    backend: Literal["fake", "localhost", "mlx"] = "fake"
    model: str = "qwen2.5-3b-mlx4"
    revision: str | None = None
    base_url: str = "http://127.0.0.1:8431"
    chat_template_kwargs: dict = Field(default_factory=dict)
    # Facts about the server copied into the scorer pin (and so the cache key). `vizor score
    # --server-commit` fills in "commit"; the server must report the same one.
    server_meta: dict[str, str] = Field(default_factory=dict)
    refs: list[int] = Field(default_factory=lambda: [0, 1])
    # Prompt sets to score (empty: those the comparisons name, or every set in the run).
    sets: list[str] = Field(default_factory=list)
    comparisons: list[Comparison] = Field(default_factory=list)
    # Prompt-only sets `vizor experiment` builds for scoring without sampling answers:
    # extra arms (e.g. content-only twins) and extra passage policies.
    prompt_arms: list[str] = Field(default_factory=list)
    policies: list[str] = Field(default_factory=list)
    bootstrap: int = 5000
    draws: int = 20000
    workers: int = 1


class GateConfig(BaseModel):
    """The Study 2 validation gate (`scripts/validation_gate.py`): comparison names refer to
    `score.comparisons`, arm names to the re-scored run."""

    slot: str = "slot 5 vs 1"
    faq_ap: str = "pinned:faq_rewrite"
    faq_c_share: str = "content:faq_rewrite"
    # AP comparison name -> the run's arm whose citation-share deltas it is correlated with
    pairs: dict[str, str] = Field(default_factory=dict)
    slot_alpha: float = 0.01
    # criterion 3: C-SoV's relative precision may be at most this fraction of AP's
    ci_ratio: float = 0.6
    # criterion 4: threshold on rho / sqrt(rel), plus rho > 0 with one-sided p < spearman_alpha
    spearman: float = 0.5
    rel: float | None = None
    spearman_alpha: float = 0.05
    mde_family: int = 3


class BanditConfig(BaseModel):
    rounds: int = 2000
    runs: int = 20
    greedy_steps: int = 4


class Config(BaseModel):
    # Heading for this run in the generated reports (empty: derived from the model).
    label: str = ""
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
    render: RenderConfig = Field(default_factory=RenderConfig)
    score: ScoreConfig = Field(default_factory=ScoreConfig)
    gate: GateConfig = Field(default_factory=GateConfig)
    # A separate model for LLM page rewrites (Study 2 runs them at temperature 0 through Localhost
    # AI). Unset: rewrites use the answer model, as in Study 1.
    rewriter: LLMConfig | None = None
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


def make_llm(cfg: Config, embedder=None, llm_cfg: LLMConfig | None = None):
    """The cached answer model for `cfg.llm`, or for `llm_cfg` when given (the rewriter)."""
    from vizor.generate.llm import CachedLLM, OllamaChat, OpenAIChat

    lc = llm_cfg or cfg.llm
    if lc.backend == "fake":
        from vizor.generate.fake_llm import FakeLLM

        assert embedder is not None
        return FakeLLM(embedder)
    if lc.backend == "openai":
        inner = OpenAIChat(lc.model, base_url=lc.base_url)
    elif lc.backend == "openai_compat":
        if not lc.base_url:
            raise ValueError("llm.base_url is required for backend openai_compat")
        inner = OpenAIChat(lc.model, base_url=lc.base_url, timeout=600, api_key="local")
        extra = {"backend": lc.backend, "base_url": lc.base_url, **lc.server_meta}
        return CachedLLM(inner, cache_dir() / "llm", lc.max_cost_usd, key_extra=extra)
    else:
        inner = OllamaChat(lc.model, base_url=lc.base_url or "http://localhost:11434")
    return CachedLLM(inner, cache_dir() / "llm", lc.max_cost_usd)


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
        system_prompt=cfg.llm.system_prompt,
        passage_policy=cfg.render.passage_policy,
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
