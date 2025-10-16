"""`vizor run`: baseline visibility for a project, written as a run directory."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

import vizor
from vizor.config import Config, build
from vizor.metrics.sentiment import make_sentiment
from vizor.metrics.visibility import domain_summary
from vizor.optimize.sandbox import Sandbox
from vizor.optimize.transforms import TransformContext
from vizor.runstore import git_commit, write_csv, write_jsonl_gz


def run_baseline(cfg: Config, out: Path, log=print) -> pd.DataFrame:
    t0 = time.time()
    project, docs, queries, engine = build(cfg)
    sentiment = make_sentiment(cfg.sentiment)
    ctx = TransformContext({d.doc_id: d for d in docs}, queries, engine.cascade.embedder)
    sb = Sandbox(engine, queries, cfg.samples, project.domains, ctx, sentiment, cfg.decay)
    base = sb.run_baseline()
    out.mkdir(parents=True, exist_ok=True)
    prompts = {r.answer.prompt_hash: r.prompt for r in base.results}
    write_jsonl_gz(
        out / "responses.jsonl.gz",
        ({"arm": "baseline", **r.answer.to_dict()} for r in base.results),
    )
    write_jsonl_gz(
        out / "prompts.jsonl.gz", ({"prompt_hash": h, "prompt": p} for h, p in prompts.items())
    )
    write_csv(out / "metrics.csv", base.rows)
    summary = domain_summary(base.rows)
    write_csv(out / "domains.csv", summary)
    manifest = {
        "vizor_version": vizor.__version__,
        "git_commit": git_commit(),
        "config": cfg.model_dump(),
        "embedder_id": engine.cascade.embedder.embedder_id,
        "reranker_id": engine.cascade.reranker.reranker_id,
        "llm_model": engine.llm.model_id,
        "llm_is_fake": cfg.llm.backend == "fake",
        "sentiment_backend": sentiment.backend_id,
        "n_queries": len(queries),
        "samples": cfg.samples,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"baseline: {len(base.results)} answers in {time.time() - t0:.0f}s -> {out}")
    return summary
