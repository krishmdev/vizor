"""Generate the guarded LLM rewrites of every target page before a run (Study 2, amendment of
2026-09-18): each page goes through `evidence_surface_llm` once, with the run's own rewriter
config. Writes rewrites.json (the rewriter settings, and every page's raw output, verdict,
violations and thinking-token count), which a run reads back through `sandbox.frozen_rewrites`,
and reasoning.jsonl (a thinking rewriter's reasoning blocks, kept for provenance and never shown
to an answer model). Prints the accepted count against the pre-registered minimum
(`sandbox.rewrite_min_accepted`) and how many rewrites reached the thinking budget.

    uv run python scripts/pregenerate_rewrites.py --config configs/study2_qwen3b_localhost_ai.yaml \
        --server-commit <localhost-ai HEAD> --out <dir>

Exit code 0 when at least the minimum is accepted, 4 when fewer are (the arm is then
inconclusive), 1 on errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vizor.config import Config, make_llm
from vizor.embed import make_embedder
from vizor.ingest.corpus import load_docs, load_project
from vizor.optimize.transforms import TransformContext, evidence_surface_llm, rewriter_fingerprint


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--server-commit", help="Localhost AI commit (openai_compat rewriters)")
    a = ap.parse_args()
    cfg = Config.load(a.config)
    if cfg.rewriter is None:
        ap.error("the config has no rewriter")
    if cfg.rewriter.backend == "openai_compat":
        if not a.server_commit:
            ap.error("--server-commit is required for an openai_compat rewriter")
        cfg.rewriter.server_meta["commit"] = a.server_commit
    cfg = Config.model_validate(cfg.model_dump())
    embedder = make_embedder("hashing")  # the rewrite does not use the embedder
    llm = make_llm(cfg, embedder, cfg.rewriter)
    docs = load_docs(load_project(cfg.project_path()))
    events: list[dict] = []
    rc = cfg.rewriter
    ctx = TransformContext(
        {d.doc_id: d for d in docs},
        [],
        embedder,
        llm=llm,
        events=events,
        rewrite={"temperature": rc.temperature, "seed": rc.seed, "max_tokens": rc.max_tokens},
    )
    reasoning = []
    pages = []
    for d in docs:
        if d.role != "target":
            continue
        new, _ = evidence_surface_llm(d, ctx)
        e = dict(events[-1])
        reasoning.append({"doc_id": d.doc_id, "reasoning": e.pop("reasoning", "")})
        budget = rc.max_thinking_tokens
        e["hit_thinking_cap"] = bool(budget and (e.get("thinking_tokens") or 0) >= budget)
        pages.append(
            {"doc_id": d.doc_id, "url": d.url, **e, "body": new.body if new is not d else ""}
        )
    accepted = sum(p["accepted"] for p in pages)
    capped = sum(p["hit_thinking_cap"] for p in pages)
    need = cfg.sandbox.rewrite_min_accepted
    out = {
        "rewriter": {**rc.model_dump(), "model_id": llm.model_id},
        "fingerprint": rewriter_fingerprint(rc),
        "hit_thinking_cap": capped,
        "hit_thinking_cap_accepted": sum(p["hit_thinking_cap"] and p["accepted"] for p in pages),
        "pages": len(pages),
        "accepted": accepted,
        "min_accepted": need,
        "conclusive": accepted >= need,
        "rewrites": pages,
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "rewrites.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    with (a.out / "reasoning.jsonl").open("w") as f:
        for r in reasoning:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"evidence_surface_llm: {accepted} of {len(pages)} pages accepted (minimum {need})")
    print(f"{capped} of {len(pages)} rewrites reached the thinking budget")
    return 0 if accepted >= need else 4


if __name__ == "__main__":
    sys.exit(main())
