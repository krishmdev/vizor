"""Check Localhost AI's /v1/score against in-process mlx-lm scoring (MLXScorer) on one model.

The prompts are the ones `scripts/determinism_check.py` sends (the run's bench prompts with its
system prompt); the continuations are that check's answers, read from its determinism.json, and
every "[n]" marker in them is a site with candidates 1-5. Both scorers get the same messages,
continuation, sites and chat-template options, and the check passes when every candidate's
log-probability agrees to --tol. Needs mlx-lm, e.g.

    uv run --with mlx-lm==0.31.3 --with mlx==0.32.2 python scripts/score_equivalence.py \
        --config configs/gemma_e4b_study2.yaml --determinism <dir>/determinism.json \
        --repo <mlx checkpoint dir> --server-commit <localhost-ai HEAD> --out <dir>

Exit code 0 when all sites agree, 2 when any differ, 1 on errors.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from vizor.config import Config, build
from vizor.generate.scorer import LocalhostScorer, MLXScorer, Site

CANDIDATES = ("1", "2", "3", "4", "5")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--determinism", type=Path, required=True)
    ap.add_argument("--repo", required=True, help="MLX checkpoint directory or HF repo")
    ap.add_argument("--revision")
    ap.add_argument("--server-commit", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tol", type=float, default=1e-4)
    a = ap.parse_args()

    cfg = Config.load(a.config)
    answers = {r["query_id"]: r["text"] for r in json.loads(a.determinism.read_text())["alone"]}
    fake = Config.model_validate({**cfg.model_dump(), "llm": {"backend": "fake"}})
    _, _, queries, engine = build(fake)
    sc = cfg.score
    kw = dict(sc.chat_template_kwargs)
    server = LocalhostScorer(sc.model, sc.base_url, chat_template_kwargs=kw)
    local = MLXScorer(a.repo, a.revision, chat_template_kwargs=kw)
    pin = server.pin()
    if pin.get("commit") != a.server_commit:
        print(f"server reports commit {pin.get('commit')}, expected {a.server_commit}")
        return 1
    rows, worst = [], 0.0
    for q in queries:
        text = answers.get(q.query_id)
        if not text:
            continue
        prompt, _ = engine.build_prompt(q)
        messages = [{"role": "user", "content": prompt}]
        if cfg.llm.system_prompt:
            messages.insert(0, {"role": "system", "content": cfg.llm.system_prompt})
        sites = [Site(m.start() + 1, CANDIDATES) for m in re.finditer(r"\[\d+\]", text)]
        if not sites:
            continue
        s_res = server.score(messages, text, sites)
        m_res = local.score(messages, text, sites)
        diff = max(
            abs(s.logprobs[c] - m.logprobs[c])
            for s, m in zip(s_res.sites, m_res.sites, strict=True)
            for c in CANDIDATES
        )
        worst = max(worst, diff)
        rows.append({"query_id": q.query_id, "sites": len(sites), "max_abs_diff": diff})
    ok = bool(rows) and worst <= a.tol
    res = {
        "model": sc.model,
        "server_commit": a.server_commit,
        "server_pin": pin,
        "mlx_pin": local.pin(),
        "chat_template_kwargs": kw,
        "prompts": len(rows),
        "sites": sum(r["sites"] for r in rows),
        "max_abs_logprob_diff": worst,
        "tol": a.tol,
        "equivalent": ok,
        "rows": rows,
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "score_equivalence.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("rows", "server_pin", "mlx_pin")}))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
