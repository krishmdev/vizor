"""Study 2 validation gate (docs/bench-design-study2.md): re-score the pre-registered Study 1
run with attribution propensity and check the four pre-registered criteria.

    uv run python scripts/validation_gate.py --run experiments/results/2026-09-05_bench-qwen3b \
        --config configs/study2_gate.yaml --out experiments/results/<date>_study2-gate \
        --server-commit <localhost-ai HEAD>

Writes ap_rows.jsonl.gz, ap_deltas.csv, ap_summary.md, ap_manifest.json, gate.json and gate.md
into --out. Exit code 0 when the gate passes, 3 when it fails (Study 2 then falls back to the
sampled citation share as its primary metric), 1 on errors. `--backend fake` runs the same code
offline with the deterministic FakeScorer, which checks the plumbing and says nothing about AP.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vizor.config import Config
from vizor.gate import run_gate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=Path, required=True, help="Study 1 results directory")
    ap.add_argument("--config", type=Path, default=Path("configs/study2_gate.yaml"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--server-commit", help="Localhost AI commit; the server must report it")
    ap.add_argument("--backend", choices=["localhost", "mlx", "fake"])
    a = ap.parse_args()
    cfg = Config.load(a.config)
    if a.backend:
        cfg.score.backend = a.backend
    if cfg.score.backend == "localhost":
        if not a.server_commit:
            ap.error("--server-commit is required for the Localhost AI scorer")
        cfg.score.server_meta["commit"] = a.server_commit
    cfg = Config.model_validate(cfg.model_dump())
    a.out.mkdir(parents=True, exist_ok=True)
    r = run_gate(a.run, cfg, a.out, log=lambda m: print(m, file=sys.stderr))
    return 0 if r["passed"] else 3


if __name__ == "__main__":
    sys.exit(main())
