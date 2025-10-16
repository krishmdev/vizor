"""Read answers back from a run directory: pretty-print one, or recompute every metric."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vizor.attribution.citations import parse_answer
from vizor.metrics.impression import decay_weights, impressions
from vizor.metrics.visibility import answer_rows, source_labels
from vizor.runstore import load_prompts, read_jsonl_gz
from vizor.types import Answer


def load_answer(run_dir: Path, arm: str = "baseline", query_id: str | None = None, sample: int = 0):
    prompts = load_prompts(run_dir / "prompts.jsonl.gz")
    for r in read_jsonl_gz(run_dir / "responses.jsonl.gz"):
        if (
            r["arm"] == arm
            and r["sample"] == sample
            and (query_id is None or r["query_id"] == query_id)
        ):
            return Answer.from_dict(r), prompts.get(r["prompt_hash"], "")
    raise KeyError(f"no answer for arm={arm} query={query_id} sample={sample}")


def format_answer(ans: Answer, prompt: str = "") -> str:
    n = len(ans.sources)
    imp = impressions(ans.sentences, n)
    labels = source_labels(ans)
    d = decay_weights(len(ans.sentences))
    q = next((ln[10:] for ln in prompt.splitlines() if ln.startswith("Question: ")), ans.query_id)
    lines = [f"Q: {q}   [{ans.model}, sample {ans.sample}, seed {ans.seed}]", "", "Sources:"]
    for s, lab, share in zip(ans.sources, labels, imp.pwc, strict=True):
        lines.append(f"  [{s.position}] {lab:<10} pawc={share:6.1%}  {s.domain:<22} {s.url}")
    lines += ["", "Sentences (pos | words | decay | citations):"]
    for s, w in zip(ans.sentences, d, strict=True):
        cites = "".join(f"[{c}]" for c in s.citations) or "-"
        lines.append(f"  {s.pos:>2} | {s.n_words:>3} | {w:.3f} | {cites:<8} {s.text}")
    if ans.hallucinated_citations:
        lines.append(f"\nhallucinated citation indices: {ans.hallucinated_citations}")
    return "\n".join(lines)


def recompute_rows(run_dir: Path, rtol: float = 1e-6) -> tuple[bool, int]:
    """Parse the raw answer text again and recompute attribution rows; compare with rows.csv.gz.
    Sentiment columns are skipped (they need the sentiment model)."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    from vizor.config import Config
    from vizor.ingest.corpus import load_project

    cfg = Config.model_validate(manifest["config"])
    domains = load_project(cfg.project_path()).domains
    rows = []
    for r in read_jsonl_gz(run_dir / "responses.jsonl.gz"):
        ans = Answer.from_dict(r)
        parsed = parse_answer(ans.text, len(ans.sources))
        if parsed.sentences != ans.sentences:
            return False, 0
        for row in answer_rows(ans, domains, cfg.decay):
            row["arm"] = r["arm"]
            rows.append(row)
    new = pd.DataFrame(rows)
    path = run_dir / ("rows.csv.gz" if (run_dir / "rows.csv.gz").exists() else "metrics.csv")
    old = pd.read_csv(path)
    cols = ["imp_pwc", "imp_word", "imp_pos", "c_share", "n_markers"]
    key = ["arm", "query_id", "sample", "domain"]
    m = old.merge(new, on=key, suffixes=("_old", "_new"))
    ok = len(m) == len(new) == len(old) and all(
        np.allclose(m[f"{c}_old"], m[f"{c}_new"], rtol=rtol, atol=1e-6) for c in cols
    )
    return ok, len(new)
