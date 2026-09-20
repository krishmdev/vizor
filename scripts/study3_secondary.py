"""Study 3's secondary metrics (docs/bench-design-study3.md), reported with no claims.

For each page arm, on the 24 page units (unweighted mean of page means, 95% percentile
bootstrap interval, raw exact Wilcoxon p):
- the named rate (the answer names the target brand), arm minus baseline;
- the content-vs-rank split: the full-mode citation-share delta, the content-only twin's delta
  (sources and order pinned to the baseline) and the rank-mediated rest (full minus twin);
- from retrieval.csv: the focus page's retrieval rate (shown among the 5 sources) and mean
  candidate rank, arm minus baseline, and the passage-selection share (how often an edited or
  new passage of the page is among the passages shown).

Exploratory, not pre-registered: the same split on sample 0 alone. The twins have 1 sample and
the full arms 2, so full minus twin also contains half the difference between samples 1 and 0
wherever the two prompts are identical; on sample 0 both use the same seed, so the rank-mediated
part is exactly 0 wherever retrieval did not change.

    uv run python scripts/study3_secondary.py <run dir> --config <config>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from vizor.config import Config
from vizor.optimize.stats import bootstrap_ci, unit_means, wilcoxon_exact_p
from vizor.scoring import load_units

ARMS = ["fact_passage", "entity_anchor", "retrieval_meta"]


def page_deltas(frame: pd.DataFrame, col: str, arm: str, ref: str, units: dict) -> pd.Series:
    """Per-page mean of (arm - ref) over the page's queries, for a per-query frame with `arm`,
    `query_id` and `col`. Queries missing from either side drop out."""
    a = frame[frame["arm"] == arm].set_index("query_id")[col].astype(float)
    b = frame[frame["arm"] == ref].set_index("query_id")[col].astype(float)
    d = (a - b.reindex(a.index)).dropna()
    if d.empty:
        return pd.Series(dtype=float)
    labels, means = unit_means(d.to_numpy(), np.array([units[q] for q in d.index]))
    return pd.Series(means, index=labels)


def describe(d: pd.Series, scale: float, b: int) -> dict:
    x = d.to_numpy() * scale
    if len(x) == 0:
        return {"n_pages": 0, "est": None, "lo": None, "hi": None, "p_wilcoxon": None}
    est, lo, hi = bootstrap_ci(x, b=b)
    return {"n_pages": len(x), "est": est, "lo": lo, "hi": hi, "p_wilcoxon": wilcoxon_exact_p(x)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", default=ARMS)
    a = ap.parse_args()
    cfg = Config.load(a.config)
    b = cfg.sandbox.bootstrap
    units = load_units(a.run)
    pq = pd.read_csv(a.run / "per_query.csv")
    ret = pd.read_csv(a.run / "retrieval.csv")
    arms_present = set(pq["arm"])
    out: dict = {"run": a.run.name, "arms": {}}
    for arm in a.arms:
        r: dict = {
            "named_pp": describe(page_deltas(pq, "mentioned", arm, "baseline", units), 100, b),
            "full_pp": describe(page_deltas(pq, "c_share", arm, "baseline", units), 100, b),
        }
        twin = f"content:{arm}"
        if twin in arms_present:
            r["content_pp"] = describe(page_deltas(pq, "c_share", twin, "baseline", units), 100, b)
            r["rank_pp"] = describe(page_deltas(pq, "c_share", arm, twin, units), 100, b)
        r["retrieved_pp"] = describe(
            page_deltas(ret, "focus_retrieved", arm, "baseline", units), 100, b
        )
        r["rank_delta"] = describe(page_deltas(ret, "focus_rank", arm, "baseline", units), 1, b)
        for name in (arm, twin):
            sub = ret[ret["arm"] == name]
            if sub.empty:
                continue
            shown = sub[sub["focus_retrieved"].astype(bool)]
            r.setdefault("passages", {})[name] = {
                "focus_retrieved": float(sub["focus_retrieved"].mean()),
                "focus_rank_mean": float(sub["focus_rank"].mean()),
                "shown_new_all": float(sub["shown_new"].astype(float).mean()),
                "shown_new_if_shown": (
                    float(shown["shown_new"].astype(float).mean()) if len(shown) else None
                ),
                "n_shown_queries": int(len(shown)),
            }
        out["arms"][arm] = r
    base = ret[ret["arm"] == "baseline"]
    out["baseline"] = {
        "focus_retrieved": float(base["focus_retrieved"].mean()),
        "focus_rank_mean": float(base["focus_rank"].mean()),
    }
    rows_path = a.run / "rows.csv.gz"
    if rows_path.exists():
        from vizor.optimize.sandbox import target_frame

        rows = pd.read_csv(rows_path)
        s0 = []
        for arm in rows["arm"].unique():
            t = target_frame(rows[rows["arm"] == arm])
            s0.append(t[t["sample"] == 0][["query_id", "c_share"]].assign(arm=arm))
        s0 = pd.concat(s0, ignore_index=True)
        for arm in a.arms:
            twin = f"content:{arm}"
            if twin not in arms_present:
                continue
            out["arms"][arm]["sample0"] = {
                "full_pp": describe(page_deltas(s0, "c_share", arm, "baseline", units), 100, b),
                "content_pp": describe(page_deltas(s0, "c_share", twin, "baseline", units), 100, b),
                "rank_pp": describe(page_deltas(s0, "c_share", arm, twin, units), 100, b),
            }
    (a.run / "study3_secondary.json").write_text(json.dumps(out, indent=1) + "\n")

    def f(x: dict, unit: str = "pp") -> str:
        if x.get("est") is None:
            return "n/a"
        return f"{x['est']:+.1f} [{x['lo']:+.1f}, {x['hi']:+.1f}] {unit} (p {x['p_wilcoxon']:.3f})"

    lines = [
        f"# Study 3 secondary metrics on {a.run.name} (no claims; raw p values)",
        "",
        f"Baseline: focus page shown in {100 * out['baseline']['focus_retrieved']:.0f}% of "
        f"queries, mean candidate rank {out['baseline']['focus_rank_mean']:.2f}.",
        "",
        "| Arm | Named rate | Full dC-SoV | Content-only | Rank-mediated | Retrieved | Rank |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm, r in out["arms"].items():
        lines.append(
            f"| `{arm}` | {f(r['named_pp'])} | {f(r['full_pp'])} | "
            f"{f(r['content_pp']) if 'content_pp' in r else 'n/a'} | "
            f"{f(r['rank_pp']) if 'rank_pp' in r else 'n/a'} | {f(r['retrieved_pp'])} | "
            f"{f(r['rank_delta'], 'ranks')} |"
        )
    lines += [
        "",
        "Passage selection: share of queries with an edited or new passage of the page among "
        "those shown (all queries; queries where the page is shown).",
        "",
        "| Set | Page shown | Mean rank | Edited/new shown (all) | Edited/new shown (if shown) |",
        "|---|---|---|---|---|",
    ]
    for r in out["arms"].values():
        for name, p in r.get("passages", {}).items():
            ifs = p["shown_new_if_shown"]
            lines.append(
                f"| `{name}` | {100 * p['focus_retrieved']:.0f}% | {p['focus_rank_mean']:.2f} | "
                f"{100 * p['shown_new_all']:.0f}% | "
                f"{'n/a' if ifs is None else f'{100 * ifs:.0f}%'} ({p['n_shown_queries']}) |"
            )
    if any("sample0" in r for r in out["arms"].values()):
        lines += [
            "",
            "Exploratory, not pre-registered: the content-vs-rank split on sample 0 alone (the "
            "twins' only sample; the same seed in every arm, so the rank-mediated part is exactly "
            "0 wherever retrieval did not change).",
            "",
            "| Arm | Full dC-SoV (sample 0) | Content-only | Rank-mediated |",
            "|---|---|---|---|",
        ]
        for arm, r in out["arms"].items():
            if "sample0" in r:
                z = r["sample0"]
                lines.append(
                    f"| `{arm}` | {f(z['full_pp'])} | {f(z['content_pp'])} | {f(z['rank_pp'])} |"
                )
    (a.run / "study3_secondary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
