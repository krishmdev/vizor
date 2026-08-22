"""Study 2's fallback primary analysis (docs/bench-design-study2.md, "Validation gate"): with the
gate failed, the primary metric is the sampled citation share, with the same page units and
weighting, the same two tests and Holm over the three arms as the AP analysis would have had.

For each arm: per-page citation-share deltas (arm minus baseline, pp, from per_query.csv), the
unweighted mean over pages with a 95% percentile bootstrap interval, the exact two-sided
Wilcoxon signed-rank p and the two-sided sign-flip permutation p, each Holm-adjusted over the
arms. An arm counts only if both adjusted p values are below 0.05, and a guarded rewrite arm
with fewer accepted pages than `sandbox.rewrite_min_accepted` is inconclusive. The MDE comes from
the A/A re-sample's page-level SD. Writes sampled_primary.json and sampled_primary.md.

    uv run python scripts/sampled_primary.py <run dir> --config <config>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from vizor.config import Config
from vizor.gate import c_share_page_deltas
from vizor.optimize.stats import bootstrap_ci, holm, mde, sign_flip_p, wilcoxon_exact_p
from vizor.scoring import load_units

PRIMARY = ["answer_first", "evidence_surface_llm", "faq_rewrite_v2"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", default=PRIMARY)
    ap.add_argument("--controls", nargs="+", default=["aa_resample", "noop"])
    a = ap.parse_args()
    cfg = Config.load(a.config)
    units = load_units(a.run)
    manifest = json.loads((a.run / "manifest.json").read_text())
    guard = manifest.get("rewrite_guard") or {}
    b, draws = cfg.sandbox.bootstrap, cfg.score.draws

    def one(arm: str) -> dict:
        d = c_share_page_deltas(a.run, arm, units).to_numpy()
        est, lo, hi = bootstrap_ci(d, b=b)
        return {
            "arm": arm,
            "n_pages": len(d),
            "d_c_share_pp": est,
            "lo": lo,
            "hi": hi,
            "sd_page_pp": float(np.std(d, ddof=1)),
            "p_wilcoxon": wilcoxon_exact_p(d),
            "p_perm": sign_flip_p(d, draws=draws),
            "n_zero_pages": int(np.sum(np.abs(d) < 1e-9)),
        }

    rows = [one(x) for x in a.arms]
    for key in ("p_wilcoxon", "p_perm"):
        for r, p in zip(rows, holm([r[key] for r in rows]), strict=True):
            r[key + "_holm"] = p
    need = cfg.sandbox.rewrite_min_accepted
    for r in rows:
        g = guard.get(r["arm"])
        accepted = g["pages"] - g["rejected"] if g else None
        r["accepted_pages"] = accepted
        r["inconclusive"] = bool(g and need and accepted < need)
        r["significant"] = (
            r["p_wilcoxon_holm"] < 0.05 and r["p_perm_holm"] < 0.05 and not r["inconclusive"]
        )
    controls = [one(x) for x in a.controls]
    aa = next((c for c in controls if c["arm"] == "aa_resample"), None)
    out = {
        "run": a.run.name,
        "metric": "c_share",
        "weighting": "page",
        "family": a.arms,
        "arms": rows,
        "controls": controls,
        "mde_pp": mde(aa["sd_page_pp"], aa["n_pages"], len(a.arms), t=True) if aa else None,
        "mde_method": "A/A page-level SD, t quantiles, 80% power, two-sided alpha 0.05 / family",
    }
    (a.run / "sampled_primary.json").write_text(json.dumps(out, indent=1) + "\n")
    lines = [
        f"# Study 2 primary (sampled citation share, page units) on {a.run.name}",
        "",
        "| Arm | dC-SoV pp [95% CI] | Holm p (Wilcoxon) | Holm p (sign-flip) | Pages | Zero "
        "pages | Verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        verdict = (
            "inconclusive"
            if r["inconclusive"]
            else ("effect" if r["significant"] else "no effect shown")
        )
        lines.append(
            f"| `{r['arm']}` | {r['d_c_share_pp']:+.1f} [{r['lo']:+.1f}, {r['hi']:+.1f}] | "
            f"{r['p_wilcoxon_holm']:.3f} | {r['p_perm_holm']:.3f} | {r['n_pages']} | "
            f"{r['n_zero_pages']} | {verdict} |"
        )
    for c in controls:
        lines.append(
            f"| `{c['arm']}` (control) | {c['d_c_share_pp']:+.1f} [{c['lo']:+.1f}, "
            f"{c['hi']:+.1f}] | {c['p_wilcoxon']:.3f} (raw) | {c['p_perm']:.3f} (raw) | "
            f"{c['n_pages']} | {c['n_zero_pages']} | control |"
        )
    if out["mde_pp"] is not None:
        lines += ["", f"MDE (A/A page SD {aa['sd_page_pp']:.2f} pp): {out['mde_pp']:.1f} pp."]
    (a.run / "sampled_primary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
