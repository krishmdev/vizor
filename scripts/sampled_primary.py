"""Study 2's fallback primary analysis (docs/bench-design-study2.md, "Validation gate"): with the
gate failed, the primary metric is the sampled citation share, with the same page units and
weighting, the same two tests and Holm over the three arms as the AP analysis would have had.

For each arm: per-page citation-share deltas (arm minus baseline, pp, from per_query.csv), the
unweighted mean over pages with a 95% percentile bootstrap interval, the exact two-sided
Wilcoxon signed-rank p and the two-sided sign-flip permutation p, each Holm-adjusted over the
arms. An arm counts only if both adjusted p values are below 0.05, and a guarded rewrite arm
with fewer accepted pages than `sandbox.rewrite_min_accepted` is inconclusive. The MDE comes from
the A/A re-sample's page-level SD. Writes sampled_primary.json and sampled_primary.md.

Two additions are reported next to the pre-registered numbers and change none of them:
- `p_perm_exact`: the sign-flip p over all 2^n sign patterns (meet in the middle), next to the
  pre-registered Monte Carlo value.
- `exploratory_reference`: not pre-registered. The same page deltas and both tests (with Holm
  over the arms) with the A/A re-sample, and then the mean of baseline and A/A, as the
  reference instead of the baseline alone. The baseline is one draw of 3 samples per query, so a
  lucky baseline shifts every arm's delta the same way; this shows how much the verdicts lean on
  that one draw.

    uv run python scripts/sampled_primary.py <run dir> --config <config> [--gate <gate.json>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from vizor.config import Config
from vizor.gate import c_share_page_deltas
from vizor.optimize.stats import (
    bootstrap_ci,
    holm,
    mde,
    sign_flip_exact_p,
    sign_flip_p,
    wilcoxon_exact_p,
)
from vizor.scoring import load_units

PRIMARY = ["answer_first", "evidence_surface_llm", "faq_rewrite_v2"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", default=PRIMARY)
    ap.add_argument("--controls", nargs="+", default=["aa_resample", "noop"])
    ap.add_argument("--gate", type=Path, help="the validation gate's gate.json, to record why")
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
            "p_perm_exact": sign_flip_exact_p(d),
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
    if a.gate:
        g = json.loads(a.gate.read_text())
        out["gate"] = {
            "dir": a.gate.parent.name,
            "passed": bool(g["passed"]),
            "failed": [k for k, v in g["criteria"].items() if not v["pass"]],
        }
    if aa is not None:
        out["exploratory_reference"] = exploratory_reference(a.run, a.arms, units)
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
    lines += [
        "",
        f"Exact sign-flip p (raw, all 2^{rows[0]['n_pages']} sign patterns; the table uses the "
        f"pre-registered Monte Carlo value with {draws} draws): "
        + ", ".join(f"`{r['arm']}` {r['p_perm_exact']:.4f}" for r in rows + controls)
        + ".",
    ]
    if "exploratory_reference" in out:
        lines += [
            "",
            "## Exploratory, not pre-registered: the choice of reference",
            "",
            "Same page deltas and tests, Holm over the three arms, with the A/A re-sample or the "
            "mean of baseline and A/A as the reference instead of the baseline alone. This does "
            "not replace the verdicts above.",
            "",
            "| Reference | Arm | dC-SoV pp | p Wilcoxon (Holm) | p sign-flip exact (Holm) | "
            "Both Holm p < 0.05 |",
            "|---|---|---|---|---|---|",
        ]
        for ref, rs in out["exploratory_reference"].items():
            for r in rs:
                lines.append(
                    f"| {ref} | `{r['arm']}` | {r['d_c_share_pp']:+.1f} | "
                    f"{r['p_wilcoxon']:.3f} ({r['p_wilcoxon_holm']:.3f}) | "
                    f"{r['p_perm_exact']:.3f} ({r['p_perm_exact_holm']:.3f}) | "
                    f"{'yes' if r['both_holm'] else 'no'} |"
                )
    (a.run / "sampled_primary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def exploratory_reference(run: Path, arms: list[str], units: dict[str, str]) -> dict:
    """Page deltas of each arm against the A/A arm, and against the mean of baseline and A/A.
    Both are linear in the per-page means, so arm - aa = (arm - base) - (aa - base) and
    arm - (base + aa) / 2 = (arm - base) - (aa - base) / 2."""
    aa = c_share_page_deltas(run, "aa_resample", units)
    out = {}
    for name, w in (("A/A re-sample", 1.0), ("mean of baseline and A/A", 0.5)):
        rs = []
        for arm in arms:
            d0 = c_share_page_deltas(run, arm, units)
            d = (d0 - w * aa.loc[d0.index]).to_numpy()
            rs.append(
                {
                    "arm": arm,
                    "d_c_share_pp": float(d.mean()),
                    "p_wilcoxon": wilcoxon_exact_p(d),
                    "p_perm_exact": sign_flip_exact_p(d),
                }
            )
        for key in ("p_wilcoxon", "p_perm_exact"):
            for r, p in zip(rs, holm([r[key] for r in rs]), strict=True):
                r[key + "_holm"] = p
        for r in rs:
            r["both_holm"] = r["p_wilcoxon_holm"] < 0.05 and r["p_perm_exact_holm"] < 0.05
        out[name] = rs
    return out


if __name__ == "__main__":
    sys.exit(main())
