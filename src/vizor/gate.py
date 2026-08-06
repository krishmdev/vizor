"""The Study 2 validation gate: does attribution propensity (AP) reproduce Study 1?

Study 1's baseline samples 0 and 1 are the reference answers (text only), and Study 1's stored
prompts are re-scored with AP. The gate passes if all four pre-registered criteria hold
(`docs/bench-design-study2.md`, thresholds in the `gate:` section of the config):

1. AP detects the slot positive control: slot 5 minus slot 1 is negative with both Holm p
   values below `slot_alpha`.
2. AP gives the FAQ rewrite (content-pinned) a negative sign.
3. The page-level 95% CI half-width of that AP delta is at most `ci_ratio` times the
   half-width of the sampled citation-share delta for the same prompts (Study 1's
   `content:faq_rewrite`), both from unweighted page means with the same bootstrap.
4. Across Study 1's page arms, the Spearman correlation between per-page AP deltas and
   per-page citation-share deltas is at least `spearman`.

It also gives the Study 2 MDE range from the page-level SDs of the content-pinned AP deltas of
Study 1's page arms (t quantiles, 24 pages, the strictest Holm step over `mde_family` arms).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from vizor.optimize.stats import bootstrap_ci, mde, unit_means
from vizor.runstore import read_jsonl_gz
from vizor.scoring import Log, load_units, paired_deltas, rows_frame, score_run


def c_share_page_deltas(run_dir: Path, arm: str, units: dict[str, str]) -> pd.Series:
    """Per-page citation-share delta (pp), arm minus baseline, from the run's per-query means."""
    pq = pd.read_csv(run_dir / "per_query.csv")
    a = pq[pq["arm"] == arm].set_index("query_id")["c_share"]
    b = pq[pq["arm"] == "baseline"].set_index("query_id")["c_share"]
    d = (a - b.loc[a.index]).dropna() * 100
    labels, means = unit_means(d.to_numpy(), np.array([units[q] for q in d.index]))
    return pd.Series(means, index=labels)


def ap_page_deltas(df: pd.DataFrame, name: str, ref: str, units: dict[str, str]) -> pd.Series:
    q = paired_deltas(df, name, ref, units)
    return q.groupby("unit")["d"].mean() * 100


def evaluate(run_dir: Path, ap_dir: Path, cfg) -> dict:
    g = cfg.gate
    units = load_units(run_dir)
    res = pd.read_csv(ap_dir / "ap_deltas.csv").set_index("name")
    df = rows_frame(list(read_jsonl_gz(ap_dir / "ap_rows.jsonl.gz")))
    sets = {c.name: c for c in cfg.score.comparisons}

    slot = res.loc[g.slot]
    c1 = bool(
        slot.d_ap_pp < 0 and slot.p_perm_holm < g.slot_alpha and slot.p_wilcoxon_holm < g.slot_alpha
    )
    faq = res.loc[g.faq_ap]
    c2 = bool(faq.d_ap_pp < 0)
    cs = c_share_page_deltas(run_dir, g.faq_c_share, units)
    _, lo, hi = bootstrap_ci(cs.to_numpy(), b=cfg.score.bootstrap)
    ap_half = (faq.d_ap_hi - faq.d_ap_lo) / 2
    cs_half = (hi - lo) / 2
    c3 = bool(ap_half <= g.ci_ratio * cs_half)

    xs, ys, pairs = [], [], []
    for ap_name, arm in g.pairs.items():
        c = sets[ap_name]
        ap = ap_page_deltas(df, c.set, c.ref, units)
        share = c_share_page_deltas(run_dir, arm, units)
        common = ap.index.intersection(share.index)
        xs += ap.loc[common].tolist()
        ys += share.loc[common].tolist()
        pairs.append({"ap": ap_name, "c_share": arm, "n_pages": len(common)})
    rho = float(stats.spearmanr(xs, ys).statistic) if len(xs) > 2 else float("nan")
    c4 = bool(rho >= g.spearman)

    # MDE from the content-pinned comparisons (Study 2's primary mode); an arm that left AP
    # unchanged on every page has SD 0 and says nothing about noise, so it is left out.
    pinned = [n for n in g.pairs if n in res.index and sets[n].set.startswith("content:")]
    arm_sds = res.loc[pinned, "sd_unit_pp"].dropna()
    arm_sds = arm_sds[arm_sds > 0]
    n_pages = len(set(units.values()))
    mdes = [mde(float(s), n_pages, g.mde_family, t=True) for s in arm_sds]
    return {
        "run": run_dir.name,
        "ap_dir": str(ap_dir),
        "criteria": {
            "1_slot_detected": {
                "pass": c1,
                "d_ap_pp": float(slot.d_ap_pp),
                "ci": [float(slot.d_ap_lo), float(slot.d_ap_hi)],
                "p_perm_holm": float(slot.p_perm_holm),
                "p_wilcoxon_holm": float(slot.p_wilcoxon_holm),
                "n_units": int(slot.n_units),
                "threshold": f"negative, both Holm p < {g.slot_alpha}",
            },
            "2_faq_negative": {
                "pass": c2,
                "d_ap_pp": float(faq.d_ap_pp),
                "ci": [float(faq.d_ap_lo), float(faq.d_ap_hi)],
            },
            "3_ci_narrower": {
                "pass": c3,
                "ap_half_width_pp": float(ap_half),
                "c_share_half_width_pp": float(cs_half),
                "c_share_d_pp": float(cs.mean()),
                "ratio": float(ap_half / cs_half) if cs_half else float("nan"),
                "threshold": g.ci_ratio,
            },
            "4_spearman": {
                "pass": c4,
                "rho": rho,
                "n_points": len(xs),
                "pairs": pairs,
                "threshold": g.spearman,
            },
        },
        "passed": c1 and c2 and c3 and c4,
        "study2_mde_pp": {
            "range": [float(np.nanmin(mdes)), float(np.nanmax(mdes))] if mdes else None,
            "unit_sd_range_pp": [float(arm_sds.min()), float(arm_sds.max())]
            if len(arm_sds)
            else None,
            "n_pages": n_pages,
            "family": g.mde_family,
            "method": "t quantiles, 80% power, two-sided alpha 0.05 / family",
        },
    }


def gate_md(r: dict) -> str:
    c = r["criteria"]
    ok = {True: "pass", False: "FAIL"}
    s1, s2, s3, s4 = (c[k] for k in sorted(c))
    lines = [
        f"# Study 2 validation gate on {r['run']}: {'PASSED' if r['passed'] else 'FAILED'}",
        "",
        "| Criterion | Result | Measured | Threshold |",
        "|---|---|---|---|",
        f"| 1. AP detects slot 5 vs 1 | {ok[s1['pass']]} | {s1['d_ap_pp']:+.2f} pp, Holm p "
        f"{s1['p_perm_holm']:.2g} (sign-flip) / {s1['p_wilcoxon_holm']:.2g} (Wilcoxon), "
        f"{s1['n_units']} pages | {s1['threshold']} |",
        f"| 2. FAQ rewrite negative | {ok[s2['pass']]} | {s2['d_ap_pp']:+.2f} pp "
        f"[{s2['ci'][0]:+.2f}, {s2['ci'][1]:+.2f}] | < 0 |",
        f"| 3. CI half-width ratio | {ok[s3['pass']]} | {s3['ap_half_width_pp']:.2f} / "
        f"{s3['c_share_half_width_pp']:.2f} pp = {s3['ratio']:.2f} | <= {s3['threshold']} |",
        f"| 4. Spearman, AP vs C-SoV page deltas | {ok[s4['pass']]} | {s4['rho']:.2f} over "
        f"{s4['n_points']} (arm, page) points | >= {s4['threshold']} |",
        "",
    ]
    m = r["study2_mde_pp"]
    if m["range"]:
        lines.append(
            f"Study 2 MDE on AP ({m['method']}, {m['n_pages']} pages, family {m['family']}): "
            f"{m['range'][0]:.2f} to {m['range'][1]:.2f} pp, from page-level SDs of "
            f"{m['unit_sd_range_pp'][0]:.2f} to {m['unit_sd_range_pp'][1]:.2f} pp."
        )
    return "\n".join(lines) + "\n"


def run_gate(run_dir: Path, cfg, out: Path, log: Log = print) -> dict:
    ap_dir = score_run(run_dir, cfg, out, log)
    r = evaluate(run_dir, ap_dir, cfg)
    (out / "gate.json").write_text(json.dumps(r, indent=2) + "\n")
    (out / "gate.md").write_text(gate_md(r))
    log(gate_md(r))
    return r
