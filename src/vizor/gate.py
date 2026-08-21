"""The Study 2 validation gate: does attribution propensity (AP) reproduce Study 1?

Study 1's baseline samples 0 and 1 are the reference answers (text only), and Study 1's stored
prompts are re-scored with AP. The gate passes if all four pre-registered criteria hold
(`docs/bench-design-study2.md`, thresholds in the `gate:` section of the config):

1. AP detects the slot positive control: slot 5 minus slot 1 is negative with both Holm p
   values below `slot_alpha`.
2. AP gives the FAQ rewrite (content-pinned) a negative sign.
3. Relative precision: |delta| / 95% CI half-width of that AP delta is at least 1 / `ci_ratio`
   times the same ratio for the sampled citation-share delta of the same prompts (Study 1's
   `content:faq_rewrite`), both from unweighted page means with the same bootstrap.
4. Across Study 1's six content-pinned page arms, the Spearman correlation between per-page AP
   deltas and per-page citation-share deltas, divided by sqrt(`rel`) (the pre-registered
   reliability of the sampled deltas), is at least `spearman`; the raw correlation is positive
   and its one-sided page-permutation p is below `spearman_alpha`. (Amended 2026-09-18.)

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


def spearman_page_permutation(ap: np.ndarray, cs: np.ndarray, draws: int, seed: int = 0):
    """Spearman correlation over (arm, page) points of two arms x pages matrices, and its
    one-sided p (rho > 0) from permuting whole pages: the same page relabelling is applied to
    every arm's citation-share column, which keeps the arm structure and each page's pairing
    across arms."""
    ok = ~np.isnan(ap) & ~np.isnan(cs)
    if ok.sum() < 3:
        return float("nan"), 1.0
    rho = float(stats.spearmanr(ap[ok], cs[ok]).statistic)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(draws):
        perm = cs[:, rng.permutation(cs.shape[1])]
        m = ~np.isnan(ap) & ~np.isnan(perm)
        if stats.spearmanr(ap[m], perm[m]).statistic >= rho - 1e-12:
            hits += 1
    return rho, (1 + hits) / (1 + draws)


def reliability(run_dir: Path, arms: list[str], units: dict[str, str]) -> float:
    """Share of the pooled page-level variance of sampled citation-share deltas over `arms` that
    is not A/A noise: 1 - var(A/A page deltas) / var(pooled arm page deltas)."""
    aa = c_share_page_deltas(run_dir, "aa_resample", units).var(ddof=1)
    pooled = np.concatenate([c_share_page_deltas(run_dir, a, units).to_numpy() for a in arms])
    return float(1 - aa / np.var(pooled, ddof=1))


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
    cs_d, lo, hi = bootstrap_ci(cs.to_numpy(), b=cfg.score.bootstrap)
    ap_half = (faq.d_ap_hi - faq.d_ap_lo) / 2
    cs_half = (hi - lo) / 2
    # Relative precision (|estimate| / CI half-width), so the two metrics' different scales
    # cancel: AP must be at least 1 / ci_ratio times as precise as the sampled citation share.
    ap_prec = abs(faq.d_ap_pp) / ap_half if ap_half > 0 else float("inf")
    cs_prec = abs(cs_d) / cs_half if cs_half > 0 else float("inf")
    c3 = bool(ap_prec >= cs_prec / g.ci_ratio)

    ap_m, cs_m, pairs = {}, {}, []
    for ap_name, arm in g.pairs.items():
        c = sets[ap_name]
        ap_m[ap_name] = ap_page_deltas(df, c.set, c.ref, units)
        cs_m[ap_name] = c_share_page_deltas(run_dir, arm, units)
        common = ap_m[ap_name].index.intersection(cs_m[ap_name].index)
        pairs.append({"ap": ap_name, "c_share": arm, "n_pages": len(common)})
    A = pd.DataFrame(ap_m).T
    C = pd.DataFrame(cs_m).T.reindex(index=A.index, columns=A.columns)
    rho, p_one = spearman_page_permutation(A.to_numpy(), C.to_numpy(), cfg.score.draws)
    n_points = int((~np.isnan(A.to_numpy()) & ~np.isnan(C.to_numpy())).sum())
    rel = g.rel if g.rel is not None else float("nan")
    corrected = rho / np.sqrt(rel) if rel > 0 else float("nan")
    c4 = bool(corrected >= g.spearman and rho > 0 and p_one < g.spearman_alpha)
    rel_here = reliability(run_dir, list(g.pairs.values()), units)

    # MDE from the content-pinned comparisons (Study 2's primary mode); an arm that left AP
    # unchanged on every page has SD 0 and says nothing about noise, so it is left out.
    pinned = [n for n in g.pairs if n in res.index and sets[n].set.startswith("content:")]
    arm_sds = res.loc[pinned, "sd_unit_pp"].dropna()
    arm_sds = arm_sds[arm_sds > 0]
    n_pages = len(set(units.values()))
    mdes = [mde(float(s), n_pages, g.mde_family, t=True) for s in arm_sds]
    return {
        "run": run_dir.name,
        # the directory's name, like "run": the absolute path would name the machine's user
        "ap_dir": ap_dir.name,
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
                "ap_d_pp": float(faq.d_ap_pp),
                "ap_half_width_pp": float(ap_half),
                "ap_precision": float(ap_prec),
                "c_share_d_pp": float(cs_d),
                "c_share_half_width_pp": float(cs_half),
                "c_share_precision": float(cs_prec),
                "ratio": float(cs_prec / ap_prec) if ap_prec else float("inf"),
                "threshold": g.ci_ratio,
            },
            "4_spearman": {
                "pass": c4,
                "rho": rho,
                "rel": rel,
                "rho_corrected": float(corrected),
                "p_one_sided": p_one,
                "rel_this_run": rel_here,
                "n_points": n_points,
                "pairs": pairs,
                "threshold": g.spearman,
                "alpha": g.spearman_alpha,
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
        f"| 3. Relative precision, FAQ rewrite | {ok[s3['pass']]} | AP abs(d)/half-width "
        f"{s3['ap_precision']:.2f} vs C-SoV {s3['c_share_precision']:.2f} (ratio "
        f"{s3['ratio']:.2f}) | C-SoV / AP <= {s3['threshold']} |",
        f"| 4. Spearman, AP vs C-SoV page deltas | {ok[s4['pass']]} | rho {s4['rho']:.2f} over "
        f"{s4['n_points']} points, / sqrt(rel {s4['rel']:.3f}) = {s4['rho_corrected']:.2f}, "
        f"one-sided page-permutation p {s4['p_one_sided']:.3g} | corrected >= "
        f"{s4['threshold']}, rho > 0, p < {s4['alpha']} |",
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
