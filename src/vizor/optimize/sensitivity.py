"""What a run could detect, computed from its result files (so it can be re-derived for any
committed run): minimum detectable effects from the A/A re-sample's noise, and whether a test is
reachable at all given how few independent units it has."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vizor.optimize.stats import mde

CONTROLS = ("noop", "aa_resample")


def min_exact_wilcoxon_p(k: int) -> float:
    """Smallest two-sided p an exact signed-rank test can return with k non-zero differences."""
    return 2.0 ** (1 - k) if k > 0 else 1.0


def reachable(k: int, family: int, alpha: float = 0.05) -> bool:
    """Can any effect size pass the strictest Holm step (p * family < alpha) with k units?"""
    return min_exact_wilcoxon_p(k) * max(1, family) < alpha


def sensitivity(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    pq = pd.read_csv(run_dir / "per_query.csv")
    deltas = pd.read_csv(run_dir / "deltas.csv")
    if "aa_resample" not in set(pq.arm):
        return {}
    base = pq[pq.arm == "baseline"].set_index("query_id")["imp_pwc"]
    aa = pq[pq.arm == "aa_resample"].set_index("query_id")["imp_pwc"]
    d = ((aa - base).dropna() * 100).sort_index()
    sd = float(d.std(ddof=1))
    tested = deltas[~deltas.arm.isin(CONTROLS)]
    arm_family = len(tested)
    out: dict = {
        "aa_sd_per_query_pp": sd,
        "n_queries": int(len(d)),
        "arm_holm_family": arm_family,
        "power": 0.8,
        "alpha": 0.05,
        "engine_arm_pp": mde(sd, len(d), arm_family),
    }
    sweep_rows = []
    for name, col in (("position_sweep.csv", "position"), ("boost_sweep.csv", "boost")):
        p = run_dir / name
        if p.exists():
            df = pd.read_csv(p)
            sweep_rows.append((col, df))
    sweep_family = sum(len(df) - 1 for _, df in sweep_rows)
    out["sweep_holm_family"] = sweep_family
    for col, df in sweep_rows:
        n = int(df.n_queries.iloc[-1])
        out[f"{col}_n_queries"] = n
        out[f"{col}_pp"] = mde(sd, n, sweep_family)
    qfile = run_dir / "queries.csv"
    page_arms = tested[tested.kind == "doc"]
    if qfile.exists() and len(page_arms):
        q = pd.read_csv(qfile).set_index("query_id")
        units = q.loc[d.index, "fold"].astype(str) + "|" + q.loc[d.index, "focus_doc"].astype(str)
        cm = d.groupby(units).mean()
        k = int(len(cm))
        out["n_page_units"] = k
        out["page_arms_testable"] = reachable(k, arm_family)
        out["min_exact_p_page_units"] = min_exact_wilcoxon_p(k)
        out["page_arm_pp"] = (
            mde(float(cm.std(ddof=1)), k, arm_family) if out["page_arms_testable"] else None
        )
    return out
