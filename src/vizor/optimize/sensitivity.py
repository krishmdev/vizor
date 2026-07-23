"""What a run could detect, computed from its result files (so it can be re-derived for any
committed run): minimum detectable effects from the A/A re-sample's noise, and whether a test is
reachable at all given how few independent units it has."""

from __future__ import annotations

import json
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


def _manifest(run_dir: Path) -> dict:
    m = run_dir / "manifest.json"
    return json.loads(m.read_text()) if m.exists() else {}


def _metric(run_dir: Path, metric: str | None) -> str:
    return metric or _manifest(run_dir).get("primary_metric", "imp_pwc")


def planned_families(arms: list[str]) -> tuple[int, int]:
    """Holm family sizes (page and engine arms, content-only twins) of a planned arm list."""
    tested = [a for a in arms if a not in CONTROLS]
    content = sum(a.startswith("content:") for a in tested)
    return len(tested) - content, content


def page_units(run_dir: Path, d: pd.Series) -> dict[str, pd.Series]:
    """A/A deltas by underlying page, with page x fold shown only as a diagnostic.

    Both cross-fit versions share a source page and are clustered together for inference.
    """
    q = pd.read_csv(run_dir / "queries.csv").set_index("query_id").loc[d.index]
    return {
        "page": d.groupby(q["focus_doc"].astype(str)).mean(),
        "page x fold": d.groupby(q["fold"].astype(str) + "|" + q["focus_doc"].astype(str)).mean(),
    }


def sensitivity(
    run_dir: Path,
    metric: str | None = None,
    family: int | None = None,
    content_family: int | None = None,
) -> dict:
    """MDEs from the A/A re-sample. `family` / `content_family` override the Holm family sizes
    read from deltas.csv, which is how a pilot (baseline + A/A only) states what the planned
    full design can detect before any page arm is run. A pilot records the planned arms in its
    manifest (`planned_arms`), and those set the families when no override is given."""
    run_dir = Path(run_dir)
    pq = pd.read_csv(run_dir / "per_query.csv")
    deltas = pd.read_csv(run_dir / "deltas.csv")
    if "aa_resample" not in set(pq.arm):
        return {}
    metric = _metric(run_dir, metric)
    if metric not in pq.columns:
        return {}

    def aa(col: str) -> pd.Series:
        base = pq[pq.arm == "baseline"].set_index("query_id")[col].astype(float)
        a = pq[pq.arm == "aa_resample"].set_index("query_id")[col].astype(float)
        return ((a - base).dropna() * 100).sort_index()

    d = aa(metric)
    sd = float(d.std(ddof=1))
    tested = deltas[~deltas.arm.isin(CONTROLS)]
    planned = _manifest(run_dir).get("planned_arms")
    if planned and not len(tested):
        pf, pc = planned_families(planned)
        family = pf if family is None else family
        content_family = pc if content_family is None else content_family
    fam_col = tested["family"] if "family" in tested else pd.Series("arms", index=tested.index)
    arm_family = family if family is not None else int((fam_col == "arms").sum())
    c_family = content_family if content_family is not None else int((fam_col == "content").sum())
    out: dict = {
        "metric": metric,
        "aa_sd_per_query_pp": sd,
        "aa_mean_pp": float(d.mean()),
        "n_queries": int(len(d)),
        "arm_holm_family": arm_family,
        "content_holm_family": c_family,
        "families_from_plan": family is not None,
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
    page_arms = tested[(tested.kind == "doc")] if "kind" in tested else tested.iloc[:0]
    if qfile.exists() and (len(page_arms) or family is not None):
        units = page_units(run_dir, d)
        if len(page_arms) and "n_units" in page_arms:
            k = int(page_arms["n_units"].max())
            kmin = int(page_arms["n_units"].min())
        else:
            k = kmin = len(units["page"])
        out["n_page_units"] = k
        out["n_page_units_min"] = kmin
        out["page_arms_testable"] = reachable(kmin, max(arm_family, 1))
        out["min_exact_p_page_units"] = min_exact_wilcoxon_p(k)
        for name, cm in units.items():
            key = name.replace(" ", "_")
            n = len(cm)
            out[f"unit_sd_{key}_pp"] = float(cm.std(ddof=1)) if n > 1 else float("nan")
            out[f"n_units_{key}"] = n
            ok = reachable(n, max(arm_family, 1))
            out[f"page_arm_{key}_pp"] = mde(float(cm.std(ddof=1)), n, arm_family) if ok else None
            if c_family:
                out[f"content_arm_{key}_pp"] = (
                    mde(float(cm.std(ddof=1)), n, c_family) if reachable(n, c_family) else None
                )
        out["page_arm_pp"] = out["page_arm_page_pp"]
    if "mentioned" in pq.columns and metric != "mentioned":
        dm = aa("mentioned")
        out["mention_aa_sd_per_query_pp"] = float(dm.std(ddof=1))
        if qfile.exists() and arm_family:
            cm = page_units(run_dir, dm)["page"]
            out["mention_page_arm_pp"] = (
                mde(float(cm.std(ddof=1)), len(cm), arm_family)
                if reachable(len(cm), max(arm_family, 1))
                else None
            )
    return out


def position_robustness(run_dir: Path) -> dict:
    """How much the last-slot result depends on analysis choices: the slot-1 vs last-slot delta
    per focus page, an exact Wilcoxon on those page means, and the Holm p the last slot would get
    if the sweeps shared one family with the arms."""
    from vizor.optimize.stats import holm, wilcoxon_p

    run_dir = Path(run_dir)
    pos_path = run_dir / "position_sweep.csv"
    if not pos_path.exists() or not (run_dir / "queries.csv").exists():
        return {}
    pos = pd.read_csv(pos_path)
    first, last = int(pos.position.iloc[0]), int(pos.position.iloc[-1])
    metric = str(pos.primary.iloc[0]) if "primary" in pos else "imp_pwc"
    if metric not in ("imp_pwc", "c_share"):
        return {}
    rows = pd.read_csv(run_dir / "rows.csv.gz")
    t = rows[rows.role == "target"]

    def per_query(arm: str) -> pd.DataFrame:
        a = t[t.arm == arm]
        return a.groupby("query_id").agg(value=(metric, "mean"), retrieved=("retrieved", "max"))

    a, b = per_query(f"engine:target_at:{first}"), per_query(f"engine:target_at:{last}")
    forced = a.index[a.retrieved.astype(bool)]
    d = (b.loc[forced, "value"] - a.loc[forced, "value"]) * 100
    q = pd.read_csv(run_dir / "queries.csv").set_index("query_id")
    by_page = d.groupby(q.loc[d.index, "focus_url"]).mean()
    deltas = pd.read_csv(run_dir / "deltas.csv")
    tested = deltas[~deltas.arm.isin(CONTROLS)]
    boost = (
        pd.read_csv(run_dir / "boost_sweep.csv") if (run_dir / "boost_sweep.csv").exists() else None
    )
    sweep_p = list(pos.p.iloc[1:]) + (list(boost.p.iloc[1:]) if boost is not None else [])
    joint = holm(list(tested.p) + sweep_p)
    return {
        "metric": metric,
        "first_slot": first,
        "last_slot": last,
        "n_queries": int(len(d)),
        "per_page_delta_pp": {k.rsplit("/", 1)[-1]: float(v) for k, v in by_page.items()},
        "n_pages": int(len(by_page)),
        "page_level_p": wilcoxon_p(by_page.to_numpy()),
        "raw_p": float(pos.p.iloc[-1]),
        "joint_family_size": len(joint),
        "joint_holm_p": float(joint[len(tested) + len(pos) - 2]),
    }
