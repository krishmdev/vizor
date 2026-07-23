import numpy as np
import pytest

from vizor.optimize.stats import bootstrap_ci, holm, paired, wilcoxon_p


def test_bootstrap_ci_covers_known_mean_most_of_the_time():
    rng = np.random.default_rng(1)
    covered = 0
    for trial in range(200):
        x = rng.normal(0.3, 1.0, size=40)
        _, lo, hi = bootstrap_ci(x, b=1000, seed=trial)
        covered += lo <= 0.3 <= hi
    assert 0.88 <= covered / 200 <= 0.99


def test_all_zero_deltas_give_degenerate_ci_and_p_one():
    r = paired(np.zeros(40))
    assert (r.mean, r.lo, r.hi, r.p) == (0.0, 0.0, 0.0, 1.0)
    assert not r.excludes_zero


def test_clear_shift_is_detected():
    rng = np.random.default_rng(0)
    r = paired(rng.normal(2.0, 1.0, size=40))
    assert r.excludes_zero and r.p < 1e-4


def test_wilcoxon_ignores_nan():
    assert wilcoxon_p(np.array([np.nan, 0.0, 0.0])) == 1.0


def test_holm_matches_hand_computation():
    # sorted p: 0.01, 0.02, 0.04 -> 3*0.01, max(0.03, 2*0.02), max(0.04, 1*0.04)
    assert holm([0.04, 0.01, 0.02]) == pytest.approx([0.04, 0.03, 0.04])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_clustered_reduces_to_paired_with_singletons():
    from vizor.optimize.stats import paired_clustered

    rng = np.random.default_rng(3)
    d = rng.normal(1, 1, 30)
    a, b = paired(d, b=2000), paired_clustered(d, np.arange(30), b=2000)
    assert a.n == b.n == 30 and a.mean == pytest.approx(b.mean) and a.p == pytest.approx(b.p)


def test_clustered_ci_is_wider_when_queries_share_a_page():
    from vizor.optimize.stats import paired_clustered

    rng = np.random.default_rng(4)
    page_effect = np.repeat(rng.normal(0, 3, 4), 10)
    d = page_effect + rng.normal(0, 0.5, 40)
    flat = paired(d, b=2000)
    clus = paired_clustered(d, np.repeat(np.arange(4), 10), b=2000)
    assert clus.n == 4 and (clus.hi - clus.lo) > 2 * (flat.hi - flat.lo)


def test_mde_formula():
    from vizor.optimize.stats import mde

    # z_{0.025} + z_{0.8} = 1.95996 + 0.84162
    assert mde(10.0, 25, 1) == pytest.approx((1.959964 + 0.841621) * 10 / 5, rel=1e-5)
    assert mde(10.0, 25, 5) > mde(10.0, 25, 1)


def test_exact_wilcoxon_floor_and_reachability():
    from scipy import stats as sps

    from vizor.optimize.sensitivity import min_exact_wilcoxon_p, reachable

    for k in (4, 7, 9):
        assert sps.wilcoxon(np.arange(1, k + 1), method="exact").pvalue == pytest.approx(
            min_exact_wilcoxon_p(k)
        )
    assert not reachable(4, 1) and not reachable(7, 7) and reachable(12, 7)


def test_pilot_sensitivity_uses_planned_families(tmp_path):
    import pandas as pd

    from vizor.optimize.sensitivity import sensitivity
    from vizor.optimize.stats import mde

    rng = np.random.default_rng(0)
    qids = [f"q{i:02d}" for i in range(48)]
    base = rng.uniform(0, 0.6, len(qids))
    aa = np.clip(base + rng.normal(0, 0.1, len(qids)), 0, 1)
    rows = [
        {"arm": arm, "query_id": q, "c_share": v, "imp_pwc": v, "mentioned": v > 0.3}
        for arm, vals in (("baseline", base), ("aa_resample", aa))
        for q, v in zip(qids, vals, strict=True)
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "per_query.csv", index=False)
    pd.DataFrame([{"arm": "noop", "kind": "doc", "family": "arms"}]).to_csv(
        tmp_path / "deltas.csv", index=False
    )
    pd.DataFrame(
        {"query_id": qids, "fold": [1, 2] * 24, "focus_doc": [f"p{i // 2}" for i in range(48)]}
    ).to_csv(tmp_path / "queries.csv", index=False)
    s = sensitivity(tmp_path, metric="c_share", family=6, content_family=6)
    assert s["metric"] == "c_share" and s["arm_holm_family"] == 6
    assert s["n_units_page"] == 24 and s["n_units_page_x_fold"] == 48
    assert s["n_page_units"] == s["n_page_units_min"] == 24
    d = (pd.Series(aa - base, index=qids) * 100).groupby([f"p{i // 2}" for i in range(48)]).mean()
    assert s["page_arm_page_pp"] == pytest.approx(mde(float(d.std(ddof=1)), 24, 6))
    assert s["content_arm_page_pp"] == pytest.approx(s["page_arm_page_pp"])
    assert s["page_arm_pp"] == pytest.approx(s["page_arm_page_pp"])
    assert "mention_page_arm_pp" in s
    assert s["families_from_plan"]

    # Without planned families a pilot states no page-arm MDE, named rate included.
    bare = sensitivity(tmp_path, metric="c_share")
    assert bare["arm_holm_family"] == 0 and "mention_page_arm_pp" not in bare
    # A pilot's manifest records the planned arms, which set the families.
    import json

    arms = ["noop", "aa_resample", "faq_rewrite", "metadata", "content:faq_rewrite"]
    (tmp_path / "manifest.json").write_text(
        json.dumps({"primary_metric": "c_share", "planned_arms": arms})
    )
    p = sensitivity(tmp_path)
    assert p["metric"] == "c_share" and p["families_from_plan"]
    assert (p["arm_holm_family"], p["content_holm_family"]) == (2, 1)
    assert p["page_arm_page_pp"] == pytest.approx(mde(float(d.std(ddof=1)), 24, 2))
