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
