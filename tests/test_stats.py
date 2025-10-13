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
