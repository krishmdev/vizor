"""Paired statistics for sandbox comparisons: bootstrap intervals, Wilcoxon and sign-flip
permutation tests, Holm, and the MDE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class PairedResult:
    n: int
    mean: float
    lo: float
    hi: float
    p: float

    @property
    def excludes_zero(self) -> bool:
        return self.lo > 0 or self.hi < 0


def bootstrap_ci(
    deltas: np.ndarray, b: int = 5000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean of per-query paired differences."""
    d = np.asarray(deltas, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(b, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)


def wilcoxon_p(deltas: np.ndarray) -> float:
    d = np.asarray(deltas, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) == 0 or np.allclose(d, 0):
        return 1.0
    return float(stats.wilcoxon(d, zero_method="zsplit").pvalue)


def paired(deltas: np.ndarray, b: int = 5000, seed: int = 0) -> PairedResult:
    d = np.asarray(deltas, dtype=float)
    d = d[~np.isnan(d)]
    mean, lo, hi = bootstrap_ci(d, b=b, seed=seed)
    return PairedResult(len(d), mean, lo, hi, wilcoxon_p(d))


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def sign_flip_p(unit_deltas: np.ndarray, draws: int = 20000, seed: int = 0) -> float:
    """Two-sided sign-flip permutation p for a zero mean of paired unit differences.

    Under the null each unit's difference is symmetric around 0, so its sign is exchangeable.
    Exact over all 2^n sign patterns when that is at most `draws`, otherwise Monte Carlo with the
    (1 + hits) / (1 + draws) correction, which keeps the test valid."""
    d = np.asarray(unit_deltas, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n == 0 or np.allclose(d, 0):
        return 1.0
    obs = abs(d.mean())
    tol = 1e-12 * max(1.0, float(np.abs(d).max()))
    if 2**n <= draws:
        signs = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1) * 2 - 1
        stats_ = np.abs((signs * d).mean(axis=1))
        return float(np.mean(stats_ >= obs - tol))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(draws, n))
    hits = int(np.sum(np.abs((signs * d).mean(axis=1)) >= obs - tol))
    return (1 + hits) / (1 + draws)


def sign_flip_exact_p(unit_deltas: np.ndarray, max_n: int = 40) -> float:
    """Exact two-sided sign-flip p over all 2^n sign patterns, by meet in the middle.

    The sums over each half of the units are enumerated (2^(n/2) each), one side is sorted, and
    the patterns whose |total| reaches the observed |sum| are counted with a binary search. This
    makes n = 24 (16.8 million patterns) cheap. It gives the value that `sign_flip_p`
    approximates by Monte Carlo when 2^n exceeds its draws."""
    d = np.asarray(unit_deltas, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n == 0 or np.allclose(d, 0):
        return 1.0
    if n > max_n:
        raise ValueError(f"{n} units is too many for exact enumeration (max {max_n})")

    def half_sums(x: np.ndarray) -> np.ndarray:
        k = len(x)
        signs = ((np.arange(2**k)[:, None] >> np.arange(k)) & 1) * 2 - 1
        return (signs * x).sum(axis=1) if k else np.zeros(1)

    left, right = half_sums(d[: n // 2]), np.sort(half_sums(d[n // 2 :]))
    tol = 1e-12 * max(1.0, float(np.abs(d).sum()))
    t = abs(d.sum()) - tol
    if t <= 0:
        return 1.0
    upper = len(right) - np.searchsorted(right, t - left, side="left")
    lower = np.searchsorted(right, -t - left, side="right")
    hits = int(np.sum(upper) + np.sum(lower))
    return hits / float(2**n)


def wilcoxon_exact_p(unit_deltas: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p on unit differences, exact when scipy can (no ties among
    the non-zero |d|), normal approximation otherwise. Zero differences are dropped (Wilcox)."""
    d = np.asarray(unit_deltas, dtype=float)
    d = d[~np.isnan(d)]
    d = d[np.abs(d) > 1e-12]
    if len(d) == 0:
        return 1.0
    ties = len(np.unique(np.round(np.abs(d), 12))) < len(d)
    method = "approx" if ties else "exact"
    return float(stats.wilcoxon(d, zero_method="wilcox", method=method).pvalue)


def paired_clustered(
    deltas: np.ndarray,
    clusters: np.ndarray,
    b: int = 5000,
    seed: int = 0,
    weighting: str = "query",
) -> PairedResult:
    """Paired comparison when queries share a treated unit (e.g. one edited page serves several
    queries). The CI resamples whole clusters; the Wilcoxon test runs on cluster means. `n` is
    the number of clusters. With one query per cluster this reduces to `paired`.

    `weighting="query"` (Study 1): the estimate is the per-query mean and the bootstrap a ratio
    of sums, so pages with more queries weigh more, while the test ranks unweighted page means.
    `weighting="page"`: estimate, bootstrap and test all use the unweighted mean of page means."""
    if weighting not in ("query", "page"):
        raise ValueError(f"unknown weighting {weighting!r}")
    d = np.asarray(deltas, dtype=float)
    c = np.asarray(clusters)
    ok = ~np.isnan(d)
    d, c = d[ok], c[ok]
    if len(d) == 0:
        return PairedResult(0, np.nan, np.nan, np.nan, 1.0)
    labels, inv = np.unique(c, return_inverse=True)
    k = len(labels)
    sums = np.bincount(inv, weights=d, minlength=k)
    counts = np.bincount(inv, minlength=k).astype(float)
    unit = sums / counts
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(b, k))
    if weighting == "page":
        means = unit[idx].mean(axis=1)
        est = float(unit.mean())
    else:
        means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
        est = float(d.mean())
    lo, hi = np.quantile(means, [0.025, 0.975])
    return PairedResult(k, est, float(lo), float(hi), wilcoxon_p(unit))


def unit_means(deltas: np.ndarray, clusters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(labels, mean delta per cluster), NaN deltas dropped."""
    d = np.asarray(deltas, dtype=float)
    c = np.asarray(clusters)
    ok = ~np.isnan(d)
    labels, inv = np.unique(c[ok], return_inverse=True)
    sums = np.bincount(inv, weights=d[ok], minlength=len(labels))
    counts = np.bincount(inv, minlength=len(labels)).astype(float)
    return labels, sums / np.maximum(counts, 1)


def mde(
    sd: float, n: int, family: int, power: float = 0.8, alpha: float = 0.05, t: bool = False
) -> float:
    """Minimum detectable mean difference for a two-sided test at the strictest Holm step
    (alpha / family): (z_{alpha_Holm/2} + z_power) * sd / sqrt(n). With `t`, Student t quantiles
    on n - 1 degrees of freedom, which is larger for few units."""
    if n <= 1 or not np.isfinite(sd):
        return float("nan")
    q = stats.t(n - 1).ppf if t else stats.norm.ppf
    z = q(1 - alpha / (2 * max(1, family))) + q(power)
    return float(z * sd / np.sqrt(n))
