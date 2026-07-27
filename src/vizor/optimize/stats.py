"""Paired statistics for sandbox comparisons. The unit of analysis is the query."""

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


def paired_clustered(
    deltas: np.ndarray, clusters: np.ndarray, b: int = 5000, seed: int = 0
) -> PairedResult:
    """Paired comparison when queries share a treated unit (e.g. one edited page serves several
    queries). The CI resamples whole clusters; the Wilcoxon test runs on cluster means. `n` is
    the number of clusters. With one query per cluster this reduces to `paired`."""
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
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(b, k))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return PairedResult(k, float(d.mean()), float(lo), float(hi), wilcoxon_p(sums / counts))


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
