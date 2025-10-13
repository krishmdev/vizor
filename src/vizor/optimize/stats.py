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
