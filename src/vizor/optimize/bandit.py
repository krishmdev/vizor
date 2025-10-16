"""Contextual bandits over optimization arms, evaluated by offline replay.

This is a contextual bandit with replay evaluation, not deep RL. Contexts are tracked queries
(with features of the target page that serves each one), arms are page-side optimizations, and the
reward table R[query, arm, sample] comes straight from the sandbox runs, so replay costs no extra
LLM calls. Each round draws a query and a sample, the policy picks an arm, and it observes that
arm's measured reward. Regret is against the oracle arm for the drawn query (best mean reward).

LinUCB is ported from the old evaluation/policy.py, with the per-select and per-update
np.linalg.inv replaced by a Sherman-Morrison update of A^-1. LinTS is rewritten: the old version
computed the posterior mean with the *updated* precision applied to the old mean, which is wrong;
here mu = B^-1 f with B and f accumulated separately.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd


def sherman_morrison(a_inv: np.ndarray, x: np.ndarray) -> np.ndarray:
    """(A + x x^T)^-1 from A^-1."""
    ax = a_inv @ x
    return a_inv - np.outer(ax, ax) / (1.0 + x @ ax)


class LinUCB:
    def __init__(self, arms: Sequence[str], d: int, alpha: float = 1.0, lam: float = 1.0) -> None:
        self.arms = list(arms)
        self.alpha = alpha
        self.a_inv = {a: np.eye(d) / lam for a in self.arms}
        self.b = {a: np.zeros(d) for a in self.arms}
        self.name = f"linucb(a={alpha:g})"

    def scores(self, x: np.ndarray) -> dict[str, float]:
        out = {}
        for a in self.arms:
            theta = self.a_inv[a] @ self.b[a]
            out[a] = float(theta @ x + self.alpha * np.sqrt(x @ self.a_inv[a] @ x))
        return out

    def select(self, x: np.ndarray, rng: np.random.Generator) -> str:
        s = self.scores(x)
        best = max(s.values())
        return str(rng.choice([a for a, v in s.items() if v >= best - 1e-12]))

    def update(self, x: np.ndarray, arm: str, r: float) -> None:
        self.a_inv[arm] = sherman_morrison(self.a_inv[arm], x)
        self.b[arm] = self.b[arm] + r * x

    def predict(self, x: np.ndarray) -> dict[str, float]:
        return {a: float((self.a_inv[a] @ self.b[a]) @ x) for a in self.arms}

    def save(self, path: Path) -> None:
        state = {
            "alpha": self.alpha,
            "arms": self.arms,
            "a_inv": {a: m.tolist() for a, m in self.a_inv.items()},
            "b": {a: v.tolist() for a, v in self.b.items()},
        }
        path.write_text(json.dumps(state))

    @classmethod
    def load(cls, path: Path) -> LinUCB:
        s = json.loads(path.read_text())
        d = len(next(iter(s["b"].values())))
        p = cls(s["arms"], d, s["alpha"])
        p.a_inv = {a: np.array(m) for a, m in s["a_inv"].items()}
        p.b = {a: np.array(v) for a, v in s["b"].items()}
        return p


class LinTS:
    """Per-arm Bayesian linear regression with Thompson sampling."""

    def __init__(self, arms: Sequence[str], d: int, v: float = 0.1, lam: float = 1.0) -> None:
        self.arms = list(arms)
        self.v = v
        self.b_inv = {a: np.eye(d) / lam for a in self.arms}
        self.f = {a: np.zeros(d) for a in self.arms}
        self.name = f"lints(v={v:g})"

    def mean(self, arm: str) -> np.ndarray:
        return self.b_inv[arm] @ self.f[arm]

    def select(self, x: np.ndarray, rng: np.random.Generator) -> str:
        best, arg = -np.inf, self.arms[0]
        for a in self.arms:
            cov = self.v**2 * self.b_inv[a]
            theta = rng.multivariate_normal(self.mean(a), (cov + cov.T) / 2, method="cholesky")
            s = float(theta @ x)
            if s > best:
                best, arg = s, a
        return arg

    def update(self, x: np.ndarray, arm: str, r: float) -> None:
        self.b_inv[arm] = sherman_morrison(self.b_inv[arm], x)
        self.f[arm] = self.f[arm] + r * x


class EpsGreedy:
    """Ridge regression per arm, explores uniformly with probability eps."""

    def __init__(self, arms: Sequence[str], d: int, eps: float = 0.1) -> None:
        self.inner = LinUCB(arms, d, alpha=0.0)
        self.arms = list(arms)
        self.eps = eps
        self.name = f"eps-greedy({eps:g})"

    def select(self, x: np.ndarray, rng: np.random.Generator) -> str:
        if rng.random() < self.eps:
            return str(rng.choice(self.arms))
        return self.inner.select(x, rng)

    def update(self, x: np.ndarray, arm: str, r: float) -> None:
        self.inner.update(x, arm, r)


class RandomPolicy:
    def __init__(self, arms: Sequence[str], d: int) -> None:
        self.arms = list(arms)
        self.name = "random"

    def select(self, x: np.ndarray, rng: np.random.Generator) -> str:
        return str(rng.choice(self.arms))

    def update(self, x: np.ndarray, arm: str, r: float) -> None:
        pass


class FixedArm:
    def __init__(self, arm: str, label: str) -> None:
        self.arm = arm
        self.name = label

    def select(self, x: np.ndarray, rng: np.random.Generator) -> str:
        return self.arm

    def update(self, x: np.ndarray, arm: str, r: float) -> None:
        pass


def make_policies(arms: Sequence[str], d: int, best_fixed: str) -> dict:
    return {
        "linucb(a=0.1)": lambda: LinUCB(arms, d, 0.1),
        "linucb(a=0.5)": lambda: LinUCB(arms, d, 0.5),
        "linucb(a=1)": lambda: LinUCB(arms, d, 1.0),
        "lints(v=0.1)": lambda: LinTS(arms, d, 0.1),
        "eps-greedy(0.1)": lambda: EpsGreedy(arms, d, 0.1),
        "random": lambda: RandomPolicy(arms, d),
        "best-fixed-arm": lambda: FixedArm(best_fixed, "best-fixed-arm"),
    }


def replay(
    rewards: np.ndarray,
    contexts: np.ndarray,
    arms: Sequence[str],
    rounds: int = 300,
    runs: int = 20,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """rewards: (n_contexts, n_arms, n_samples). Returns (curve, summary)."""
    n_ctx, n_arms, n_samp = rewards.shape
    mu = rewards.mean(axis=2)
    oracle = mu.max(axis=1)
    best_fixed = arms[int(mu.mean(axis=0).argmax())]
    factories = make_policies(arms, contexts.shape[1], best_fixed)
    curves, summary = [], []
    for name, make in factories.items():
        regrets = np.zeros((runs, rounds))
        picks = np.zeros(n_arms)
        total = np.zeros(runs)
        for run in range(runs):
            rng = np.random.default_rng([seed, run])
            draw_c = rng.integers(0, n_ctx, rounds)
            draw_k = rng.integers(0, n_samp, rounds)
            pol = make()
            prng = np.random.default_rng([seed, run, 1])
            for t in range(rounds):
                c, k = draw_c[t], draw_k[t]
                a = arms.index(pol.select(contexts[c], prng))
                r = rewards[c, a, k]
                pol.update(contexts[c], arms[a], r)
                regrets[run, t] = oracle[c] - mu[c, a]
                total[run] += r
                picks[a] += 1
        cum = regrets.cumsum(axis=1)
        m = cum.mean(axis=0)
        half = 1.96 * cum.std(axis=0, ddof=1) / np.sqrt(runs) if runs > 1 else np.zeros(rounds)
        curves.append(
            pd.DataFrame(
                {
                    "policy": name,
                    "t": np.arange(1, rounds + 1),
                    "cum_regret": m,
                    "lo": m - half,
                    "hi": m + half,
                }
            )
        )
        summary.append(
            {
                "policy": name,
                "final_regret": float(m[-1]),
                "final_regret_ci": float(half[-1]),
                "mean_reward_pp": float(total.mean() / rounds * 100),
                "top_arm": arms[int(picks.argmax())],
                "top_arm_share": float(picks.max() / picks.sum()),
            }
        )
    oracle_row = {
        "policy": "oracle",
        "final_regret": 0.0,
        "final_regret_ci": 0.0,
        "mean_reward_pp": float(oracle.mean() * 100),
        "top_arm": "",
        "top_arm_share": np.nan,
    }
    return pd.concat(curves, ignore_index=True), pd.DataFrame([*summary, oracle_row])
