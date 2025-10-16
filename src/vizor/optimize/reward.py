"""Rewards for the bandit, from before/after visibility of the target.

Ported in spirit from the old DeltaVComputer (delta SoV, PAWC, sentiment, citations), minus its
invented business multipliers. The bandit's reward is the change in the target's PAWC share for
one (query, sample), minus an optional token-cost penalty, clipped to [-1, 1].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

INTENTS = ["informational", "comparison", "transactional", "troubleshooting"]
N_FEATURES = 13


@dataclass(frozen=True)
class Deltas:
    pwc: float
    c_share: float
    cited: float
    sentiment: float


def deltas(before: pd.Series, after: pd.Series) -> Deltas:
    def d(k: str) -> float:
        a, b = after.get(k, np.nan), before.get(k, np.nan)
        return float(a - b) if not (np.isnan(a) or np.isnan(b)) else 0.0

    return Deltas(d("imp_pwc"), d("c_share"), d("cited"), d("answer_sentiment"))


def reward(delta: Deltas, cost_norm: float = 0.0, lam: float = 0.0) -> float:
    return float(np.clip(delta.pwc - lam * cost_norm, -1.0, 1.0))


def context_features(
    doc, intent: str, retrieval_rate: float, cite_rate: float, pwc: float
) -> np.ndarray:
    """13 features in [0, 1]: bias, has_faq, has_jsonld, meta description quality (70-160 chars),
    log(words)/8, internal links/5, the target's baseline retrieval rate, citation rate and PAWC
    share on this query, and the query intent one-hot."""
    words = max(1, len(doc.body.split()))
    x = [
        1.0,
        float(bool(doc.faq)),
        float(bool(doc.jsonld)),
        float(70 <= len(doc.meta_description) <= 160),
        min(1.0, np.log(words) / 8),
        min(1.0, len(doc.links) / 5),
        retrieval_rate,
        cite_rate,
        pwc,
        *[float(intent == i) for i in INTENTS],
    ]
    return np.array(x, dtype=float)
