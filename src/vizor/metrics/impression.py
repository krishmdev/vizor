"""Impression metrics from Aggarwal et al., "GEO: Generative Engine Optimization" (2311.09735).

For a response with sentences S (N = |S|), sentence position i (zero-indexed) and citation multiset
C(s), the position-adjusted word count of source c is

    raw_pwc(c) = sum over s with c in C(s) of |s| * d(i) / |C(s)|

where d(i) = exp(-i / N) as written in the paper, or exp(-i / (N - 1)) as in the authors' released
impression_wordpos_count_simple ("reference" mode). Each metric is normalized so the sources' shares
sum to 1; the paper's extra division by sum |s| is a constant that normalization cancels.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from vizor.types import CitedSentence

Decay = Literal["paper", "reference"]
NoCitation = Literal["zeros", "uniform"]


@dataclass(frozen=True)
class Impressions:
    word: list[float]
    pos: list[float]
    pwc: list[float]
    raw_pwc: list[float]
    uncited: bool


def decay_weights(n_sentences: int, mode: Decay = "paper") -> list[float]:
    if n_sentences == 0:
        return []
    if mode == "paper":
        return [math.exp(-i / n_sentences) for i in range(n_sentences)]
    if n_sentences == 1:
        return [1.0]
    return [math.exp(-i / (n_sentences - 1)) for i in range(n_sentences)]


def _normalize(raw: list[float], no_citation: NoCitation) -> list[float]:
    total = sum(raw)
    if total > 0:
        return [x / total for x in raw]
    if no_citation == "uniform":
        return [1.0 / len(raw)] * len(raw) if raw else []
    return [0.0] * len(raw)


def impressions(
    sentences: Sequence[CitedSentence],
    n_sources: int,
    decay: Decay = "paper",
    no_citation: NoCitation = "zeros",
) -> Impressions:
    d = decay_weights(len(sentences), decay)
    word = [0.0] * n_sources
    pos = [0.0] * n_sources
    pwc = [0.0] * n_sources
    n_cites = 0
    for s, w in zip(sentences, d, strict=True):
        k = len(s.citations)
        for c in s.citations:
            if not 1 <= c <= n_sources:
                continue
            n_cites += 1
            word[c - 1] += s.n_words / k
            pos[c - 1] += w / k
            pwc[c - 1] += s.n_words * w / k
    uncited = n_cites == 0
    return Impressions(
        word=_normalize(word, no_citation),
        pos=_normalize(pos, no_citation),
        pwc=_normalize(pwc, no_citation),
        raw_pwc=pwc,
        uncited=uncited,
    )


def citation_share(sentences: Sequence[CitedSentence], n_sources: int) -> list[float]:
    """Share of valid [n] markers pointing at each source (C-SoV within one response)."""
    counts = [0] * n_sources
    for s in sentences:
        for c in s.citations:
            if 1 <= c <= n_sources:
                counts[c - 1] += 1
    total = sum(counts)
    return [x / total if total else 0.0 for x in counts]


def relative_improvement(before: float, after: float) -> float | None:
    """The paper's (Imp' - Imp) / Imp * 100. Undefined when the baseline share is 0."""
    if before == 0:
        return None
    return (after - before) / before * 100.0
