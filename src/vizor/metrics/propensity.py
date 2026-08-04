"""Attribution propensity (AP): a teacher-forced, sampling-free citation metric.

Take a fixed reference answer R (a baseline sample) and a prompt. At every citation site of R
(the first index digit of each marker whose first index is a valid source), ask the scorer for
the probability of each source index, renormalized over the prompt's valid indices. AP is the
mean over R's sites of the total probability on the target site's slots.

Because R is fixed and scoring does not sample, AP(arm) - AP(baseline) is a deterministic
function of the two prompts and R. Identical prompts give a difference of exactly 0, so the
A/A difference is 0 by construction and the only variance left is between reference answers
and between pages.

In content-pinned mode the arm's sources and their order are the baseline's (the `content:`
prompt of the edit), so the target's slot is the same in both prompts and only the edited page's
text differs. In full mode the arm's own retrieval decides the sources and the target's slot.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence

import numpy as np

from vizor.attribution.citations import find_markers
from vizor.generate.scorer import ScoreResult, Site, renormalize

_DIGIT = re.compile(r"\d")


def citation_sites(text: str, n_sources: int, n_candidates: int | None = None) -> list[Site]:
    """One site per marker whose first index is in 1..n_sources, at that index's first digit.
    Candidates are the index strings "1".."n_candidates" (default n_sources)."""
    k = n_candidates or n_sources
    cands = tuple(str(i) for i in range(1, k + 1))
    out: list[Site] = []
    for a, b, _form, idx in find_markers(text):
        if not idx or not 1 <= idx[0] <= n_sources:
            continue
        m = _DIGIT.search(text, a, b)
        if m:
            out.append(Site(m.start(), cands))
    return out


def ap_from_probs(site_probs: Sequence[dict[str, float]], target_slots: Iterable[int]) -> float:
    """Mean over sites of the probability mass on the target's slots (NaN without sites)."""
    slots = [str(s) for s in target_slots]
    if not site_probs:
        return float("nan")
    return float(np.mean([sum(p.get(s, 0.0) for s in slots) for p in site_probs]))


def ap_from_result(res: ScoreResult, target_slots: Iterable[int]) -> float:
    return ap_from_probs([s.renorm for s in res.sites], target_slots)


def ap_from_logprobs(
    site_logprobs: Sequence[dict[str, float]], target_slots: Iterable[int]
) -> float:
    """AP rebuilt from stored per-site candidate log-probabilities (what `vizor recompute`
    checks): renormalize each site over its candidates, then average the target's mass."""
    return ap_from_probs([renormalize(lp) for lp in site_logprobs], target_slots)


def target_slots(sources: Sequence[dict], target_domains: Iterable[str]) -> tuple[int, ...]:
    """1-based positions of target-domain sources in a prompt's source list."""
    t = set(target_domains)
    return tuple(int(s["position"]) for s in sources if s["domain"] in t)


def is_close(a: float, b: float, tol: float = 1e-9) -> bool:
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= tol
