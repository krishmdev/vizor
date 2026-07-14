"""Brand mentions: does an answer name a site's brand, whether or not it cites that site?

Citations and mentions can disagree in both directions. An answer can cite a Brewline page and
never say "Brewline", or name the Brewline Duo from a review site's page without citing Brewline
at all. The question "how often does the site show up in the answer" needs both.

Brand names come from the project YAML (`brands: {domain: [name, ...]}`). A domain without an
entry falls back to its first label, with hyphens read as optional spaces ("crema-lab" matches
"Crema Lab", "crema-lab" and "CremaLab"). Matching is case-insensitive on whole words, spaces
and hyphens are interchangeable, and "&" also matches "and".
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence

from vizor.attribution.citations import strip_markers
from vizor.types import CitedSentence


def _name_pattern(name: str) -> str:
    parts = [p for p in re.split(r"[\s-]+", name.strip()) if p]
    out = []
    for p in parts:
        out.append("(?:&|and)" if p in ("&", "and") else re.escape(p))
    return r"[\s-]*".join(out)


def brand_regex(names: Iterable[str]) -> re.Pattern:
    alts = sorted({_name_pattern(n) for n in names if n.strip()}, key=len, reverse=True)
    return re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(alts) + r")(?![A-Za-z0-9])", re.I)


def default_names(domain: str) -> list[str]:
    label = domain.split(".")[0]
    return [label.replace("-", " ")]


def brand_patterns(
    domains: Iterable[str], brands: Mapping[str, Sequence[str]] | None = None
) -> dict[str, re.Pattern]:
    brands = brands or {}
    return {d: brand_regex(list(brands.get(d) or default_names(d))) for d in domains}


def mention_counts(sentences: Sequence[CitedSentence], pattern: re.Pattern) -> tuple[int, int]:
    """(sentences naming the brand, total brand mentions) with citation markers stripped."""
    hits = [len(pattern.findall(strip_markers(s.text))) for s in sentences]
    return sum(1 for h in hits if h), sum(hits)
