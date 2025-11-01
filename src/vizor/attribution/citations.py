"""Split a generated answer into sentences and pull out the [n] citation markers.

Every sentence counts toward N and positions, cited or not, because the paper's decay is over the
whole response. Word counts exclude the markers themselves.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from vizor.types import CitedSentence

# Same bracket pattern as GEO's extract_citations_new, widened to also accept "1, 2" and "1-3".
_MARKER = re.compile(r"\[[^\w\s\]]*(\d+(?:\s*[-–,]\s*\d+)*)[^\w\s\]]*\]")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_HEADING = re.compile(r"^\s*#{1,6}\s+|^\s*\*\*[^*]+\*\*:?\s*$")
_ABBREV = {
    "e.g",
    "i.e",
    "etc",
    "vs",
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "inc",
    "ltd",
    "co",
    "approx",
    "fig",
    "st",
    "jr",
    "sr",
    "u.s",
    "oz",
    "lb",
}
# Abbreviations only when a number follows ("No. 5", "min. 3 bars"); otherwise ordinary words.
_ABBREV_BEFORE_DIGIT = {"no", "min", "max", "approx"}
MAX_RANGE = 20
_TRAILING = re.compile(r"([.!?])((?:\s*\[[^\]]*\d[^\]]*\])+)")
_WORD = re.compile(r"[A-Za-z0-9]+(?:['’.][A-Za-z0-9]+)*")


@dataclass(frozen=True)
class ParsedAnswer:
    sentences: list[CitedSentence]
    hallucinated: list[int]


def _expand(group: str) -> list[int]:
    out: list[int] = []
    for part in re.split(r"\s*,\s*", group):
        if re.search(r"[-–]", part):
            a, b = (int(x) for x in re.split(r"\s*[-–]\s*", part))
            # [1-3] expands; a huge or reversed range is kept as its two endpoints
            out.extend(range(a, b + 1) if a <= b <= a + MAX_RANGE else [a, b])
        elif part:
            out.append(int(part))
    return out


def normalize_markers(text: str) -> str:
    """Move markers before the terminator: 'x.[1]' and 'x. [1]' both become 'x[1].'"""
    return _TRAILING.sub(lambda m: m.group(2).lstrip() + m.group(1), text)


def _split_sentences(para: str) -> list[str]:
    out: list[str] = []
    start = 0
    for m in re.finditer(r"[.!?](?=\s+|$)", para):
        end = m.end()
        before = para[start:end]
        tail = re.search(r"([A-Za-z.]+)\.$", before)
        word = tail.group(1).lower().rstrip(".") if tail else ""
        if word in _ABBREV:
            continue
        if word in _ABBREV_BEFORE_DIGIT and re.match(r"\s+\d", para[end : end + 4]):
            continue
        if re.search(r"\d\.$", before) and re.match(r"\d", para[end : end + 1] or ""):
            continue
        out.append(before.strip())
        start = end
    rest = para[start:].strip()
    if rest:
        out.append(rest)
    return [s for s in out if s]


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    text = normalize_markers(text.replace("\r\n", "\n"))
    for para in re.split(r"\n\s*\n", text):
        lines = [ln for ln in para.split("\n") if ln.strip()]
        buf: list[str] = []
        for ln in lines:
            if _BULLET.match(ln) or _HEADING.match(ln):
                if buf:
                    sentences.extend(_split_sentences(" ".join(buf)))
                    buf = []
                stripped = _BULLET.sub("", ln).strip()
                sentences.extend(_split_sentences(stripped) or [stripped])
            else:
                buf.append(ln.strip())
        if buf:
            sentences.extend(_split_sentences(" ".join(buf)))
    return sentences


def strip_markers(sentence: str) -> str:
    return re.sub(r"\s+([.!?,;:])", r"\1", _MARKER.sub("", sentence)).strip()


def count_words(sentence: str, mode: str = "alnum") -> int:
    """'alnum' counts word tokens; 'geo_reference' mimics GEO's get_num_words (tokens > 2 chars)."""
    words = _WORD.findall(strip_markers(sentence))
    if mode == "geo_reference":
        return sum(1 for w in words if len(w) > 2)
    return len(words)


def parse_answer(text: str, n_sources: int, wordcount: str = "alnum") -> ParsedAnswer:
    """Split into sentences and collect valid citations. Indices outside 1..n_sources are dropped
    and returned as `hallucinated`, so they don't count toward |C(s)|. GEO's reference code keeps
    them in the divisor (and a [0] there wraps around to the last source)."""
    sentences: list[CitedSentence] = []
    hallucinated: list[int] = []
    for pos, s in enumerate(split_sentences(text)):
        cites: list[int] = []
        for m in _MARKER.finditer(s):
            for c in _expand(m.group(1)):
                if 1 <= c <= n_sources:
                    cites.append(c)
                else:
                    hallucinated.append(c)
        sentences.append(CitedSentence(pos, s, count_words(s, wordcount), tuple(cites)))
    return ParsedAnswer(sentences, hallucinated)
