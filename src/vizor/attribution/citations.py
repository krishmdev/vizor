"""Split a generated answer into sentences and pull out its citation markers.

Every sentence counts toward N and positions, cited or not, because the paper's decay is over the
whole response. Word counts exclude the markers themselves.

Accepted marker forms (the index list may use commas, semicolons, "and", "&", ranges "1-3" or
"1 to 3", and may repeat the keyword):
  [1]  [1][2]  [1, 2]  [1-3]  [^1]  【1】
  [Source 1]  [Sources 1 and 3]  [source: 2]  [Source #2]  [Search result 4]  [1, Source 2]
  [Source [3]]  and the unbalanced [Source [3]
  (Source 2)  (Sources 1, 3)  (see Source 2; Source 4)
  bare "Source 3" / "Search result 3" in running text (capitalised, one or two digits)
A bare parenthesised number such as "(2)" is not a citation: it is too often a count or a list
item. Anything that still looks like a citation after parsing (for example "[Source A]") is kept
as `unparsed`, so format failures can be counted instead of silently read as "uncited".
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from vizor.types import CitedSentence

_KW = r"(?:search[\s-]+results?|sources?|results?|references?|refs?\.?|documents?|docs?|citations?)"
_SEP = r"(?:\s*(?:[-–—,;&]|\band\b|\bto\b)\s*)"
_ITEM = rf"(?:{_KW}\s*[:#]?\s*)?#?\d+"
_LIST = rf"{_ITEM}(?:{_SEP}{_ITEM})*"
# [Source [3]] / [Source [3] -> [Source 3]
_NESTED = re.compile(rf"\[\s*({_KW})\s*[:#]?\s*\[\s*(\d[^\[\]]*?)\s*\]\s*\]?", re.I)
_BRACKET = re.compile(rf"[\[【]\s*\^?\s*({_LIST})\s*[\]】]", re.I)
_PAREN = re.compile(
    rf"\(\s*(?:see\s+|cf\.?\s+|per\s+)?((?:{_KW})\s*[:#]?\s*#?\d+(?:{_SEP}{_ITEM})*)\s*\)",
    re.I,
)
# Bare references are read only when capitalised ("Source 3", "Search result 4"), with at most
# two digits, and not right after a number, so "2 sources 3 times" or "Source 800 lumens" are
# not citations.
_BARE_NUM = r"#?\d{1,2}(?!\d)"
_BARE = re.compile(
    r"(?<!\d\s)(?<!\d)\b((?:Search[\s-]+[Rr]esults?|SEARCH[\s-]+RESULTS?|Sources?|SOURCES?)"
    rf"\s*{_BARE_NUM}(?:\s*(?:,|\band\b|&)\s*{_BARE_NUM})*)\b(?!\s*times\b)"
)
_FORMS = (("bracket", _BRACKET), ("paren", _PAREN), ("bare", _BARE))
# Leftovers that look like an attempt at a citation: brackets holding a digit or a keyword.
_RESIDUE = re.compile(rf"[\[【][^\]】\n]{{0,40}}(?:\d|{_KW})[^\]】\n]{{0,40}}[\]】]", re.I)
# A single marker of any accepted form (bare excluded), used when moving and stripping markers.
_ANY = rf"(?:{_BRACKET.pattern}|{_PAREN.pattern})"
_MARKER = re.compile(_ANY, re.I)
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
# Markers right after a terminator belong to the sentence before ("x. [1] Next" -> "x [1]. Next"),
# unless running text continues in lowercase ("x. [3] states that ..."). Markers that name the
# source ("[Source 3]") are how small models open a sentence, so after a terminator they stay
# with the text that follows and only move back at the end of a line.
_TRAILING = re.compile(rf"([.!?])((?:\s*{_ANY})+)", re.I)
_HAS_KW = re.compile(_KW, re.I)
_WORD = re.compile(r"[A-Za-z0-9]+(?:['’.][A-Za-z0-9]+)*")


@dataclass(frozen=True)
class ParsedAnswer:
    sentences: list[CitedSentence]
    hallucinated: list[int]
    unparsed: list[str] = field(default_factory=list)
    forms: Counter = field(default_factory=Counter)


def _expand(group: str) -> list[int]:
    """'1, 2' -> [1, 2]; '1-3' and '1 to 3' -> [1, 2, 3]; keywords inside the list are ignored."""
    toks = re.findall(r"\d+|[-–—]|\bto\b", group, re.I)
    out: list[int] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if not t.isdigit():
            i += 1
            continue
        a = int(t)
        if i + 2 < len(toks) and not toks[i + 1].isdigit() and toks[i + 2].isdigit():
            b = int(toks[i + 2])
            # [1-3] expands; a huge or reversed range is kept as its two endpoints
            out.extend(range(a, b + 1) if a <= b <= a + MAX_RANGE else [a, b])
            i += 3
        else:
            out.append(a)
            i += 1
    return out


def _unnest(text: str) -> str:
    return _NESTED.sub(lambda m: f"[{m.group(1)} {m.group(2)}]", text)


def find_markers(text: str) -> list[tuple[int, int, str, list[int]]]:
    """(start, end, form, indices) for every citation marker, in order, without overlaps."""
    spans: list[tuple[int, int, str, list[int]]] = []
    taken: list[tuple[int, int]] = []
    for form, pat in _FORMS:
        for m in pat.finditer(text):
            if any(m.start() < b and a < m.end() for a, b in taken):
                continue
            spans.append((m.start(), m.end(), form, _expand(m.group(1))))
            taken.append((m.start(), m.end()))
    return sorted(spans)


def normalize_markers(text: str) -> str:
    """Move markers before the terminator: 'x.[1]' and 'x. [1]' both become 'x[1].'"""

    def move(m: re.Match) -> str:
        after = m.string[m.end() :]
        nxt = after.lstrip(" \t")[:1]
        at_line_end = nxt in ("", "\n", "\r")
        keep = nxt.islower() or (bool(_HAS_KW.search(m.group(2))) and not at_line_end)
        return m.group(0) if keep else m.group(2).lstrip() + m.group(1)

    return _TRAILING.sub(move, _unnest(text))


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
    text = _unnest(sentence)
    for _, pat in _FORMS[:2]:
        text = pat.sub("", text)
    return re.sub(r"\s+([.!?,;:])", r"\1", text).strip()


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
    unparsed: list[str] = []
    forms: Counter = Counter()
    for pos, s in enumerate(split_sentences(text)):
        cites: list[int] = []
        rest, last = [], 0
        for a, b, form, idx in find_markers(s):
            forms[form] += 1
            rest.append(s[last:a])
            last = b
            for c in idx:
                if 1 <= c <= n_sources:
                    cites.append(c)
                else:
                    hallucinated.append(c)
        rest = " ".join([*rest, s[last:]])
        unparsed += _RESIDUE.findall(rest)
        sentences.append(CitedSentence(pos, s, count_words(s, wordcount), tuple(cites)))
    return ParsedAnswer(sentences, hallucinated, unparsed, forms)
