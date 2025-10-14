"""Turn a SourceDoc into retrievable passages.

Passage kinds: `head` (title, description, headings), `body` (120-word windows with 30-word overlap
that never cross a paragraph boundary unless a paragraph is short), `faq` (one per Q/A pair),
`jsonld` (flattened structured data) and `links` (a doc's related-links block).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from vizor.types import SourceDoc

WINDOW = 120
OVERLAP = 30


@dataclass(frozen=True)
class Passage:
    passage_id: str
    doc_id: str
    kind: str
    order: int
    text: str


def flatten_jsonld(items: tuple[dict, ...] | list[dict], limit: int | None = None) -> str:
    parts: list[str] = []

    def walk(x: object, key: str = "") -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("@context", "@id", "url", "image", "logo", "sameAs"):
                    continue
                walk(v, k)
        elif isinstance(x, list):
            for v in x:
                walk(v, key)
        elif isinstance(x, str | int | float) and str(x).strip():
            label = key.lstrip("@")
            parts.append(f"{label}: {x}" if label else str(x))

    for item in items:
        walk(item)
    text = "; ".join(parts)
    if limit and len(text) > limit:
        text = text[: limit - 1].rsplit(" ", 1)[0] + "…"
    return text


def _windows(paragraphs: list[str]) -> list[str]:
    chunks: list[str] = []
    buf: list[str] = []
    for para in paragraphs:
        words = para.split()
        if len(buf) + len(words) <= WINDOW:
            buf.extend(words)
            continue
        if buf:
            chunks.append(" ".join(buf))
            buf = []
        if len(words) <= WINDOW:
            buf = list(words)
            continue
        step = WINDOW - OVERLAP
        for start in range(0, len(words), step):
            piece = words[start : start + WINDOW]
            if len(piece) < OVERLAP and chunks:
                break
            chunks.append(" ".join(piece))
            if start + WINDOW >= len(words):
                break
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def passages(doc: SourceDoc) -> list[Passage]:
    out: list[Passage] = []

    def add(kind: str, text: str) -> None:
        if text.strip():
            n = len(out)
            out.append(Passage(f"{doc.doc_id}:{n}", doc.doc_id, kind, n, text.strip()))

    head = " ".join([doc.title, doc.meta_description, *doc.headings[:6]])
    add("head", head)
    for chunk in _windows(doc.paragraphs):
        add("body", chunk)
    for q, a in doc.faq:
        add("faq", f"Q: {q} A: {a}")
    if doc.jsonld:
        add("jsonld", flatten_jsonld(doc.jsonld, limit=1200))
    if doc.links:
        add("links", "Related: " + "; ".join(anchor for anchor, _ in doc.links))
    return out


def doc_fingerprint(doc: SourceDoc) -> str:
    import hashlib

    blob = json.dumps(doc.to_dict(), sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]
