"""Source-tagged answer prompt.

The instruction is adapted from GEO's `query_prompt` (src/generative_le.py, Apache-2.0; see NOTICE).
Each source is rendered with its title, URL, meta description, flattened JSON-LD and the passages
most relevant to the question. Showing metadata and structured data to the model is a modeling
assumption of this sandbox engine, not a claim about how any production answer engine works.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np

from vizor.retrieve.cascade import Candidate, Cascade
from vizor.retrieve.chunk import flatten_jsonld

INSTRUCTION = (
    "Write an accurate and concise answer for the given user question, using _only_ the provided "
    "search results. The answer should be correct, high-quality, and written by an expert using "
    "an unbiased and journalistic tone. Write in plain prose paragraphs of about 120 to 200 words, "
    "without headings. Every sentence in the answer should be _immediately followed_ by an in-line "
    "citation to the search result(s). The cited search result(s) should fully support _all_ the "
    "information in the sentence. Search results need to be cited using [index]. When citing "
    "several search results, use [1][2][3] format rather than [1, 2, 3]. You can use multiple "
    "search results to respond comprehensively while avoiding irrelevant search results."
)

MAX_SOURCE_WORDS = 350
MAX_JSONLD_CHARS = 400
N_PASSAGES = 3


@dataclass(frozen=True)
class RenderedSource:
    index: int
    title: str
    url: str
    description: str
    structured: str
    content: str


def _cap_words(text: str, limit: int) -> str:
    words = text.split(" ")
    if len(words) <= limit:
        return text
    return " ".join(words[:limit]).rstrip() + " …"


# How the passages shown for a source are chosen (`render.passage_policy`):
#   query-top3  the 3 body or FAQ passages most similar to the question (Study 1's engine)
#   body-top3   the 3 most similar body passages; FAQ passages never compete for a slot
#   top2+faq1   the 2 most similar body passages plus the most similar FAQ passage (a third body
#               passage when the page has no FAQ)
PASSAGE_POLICIES = ("query-top3", "body-top3", "top2+faq1")


def choose_passages(scored: list, policy: str = "query-top3") -> list:
    """The (passage, score) pairs to render, in page order."""
    if policy not in PASSAGE_POLICIES:
        raise ValueError(f"unknown passage policy {policy!r}")
    ranked = sorted(scored, key=lambda x: -x[1])
    body = [x for x in ranked if x[0].kind == "body"]
    faq = [x for x in ranked if x[0].kind == "faq"]
    if policy == "query-top3":
        chosen = [x for x in ranked if x[0].kind in ("body", "faq")][:N_PASSAGES]
    elif policy == "body-top3":
        chosen = body[:N_PASSAGES]
    else:
        chosen = body[: N_PASSAGES - 1] + (faq[:1] or body[N_PASSAGES - 1 : N_PASSAGES])
    return sorted(chosen, key=lambda x: x[0].order)


def render_source(
    index: int,
    cand: Candidate,
    cascade: Cascade,
    qvec: np.ndarray,
    passage_policy: str = "query-top3",
) -> RenderedSource:
    doc = cand.doc
    chosen = choose_passages(cascade.passage_scores(qvec, doc.doc_id), passage_policy)
    lines = [p.text for p, _ in chosen]
    if doc.links:
        lines.append("Related: " + "; ".join(f"{a} ({h})" for a, h in doc.links))
    content = _cap_words("\n".join(lines), MAX_SOURCE_WORDS)
    return RenderedSource(
        index=index,
        title=doc.title,
        url=doc.url,
        description=doc.meta_description,
        structured=flatten_jsonld(doc.jsonld, MAX_JSONLD_CHARS) if doc.jsonld else "",
        content=content,
    )


def format_prompt(question: str, sources: list[RenderedSource]) -> str:
    blocks = []
    for s in sources:
        block = [f"### Source [{s.index}]", f"Title: {s.title}", f"URL: {s.url}"]
        if s.description:
            block.append(f"Description: {s.description}")
        if s.structured:
            block.append(f"Structured data: {s.structured}")
        block.append("Content:")
        block.append(s.content)
        blocks.append("\n".join(block))
    return (
        f"{INSTRUCTION}\n\nQuestion: {question}\n\nSearch Results:\n\n" + "\n\n".join(blocks) + "\n"
    )


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


_SOURCE_HEADER = re.compile(r"^### Source \[(\d+)\]$", re.M)
_FIELD = re.compile(r"^(Title|URL|Description|Structured data): (.*)$")


def parse_prompt(prompt: str) -> tuple[str, list[RenderedSource]]:
    """Inverse of format_prompt: the question and the rendered sources."""
    q = re.search(r"^Question: (.*)$", prompt, re.M)
    question = q.group(1).strip() if q else ""
    starts = list(_SOURCE_HEADER.finditer(prompt))
    out = []
    for i, m in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(prompt)
        body = prompt[m.end() : end].strip("\n")
        fields = {"Title": "", "URL": "", "Description": "", "Structured data": ""}
        head, _, content = body.partition("\nContent:\n")
        for line in head.split("\n"):
            fm = _FIELD.match(line)
            if fm:
                fields[fm.group(1)] = fm.group(2)
        out.append(
            RenderedSource(
                index=int(m.group(1)),
                title=fields["Title"],
                url=fields["URL"],
                description=fields["Description"],
                structured=fields["Structured data"],
                content=content.rstrip("\n"),
            )
        )
    return question, out
