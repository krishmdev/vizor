"""Page summaries used for query generation.

With an LLM: a neutral summary of at most 120 words covering offerings, claims and entities.
Offline: extractive centroid summary (embed the sentences, keep the k closest to the page
centroid, in their original order).
"""

from __future__ import annotations

import numpy as np

from vizor.attribution.citations import split_sentences
from vizor.embed import Embedder
from vizor.generate.llm import LLM
from vizor.types import SourceDoc

PROMPT = (
    "Summarize the web page below in at most 120 words, in a neutral tone. Cover what it offers, "
    "its main factual claims (numbers, prices, specs) and the brands or products it names. Use "
    "only information from the page.\n\nTitle: {title}\n\n{body}"
)


def extractive_summary(doc: SourceDoc, embedder: Embedder, k: int = 4) -> str:
    sents = [s for p in doc.paragraphs for s in split_sentences(p) if len(s.split()) >= 5]
    if len(sents) <= k:
        return " ".join(sents)
    v = embedder.encode(sents)
    c = v.mean(axis=0)
    keep = sorted(np.argsort(-(v @ c), kind="stable")[:k])
    return " ".join(sents[i] for i in keep)


def llm_summary(doc: SourceDoc, llm: LLM) -> str:
    c = llm.complete(
        [{"role": "user", "content": PROMPT.format(title=doc.title, body=doc.body[:6000])}],
        temperature=0.0,
        seed=0,
        max_tokens=220,
    )
    return c.text.strip()


def summarize(doc: SourceDoc, embedder: Embedder, llm: LLM | None = None) -> str:
    return llm_summary(doc, llm) if llm is not None else extractive_summary(doc, embedder)
