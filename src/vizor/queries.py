"""Query generation from a corpus.

1. For each topic x intent, ask the LLM for JSON [{"query", "intent"}] from page summaries,
   oversampling 2x.
2. Greedy dedup: keep a query only if its max cosine to already-kept queries is below 0.88.
3. Relevance filter: max cosine to any corpus passage must be at least 0.30.
4. Balance intents, then cluster with KMeans (k by silhouette, at least 2).
Without an LLM this falls back to page headings and FAQ questions as candidates.
"""

from __future__ import annotations

import json
import re

import numpy as np

from vizor.embed import Embedder
from vizor.generate.llm import LLM
from vizor.retrieve.chunk import passages
from vizor.summarize import summarize
from vizor.types import Query, SourceDoc, stable_id

INTENTS = ["informational", "comparison", "transactional", "troubleshooting"]
DEDUP_COS = 0.88
MIN_RELEVANCE = 0.30

PROMPT = """You write realistic search queries that people type into an AI answer engine.
Topic: {topic}
Intent: {intent}
Here are summaries of pages in this space:
{summaries}

Write {n} distinct queries with this intent that these pages could help answer. Mix generic
queries with a few that name brands or products from the summaries. Return only a JSON list like
[{{"query": "...", "intent": "{intent}"}}]."""


def _parse_json_list(text: str) -> list[dict]:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return []
    return [d for d in data if isinstance(d, dict) and d.get("query")]


def dedup(texts: list[str], vecs: np.ndarray, threshold: float = DEDUP_COS) -> list[int]:
    kept: list[int] = []
    for i in range(len(texts)):
        if not kept or float(np.max(vecs[kept] @ vecs[i])) < threshold:
            kept.append(i)
    return kept


def cluster(vecs: np.ndarray, k_max: int = 8, seed: int = 0) -> np.ndarray:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = len(vecs)
    if n < 4:
        return np.zeros(n, dtype=int)
    best, labels = -1.0, np.zeros(n, dtype=int)
    for k in range(2, min(k_max, n - 1) + 1):
        lab = KMeans(k, n_init=10, random_state=seed).fit_predict(vecs)
        s = silhouette_score(vecs, lab)
        if s > best:
            best, labels = s, lab
    return labels


def candidates_from_llm(docs, topics, embedder, llm, per_intent) -> list[tuple[str, str]]:
    summaries = "\n".join(f"- {d.title}: {summarize(d, embedder, llm)}" for d in docs[:20])
    out = []
    for topic in topics or ["the products and services in this corpus"]:
        for intent in INTENTS:
            c = llm.complete(
                [
                    {
                        "role": "user",
                        "content": PROMPT.format(
                            topic=topic, intent=intent, summaries=summaries, n=2 * per_intent
                        ),
                    }
                ],
                temperature=0.7,
                seed=0,
                max_tokens=800,
            )
            out += [(d["query"].strip(), d.get("intent", intent)) for d in _parse_json_list(c.text)]
    return out


def candidates_offline(docs: list[SourceDoc]) -> list[tuple[str, str]]:
    out = []
    for d in docs:
        for q, _ in d.faq:
            out.append((q.rstrip("?").lower(), "informational"))
        for h in d.headings[1:]:
            if len(h.split()) >= 3:
                out.append((h.lower(), "informational"))
    return out


def generate_queries(cfg, per_intent: int = 10, llm: LLM | None = None) -> list[Query]:
    from vizor.config import build

    project, docs, _, engine = build(cfg)
    embedder: Embedder = engine.cascade.embedder
    if llm is None and cfg.llm.backend != "fake":
        llm = engine.llm
    cands = (
        candidates_from_llm(docs, project.topics, embedder, llm, per_intent)
        if llm is not None
        else candidates_offline(docs)
    )
    cands = [(q, i if i in INTENTS else "informational") for q, i in cands if q]
    if not cands:
        return []
    qv = embedder.encode([q for q, _ in cands], kind="query")
    keep = dedup([q for q, _ in cands], qv)
    pv = np.stack([engine.cascade._vec[p.passage_id] for d in docs for p in passages(d)])
    keep = [i for i in keep if float(np.max(pv @ qv[i])) >= MIN_RELEVANCE]
    by_intent: dict[str, list[int]] = {}
    for i in keep:
        by_intent.setdefault(cands[i][1], []).append(i)
    chosen = [i for group in by_intent.values() for i in group[:per_intent]]
    labels = cluster(qv[chosen])
    return [
        Query(f"{cands[i][1][:4]}-{stable_id(cands[i][0], 6)}", cands[i][0], cands[i][1], int(c))
        for i, c in zip(chosen, labels, strict=True)
    ]
