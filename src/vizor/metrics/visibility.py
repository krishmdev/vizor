"""Per-answer attribution rows and domain-level visibility aggregates.

`answer_rows` turns one cited answer into one row per domain in the project (so domains that were
never retrieved still contribute zeros to the averages). `domain_summary` aggregates those rows
over a query set into Citation Share-of-Voice, PAWC share, citation rate, retrieval-to-citation
conversion and first-citation position.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from vizor.attribution.citations import strip_markers
from vizor.metrics.impression import Decay, decay_weights, impressions
from vizor.types import Answer

SentimentFn = Callable[[Sequence[str]], list[float]]


def source_labels(answer: Answer, decay: Decay = "paper") -> list[str]:
    """'emphasized' (largest PAWC share), 'cited', or 'ignored' for each source in the prompt."""
    n = len(answer.sources)
    imp = impressions(answer.sentences, n, decay)
    cited = {c for s in answer.sentences for c in s.citations}
    best = max(range(n), key=lambda i: imp.pwc[i]) if n and not imp.uncited else None
    labels = []
    for i in range(n):
        if i == best:
            labels.append("emphasized")
        elif i + 1 in cited:
            labels.append("cited")
        else:
            labels.append("ignored")
    return labels


def sentence_weights(answer: Answer, positions: set[int], decay: Decay = "paper") -> list[float]:
    """PAWC contribution of each sentence toward the given source positions."""
    d = decay_weights(len(answer.sentences), decay)
    out = []
    for s, w in zip(answer.sentences, d, strict=True):
        k = len(s.citations)
        share = sum(1 for c in s.citations if c in positions) / k if k else 0.0
        out.append(s.n_words * w * share)
    return out


def answer_rows(
    answer: Answer,
    domains: dict[str, str],
    decay: Decay = "paper",
    sentiment: SentimentFn | None = None,
    source_text: dict[int, str] | None = None,
) -> list[dict]:
    """One row per project domain. `domains` maps domain -> role."""
    n = len(answer.sources)
    imp = impressions(answer.sentences, n, decay)
    markers = [0] * n
    for s in answer.sentences:
        for c in s.citations:
            markers[c - 1] += 1
    total_markers = sum(markers)
    labels = source_labels(answer, decay)
    by_domain: dict[str, list[int]] = defaultdict(list)
    for ref in answer.sources:
        by_domain[ref.domain].append(ref.position)

    rows = []
    for domain, role in domains.items():
        positions = by_domain.get(domain, [])
        idx = [p - 1 for p in positions]
        n_markers = sum(markers[i] for i in idx)
        first = next(
            (s.pos for s in answer.sentences if any(c in positions for c in s.citations)), None
        )
        row = {
            "query_id": answer.query_id,
            "sample": answer.sample,
            "domain": domain,
            "role": role,
            "retrieved": bool(positions),
            "n_sources": len(positions),
            "best_position": min(positions) if positions else np.nan,
            "cited": n_markers > 0,
            "n_markers": n_markers,
            "c_share": n_markers / total_markers if total_markers else 0.0,
            "imp_word": sum(imp.word[i] for i in idx),
            "imp_pos": sum(imp.pos[i] for i in idx),
            "imp_pwc": sum(imp.pwc[i] for i in idx),
            "first_cite_sentence": first if first is not None else np.nan,
            "n_sentences": len(answer.sentences),
            "emphasized": any(labels[i] == "emphasized" for i in idx),
            "ignored": bool(idx) and all(labels[i] == "ignored" for i in idx),
            "answer_sentiment": np.nan,
            "source_sentiment": np.nan,
        }
        if sentiment is not None and n_markers:
            w = sentence_weights(answer, set(positions), decay)
            texts = [strip_markers(s.text) for s, wi in zip(answer.sentences, w, strict=True) if wi]
            ws = [wi for wi in w if wi]
            if texts and sum(ws) > 0:
                row["answer_sentiment"] = float(np.average(sentiment(texts), weights=ws))
        if sentiment is not None and source_text and positions:
            texts = [source_text[p] for p in positions if p in source_text]
            if texts:
                row["source_sentiment"] = float(np.mean(sentiment(texts)))
        rows.append(row)
    return rows


def rows_frame(rows: Iterable[dict]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def domain_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate answer rows (one per answer x domain) into per-domain visibility metrics."""
    out = []
    total_markers = df["n_markers"].sum()
    for (domain, role), g in df.groupby(["domain", "role"], sort=False):
        retrieved = g[g["retrieved"]]
        out.append(
            {
                "domain": domain,
                "role": role,
                "answers": len(g),
                "retrieval_rate": g["retrieved"].mean(),
                "citation_rate": g["cited"].mean(),
                "conversion": retrieved["cited"].mean() if len(retrieved) else np.nan,
                "c_sov": g["n_markers"].sum() / total_markers if total_markers else 0.0,
                "pawc_sov": g["imp_pwc"].mean(),
                "word_sov": g["imp_word"].mean(),
                "citations_per_answer": g["n_markers"].mean(),
                "first_cite_sentence": g["first_cite_sentence"].mean() + 1,
                "emphasized_rate": g["emphasized"].mean(),
                "ignored_rate": retrieved["ignored"].mean() if len(retrieved) else np.nan,
                "answer_sentiment": g["answer_sentiment"].mean(),
                "source_sentiment": g["source_sentiment"].mean(),
            }
        )
    return pd.DataFrame(out).sort_values("pawc_sov", ascending=False).reset_index(drop=True)
