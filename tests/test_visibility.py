import numpy as np
import pytest

from vizor.metrics.sentiment import VaderSentiment
from vizor.metrics.visibility import answer_rows, domain_summary, rows_frame, source_labels
from vizor.types import Answer, CitedSentence, SourceRef

DOMAINS = {"t.example": "target", "a.example": "competitor", "b.example": "competitor"}


def _answer(qid, sources, sents):
    refs = [
        SourceRef(i + 1, f"d{i}", d, f"https://{d}/{i}", 0.5, 0.0, 0.5)
        for i, d in enumerate(sources)
    ]
    cs = [CitedSentence(i, t, len(t.split()), tuple(c)) for i, (t, c) in enumerate(sents)]
    return Answer(qid, 0, 1, "m", "h", refs, " ".join(t for t, _ in sents), cs, [])


A1 = _answer(
    "q1",
    ["t.example", "a.example", "t.example"],
    [("one two three four", [1]), ("five six seven eight nine ten", [1, 2]), ("x y", [2])],
)
A2 = _answer("q2", ["a.example", "b.example"], [("great machine really good", [2])])


def test_labels_emphasized_cited_ignored():
    assert source_labels(A1) == ["emphasized", "cited", "ignored"]


def test_rows_aggregate_multiple_docs_of_one_domain():
    rows = {r["domain"]: r for r in answer_rows(A1, DOMAINS)}
    t = rows["t.example"]
    assert t["retrieved"] and t["n_sources"] == 2 and t["best_position"] == 1
    assert t["n_markers"] == 2 and t["c_share"] == pytest.approx(0.5)
    assert not t["ignored"]  # one of its two docs was cited
    b = rows["b.example"]
    assert not b["retrieved"] and b["imp_pwc"] == 0 and np.isnan(b["best_position"])


def test_domain_summary():
    df = rows_frame([*answer_rows(A1, DOMAINS), *answer_rows(A2, DOMAINS)])
    s = domain_summary(df).set_index("domain")
    assert s.loc["t.example", "retrieval_rate"] == 0.5
    assert s.loc["t.example", "citation_rate"] == 0.5
    assert s.loc["a.example", "conversion"] == 0.5
    assert s["c_sov"].sum() == pytest.approx(1.0)
    assert s.loc["b.example", "pawc_sov"] == pytest.approx(0.5)
    assert s.loc["t.example", "first_cite_sentence"] == 1.0


def test_answer_sentiment_uses_only_citing_sentences():
    vader = VaderSentiment()
    rows = {r["domain"]: r for r in answer_rows(A2, DOMAINS, sentiment=vader)}
    assert rows["b.example"]["answer_sentiment"] > 0.5
    assert np.isnan(rows["a.example"]["answer_sentiment"])
