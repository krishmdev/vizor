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


def test_rows_count_brand_mentions_with_or_without_a_citation():
    a = _answer(
        "q3",
        ["a.example", "t.example"],
        [("The T brand pump is quiet", [1]), ("A is cheaper", [1]), ("Both last years", [])],
    )
    rows = {r["domain"]: r for r in answer_rows(a, DOMAINS, brands={"t.example": ["T brand"]})}
    t = rows["t.example"]
    assert t["mentioned"] and not t["cited"]
    assert t["n_mention_sentences"] == 1 and t["mention_share"] == pytest.approx(1 / 3)
    # without a brands entry the domain label is the brand
    assert rows["a.example"]["mentioned"] and rows["a.example"]["cited"]
    assert not rows["b.example"]["mentioned"]


def test_rows_flag_answers_that_cite_only_on_the_last_sentence():
    last = _answer("q4", ["t.example"], [("one two", []), ("three four", [1])])
    assert answer_rows(last, DOMAINS)[0]["cites_last_only"]
    assert not answer_rows(A1, DOMAINS)[0]["cites_last_only"]
    one = _answer("q5", ["t.example"], [("only one sentence", [1])])
    assert not answer_rows(one, DOMAINS)[0]["cites_last_only"]


def test_rows_count_unparsed_markers():
    a = _answer("q6", ["t.example"], [("quiet", [])])
    a.unparsed_markers = ["[Source A]"]
    assert answer_rows(a, DOMAINS)[0]["n_unparsed"] == 1


def test_domain_summary_reports_mentions_apart_from_citations():
    a = _answer("q7", ["a.example"], [("T is loud", [1])])
    s = domain_summary(rows_frame(answer_rows(a, DOMAINS))).set_index("domain")
    assert s.loc["t.example", "mention_rate"] == 1.0
    assert s.loc["t.example", "mentioned_not_cited_rate"] == 1.0
    assert s.loc["a.example", "cited_not_mentioned_rate"] == 1.0
