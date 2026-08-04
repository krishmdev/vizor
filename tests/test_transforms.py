import re

import pytest

from vizor.attribution.citations import split_sentences
from vizor.optimize.transforms import DETERMINISTIC, TRANSFORMS, TransformContext, apply_to_targets


@pytest.fixture(scope="module")
def ctx(docs, queries, hashing):
    return TransformContext({d.doc_id: d for d in docs}, queries, hashing)


def _targets(docs):
    return [d for d in docs if d.role == "target"]


def _page_sentences(doc):
    return {s for p in doc.paragraphs for s in split_sentences(p)}


def test_noop_returns_the_same_object(docs, ctx):
    for d in _targets(docs):
        assert TRANSFORMS["noop"].apply(d, ctx)[0] is d


def test_faq_answers_are_page_sentences(docs, ctx):
    for d in _targets(docs):
        new, _ = TRANSFORMS["faq_rewrite"].apply(d, ctx)
        sents = _page_sentences(d)
        for q, a in new.faq:
            assert q.endswith("?")
            assert all(s in sents for s in split_sentences(a)), a


def test_internal_links_point_at_real_same_site_pages(docs, ctx):
    urls = {x.url for x in docs}
    for d in _targets(docs):
        new, _ = TRANSFORMS["internal_links"].apply(d, ctx)
        assert new.links
        for _, href in new.links:
            assert href in urls and href.startswith(f"https://{d.domain}/") and href != d.url


def test_jsonld_only_uses_prices_on_the_page(docs, ctx):
    for d in _targets(docs):
        new, _ = TRANSFORMS["jsonld_insert"].apply(d, ctx)
        for item in new.jsonld:
            if item["@type"] == "Product":
                price = item["offers"]["price"]
                assert re.search(r"\$" + f"{int(float(price)):,}", d.body)
            assert "aggregateRating" not in item


def test_metadata_limits(docs, ctx):
    for d in _targets(docs):
        new, _ = TRANSFORMS["metadata"].apply(d, ctx)
        assert len(new.title) <= 60 and 0 < len(new.meta_description) <= 155
        assert new.body == d.body


def test_surfacing_only_reorders(docs, ctx):
    for name in ("stats_surface", "quote_surface"):
        for d in _targets(docs):
            new, _ = TRANSFORMS[name].apply(d, ctx)
            assert sorted(_page_sentences(new)) == sorted(_page_sentences(d)) or new is d


def test_every_deterministic_arm_versions_the_doc(docs, ctx):
    for name in DETERMINISTIC:
        changed, _ = apply_to_targets([name], ctx)
        for doc in changed.values():
            assert doc.version == 1 and doc.applied_transforms == (name,)


class _EchoLLM:
    """Rewriter stub: returns the page with a prefix line, optionally with an invented fact."""

    model_id = "stub-rewriter"

    def __init__(self, extra: str = ""):
        self.extra = extra

    def complete(self, messages, *, temperature, seed, max_tokens):
        from vizor.generate.llm import Completion

        assert temperature == 0.0
        body = messages[-1]["content"].split("```\n", 1)[1].rsplit("\n```", 1)[0]
        return Completion(f"```\n{self.extra}{body}\n```", self.model_id)


def test_grounding_guard_rejects_invented_numbers_and_names():
    from vizor.optimize.transforms import grounding_violations

    old = "The Harrow weighs 1.9 kg. It costs $1,399 and ships in May."
    assert grounding_violations("It costs $1399. The Harrow weighs 1.9 kg.", old) == []
    assert grounding_violations("The Harrow weighs 2.1 kg.", old) == ["2.1"]
    assert grounding_violations("It ships from Shimano in May.", old) == ["Shimano"]
    # sentence-initial capitals only need the word, in any case
    assert grounding_violations("Ships in May.", old) == []


def test_evidence_surface_keeps_the_page_when_the_guard_rejects(docs, ctx):
    from dataclasses import replace

    t = TRANSFORMS["evidence_surface_llm"]
    doc = _targets(docs)[0]
    events = []
    ok_ctx = replace(ctx, llm=_EchoLLM(), events=events)
    new, diff = t.apply(doc, ok_ctx)
    assert new is not doc and diff and events[-1]["accepted"]
    bad_ctx = replace(ctx, llm=_EchoLLM("Rated 4.9 stars by Acme Labs.\n\n"), events=events)
    new, diff = t.apply(doc, bad_ctx)
    assert new is doc and diff == ""
    assert not events[-1]["accepted"] and "4.9" in events[-1]["violations"]
    assert "Acme" in events[-1]["violations"]
    assert not t.uses_queries and t.requires_llm and not t.fabrication_risk


def test_answer_first_moves_sentences_and_adds_none(docs, ctx):
    from vizor.optimize.transforms import MAX_LEAD_FACTS, answer_first, key_fact

    for doc in _targets(docs):
        new, diff = answer_first(doc, ctx)
        if new is doc:
            continue
        before = sorted(_page_sentences(doc))
        assert sorted(_page_sentences(new)) == before
        lead = split_sentences(new.paragraphs[0])
        assert 1 <= len(lead) <= MAX_LEAD_FACTS and all(key_fact(s) for s in lead)
    assert key_fact("It weighs 11.6 kg.") and key_fact("It costs $49.")
    assert key_fact("The drivetrain is 2 x 10.") and not key_fact("It has 3 modes.")


def test_faq_v2_is_query_blind_and_short(docs, ctx):
    from dataclasses import replace

    from vizor.optimize.transforms import faq_rewrite_v2

    blind = replace(ctx, queries=[])
    for doc in _targets(docs):
        new, _ = faq_rewrite_v2(doc, blind)
        added = new.faq[len(doc.faq) :]
        assert len(added) <= 2
        answers = [a for _, a in added]
        assert len(set(answers)) == len(answers)
        assert all(a in _page_sentences(doc) for a in answers)
        assert new.faq == faq_rewrite_v2(doc, ctx)[0].faq  # queries change nothing
    assert not TRANSFORMS["faq_rewrite_v2"].uses_queries
