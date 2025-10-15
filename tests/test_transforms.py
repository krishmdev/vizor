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
