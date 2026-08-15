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
    assert grounding_violations("The Harrow weighs 2.1 kg.", old) == ["2.1", "2.1 kg"]
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


def test_guard_summary_counts_each_page_once():
    from vizor.experiment import guard_summary

    ev = [
        {"transform": "evidence_surface_llm", "doc_id": "a", "accepted": True, "violations": []},
        {
            "transform": "evidence_surface_llm",
            "doc_id": "b",
            "accepted": False,
            "violations": ["7"],
        },
        {
            "transform": "evidence_surface_llm",
            "doc_id": "b",
            "accepted": False,
            "violations": ["7"],
        },
    ]
    s = guard_summary(ev)["evidence_surface_llm"]
    assert (s["pages"], s["rejected"], s["rejection_rate"]) == (2, 1, 0.5)
    assert s["rejected_pages"] == {"b": ["7"]}


def test_rewriter_config_builds_its_own_model():
    from vizor.config import Config, make_llm

    cfg = Config.model_validate(
        {
            "llm": {"backend": "ollama", "model": "answer-model"},
            "rewriter": {"backend": "ollama", "model": "rewrite-model", "temperature": 0.0},
        }
    )
    assert make_llm(cfg).model_id == "answer-model"
    assert make_llm(cfg, llm_cfg=cfg.rewriter).model_id == "rewrite-model"


def test_evidence_surface_rejects_a_rewrite_that_drops_numbers(docs, ctx):
    from dataclasses import replace

    from vizor.optimize.transforms import evidence_surface_llm

    class _Short:
        model_id = "stub"

        def complete(self, messages, **kw):
            from vizor.generate.llm import Completion

            return Completion("A short page with no numbers at all.", "stub")

    doc = next(d for d in _targets(docs) if any(c.isdigit() for c in d.body))
    events = []
    new, _ = evidence_surface_llm(doc, replace(ctx, llm=_Short(), events=events))
    assert new is doc and any("numbers>" in v for v in events[-1]["violations"])


def test_guard_allows_ordinary_rewording_and_catches_swaps():
    from vizor.optimize.transforms import grounding_violations

    old = "Floor Pump\nThe pump reaches 160 psi (11 bar). It weighs 1.9 kg and costs $49."
    ok = (
        "This pump reaches 160 psi, or 11 bar. Additionally, it weighs 1.9 kg. "
        "Overall, it costs $49. Floor pump owners like it."
    )
    assert grounding_violations(ok, old) == []
    assert grounding_violations("The pump reaches 11 psi.", old) == ["11 psi"]
    assert grounding_violations("It costs $1.9.", old) == ["1.9 $"]
    assert grounding_violations("It weighs two kilograms.", old) == ["two"]
    # a sentence-initial name followed by another capitalized word is still a name
    assert grounding_violations("Acme Labs rated it highly.", old) == ["Acme", "Labs"]


def test_key_fact_units_are_tight():
    from vizor.optimize.transforms import key_fact

    assert not key_fact("Ride 2 in a row.") and not key_fact("Top 5 m of the path.")
    assert key_fact("It holds 20 litres.") and key_fact("Assist stops at 32 km/h.")


def test_guard_accepts_every_bench_page_unchanged():
    from tests.conftest import ROOT
    from vizor.ingest.corpus import load_docs, load_project
    from vizor.optimize.transforms import grounding_violations

    docs = load_docs(load_project(ROOT / "data" / "bench" / "project.yaml"))
    assert all(grounding_violations(d.body, d.body) == [] for d in docs if d.role == "target")


def test_mark_inconclusive_blocks_claims_for_a_rejected_rewrite():
    import pandas as pd

    from vizor.scoring import mark_inconclusive

    res = pd.DataFrame(
        {
            "set": ["content:evidence_surface_llm", "content:answer_first"],
            "significant": [True, True],
        }
    )
    guard = {"evidence_surface_llm": {"pages": 24, "rejected": 20}}
    out = mark_inconclusive(res, guard, 12)
    assert list(out["inconclusive"]) == [True, False]
    assert list(out["significant"]) == [False, True]
    assert mark_inconclusive(res, guard, 0)["significant"].all()


def test_pregenerate_script_reports_acceptance(tmp_path):
    import json
    import subprocess
    import sys

    import yaml

    from tests.conftest import ROOT

    cfg = {
        "project": "data/bench/project.yaml",
        "rewriter": {"backend": "fake"},
        "sandbox": {"rewrite_min_accepted": 12},
    }
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "pregenerate_rewrites.py"),
            "--config",
            str(tmp_path / "c.yaml"),
            "--out",
            str(tmp_path / "o"),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    # FakeLLM "rewrites" are extractive fragments, which the guard rejects
    assert res.returncode == 4, res.stderr
    out = json.loads((tmp_path / "o" / "rewrites.json").read_text())
    assert out["pages"] == 24 and out["accepted"] < 12 and not out["conclusive"]


def test_evidence_surface_uses_rewriter_settings_and_frozen_text(docs, ctx):
    from dataclasses import replace

    from vizor.generate.llm import Completion
    from vizor.optimize.transforms import evidence_surface_llm

    seen = {}

    class _Thinker:
        model_id = "thinker"

        def complete(self, messages, *, temperature, seed, max_tokens):
            seen.update(temperature=temperature, seed=seed, max_tokens=max_tokens)
            body = messages[-1]["content"].split("```\n", 1)[1].rsplit("\n```", 1)[0]
            return Completion(
                body, "thinker", {"thinking_tokens": 2048}, reasoning="Rated 9.9 by Acme."
            )

    doc = _targets(docs)[0]
    events = []
    rw = {"temperature": 0.6, "seed": 17, "max_tokens": 4096}
    new, _ = evidence_surface_llm(doc, replace(ctx, llm=_Thinker(), events=events, rewrite=rw))
    assert seen == rw and new is not doc and events[-1]["accepted"]
    # the reasoning is recorded but never guarded or used
    assert events[-1]["reasoning"] == "Rated 9.9 by Acme." and events[-1]["thinking_tokens"] == 2048
    # frozen: no call, the stored text goes through the same guard (a stray think block dropped)
    frozen = {doc.doc_id: "<think>Acme 9.9</think>\n" + events[-1]["text"]}
    again, _ = evidence_surface_llm(doc, replace(ctx, llm=None, events=events, frozen=frozen))
    assert again.body == new.body and events[-1]["accepted"] and events[-1]["frozen"]
    with pytest.raises(RuntimeError):
        evidence_surface_llm(_targets(docs)[1], replace(ctx, llm=None, frozen=frozen))


def test_frozen_rewrites_need_the_same_rewriter(tmp_path):
    import json

    from vizor.config import Config
    from vizor.experiment import load_frozen_rewrites
    from vizor.optimize.transforms import rewriter_fingerprint

    rw = {"backend": "openai_compat", "model": "r", "base_url": "http://x/v1", "temperature": 0.6}
    cfg = Config.model_validate({"rewriter": rw, "sandbox": {"frozen_rewrites": "rw.json"}})
    doc = {"rewriter": {"server_meta": {"commit": "c"}}, "accepted": 1, "pages": 1}
    doc["rewrites"] = [{"doc_id": "a", "text": "page"}]
    doc["fingerprint"] = rewriter_fingerprint(cfg.rewriter)
    (tmp_path / "rw.json").write_text(json.dumps(doc))
    texts, meta = load_frozen_rewrites(cfg, tmp_path)
    assert texts == {"a": "page"} and meta["server_commit"] == "c" and len(meta["sha256"]) == 64
    cfg.rewriter.seed = 1
    with pytest.raises(ValueError):
        load_frozen_rewrites(cfg, tmp_path)
