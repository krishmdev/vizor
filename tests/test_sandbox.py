import numpy as np
import pytest

from vizor.optimize.sandbox import Arm, Sandbox, boost_sweep, position_sweep
from vizor.optimize.transforms import TransformContext


@pytest.fixture(scope="module")
def sandbox(engine, docs, queries, project, hashing):
    doc_map = {d.doc_id: d for d in docs}
    qs = queries[::3]  # 14 queries across all intents
    sb = Sandbox(
        engine,
        qs,
        2,
        project.domains,
        lambda q: TransformContext(doc_map, list(q), hashing),
        bootstrap=500,
    )
    sb.run_baseline()
    return sb


def test_noop_gives_exactly_zero_under_common_random_numbers(sandbox):
    cmp = sandbox.compare(sandbox.baseline, sandbox.run(Arm.parse("noop")))
    assert cmp["d_pwc_pp"] == 0.0 and cmp["d_pwc_lo"] == 0.0 and cmp["d_pwc_hi"] == 0.0
    assert cmp["p"] == 1.0


def test_same_prompts_same_seeds_same_answers(sandbox):
    again = sandbox.run(Arm.parse("noop"))
    a = [r.answer.text for r in sandbox.baseline.results]
    assert a == [r.answer.text for r in again.results]


def test_aa_resample_changes_seeds_not_prompts(sandbox):
    aa = sandbox.run(Arm.parse("aa_resample"))
    base = sandbox.baseline.results
    assert [r.answer.prompt_hash for r in aa.results] == [r.answer.prompt_hash for r in base]
    assert all(x.answer.seed != y.answer.seed for x, y in zip(aa.results, base, strict=True))


def test_doc_arm_edits_only_the_focus_page_of_each_query(sandbox):
    run = sandbox.run(Arm.parse("faq_rewrite"))
    assert set(run.changed) == {"fold1", "fold2"}
    for pages in run.changed.values():
        assert all(d.role == "target" for d in pages.values())
    for qid, (_, doc_id) in run.scored_against.items():
        assert doc_id == sandbox.focus[qid]
    assert set(sandbox.focus) == {q.query_id for q in sandbox.queries}


def _page_text(doc):
    parts = [doc.title, doc.meta_description, doc.body, *(f"{q} {a}" for q, a in doc.faq)]
    return " ".join(parts).lower()


@pytest.mark.parametrize(
    "arm", ["faq_rewrite", "metadata", "keyword_stuffing", "faq_rewrite+metadata"]
)
def test_no_query_is_scored_against_a_page_built_from_it(sandbox, arm):
    run = sandbox.run(Arm.parse(arm))
    base_docs = sandbox.engine.cascade.docs
    text = {q.query_id: q.text.lower() for q in sandbox.queries}
    checked = 0
    for qid, (fold, doc_id) in run.scored_against.items():
        new = run.changed.get(fold, {}).get(doc_id)
        if new is None or text[qid] in _page_text(base_docs[doc_id]):
            continue
        checked += 1
        assert text[qid] not in _page_text(new), (arm, qid, fold)
    assert checked > 0


def test_folds_partition_queries(sandbox):
    a, b = sandbox.folds
    ids_a, ids_b = {q.query_id for q in a}, {q.query_id for q in b}
    assert not ids_a & ids_b and ids_a | ids_b == {q.query_id for q in sandbox.queries}


def test_holm_excludes_controls(sandbox):
    rows = [
        sandbox.compare(sandbox.baseline, sandbox.run(Arm.parse(a)))
        for a in ["noop", "aa_resample", "faq_rewrite", "jsonld_insert"]
    ]
    df = Sandbox.with_holm(rows).set_index("arm")
    assert df.loc[["noop", "aa_resample"], "p_holm"].isna().all()
    assert df.loc[["faq_rewrite", "jsonld_insert"], "p_holm"].notna().all()
    assert "beyond_aa" not in df.columns
    assert (df["significant"] == (df["p_holm"] < 0.05)).all()


def test_page_arms_are_tested_per_edited_page(sandbox):
    cmp = sandbox.compare(sandbox.baseline, sandbox.run(Arm.parse("faq_rewrite")))
    units = {doc_id for _, doc_id in sandbox.run(Arm.parse("faq_rewrite")).scored_against.values()}
    assert cmp["unit"] == "page" and cmp["n_units"] == len(units) < cmp["n_queries"]
    eng = sandbox.compare(sandbox.baseline, sandbox.run(Arm.parse("engine:reverse")))
    assert eng["unit"] == "query" and eng["n_units"] == eng["n_queries"]
    assert "d_pwc_given_cited_pp" in cmp


def test_position_sweep_places_target(sandbox):
    df, runs = position_sweep(sandbox, [1, 3, 5])
    assert list(df.position) == [1, 3, 5]
    for p, run in runs.items():
        for r in run.results:
            if r.answer.usage.get("forced"):
                assert r.answer.sources[p - 1].domain == "brewline.example"


def test_boost_sweep_is_monotone_in_retrieval(sandbox):
    df, _ = boost_sweep(sandbox, [0.0, 0.05, 0.5])
    assert np.all(np.diff(df.retrieval_pct.to_numpy()) >= 0)
    # a large post-rerank boost puts a target page in the prompt exactly when one is a candidate
    with_candidate = {
        r.answer.query_id
        for r in sandbox.baseline.results
        if any(c.doc.role == "target" for c in r.selection.candidates)
    }
    assert df.retrieval_pct.iloc[-1] == pytest.approx(
        100 * len(with_candidate) / len(sandbox.queries)
    )
    last = df.iloc[-1]
    assert last.n_set_changed + last.n_order_only + last.n_unchanged == len(sandbox.queries)


def _sources(run):
    return {r.answer.query_id: [s.doc_id for s in r.answer.sources] for r in run.results}


def test_content_arm_pins_sources_and_order_to_the_baseline(sandbox):
    base = _sources(sandbox.baseline)
    full = sandbox.run(Arm.parse("keyword_stuffing"))
    content = sandbox.run(Arm.parse("content:keyword_stuffing"))
    assert content.arm.mode == "content" and content.arm.base_name == "keyword_stuffing"
    assert _sources(content) == base
    # same edited pages as the full arm, so the two are paired page by page
    assert content.scored_against == full.scored_against
    assert {f: set(p) for f, p in content.changed.items()} == {
        f: set(p) for f, p in full.changed.items()
    }


def test_content_arm_renders_the_edited_page(sandbox):
    content = sandbox.run(Arm.parse("content:keyword_stuffing"))
    shown = 0
    for r in content.results:
        fold, doc_id = content.scored_against[r.answer.query_id]
        if doc_id in [s.doc_id for s in r.answer.sources]:
            new = content.changed[fold][doc_id]
            assert new.applied_transforms == ("keyword_stuffing",)
            shown += "Related searches:" in r.prompt
    assert shown > 0


def test_content_arm_equals_baseline_when_the_edit_is_a_noop(sandbox):
    cmp = sandbox.compare(sandbox.baseline, sandbox.run(Arm.parse("content:noop")))
    assert cmp["d_pwc_pp"] == 0.0 and cmp["mode"] == "content"


def test_compare_reports_mentions_and_format_rates(sandbox):
    cmp = sandbox.compare(sandbox.baseline, sandbox.run(Arm.parse("aa_resample")))
    for k in (
        "d_mention_pp",
        "d_mention_lo",
        "p_mention",
        "d_mention_share_pp",
        "unparsed_rate_pct",
        "last_only_rate_pct",
        "mentioned_not_cited_pct",
        "base_csov_pct",
        "p_csov",
    ):
        assert k in cmp, k
    assert cmp["primary"] == "imp_pwc" and cmp["p"] == cmp["p_pwc"]


def test_primary_metric_sets_the_verdict_p(engine, docs, queries, project, hashing):
    doc_map = {d.doc_id: d for d in docs}
    sb = Sandbox(
        engine,
        queries[::5],
        2,
        project.domains,
        lambda q: TransformContext(doc_map, list(q), hashing),
        bootstrap=200,
        primary="c_share",
    )
    sb.run_baseline()
    cmp = sb.compare(sb.baseline, sb.run(Arm.parse("engine:reverse")))
    assert cmp["primary"] == "c_share" and cmp["p"] == cmp["p_csov"]
    with pytest.raises(ValueError):
        Sandbox(engine, queries[:2], 1, project.domains, lambda q: None, primary="clicks")


def test_content_arms_form_their_own_holm_family(sandbox):
    rows = [
        sandbox.compare(sandbox.baseline, sandbox.run(Arm.parse(a)))
        for a in ["aa_resample", "faq_rewrite", "jsonld_insert", "content:faq_rewrite"]
    ]
    df = Sandbox.with_holm(rows).set_index("arm")
    assert df.loc["content:faq_rewrite", "family"] == "content"
    # alone in its family, the content arm's Holm p is its raw p
    assert df.loc["content:faq_rewrite", "p_holm"] == pytest.approx(
        df.loc["content:faq_rewrite", "p"]
    )
    assert df.loc[["faq_rewrite", "jsonld_insert"], "p_mention_holm"].notna().all()


def test_decomposition_adds_up(sandbox):
    base = sandbox.baseline
    full = sandbox.run(Arm.parse("stats_surface"))
    content = sandbox.run(Arm.parse("content:stats_surface"))
    d = sandbox.decompose(base, full, content)
    for m in ("csov", "pwc", "mention"):
        assert d[f"total_{m}_pp"] == pytest.approx(d[f"content_{m}_pp"] + d[f"rank_{m}_pp"])
    cmp = sandbox.compare(base, content)
    assert cmp["n_sources_changed"] == 0
