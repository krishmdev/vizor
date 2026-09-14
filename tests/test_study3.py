"""Study 3: the three query-blind arms, the retrieval log, per-arm samples, the pre-registered
decision rule and the committed queries and configs. Offline (hashing embedder, FakeLLM)."""

import json
import re
import runpy
import sys
from collections import Counter

import pandas as pd
import pytest

from tests.conftest import ROOT
from vizor.attribution.citations import split_sentences
from vizor.config import Config
from vizor.ingest.corpus import load_docs, load_project, load_queries
from vizor.optimize.transforms import (
    FACT_PASSAGE_WORDS,
    MAX_ANCHOR_ADDED,
    TRANSFORMS,
    TransformContext,
    grounding_violations,
    key_fact,
    page_terms,
)
from vizor.retrieve.chunk import _windows, window_starts

BENCH3 = ROOT / "data" / "bench" / "project_study3.yaml"
ARMS = ["fact_passage", "entity_anchor", "retrieval_meta"]


@pytest.fixture(scope="module")
def bench():
    project = load_project(BENCH3)
    docs = load_docs(project)
    return project, docs, load_queries(project.queries_path)


@pytest.fixture(scope="module")
def ctx(bench, hashing):
    _, docs, _ = bench
    return TransformContext({d.doc_id: d for d in docs}, [], hashing)


def targets(bench):
    return [d for d in bench[1] if d.role == "target"]


def is_product(doc) -> bool:
    return doc.headings[0].startswith("Larkspur ")


# ------------------------------------------------------------------ queries and configs


def test_study3_queries_are_fresh_balanced_and_cover_every_page(bench):
    project, docs, qs = bench
    raw = [json.loads(line) for line in project.queries_path.read_text().splitlines()]
    assert len(qs) == 72 and len({q.query_id for q in qs}) == 72
    assert set(Counter(q.intent for q in qs).values()) == {18}
    pages = {d.url.split("larkspur.example/", 1)[1] for d in docs if d.role == "target"}
    per_topic = Counter(r["topic"] for r in raw)
    assert set(per_topic) == pages and set(per_topic.values()) == {3}
    old = set()
    for name in ("queries.jsonl", "queries_study2.jsonl"):
        lines = (ROOT / "data" / "bench" / name).read_text().splitlines()
        old |= {json.loads(line)["query"].strip().lower() for line in lines}
    assert not old & {q.text.strip().lower() for q in qs}
    assert not any("larkspur" in q.text.lower() for q in qs)


def test_study3_configs_agree():
    main = Config.load(ROOT / "configs" / "study3_qwen9b.yaml")
    pilot = Config.load(ROOT / "configs" / "study3_qwen9b_pilot.yaml")
    dry = Config.load(ROOT / "configs" / "study3_dryrun.yaml")
    # the pilot's answers are reused by the main run, so everything that keys the cache agrees
    assert main.llm == pilot.llm
    assert main.llm.model == "qwen3.5-9b-mlx4" and main.llm.workers == 1
    assert main.llm.chat_template_kwargs == {"enable_thinking": False}
    assert (main.llm.temperature, main.llm.max_tokens, main.seed) == (0.7, 450, 0)
    assert main.project == pilot.project == dry.project == "data/bench/project_study3.yaml"
    assert (main.samples, pilot.samples) == (3, 2)
    assert pilot.sandbox.arms == ["noop", "aa_resample"]
    twins = [f"content:{a}" for a in ARMS]
    assert main.sandbox.arms == dry.sandbox.arms == ["noop", "aa_resample", *ARMS, *twins]
    assert main.sandbox.arm_samples == dry.sandbox.arm_samples == dict.fromkeys(twins, 1)
    assert main.sandbox.primary_metric == "c_share" and main.sandbox.weighting == "page"
    assert main.render.passage_policy == "query-top3"
    assert not main.sandbox.position_sweep and not main.sandbox.boost_sweep
    assert main.rewriter is None
    s2 = Config.load(ROOT / "configs" / "study2_qwen3b_localhost_ai.yaml")
    assert main.llm.system_prompt == s2.llm.system_prompt


# ------------------------------------------------------------------ the three arms


def test_new_arms_are_registered_query_blind_and_deterministic(bench, hashing):
    _, docs, qs = bench
    dm = {d.doc_id: d for d in docs}
    for arm in ARMS:
        t = TRANSFORMS[arm]
        assert not t.uses_queries and not t.requires_llm and not t.fabrication_risk
    for doc in targets(bench)[:6]:
        for arm in ARMS:
            a, _ = TRANSFORMS[arm].apply(doc, TransformContext(dm, [], hashing))
            b, _ = TRANSFORMS[arm].apply(doc, TransformContext(dm, list(qs), hashing))
            assert a == b


def fact_paragraph(doc, new):
    assert new.paragraphs[0] == doc.paragraphs[0]
    assert new.paragraphs[:1] + new.paragraphs[2:] == doc.paragraphs
    return new.paragraphs[1]


def test_fact_passage_one_window_named_sentences_from_the_page(bench, ctx):
    for doc in targets(bench):
        new, diff = TRANSFORMS["fact_passage"].apply(doc, ctx)
        assert new is not doc and diff
        para = fact_paragraph(doc, new)
        assert len(para.split()) <= FACT_PASSAGE_WORDS
        # the whole passage falls inside one body window (alone, or joined to a short intro)
        assert any(" ".join(para.split()) in w for w in _windows(new.paragraphs))
        for sent in split_sentences(para):
            if is_product(doc):
                assert doc.headings[0].lower() in sent.lower(), sent
            else:
                assert "larkspur" in sent.lower(), sent
            assert key_fact(sent)
        assert not grounding_violations(new.body, doc.body + "\n" + "\n".join(doc.headings))


def test_entity_anchor_only_touches_window_leads_under_the_cap(bench, ctx):
    changed = 0
    for doc in targets(bench):
        new, _ = TRANSFORMS["entity_anchor"].apply(doc, ctx)
        added = new.n_words - doc.n_words
        assert 0 <= added <= MAX_ANCHOR_ADDED
        if new is doc:
            continue
        changed += 1
        leads = set()
        for pi, off in window_starts(doc.paragraphs):
            pos = 0
            for s in split_sentences(doc.paragraphs[pi]):
                if pos >= off:
                    leads.add(s)
                    break
                pos += len(s.split())
        old_s = [s for p in doc.paragraphs for s in split_sentences(p)]
        new_s = [s for p in new.paragraphs for s in split_sentences(p)]
        assert len(old_s) == len(new_s)
        for a, b in zip(old_s, new_s, strict=True):
            if a != b:
                assert a in leads
                assert "Larkspur" in b and b.count("Larkspur") > a.count("Larkspur")
    # every product page names itself somewhere; the guides have no product to anchor
    assert changed == sum(is_product(d) for d in targets(bench))


def test_entity_anchor_rewrites_pronouns_and_bare_references(ctx, bench):
    from vizor.optimize.transforms import _entity, anchor_sentence

    doc = next(d for d in targets(bench) if d.headings[0] == "Larkspur Beam 800")
    ent = _entity(doc, ctx)
    assert anchor_sentence("It charges by USB-C.", ent) == "The Larkspur Beam 800 charges by USB-C."
    assert anchor_sentence("The Beam 800 has four modes.", ent).startswith("The Larkspur Beam 800")
    assert anchor_sentence("Its lens is shaped.", ent) == "The Larkspur Beam 800's lens is shaped."
    assert anchor_sentence("The lens is shaped.", ent) == (
        "On the Larkspur Beam 800, the lens is shaped."
    )
    # other products of the site get the brand too, but never twice
    assert anchor_sentence("Pair it with the Lock D9.", ent) == "Pair it with the Larkspur Lock D9."
    assert anchor_sentence("The Larkspur Lock D9 is heavy.", ent).count("Larkspur") == 1


def test_retrieval_meta_rewrites_only_head_fields_from_page_terms(bench, ctx):
    for doc in targets(bench):
        terms = page_terms(doc, ctx)
        page = " ".join([*doc.headings, doc.body]).lower()
        assert terms and all(t.lower() in page for t in terms)
        new, _ = TRANSFORMS["retrieval_meta"].apply(doc, ctx)
        assert new.body == doc.body and new.faq == doc.faq and new.jsonld == doc.jsonld
        assert len(new.title) <= 70 and len(new.meta_description) <= 155
        assert new.title.startswith(doc.headings[0]) and new.title != doc.title
        assert new.headings[0].startswith(doc.headings[0] + ": ")
        assert len(new.headings) == len(doc.headings)
        for h_old, h_new in zip(doc.headings[1:], new.headings[1:], strict=True):
            assert h_new == h_old or h_new.startswith(h_old + ": ")
        # the description is made of page sentences
        assert all(s in doc.body for s in split_sentences(new.meta_description))


def test_guard_rejection_keeps_the_page_and_is_logged(bench, hashing):
    from vizor.optimize.transforms import _guarded

    doc = targets(bench)[0]
    c = TransformContext({doc.doc_id: doc}, [], hashing)
    bad = doc.evolve("fact_passage", body=doc.body + "\n\nIt costs $12345.")
    assert _guarded("fact_passage", doc, bad, c, doc.body, bad.body) is doc
    ok = doc.evolve("fact_passage", body=doc.body)
    assert _guarded("fact_passage", doc, ok, c, doc.body, doc.body) is ok
    assert [(e["accepted"], e["change"]) for e in c.events] == [
        (False, "changed"),
        (True, "unchanged"),
    ]
    assert "12345" in c.events[0]["violations"]


def test_arms_log_guard_verdicts_and_no_op_pages(bench, ctx):
    from vizor.experiment import guard_summary

    ctx.events.clear()
    for doc in targets(bench):
        for arm in ARMS:
            TRANSFORMS[arm].apply(doc, ctx)
    g = guard_summary(ctx.events)
    assert set(g) == set(ARMS)
    assert all(v["pages"] == 24 and v["rejected"] == 0 for v in g.values())
    guides = {d.doc_id for d in targets(bench) if not is_product(d)}
    assert set(g["entity_anchor"]["unchanged_pages"]) == guides
    assert not g["fact_passage"]["unchanged_pages"] and not g["retrieval_meta"]["unchanged_pages"]
    ctx.events.clear()


def test_window_starts_match_windows(bench):
    for doc in bench[1]:
        ps = doc.paragraphs
        chunks, starts = _windows(ps), window_starts(ps)
        assert len(chunks) == len(starts)
        for c, (pi, off) in zip(chunks, starts, strict=True):
            assert c.split()[:3] == ps[pi].split()[off : off + 3]


# ------------------------------------------------------------------ sandbox: log and samples


@pytest.fixture(scope="module")
def sandbox3(bench, hashing):
    from vizor.generate.engine import Engine
    from vizor.generate.fake_llm import FakeLLM
    from vizor.optimize.sandbox import Sandbox
    from vizor.retrieve.cascade import Cascade
    from vizor.retrieve.rerank import NoopReranker

    project, docs, qs = bench
    dm = {d.doc_id: d for d in docs}
    cascade = Cascade(docs, hashing, NoopReranker(), backend="numpy")
    sb = Sandbox(
        Engine(cascade, FakeLLM(hashing)),
        qs[::4],
        2,
        project.domains,
        lambda q: TransformContext(dm, list(q), hashing),
        bootstrap=200,
        primary="c_share",
        weighting="page",
        arm_samples={"content:fact_passage": 1},
    )
    sb.run_baseline()
    return sb, dm


def test_arm_samples_override_and_common_seeds(sandbox3):
    from vizor.optimize.sandbox import Arm

    sb, _ = sandbox3
    twin = sb.run(Arm.parse("content:fact_passage"))
    assert set(twin.rows["sample"]) == {0}
    base0 = {r.answer.query_id: r.answer.seed for r in sb.baseline.results if r.answer.sample == 0}
    assert {r.answer.query_id: r.answer.seed for r in twin.results} == base0
    full = sb.run(Arm.parse("fact_passage"))
    assert set(full.rows["sample"]) == {0, 1}


def test_retrieval_log_rows(sandbox3):
    from vizor.metrics.retrieval_log import retrieval_frame, retrieval_summary
    from vizor.optimize.sandbox import Arm

    sb, dm = sandbox3
    runs = [sb.baseline, sb.run(Arm.parse("content:fact_passage")), sb.run(Arm.parse("noop"))]
    runs.append(sb.run(Arm.parse("retrieval_meta")))
    df = retrieval_frame(
        runs,
        lambda r: (
            {q: d for q, (_, d) in r.scored_against.items()} if r.scored_against else sb.focus
        ),
        dm,
    )
    assert len(df) == 4 * len(sb.queries)
    by = {a: g.set_index("query_id") for a, g in df.groupby("arm")}
    base, twin, meta = by["baseline"], by["content:fact_passage"], by["retrieval_meta"]
    assert not base["shown_new"].any() and not by["noop"]["shown_new"].any()
    # pinned retrieval: same slots and ranks; the new key-facts passage is what is shown
    assert (twin["focus_slot"].fillna(0) == base["focus_slot"].fillna(0)).all()
    shown = twin[twin["focus_retrieved"]]
    assert len(shown) and shown["shown_new"].all()
    # retrieval_meta changes no passage, only the rendered title and description
    assert not meta["shown_new"].any()
    assert meta.loc[meta["focus_retrieved"], "title_changed"].astype(bool).all()
    assert not base.loc[base["focus_retrieved"], "title_changed"].astype(bool).any()
    for _, r in base.iterrows():
        if r["focus_retrieved"]:
            assert r["focus_rank"] >= 1 and 1 <= r["focus_slot"] <= 5
    s = retrieval_summary(df).set_index("arm")
    assert s.loc["baseline", "shown_new_all"] == 0
    assert s.loc["content:fact_passage", "shown_new_if_shown"] == 1


def test_retrieval_log_counts_cut_passages_as_original():
    from vizor.metrics.retrieval_log import _is_original

    texts = {"one two three four five six"}
    assert _is_original("one two three …", texts)
    assert not _is_original("one two seven …", texts)
    assert not _is_original("one two three", texts)


# ------------------------------------------------------------------ decision rule


def run_primary(tmp_path, monkeypatch, extra):
    argv = ["sampled_primary.py", str(tmp_path), "--config"]
    argv += [str(ROOT / "configs" / "study3_dryrun.yaml"), "--arms", *ARMS, *extra]
    monkeypatch.setattr(sys, "argv", argv)
    try:
        runpy.run_path(str(ROOT / "scripts" / "sampled_primary.py"), run_name="__main__")
    except SystemExit as e:
        assert e.code == 0
    return json.loads((tmp_path / "sampled_primary.json").read_text())


def test_study3_rule_needs_both_holm_tests_and_the_same_sign_vs_aa(tmp_path, monkeypatch):
    pages = [f"p{i}" for i in range(12)]
    qs = [(f"q{i}", pages[i % 12]) for i in range(24)]
    pd.DataFrame(qs, columns=["query_id", "focus_doc"]).to_csv(tmp_path / "queries.csv")
    rows = []
    for i, (q, _) in enumerate(qs):
        rows.append({"arm": "baseline", "query_id": q, "c_share": 0.5})
        # the A/A arm sits 10 points above the baseline: a lucky-low baseline
        rows.append({"arm": "aa_resample", "query_id": q, "c_share": 0.6})
        rows.append({"arm": "noop", "query_id": q, "c_share": 0.5})
        rows.append({"arm": "fact_passage", "query_id": q, "c_share": 0.3 + 0.001 * i})
        # +5 vs the baseline (both tests pass) but -5 vs the A/A arm: fails the sign condition
        rows.append({"arm": "entity_anchor", "query_id": q, "c_share": 0.55 + 0.001 * i})
        rows.append({"arm": "retrieval_meta", "query_id": q, "c_share": 0.5})
    pd.DataFrame(rows).to_csv(tmp_path / "per_query.csv")
    (tmp_path / "manifest.json").write_text(json.dumps({"rewrite_guard": {}}))
    loose = {r["arm"]: r for r in run_primary(tmp_path, monkeypatch, [])["arms"]}
    assert loose["entity_anchor"]["significant"] and "same_sign_vs_aa" not in loose["entity_anchor"]
    out = run_primary(tmp_path, monkeypatch, ["--require-aa-sign", "--study", "Study 3"])
    by = {r["arm"]: r for r in out["arms"]}
    assert by["fact_passage"]["significant"] and by["fact_passage"]["same_sign_vs_aa"]
    assert (
        by["entity_anchor"]["p_wilcoxon_holm"] < 0.05 and by["entity_anchor"]["p_perm_holm"] < 0.05
    )
    assert not by["entity_anchor"]["same_sign_vs_aa"] and not by["entity_anchor"]["significant"]
    assert not by["retrieval_meta"]["significant"]
    assert out["rule"].startswith("both Holm")
    md = (tmp_path / "sampled_primary.md").read_text()
    assert md.startswith("# Study 3 primary") and "`entity_anchor` -3.8 pp (opposite)" in md


def test_study3_secondary_script(tmp_path, monkeypatch):
    qs = [(f"q{i}", f"p{i % 6}") for i in range(12)]
    pd.DataFrame(qs, columns=["query_id", "focus_doc"]).to_csv(tmp_path / "queries.csv")
    pq, ret = [], []
    arms = ["baseline", *ARMS, *(f"content:{a}" for a in ARMS)]
    for i, (q, p) in enumerate(qs):
        for j, arm in enumerate(arms):
            pq.append({"arm": arm, "query_id": q, "c_share": 0.4 + 0.01 * j, "mentioned": 0.5})
            ret.append(
                {
                    "arm": arm,
                    "query_id": q,
                    "focus_doc": p,
                    "focus_rank": 1 + (i % 3) + (arm == "fact_passage"),
                    "focus_retrieved": not (arm == "fact_passage" and i == 0),
                    "shown_new": "fact_passage" in arm,
                }
            )
    pd.DataFrame(pq).to_csv(tmp_path / "per_query.csv")
    pd.DataFrame(ret).to_csv(tmp_path / "retrieval.csv")
    argv = ["study3_secondary.py", str(tmp_path), "--config"]
    argv.append(str(ROOT / "configs" / "study3_dryrun.yaml"))
    monkeypatch.setattr(sys, "argv", argv)
    try:
        runpy.run_path(str(ROOT / "scripts" / "study3_secondary.py"), run_name="__main__")
    except SystemExit as e:
        assert e.code == 0
    out = json.loads((tmp_path / "study3_secondary.json").read_text())
    fp = out["arms"]["fact_passage"]
    # full = content + rank, exactly, on page means
    assert fp["full_pp"]["est"] == pytest.approx(fp["content_pp"]["est"] + fp["rank_pp"]["est"])
    assert fp["rank_delta"]["est"] == pytest.approx(1.0)
    assert fp["retrieved_pp"]["est"] == pytest.approx(-100 / 12)
    assert fp["passages"]["fact_passage"]["shown_new_all"] == 1.0
    assert out["arms"]["retrieval_meta"]["passages"]["retrieval_meta"]["shown_new_all"] == 0.0
    assert re.search(
        r"Baseline: focus page shown in 100%", (tmp_path / "study3_secondary.md").read_text()
    )


@pytest.mark.parametrize("latency,applied", [(12.0, False), (16.0, True)])
def test_study3_pilot_mde_projection_and_contingency(tmp_path, monkeypatch, latency, applied):
    from vizor.runstore import write_jsonl_gz

    qs = [(f"q{i}", f"p{i % 12}") for i in range(24)]
    pd.DataFrame(qs, columns=["query_id", "focus_doc"]).to_csv(tmp_path / "queries.csv")
    rows = []
    for i, (q, _) in enumerate(qs):
        rows.append({"arm": "baseline", "query_id": q, "c_share": 0.5})
        rows.append({"arm": "aa_resample", "query_id": q, "c_share": 0.45 + 0.004 * i})
    pd.DataFrame(rows).to_csv(tmp_path / "per_query.csv")
    (tmp_path / "manifest.json").write_text(json.dumps({"samples": 2}))
    resp = [{"usage": {"latency_s": latency, "cached": False}} for _ in range(10)]
    resp += [{"usage": {"latency_s": 0.001, "cached": True}} for _ in range(10)]
    write_jsonl_gz(tmp_path / "responses.jsonl.gz", resp)
    monkeypatch.setattr(sys, "argv", ["study3_pilot.py", str(tmp_path)])
    try:
        runpy.run_path(str(ROOT / "scripts" / "study3_pilot.py"), run_name="__main__")
    except SystemExit as e:
        assert e.code == 0
    out = json.loads((tmp_path / "pilot.json").read_text())
    assert out["n_pages"] == 12 and out["answers_generated"] == 10
    assert out["seconds_per_answer_mean"] == pytest.approx(latency)
    assert out["contingency_applied"] is applied
    assert out["main_samples"] == (2 if applied else 3)
    ratio = out["aa_sd_page_pp_projected"] / out["aa_sd_page_pp"]
    assert ratio == pytest.approx(1.0 if applied else (2 / 3) ** 0.5)
    assert out["mde_pp_projected"] <= out["mde_pp_at_pilot_samples"]
