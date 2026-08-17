"""The committed Study 2 pre-registration inputs: queries and configs."""

import json
from collections import Counter

from tests.conftest import ROOT
from vizor.config import Config
from vizor.ingest.corpus import load_docs, load_project, load_queries


def test_study2_queries_are_fresh_and_balanced():
    project = load_project(ROOT / "data" / "bench" / "project_study2.yaml")
    qs = load_queries(project.queries_path)
    raw = [json.loads(line) for line in project.queries_path.read_text().splitlines()]
    assert len(qs) == 72 and len({q.query_id for q in qs}) == 72
    assert set(Counter(q.intent for q in qs).values()) == {18}
    targets = {
        d.url.split("larkspur.example/", 1)[1] for d in load_docs(project) if d.role == "target"
    }
    per_topic = Counter(r["topic"] for r in raw)
    assert set(per_topic) == targets and set(per_topic.values()) == {3}
    old = (ROOT / "data" / "bench" / "queries.jsonl").read_text().splitlines()
    assert not {json.loads(line)["query"] for line in old} & {q.text for q in qs}
    assert not any("larkspur" in q.text.lower() for q in qs)


def test_study2_configs_agree():
    main = Config.load(ROOT / "configs" / "study2_qwen3b_localhost_ai.yaml")
    dry = Config.load(ROOT / "configs" / "study2_dryrun.yaml")
    assert main.sandbox.arms == dry.sandbox.arms
    assert main.score.comparisons == dry.score.comparisons
    assert (
        main.score.prompt_arms
        == dry.score.prompt_arms
        == [f"content:{a}" for a in ("answer_first", "evidence_surface_llm", "faq_rewrite_v2")]
    )
    primary = [c for c in main.score.comparisons if c.family == "primary"]
    assert [c.set for c in primary] == main.score.prompt_arms
    assert main.sandbox.rewrite_min_accepted == dry.sandbox.rewrite_min_accepted == 12
    assert main.sandbox.weighting == "page" and main.render.passage_policy == "query-top3"
    assert main.llm.model == main.score.model == "qwen2.5-3b-mlx4"
    # second amendment: a thinking Qwen3.5-9B rewriter, read back from its frozen output
    rw = main.rewriter
    assert rw.model == "qwen3.5-9b-mlx4" and rw.chat_template_kwargs == {"enable_thinking": True}
    assert (rw.temperature, rw.top_p, rw.max_thinking_tokens) == (0.6, 0.95, 2048)
    assert main.sandbox.frozen_rewrites.endswith("_study2-rewrites/rewrites.json")
    assert not main.llm.chat_template_kwargs
    assert (
        main.llm.system_prompt
        == Config.load(ROOT / "configs" / "bench_qwen3b.yaml").llm.system_prompt
    )
