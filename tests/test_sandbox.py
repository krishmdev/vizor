import numpy as np
import pytest

from vizor.optimize.sandbox import Arm, Sandbox, boost_sweep, position_sweep
from vizor.optimize.transforms import TransformContext


@pytest.fixture(scope="module")
def sandbox(engine, docs, queries, project, hashing):
    ctx = TransformContext({d.doc_id: d for d in docs}, queries[:12], hashing)
    sb = Sandbox(engine, queries[:12], 2, project.domains, ctx, bootstrap=500)
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


def test_doc_arm_only_changes_target_pages(sandbox):
    run = sandbox.run(Arm.parse("faq_rewrite"))
    assert run.changed and all(d.role == "target" for d in run.changed.values())


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
    assert df.retrieval_pct.iloc[-1] == 100.0
