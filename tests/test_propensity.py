"""Attribution propensity on the offline stack: FakeLLM answers, FakeScorer scores."""

import gzip
import json

import numpy as np
import pandas as pd
import pytest

from vizor.config import Config
from vizor.experiment import run_experiment
from vizor.metrics.propensity import ap_from_logprobs, ap_from_probs, citation_sites
from vizor.scoring import load_prompt_sets, paired_deltas, recompute_ap, score_run


def test_citation_sites_point_at_the_first_valid_digit():
    text = "Light is 800 lumens [2]. It weighs 118 g [1][3]. Also [Source 4]. Bad [9]."
    sites = citation_sites(text, 5)
    assert [text[s.char_offset] for s in sites] == ["2", "1", "3", "4"]
    assert all(s.candidates == ("1", "2", "3", "4", "5") for s in sites)
    assert [s.char_offset for s in sites][0] == text.index("2]")
    assert citation_sites("No citations here.", 5) == []
    # the arm's prompt may show a different number of sources than the reference's
    assert citation_sites("x [2].", 5, 3)[0].candidates == ("1", "2", "3")


def test_ap_from_probs_and_logprobs():
    probs = [{"1": 0.5, "2": 0.5}, {"1": 0.1, "2": 0.9}]
    assert ap_from_probs(probs, [2]) == pytest.approx(0.7)
    assert ap_from_probs(probs, [1, 2]) == pytest.approx(1.0)
    assert np.isnan(ap_from_probs([], [1]))
    lps = [{k: np.log(v) - 3.0 for k, v in p.items()} for p in probs]
    assert ap_from_logprobs(lps, [2]) == pytest.approx(0.7)


def _cfg(**score):
    return Config.model_validate(
        {
            "label": "ap test",
            "embedder": "hashing",
            "reranker": "none",
            "sentiment": "vader",
            "samples": 2,
            "queries": 8,
            "llm": {"backend": "fake"},
            "sandbox": {
                "bootstrap": 200,
                "primary_metric": "c_share",
                "weighting": "page",
                "arms": ["noop", "aa_resample", "faq_rewrite", "stats_surface"],
                "position_sweep": [1, 5],
                "boost_sweep": [],
            },
            "score": {"backend": "fake", "bootstrap": 200, "draws": 2000, **score},
            "bandit": {"rounds": 20, "runs": 2, "greedy_steps": 0},
        }
    )


@pytest.fixture(scope="module")
def scored(tmp_path_factory):
    cfg = _cfg(
        prompt_arms=["content:faq_rewrite", "content:stats_surface"],
        policies=["body-top3"],
        comparisons=[
            {"name": "faq pinned", "set": "content:faq_rewrite", "family": "primary"},
            {"name": "stats pinned", "set": "content:stats_surface", "family": "primary"},
            {"name": "faq full", "set": "faq_rewrite"},
            {"name": "aa", "set": "aa_resample"},
            {"name": "noop", "set": "noop"},
            {
                "name": "slot 5 vs 1",
                "set": "engine:target_at:5",
                "ref": "engine:target_at:1",
                "family": "slot",
            },
            {
                "name": "faq body-top3",
                "set": "content:faq_rewrite@body-top3",
                "ref": "baseline@body-top3",
            },
        ],
    )
    run = tmp_path_factory.mktemp("run")
    run_experiment(cfg, run, log=lambda _: None)
    out = score_run(run, cfg, log=lambda _: None)
    return run, out


def _rows(out):
    with gzip.open(out / "ap_rows.jsonl.gz", "rt") as f:
        return pd.DataFrame([json.loads(line) for line in f])


def test_prompt_only_sets_are_written(scored):
    run, _ = scored
    sets = load_prompt_sets(run)
    for name in ("content:faq_rewrite", "baseline@body-top3", "content:faq_rewrite@body-top3"):
        assert name in sets and len(sets[name]) == 8
    # no answers were sampled for the prompt-only sets
    with gzip.open(run / "responses.jsonl.gz", "rt") as f:
        arms = {json.loads(line)["arm"] for line in f}
    assert "content:faq_rewrite" not in arms


def test_aa_and_noop_are_exactly_zero(scored):
    _, out = scored
    res = pd.read_csv(out / "ap_deltas.csv").set_index("name")
    for name in ("aa", "noop"):
        assert res.loc[name, "d_ap_pp"] == 0.0
        assert res.loc[name, "d_ap_lo"] == 0.0 and res.loc[name, "d_ap_hi"] == 0.0
        assert res.loc[name, "p_perm"] == 1.0
    q = paired_deltas(_rows(out), "aa_resample", "baseline", {})
    assert (q["d"] == 0).all()


def test_pinned_sets_keep_the_baseline_slots(scored):
    _, out = scored
    df = _rows(out)
    base = df[df.set == "baseline"].set_index(["query_id", "ref_sample"])["target_slots"]
    for name in ("content:faq_rewrite", "content:stats_surface"):
        arm = df[df.set == name].set_index(["query_id", "ref_sample"])["target_slots"]
        assert (arm.map(tuple) == base.loc[arm.index].map(tuple)).all()
        assert (df.loc[df.set == name, "mode"] == "pinned").all()


def test_slot_control_moves_ap_down(scored):
    _, out = scored
    res = pd.read_csv(out / "ap_deltas.csv").set_index("name")
    assert res.loc["slot 5 vs 1", "d_ap_pp"] < 0
    assert res.loc["slot 5 vs 1", "family"] == "slot"
    # the primary family is Holm-adjusted over its two members
    prim = res[res.family == "primary"]
    assert len(prim) == 2 and (prim.p_perm_holm >= prim.p_perm).all()


def test_recompute_rebuilds_ap_and_catches_edits(scored, tmp_path):
    _, out = scored
    ok, n = recompute_ap(out)
    assert ok and n == len(_rows(out))
    rows = _rows(out)
    i = rows.index[rows.n_sites > 0][0]
    rows.loc[i, "ap"] = rows.loc[i, "ap"] + 0.01
    bad = tmp_path / "ap"
    bad.mkdir()
    with gzip.open(bad / "ap_rows.jsonl.gz", "wt") as f:
        for r in rows.to_dict("records"):
            f.write(json.dumps(r) + "\n")
    assert recompute_ap(bad)[0] is False


def test_ap_manifest_records_the_scorer_pin(scored):
    _, out = scored
    m = json.loads((out / "ap_manifest.json").read_text())
    assert m["scorer_id"] == "fake-scorer/v1" and m["pin"]["backend"] == "fake"
    assert m["score_schema"] == "score-v1" and m["refs"] == [0, 1]
    assert "Attribution propensity" in (out / "ap_summary.md").read_text()


def test_cli_recompute_checks_ap_rows(scored):
    from typer.testing import CliRunner

    from vizor.cli import app

    run, _ = scored
    res = CliRunner().invoke(app, ["recompute", str(run)])
    assert res.exit_code == 0, res.output
    assert "AP rows recomputed in ap; match" in res.output
