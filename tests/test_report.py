"""End to end on the offline stack: a small run with content-only arms and a C-SoV primary
metric, then the generated report."""

import pandas as pd
import pytest

from vizor.config import Config
from vizor.experiment import run_experiment
from vizor.metrics.report import claim_md, load, readme_block, summary_md


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    cfg = Config.model_validate(
        {
            "label": "report test",
            "embedder": "hashing",
            "reranker": "none",
            "sentiment": "vader",
            "samples": 2,
            "queries": 8,
            "llm": {"backend": "fake"},
            "sandbox": {
                "bootstrap": 200,
                "primary_metric": "c_share",
                "arms": [
                    "noop",
                    "aa_resample",
                    "faq_rewrite",
                    "stats_surface",
                    "content:faq_rewrite",
                    "content:stats_surface",
                ],
                "position_sweep": [1, 5],
                "boost_sweep": [],
            },
            "bandit": {"rounds": 20, "runs": 2, "greedy_steps": 0},
        }
    )
    out = tmp_path_factory.mktemp("run")
    run_experiment(cfg, out, log=lambda _: None)
    return out


def test_run_writes_decomposition_and_families(run_dir):
    d = pd.read_csv(run_dir / "deltas.csv").set_index("arm")
    assert d.loc["content:faq_rewrite", "family"] == "content"
    assert d.loc["faq_rewrite", "family"] == "arms"
    assert (d["primary"] == "c_share").all()
    dec = pd.read_csv(run_dir / "decomposition.csv")
    assert set(dec.arm) == {"faq_rewrite", "stats_surface"}
    assert not (run_dir / "boost_sweep.csv").exists()
    m = load(run_dir)["manifest"]
    assert m["input_fingerprints_match"]
    assert len(m["source_sha256_start"]) == len(m["config_sha256"]) == 64
    assert len(m["corpus_sha256_start"]) == 64


def test_report_names_the_primary_metric_and_splits_content_from_rank(run_dir):
    md = summary_md(load(run_dir))
    assert md.startswith("# FakeLLM (pipeline check)")
    assert "ΔC-SoV pp [95% CI] (primary)" in md
    assert "## Content vs rank" in md
    assert "Unparsed markers %" in md and "Named %" in md


def test_position_verdict_uses_the_metric_behind_its_p_value(run_dir):
    r = load(run_dir)
    r["manifest"] = {**r["manifest"], "llm_is_fake": False}
    verdict = claim_md(r)
    assert "C-SoV slot 1" in verdict
    assert "on C-SoV is" in verdict
    assert "PAWC share slot 1" not in verdict


def test_invalid_input_fingerprint_suppresses_inferential_report(run_dir):
    r = load(run_dir)
    r["invalid_reason"] = "Source changed during this run."
    md = summary_md(r)
    block = readme_block([r])
    assert "**Invalid result:** Source changed during this run." in md
    assert "**Invalid result:** Source changed during this run." in block
    assert "Sandbox arms" not in md and "Holm-significant" not in block
