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


def test_sampled_primary_replaces_the_legacy_verdict(run_dir):
    r = load(run_dir)
    assert r["sampled_primary"] is None

    def row(arm, d, p, sig):
        return {
            "arm": arm,
            "n_pages": 4,
            "d_c_share_pp": d,
            "lo": d - 1,
            "hi": d + 1,
            "p_wilcoxon": p,
            "p_perm": p,
            "p_wilcoxon_holm": p,
            "p_perm_holm": p,
            "significant": sig,
        }

    ref = [
        {"arm": "faq_rewrite", "d_c_share_pp": -1.0, "p_wilcoxon_holm": 0.5},
        {"arm": "stats_surface", "d_c_share_pp": 0.0, "p_wilcoxon_holm": 1.0},
    ]
    for x in ref:
        x["p_perm_exact_holm"] = x["p_wilcoxon_holm"]
    r["sampled_primary"] = {
        "family": ["faq_rewrite", "stats_surface"],
        "arms": [row("faq_rewrite", -12.3, 0.01, True), row("stats_surface", 1.0, 0.9, False)],
        "controls": [row("aa_resample", -7.7, 0.2, False)],
        "gate": {"passed": False, "failed": ["3_ci_narrower", "4_spearman"]},
        "exploratory_reference": {"A/A re-sample": ref},
    }
    r["manifest"] = {**r["manifest"], "llm_is_fake": False}
    md = summary_md(r)
    assert "AP failed its validation gate on criteria 3 and 4" in md
    assert "Study 1 test (continuity, not this study's rule)" in md
    assert "| `faq_rewrite` | -12.3 [-13.3, -11.3]" in md
    assert "0.010 / 0.010 *" in md
    assert "moved C-SoV by -7.7 [-8.7, -6.7] pp on 4 page units" in md
    assert "1 of 2 have an effect under the sampled primary rule" in md
    assert "Exploratory, not pre-registered" in md
    assert "same sign" not in md
    # Study 3's rule adds the sign condition against the A/A arm
    r["sampled_primary"]["rule"] = "both Holm p < 0.05 and the delta vs A/A has the same sign"
    assert "its delta against the A/A re-sample has the same sign" in summary_md(r)


def test_bandit_line_is_marked_exploratory(run_dir):
    r = load(run_dir)
    r["manifest"] = {**r["manifest"], "llm_is_fake": False}
    line = next(x for x in claim_md(r).splitlines() if "bandit policy" in x)
    assert line.startswith("- Exploratory, not pre-registered: on the held-out queries")
