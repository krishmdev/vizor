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


def test_mde_line_names_cross_fitting_and_twin_samples_only_when_they_apply(run_dir, monkeypatch):
    import vizor.metrics.report as rep

    sens = {"arm_holm_family": 2, "content_holm_family": 2, "aa_sd_per_query_pp": 10.0}
    sens |= {"page_arm_page_pp": 5.0, "n_units_page": 4, "content_arm_page_pp": 5.0}
    monkeypatch.setattr(rep, "sensitivity", lambda *a, **k: dict(sens))
    r = load(run_dir)
    r["manifest"] = {**r["manifest"], "llm_is_fake": False}

    def mde(r):
        return next(x for x in claim_md(r).splitlines() if x.startswith("- Minimum detectable"))

    assert "(both cross-fit folds clustered by page)" in mde(r)  # faq_rewrite is cross-fitted
    assert "the twins ran at" not in mde(r)
    r["deltas"] = r["deltas"].assign(transform_scope="none")
    r["manifest"]["config"] = {"sandbox": {"arm_samples": {"content:faq_rewrite": 1}}}
    line = mde(r)
    assert "cross-fit" not in line
    assert "this assumes 2 samples per query, as for the page edits; the twins ran at 1" in line


def test_content_vs_rank_leads_with_the_same_seed_split_when_twins_have_fewer_samples(run_dir):
    r = load(run_dir)
    r["manifest"] = {**r["manifest"], "llm_is_fake": False}
    assert "- Content vs rank: for " in claim_md(r)
    r["manifest"]["config"] = {"sandbox": {"arm_samples": {"content:faq_rewrite": 1}}}

    def est(x):
        return {"est": x}

    s0 = {"content_pp": est(-2.34), "rank_pp": est(-4.28)}
    r["study3_secondary"] = {"arms": {"faq_rewrite": {"sample0": s0}}}
    r["decomposition"] = r["decomposition"].assign(n_sources_changed=[1, 5])
    lines = claim_md(r).splitlines()
    i = next(i for i, x in enumerate(lines) if "content-vs-rank split" in x)
    assert lines[i - 1].startswith("- Content vs rank, exploratory and not pre-registered")
    assert "`faq_rewrite` -2.3 content / -4.3 rank-mediated" in lines[i - 1]
    assert "confounded by the 1-sample twins" in lines[i]
    assert "`faq_rewrite` changed the sources of only 1 query, so its rank-mediated" in lines[i]
    assert "`stats_surface` changed" not in lines[i]
