"""The validation gate end to end on the offline stack (FakeLLM run, FakeScorer)."""

import json
import subprocess
import sys

import pytest
import yaml

from tests.conftest import ROOT
from vizor.config import Config
from vizor.experiment import run_experiment
from vizor.gate import gate_md, run_gate

ARMS = ["faq_rewrite", "stats_surface"]


def _cfg():
    return Config.model_validate(
        {
            "embedder": "hashing",
            "reranker": "none",
            "sentiment": "vader",
            "samples": 2,
            "queries": 8,
            "llm": {"backend": "fake"},
            "sandbox": {
                "bootstrap": 200,
                "primary_metric": "c_share",
                "arms": ["noop", "aa_resample", *ARMS, *[f"content:{a}" for a in ARMS]],
                "position_sweep": [1, 5],
                "boost_sweep": [],
            },
            "score": {
                "backend": "fake",
                "bootstrap": 200,
                "draws": 2000,
                "comparisons": [
                    *[
                        {"name": f"pinned:{a}", "set": f"content:{a}", "family": "pinned"}
                        for a in ARMS
                    ],
                    *[{"name": f"full:{a}", "set": a, "family": "full"} for a in ARMS],
                    {
                        "name": "slot 5 vs 1",
                        "set": "engine:target_at:5",
                        "ref": "engine:target_at:1",
                        "family": "slot",
                    },
                ],
            },
            "gate": {
                "pairs": {
                    **{f"pinned:{a}": f"content:{a}" for a in ARMS},
                    **{f"full:{a}": a for a in ARMS},
                },
                "mde_family": 3,
            },
            "bandit": {"rounds": 20, "runs": 2, "greedy_steps": 0},
        }
    )


@pytest.fixture(scope="module")
def gated(tmp_path_factory):
    cfg = _cfg()
    run = tmp_path_factory.mktemp("run")
    run_experiment(cfg, run, log=lambda _: None)
    out = run / "gate"
    return run, out, run_gate(run, cfg, out, log=lambda _: None)


def test_gate_reports_all_four_criteria(gated):
    _, out, r = gated
    assert set(r["criteria"]) == {
        "1_slot_detected",
        "2_faq_negative",
        "3_ci_narrower",
        "4_spearman",
    }
    assert r["passed"] == all(c["pass"] for c in r["criteria"].values())
    assert r["criteria"]["4_spearman"]["n_points"] > 0
    assert r["criteria"]["3_ci_narrower"]["c_share_half_width_pp"] >= 0
    assert json.loads((out / "gate.json").read_text())["passed"] == r["passed"]
    assert "validation gate" in gate_md(r)
    lo, hi = r["study2_mde_pp"]["range"]
    assert 0 < lo <= hi


def test_gate_config_is_complete():
    cfg = Config.load(ROOT / "configs" / "study2_gate.yaml")
    names = {c.name for c in cfg.score.comparisons}
    assert {cfg.gate.slot, cfg.gate.faq_ap} <= names
    assert set(cfg.gate.pairs) <= names and len(cfg.gate.pairs) == 12
    assert cfg.score.backend == "localhost" and cfg.score.refs == [0, 1]


def test_gate_script_offline(gated, tmp_path):
    run, _, _ = gated
    cfg = tmp_path / "gate.yaml"
    cfg.write_text(yaml.safe_dump(_cfg().model_dump(mode="json")))
    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validation_gate.py"),
            "--run",
            str(run),
            "--config",
            str(cfg),
            "--out",
            str(tmp_path / "g"),
            "--backend",
            "fake",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert res.returncode in (0, 3), res.stderr
    assert (tmp_path / "g" / "gate.md").exists()
