import json
import runpy
import sys

import pandas as pd

from tests.conftest import ROOT


def test_sampled_primary_holm_both_tests_and_rewrite_minimum(tmp_path, monkeypatch):
    pages = [f"p{i}" for i in range(12)]
    qs = [(f"q{i}", pages[i % 12]) for i in range(24)]
    pd.DataFrame(qs, columns=["query_id", "focus_doc"]).to_csv(tmp_path / "queries.csv")
    rows = []
    for i, (q, _) in enumerate(qs):
        rows.append({"arm": "baseline", "query_id": q, "c_share": 0.5})
        rows.append({"arm": "answer_first", "query_id": q, "c_share": 0.3 + 0.001 * i})
        rows.append({"arm": "evidence_surface_llm", "query_id": q, "c_share": 0.2})
        rows.append({"arm": "faq_rewrite_v2", "query_id": q, "c_share": 0.5 + (-1) ** i * 0.01})
        rows.append({"arm": "aa_resample", "query_id": q, "c_share": 0.5})
        rows.append({"arm": "noop", "query_id": q, "c_share": 0.5})
    pd.DataFrame(rows).to_csv(tmp_path / "per_query.csv")
    guard = {"evidence_surface_llm": {"pages": 12, "rejected": 11}}
    (tmp_path / "manifest.json").write_text(json.dumps({"rewrite_guard": guard}))
    cfg = ROOT / "configs" / "study2_dryrun.yaml"
    argv = ["sampled_primary.py", str(tmp_path), "--config", str(cfg)]
    monkeypatch.setattr(sys, "argv", argv)
    try:
        runpy.run_path(str(ROOT / "scripts" / "sampled_primary.py"), run_name="__main__")
    except SystemExit as e:
        assert e.code == 0
    out = json.loads((tmp_path / "sampled_primary.json").read_text())
    by = {r["arm"]: r for r in out["arms"]}
    assert by["answer_first"]["significant"] and by["answer_first"]["d_c_share_pp"] < -18
    # a large effect with too few accepted rewrites cannot count
    assert by["evidence_surface_llm"]["inconclusive"]
    assert not by["evidence_surface_llm"]["significant"]
    assert not by["faq_rewrite_v2"]["significant"]
    assert by["faq_rewrite_v2"]["p_perm_holm"] >= by["faq_rewrite_v2"]["p_perm"]
    # exact sign-flip p and the exploratory reference analysis are reported alongside
    assert 0 < by["answer_first"]["p_perm_exact"] <= 1
    ref = out["exploratory_reference"]
    aa_ref = {r["arm"]: r for r in ref["A/A re-sample"]}
    # the A/A arm equals the baseline here, so every reference gives the same delta
    assert aa_ref["answer_first"]["d_c_share_pp"] == by["answer_first"]["d_c_share_pp"]
    assert "Exploratory, not pre-registered" in (tmp_path / "sampled_primary.md").read_text()
