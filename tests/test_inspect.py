import shutil
from pathlib import Path

import pandas as pd
import pytest

from vizor.inspect import recompute_rows

PILOT = Path(__file__).parents[1] / "experiments/results/2026-09-05_bench-qwen3b-pilot"


@pytest.mark.skipif(not PILOT.exists(), reason="committed pilot run not present")
def test_recompute_checks_brand_mentions(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    for f in ("manifest.json", "responses.jsonl.gz", "rows.csv.gz"):
        shutil.copy(PILOT / f, run / f)
    ok, n = recompute_rows(run)
    assert ok and n == 5400
    rows = pd.read_csv(run / "rows.csv.gz")
    i = rows.index[rows["mentioned"]][0]
    rows.loc[i, "mentioned"] = False
    rows.to_csv(run / "rows.csv.gz", index=False)
    assert recompute_rows(run)[0] is False
