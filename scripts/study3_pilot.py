"""Study 3 pilot summary (docs/bench-design-study3.md, "Pilot and MDE"): the A/A page SD and MDE
of the sampled citation share at the pilot's 2 samples, the MDE projected to the main run's
samples, the time per answer, and the pre-registered budget contingency.

- A/A page deltas: aa_resample minus baseline, per page (unweighted mean of its queries), pp.
- MDE: t quantiles on n_pages - 1 df, 80% power, two-sided alpha 0.05 / 3 (Holm's first step).
- Projection: a page delta is a difference of two means of k samples per query, so its sampling
  SD scales with sqrt(1 / k); at k samples the SD is sd_2 * sqrt(2 / k). This assumes the A/A
  spread is all sampling noise, which is what an A/A re-sample measures.
- Time per answer: the mean `latency_s` of the answers the server generated (cache hits are
  left out). Above 15 s, the main run drops to 2 samples (the contingency).

    uv run python scripts/study3_pilot.py <pilot dir> [--family 3] [--main-samples 3]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from vizor.gate import c_share_page_deltas
from vizor.optimize.stats import mde
from vizor.runstore import read_jsonl_gz
from vizor.scoring import load_units

CONTINGENCY_S = 15.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--family", type=int, default=3)
    ap.add_argument("--main-samples", type=int, default=3)
    a = ap.parse_args()
    units = load_units(a.run)
    manifest = json.loads((a.run / "manifest.json").read_text())
    k = int(manifest["samples"])
    d = c_share_page_deltas(a.run, "aa_resample", units).to_numpy()
    sd = float(np.std(d, ddof=1))
    n = len(d)
    lat = [
        r["usage"]["latency_s"]
        for r in read_jsonl_gz(a.run / "responses.jsonl.gz")
        if not r["usage"].get("cached") and "latency_s" in r["usage"]
    ]
    per = float(np.mean(lat)) if lat else float("nan")
    over = bool(lat) and per > CONTINGENCY_S
    main_k = 2 if over else a.main_samples
    out = {
        "run": a.run.name,
        "samples": k,
        "n_pages": n,
        "aa_mean_pp": float(d.mean()),
        "aa_sd_page_pp": sd,
        "mde_pp_at_pilot_samples": mde(sd, n, a.family, t=True),
        "main_samples": main_k,
        "aa_sd_page_pp_projected": sd * float(np.sqrt(k / main_k)),
        "mde_pp_projected": mde(sd * float(np.sqrt(k / main_k)), n, a.family, t=True),
        "mde_method": "A/A page SD, t quantiles, 80% power, two-sided alpha 0.05 / family; "
        "projected by sqrt(pilot samples / main samples)",
        "family": a.family,
        "answers_generated": len(lat),
        "seconds_per_answer_mean": per,
        "seconds_per_answer_median": float(np.median(lat)) if lat else float("nan"),
        "generation_hours": float(np.sum(lat)) / 3600 if lat else 0.0,
        "contingency_threshold_s": CONTINGENCY_S,
        "contingency_applied": over,
    }
    (a.run / "pilot.json").write_text(json.dumps(out, indent=1) + "\n")
    lines = [
        f"# Study 3 pilot on {a.run.name}",
        "",
        f"- A/A (aa_resample minus baseline), {n} pages at {k} samples: mean "
        f"{out['aa_mean_pp']:+.2f} pp, page SD {sd:.2f} pp, "
        f"MDE {out['mde_pp_at_pilot_samples']:.1f} pp (t, 80% power, alpha 0.05 / {a.family}).",
        f"- Projected to {main_k} samples: page SD {out['aa_sd_page_pp_projected']:.2f} pp, "
        f"MDE {out['mde_pp_projected']:.1f} pp.",
        f"- Time per generated answer: mean {per:.1f} s, median "
        f"{out['seconds_per_answer_median']:.1f} s over {len(lat)} answers "
        f"({out['generation_hours']:.2f} h).",
        f"- Contingency (over {CONTINGENCY_S:.0f} s per answer: 2 samples in the main run): "
        + ("applied, main run at 2 samples." if over else f"not applied, main run at {main_k}."),
    ]
    (a.run / "pilot.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
