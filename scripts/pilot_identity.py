"""Compare a main run's baseline and A/A answers with its pilot's (Study 3).

The Study 3 main run regenerated the pilot's baseline and A/A answers after Localhost AI's
history was re-dated (the server commit is part of the answer cache key). Both runs use the same
seeds and the same server code, so at batch 1 the answers should be byte-identical. This counts
the (arm, query, sample) answers present in both runs and how many have the same text, and writes
`pilot_identity.json` in the main run's directory.

    uv run python scripts/pilot_identity.py experiments/results/2026-09-28_study3-qwen9b \
        --pilot experiments/results/2026-09-18_study3-pilot
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

ARMS = ("baseline", "aa_resample")


def answers(run: Path) -> dict[tuple[str, str, int], str]:
    out = {}
    with gzip.open(run / "responses.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["arm"] in ARMS:
                out[(r["arm"], r["query_id"], r["sample"])] = r["text"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path, help="main run directory")
    ap.add_argument("--pilot", type=Path, required=True, help="pilot run directory")
    a = ap.parse_args()
    pilot, main_ = answers(a.pilot), answers(a.run)
    keys = sorted(set(pilot) & set(main_))
    out = {
        "pilot": a.pilot.resolve().name,
        "compared": len(keys),
        "identical": sum(pilot[k] == main_[k] for k in keys),
    }
    text = json.dumps(out)
    (a.run / "pilot_identity.json").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
