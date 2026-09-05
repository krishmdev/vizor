"""Classify each accepted rewrite of a stored run as unchanged, reordered (the same sentences in
another order) or changed, from the run's diffs.json. The grounding guard accepts all three, so
this separates the accepted no-ops from real edits. Writes rewrite_changes.json in the run dir.

    uv run python scripts/rewrite_changes.py <run dir> [--arm evidence_surface_llm]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from vizor.optimize.transforms import rewrite_change


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--arm", default="evidence_surface_llm")
    a = ap.parse_args()
    diffs = json.loads((a.run / "diffs.json").read_text())[a.arm]
    guard = json.loads((a.run / "manifest.json").read_text()).get("rewrite_guard", {}).get(a.arm)
    pages = {
        d: rewrite_change(r["before"], r.get("after", ""))
        for f in diffs.values()
        for d, r in f.items()
    }
    counts = Counter(pages.values())
    out = {
        "run": a.run.name,
        "arm": a.arm,
        "pages_rewritten": guard["pages"] if guard else None,
        "rejected": guard["rejected"] if guard else None,
        "accepted": len(pages),
        "counts": {c: counts.get(c, 0) for c in ("changed", "reordered", "unchanged")},
        "pages": dict(sorted(pages.items())),
    }
    (a.run / "rewrite_changes.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["counts"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
