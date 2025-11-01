"""Plain-file run storage. No database: every run is a directory you can diff and commit.

<dir>/manifest.json        config, model ids, prompt hash, seeds, host, cost
<dir>/responses.jsonl.gz   every answer (arm, query, sample, seed, sources, raw text, usage)
<dir>/prompts.jsonl.gz     each distinct prompt once, keyed by hash
<dir>/rows.csv.gz          per answer x domain attribution metrics
"""

from __future__ import annotations

import gzip
import json
import os
import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd

from vizor.types import Answer


def write_jsonl_gz(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    # mtime=0 keeps the gzip bytes identical across reruns with identical content.
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        for r in records:
            gz.write((json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
            n += 1
    return n


def read_jsonl_gz(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            gz.write(df.to_csv(index=False, float_format="%.6g").encode("utf-8"))
    else:
        df.to_csv(path, index=False, float_format="%.6g")


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def src_tree() -> str:
    """Git tree hash of src/ at HEAD, plus "+dirty" if src/ has uncommitted changes: names the
    code even if commit hashes are later rewritten."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD:src"], capture_output=True, text=True, timeout=5
        )
        tree = out.stdout.strip() or "unknown"
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "src"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        return tree + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def host_manifest(**extra: str) -> dict:
    """Host/workload snapshot from an external helper named by $VIZOR_RUN_MANIFEST, if set."""
    tool = (
        Path(os.environ.get("VIZOR_RUN_MANIFEST", ""))
        if os.environ.get("VIZOR_RUN_MANIFEST")
        else None
    )
    if tool is not None and tool.exists():
        try:
            args = [f"{k}={v}" for k, v in extra.items()]
            out = subprocess.run(
                ["python3", str(tool), *args], capture_output=True, text=True, timeout=60
            )
            return json.loads(out.stdout)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    import platform

    return {
        "host": {
            "os": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "extra": extra,
    }


def load_answers(path: Path) -> list[tuple[str, Answer]]:
    return [(r["arm"], Answer.from_dict(r)) for r in read_jsonl_gz(path)]


def load_prompts(path: Path) -> dict[str, str]:
    return {r["prompt_hash"]: r["prompt"] for r in read_jsonl_gz(path)}
