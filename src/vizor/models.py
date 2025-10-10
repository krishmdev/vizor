"""Pinned Hugging Face artifacts.

Every model is fetched by repo *and* commit revision into the project-local HF cache (`.models/`,
see vizor/__init__.py) and checked against the sha256 hashes in `models.lock`. Nothing reads the
machine-wide HF cache, so a fresh clone gets exactly these bytes or fails loudly.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Pin:
    repo: str
    revision: str
    files: tuple[str, ...]


PINS: dict[str, Pin] = {
    "embedder": Pin(
        "BAAI/bge-small-en-v1.5",
        "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        (
            "1_Pooling/config.json",
            "config.json",
            "config_sentence_transformers.json",
            "model.safetensors",
            "modules.json",
            "sentence_bert_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
        ),
    ),
    "reranker": Pin(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "233902d25c440f23af6f7d6e94d2946bac0bee0a",
        (
            "config.json",
            "model.safetensors",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
        ),
    ),
    "sentiment": Pin(
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7",
        (
            "config.json",
            "merges.txt",
            "pytorch_model.bin",
            "special_tokens_map.json",
            "vocab.json",
        ),
    ),
}


def project_root() -> Path:
    env = os.environ.get("VIZOR_ROOT")
    if env:
        return Path(env)
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "models.lock").exists():
            return p
    here = Path(__file__).resolve().parents[2]
    if (here / "models.lock").exists():
        return here
    return Path.cwd()


def hf_home() -> Path:
    return Path(os.environ.get("HF_HOME", project_root() / ".models"))


def snapshot_dir(name: str) -> Path:
    pin = PINS[name]
    org, model = pin.repo.split("/")
    return hf_home() / "hub" / f"models--{org}--{model}" / "snapshots" / pin.revision


def limit_torch_threads() -> None:
    import torch

    torch.set_num_threads(int(os.environ.get("VIZOR_THREADS", "2")))


class ModelMissing(RuntimeError):
    pass


def local_path(name: str) -> str:
    """Path to the pinned snapshot. Loading from a path means no hub request at runtime."""
    snap = snapshot_dir(name)
    missing = [f for f in PINS[name].files if not (snap / f).exists()]
    if missing:
        raise ModelMissing(
            f"{PINS[name].repo}@{PINS[name].revision[:12]} is not in {hf_home()}; run `make models`"
        )
    return str(snap)


def model_available(name: str) -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    snap = snapshot_dir(name)
    return all((snap / f).exists() for f in PINS[name].files)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lock_path() -> Path:
    return project_root() / "models.lock"


def fetch(names: list[str] | None = None, write_lock: bool = False) -> dict:
    """Download pinned files, then verify them (or record their hashes with write_lock)."""
    from huggingface_hub import hf_hub_download

    names = names or list(PINS)
    lock = json.loads(lock_path().read_text()) if lock_path().exists() else {"models": {}}
    for name in names:
        pin = PINS[name]
        entry = {"repo": pin.repo, "revision": pin.revision, "files": {}}
        for f in pin.files:
            local = hf_hub_download(
                pin.repo, f, revision=pin.revision, cache_dir=str(hf_home() / "hub")
            )
            entry["files"][f] = _sha256(Path(local))
        if write_lock:
            lock["models"][name] = entry
    if write_lock:
        lock_path().write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    return verify(names)


def verify(names: list[str] | None = None) -> dict:
    lock = json.loads(lock_path().read_text())["models"]
    report = {}
    for name in names or list(PINS):
        pin, entry = PINS[name], lock[name]
        if (entry["repo"], entry["revision"]) != (pin.repo, pin.revision):
            raise RuntimeError(f"models.lock pins {entry['repo']}@{entry['revision']} for {name}")
        snap = snapshot_dir(name)
        bad = [f for f, digest in entry["files"].items() if _sha256(snap / f) != digest]
        if bad:
            raise RuntimeError(f"{name}: hash mismatch for {bad}")
        report[name] = f"{pin.repo}@{pin.revision[:12]} ok ({len(entry['files'])} files)"
    return report
