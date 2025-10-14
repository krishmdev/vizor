"""Projects and corpora.

A project (YAML) names the target and competitor domains and where their pages come from: a
directory of saved HTML (the bundled demo), or a corpus JSONL written by `vizor ingest`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from vizor.ingest.extract import extract
from vizor.types import Query, SourceDoc


@dataclass
class Project:
    name: str
    target_domains: list[str]
    competitor_domains: list[str]
    root: Path
    html_dir: Path | None = None
    corpus_jsonl: Path | None = None
    queries_path: Path | None = None
    topics: list[str] = field(default_factory=list)

    @property
    def domains(self) -> dict[str, str]:
        d = {x: "target" for x in self.target_domains}
        d.update({x: "competitor" for x in self.competitor_domains})
        return d

    def role(self, domain: str) -> str:
        return "target" if domain in self.target_domains else "competitor"


def load_project(path: str | Path) -> Project:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    root = path.parent

    def opt(key: str) -> Path | None:
        return (root / cfg[key]) if cfg.get(key) else None

    return Project(
        name=cfg["name"],
        target_domains=list(cfg["target_domains"]),
        competitor_domains=list(cfg.get("competitor_domains", [])),
        root=root,
        html_dir=opt("html_dir"),
        corpus_jsonl=opt("corpus_jsonl"),
        queries_path=opt("queries"),
        topics=list(cfg.get("topics", [])),
    )


def load_docs(project: Project) -> list[SourceDoc]:
    docs: list[SourceDoc] = []
    if project.html_dir and project.html_dir.exists():
        for f in sorted(project.html_dir.glob("*/**/*.html")):
            domain = f.relative_to(project.html_dir).parts[0]
            slug = "/".join(f.relative_to(project.html_dir / domain).with_suffix("").parts)
            url = f"https://{domain}/{'' if slug == 'index' else slug}"
            docs.append(
                extract(
                    f.read_text(encoding="utf-8"),
                    url,
                    role=project.role(domain),  # type: ignore[arg-type]
                    provenance={"source": "bundled", "path": str(f.relative_to(project.root))},
                )
            )
    if project.corpus_jsonl and project.corpus_jsonl.exists():
        docs.extend(read_corpus(project.corpus_jsonl))
    seen: set[str] = set()
    out = []
    for d in docs:
        if d.doc_id not in seen:
            seen.add(d.doc_id)
            out.append(d)
    return out


def read_corpus(path: Path) -> list[SourceDoc]:
    with open(path, encoding="utf-8") as fh:
        return [SourceDoc.from_dict(json.loads(line)) for line in fh if line.strip()]


def append_corpus(path: Path, docs: list[SourceDoc]) -> int:
    """Append docs, skipping any whose body is already present (dedup by content hash)."""
    import hashlib

    path.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if path.exists():
        for d in read_corpus(path):
            existing.add(hashlib.sha1(d.body.encode()).hexdigest())
    n = 0
    with open(path, "a", encoding="utf-8") as fh:
        for d in docs:
            h = hashlib.sha1(d.body.encode()).hexdigest()
            if h in existing or not d.body.strip():
                continue
            existing.add(h)
            fh.write(json.dumps(d.to_dict()) + "\n")
            n += 1
    return n


def load_queries(path: Path) -> list[Query]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                d = json.loads(line)
                out.append(Query(d["query_id"], d["query"], d["intent"], d.get("cluster_id", -1)))
    return out
