import os
from pathlib import Path

import pytest

os.environ.setdefault("VIZOR_THREADS", "2")

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo" / "project.yaml"


@pytest.fixture(scope="session")
def project():
    from vizor.ingest.corpus import load_project

    return load_project(DEMO)


@pytest.fixture(scope="session")
def docs(project):
    from vizor.ingest.corpus import load_docs

    return load_docs(project)


@pytest.fixture(scope="session")
def queries(project):
    from vizor.ingest.corpus import load_queries

    return load_queries(project.queries_path)


@pytest.fixture(scope="session")
def hashing():
    from vizor.embed import HashingEmbedder

    return HashingEmbedder()


@pytest.fixture(scope="session")
def cascade(docs, hashing):
    from vizor.retrieve.cascade import Cascade
    from vizor.retrieve.rerank import NoopReranker

    return Cascade(docs, hashing, NoopReranker(), backend="numpy")


@pytest.fixture(scope="session")
def engine(cascade, hashing):
    from vizor.generate.engine import Engine
    from vizor.generate.fake_llm import FakeLLM

    return Engine(cascade, FakeLLM(hashing))
