import shutil
import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module):
    root = tmp_path_factory.mktemp("root")
    for name in ("models.lock", "configs", "data"):
        src = __import__("tests.conftest", fromlist=["ROOT"]).ROOT / name
        (shutil.copytree if src.is_dir() else shutil.copy)(src, root / name)
    monkeypatch_module.setenv("VIZOR_ROOT", str(root))
    monkeypatch_module.setenv("VIZOR_CONFIG", str(root / "configs" / "ci.yaml"))
    from vizor.api import app as api

    api.STATE.update({"engine": None, "jobs": {}})
    with TestClient(api.app) as c:
        yield c
    api.STATE.update({"engine": None, "jobs": {}})


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_health_and_project(client):
    assert client.get("/health").json()["status"] == "ok"
    p = client.get("/project").json()
    assert p["is_fake_llm"] and "brewline.example" in p["domains"]
    assert len(client.get("/corpus/docs").json()) == 22


def test_answer_has_attribution(client):
    r = client.post("/answer", json={"query": "best espresso machine for milk drinks"})
    assert r.status_code == 200
    a = r.json()
    assert len(a["sources"]) == 5
    assert {s["label"] for s in a["sources"]} <= {"emphasized", "cited", "ignored"}
    assert sum(s["pwc_share"] for s in a["sources"]) == pytest.approx(1.0)
    assert all(1 <= c <= 5 for s in a["sentences"] for c in s["citations"])


def test_answer_with_policy_and_arm(client):
    r = client.post(
        "/answer", json={"query": "dual boiler", "policy": "target_at:1", "arms": ["faq_rewrite"]}
    )
    assert r.status_code == 200 and r.json()["sources"][0]["role"] == "target"
    assert (
        client.post("/answer", json={"query": "dual boiler", "policy": "sideways"}).status_code
        == 422
    )


def test_bad_ids(client):
    assert client.get("/experiments/nope").status_code == 404
    assert client.get("/experiments/..%2Fetc").status_code in (400, 404)


def test_background_sandbox_job_and_results(client):
    job = client.post(
        "/sandbox", json={"queries": 4, "samples": 1, "arms": ["noop", "faq_rewrite"]}
    ).json()
    for _ in range(120):
        st = client.get(f"/jobs/{job['job_id']}").json()
        if st["status"] != "running":
            break
        time.sleep(0.5)
    assert st["status"] == "done", st
    from vizor.api import app as api

    api.STATE["jobs"]["busy"] = {
        "job_id": "busy",
        "kind": "x",
        "status": "running",
        "out_dir": "",
        "error": None,
    }
    assert client.post("/sandbox", json={"queries": 4, "samples": 1}).status_code == 409
    del api.STATE["jobs"]["busy"]
    ids = [e["id"] for e in client.get("/experiments").json()]
    assert job["job_id"] in ids
    e = client.get(f"/experiments/{job['job_id']}").json()
    assert {d["arm"] for d in e["deltas"]} == {"noop", "faq_rewrite"}
    first = client.get(f"/experiments/{job['job_id']}/answers").json()[0]
    a = client.get(f"/experiments/{job['job_id']}/answers/{first['query_id']}/0").json()
    assert a["is_fake_llm"] and a["sources"]
