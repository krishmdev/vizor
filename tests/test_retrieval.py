import json
import subprocess
import sys

import numpy as np
import pytest

from vizor.embed import HashingEmbedder
from vizor.optimize.retrieval_policy import RetrievalPolicy
from vizor.retrieve.chunk import passages
from vizor.retrieve.index import EmbedderMismatch, IndexStore, VectorIndex


def test_hashing_embedder_is_normalized_and_deterministic(hashing):
    a = hashing.encode(["dual boiler espresso machine"])
    b = HashingEmbedder().encode(["dual boiler espresso machine"])
    assert np.allclose(a, b)
    assert np.isclose(np.linalg.norm(a[0]), 1.0)


def test_numpy_index_returns_exact_top_k(hashing):
    texts = [f"passage number {i} about grinders and boilers {i % 7}" for i in range(60)]
    idx = VectorIndex.build(hashing, [str(i) for i in range(60)], texts, backend="numpy")
    q = hashing.encode(["boilers 3"], kind="query")[0]
    got = idx.search(q, 10, hashing.embedder_id)
    brute = sorted(((str(i), float(v @ q)) for i, v in enumerate(idx.vectors)), key=lambda x: -x[1])
    assert [s for _, s in got] == pytest.approx([s for _, s in brute[:10]])


_PARITY = r"""
import json, sys
import numpy as np
from vizor.retrieve.index import VectorIndex
rng = np.random.default_rng(0)
v = rng.normal(size=(500, 64)).astype("float32")
v /= np.linalg.norm(v, axis=1, keepdims=True)
ids = [str(i) for i in range(500)]
a = VectorIndex("fixture", 64, ids, v, backend="numpy")
b = VectorIndex("fixture", 64, ids, v, backend="faiss")
out = []
for q in rng.normal(size=(5, 64)).astype("float32"):
    q /= np.linalg.norm(q)
    out.append([a.search(q, 20, "fixture"), b.search(q, 20, "fixture")])
assert "torch" not in sys.modules
print(json.dumps(out))
"""


def test_numpy_and_faiss_backends_agree():
    pytest.importorskip("faiss")
    # Run in a fresh process: on macOS faiss must never share a process with torch.
    r = subprocess.run(
        [sys.executable, "-c", _PARITY],
        capture_output=True,
        text=True,
        env={"PYTHONFAULTHANDLER": "1", "OMP_NUM_THREADS": "1", "PATH": ""},
        check=True,
    )
    for numpy_hits, faiss_hits in json.loads(r.stdout):
        assert [i for i, _ in numpy_hits] == [i for i, _ in faiss_hits]
        assert [s for _, s in numpy_hits] == pytest.approx([s for _, s in faiss_hits], abs=1e-5)


def test_index_refuses_other_embedders(hashing, tmp_path):
    idx = VectorIndex.build(hashing, ["a"], ["espresso"], backend="numpy")
    other = HashingEmbedder(dim=128)
    with pytest.raises(EmbedderMismatch):
        idx.search(other.encode(["x"])[0], 1, other.embedder_id)
    idx.save(tmp_path / "i")
    with pytest.raises(EmbedderMismatch):
        VectorIndex.load(tmp_path / "i", other.embedder_id)


class FlakyEmbedder(HashingEmbedder):
    def __init__(self, fail_after: int):
        super().__init__()
        self.embedder_id = "flaky/" + self.embedder_id
        self.calls = 0
        self.fail_after = fail_after

    def _encode(self, texts, kind):
        self.calls += 1
        if self.calls > self.fail_after:
            raise ConnectionError("embedding provider went away")
        return super()._encode(texts, kind)


def test_failed_rebuild_keeps_previous_generation_serving(tmp_path):
    emb = FlakyEmbedder(fail_after=1)
    store = IndexStore(tmp_path)
    store.build(emb, ["a", "b"], ["dual boiler", "flat burr grinder"], backend="numpy")
    first = store.current()
    with pytest.raises(ConnectionError):
        store.build(emb, ["a", "b", "c"], ["dual boiler", "flat burr grinder", "new text"], "numpy")
    assert store.current() == first
    served = store.load_current(emb.embedder_id, backend="numpy")
    assert served.ids == ["a", "b"]
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


def test_passages_cover_head_body_faq_jsonld_links(docs):
    aria = next(d for d in docs if d.url.endswith("/aria"))
    kinds = {p.kind for p in passages(aria)}
    assert kinds == {"head", "body", "faq", "jsonld", "links"}
    duo = next(d for d in docs if d.url.endswith("/duo"))
    assert {p.kind for p in passages(duo)} == {"head", "body"}
    assert all(len(p.text.split()) <= 160 for p in passages(duo))


def test_cascade_returns_five_distinct_docs(cascade):
    sel = cascade.select("single boiler vs dual boiler espresso machine")
    assert len(sel.sources) == 5
    assert len({c.doc.doc_id for c in sel.sources}) == 5
    scores = [c.final_score for c in sel.sources]
    assert scores == sorted(scores, reverse=True)


def test_target_at_places_exactly_one_target(cascade):
    for p in range(1, 6):
        sel = cascade.select(
            "best espresso machine for milk drinks", RetrievalPolicy.parse(f"target_at:{p}")
        )
        roles = [c.doc.role for c in sel.sources]
        assert sel.forced and roles.count("target") == 1 and roles.index("target") == p - 1


def test_boost_only_moves_target_docs_up(cascade):
    q = "how long should an espresso shot take"
    base = cascade.candidates(q)
    boosted = cascade.candidates(q, RetrievalPolicy(target_boost=0.1))
    b = {c.doc.doc_id: c.final_score for c in base}
    for c in boosted:
        delta = c.final_score - b[c.doc.doc_id]
        assert delta == pytest.approx(0.1 if c.doc.role == "target" else 0.0)


def test_policy_parse_round_trip():
    for spec in [
        "relevance",
        "reverse",
        "random",
        "target_at:3",
        "boost:0.05@post",
        "boost:0.02@pre",
    ]:
        assert RetrievalPolicy.parse(spec).name == spec
