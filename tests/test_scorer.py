import json
import math
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from vizor.generate.prompt import RenderedSource, format_prompt
from vizor.generate.scorer import (
    CachedScorer,
    FakeScorer,
    LocalhostScorer,
    Site,
    renormalize,
)


def _prompt() -> str:
    return format_prompt(
        "how heavy is the pump",
        [
            RenderedSource(1, "Road bike", "https://a.example/x", "", "", "Frames and forks."),
            RenderedSource(
                2, "Floor pump", "https://b.example/y", "", "", "The pump weighs 1.9 kg."
            ),
        ],
    )


class _Stub(BaseHTTPRequestHandler):
    requests: list = []
    drop: tuple = ()
    renorm: dict | None = None

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Stub.requests.append((self.path, body))
        sites = []
        for s in body["sites"]:
            cands = {c: -float(i + 1) for i, c in enumerate(s["candidates"])}
            site = {"token_index": 7 + s["char_offset"], "candidates": cands}
            if _Stub.renorm is not None:
                site["renorm"] = _Stub.renorm
            sites.append(site)
        out = {
            "model": body["model"],
            "revision": "abc123",
            "commit": "deadbeefcafe",
            "tokenizer_sha": "t0k",
            "sites": sites,
        }
        for k in _Stub.drop:
            out.pop(k)
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture()
def stub_server():
    _Stub.requests, _Stub.drop, _Stub.renorm = [], (), None
    srv = HTTPServer(("127.0.0.1", 0), _Stub)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def test_localhost_scorer_speaks_the_contract(stub_server):
    sc = LocalhostScorer("qwen2.5-3b-mlx4", base_url=stub_server, server_meta={"commit": "deadbee"})
    msgs = [{"role": "user", "content": _prompt()}]
    res = sc.score(msgs, "It weighs 1.9 kg [2].", [Site(17, ("1", "2"))])
    path, body = _Stub.requests[-1]
    assert path == "/v1/score"
    assert body["model"] == "qwen2.5-3b-mlx4"
    assert body["messages"] == msgs
    assert body["continuation"] == "It weighs 1.9 kg [2]."
    assert body["sites"] == [{"char_offset": 17, "candidates": ["1", "2"]}]
    assert body["chat_template_kwargs"] == {}
    (s,) = res.sites
    assert s.token_index == 24 and s.logprobs == {"1": -1.0, "2": -2.0}
    # renorm is filled in when the server leaves it out
    assert s.renorm["1"] == pytest.approx(1 / (1 + math.exp(-1)))
    pin = sc.pin()
    assert pin["commit"] == "deadbeefcafe" and pin["revision"] == "abc123"
    assert pin["tokenizer_sha"] == "t0k"


def test_localhost_scorer_refuses_a_different_commit(stub_server):
    sc = LocalhostScorer("m", base_url=stub_server, server_meta={"commit": "0123456"})
    with pytest.raises(RuntimeError, match="commit"):
        sc.pin()


def test_fake_scorer_prefers_the_matching_source():
    res = FakeScorer().score(
        [{"role": "user", "content": _prompt()}],
        "The pump weighs 1.9 kg [2].",
        [Site(24, ("1", "2"))],
    )
    (s,) = res.sites
    assert s.renorm["2"] > 0.9
    assert sum(s.renorm.values()) == pytest.approx(1.0)
    assert sum(math.exp(v) for v in s.logprobs.values()) == pytest.approx(1.0)


def test_cached_scorer_keys_on_the_pin(tmp_path):
    msgs = [{"role": "user", "content": _prompt()}]
    sites = [Site(24, ("1", "2"))]
    a = CachedScorer(FakeScorer(), tmp_path)
    r1 = a.score(msgs, "The pump weighs 1.9 kg [2].", sites)
    r2 = a.score(msgs, "The pump weighs 1.9 kg [2].", sites)
    assert (a.calls, a.hits) == (1, 1) and r2.cached
    assert r1.sites[0].renorm == pytest.approx(r2.sites[0].renorm)
    b = CachedScorer(FakeScorer(position_prior=0.3), tmp_path)
    b.score(msgs, "The pump weighs 1.9 kg [2].", sites)
    assert (b.calls, b.hits) == (1, 0)


def test_renormalize():
    p = renormalize({"1": math.log(0.2), "2": math.log(0.2)})
    assert p == pytest.approx({"1": 0.5, "2": 0.5})


@pytest.mark.slow
def test_mlx_scorer_on_real_weights():
    """Run by hand: VIZOR_MLX_REPO=mlx-community/Qwen2.5-3B-Instruct-4bit pytest -m slow"""
    import os

    pytest.importorskip("mlx_lm")
    repo = os.environ.get("VIZOR_MLX_REPO")
    if not repo:
        pytest.skip("set VIZOR_MLX_REPO to an MLX checkpoint")
    from vizor.generate.scorer import MLXScorer

    sc = MLXScorer(repo)
    msgs = [{"role": "user", "content": _prompt()}]
    res = sc.score(msgs, "The pump weighs 1.9 kg [2].", [Site(24, ("1", "2"))])
    (s,) = res.sites
    assert sum(s.renorm.values()) == pytest.approx(1.0)
    assert s.renorm["2"] > s.renorm["1"]
    again = sc.score(msgs, "The pump weighs 1.9 kg [2].", [Site(24, ("1", "2"))])
    assert again.sites[0].logprobs == pytest.approx(s.logprobs, abs=1e-5)


def test_localhost_pin_fails_closed(stub_server):
    for field in ("revision", "tokenizer_sha", "commit"):
        _Stub.drop = (field,)
        with pytest.raises(RuntimeError, match=field):
            LocalhostScorer("m", base_url=stub_server).pin()
    _Stub.drop = ()
    pin = LocalhostScorer("m", base_url=stub_server, chat_template_kwargs={"x": 1}).pin()
    assert pin["chat_template_kwargs"] == {"x": 1}


def test_localhost_renorm_is_client_side_and_checked(stub_server):
    sc = LocalhostScorer("m", base_url=stub_server)
    msgs = [{"role": "user", "content": _prompt()}]
    p1 = 1 / (1 + math.exp(-1))
    _Stub.renorm = {"1": p1 + 5e-6, "2": 1 - p1}  # within 1e-5: accepted, not used
    (s,) = sc.score(msgs, "x [1].", [Site(3, ("1", "2"))]).sites
    assert s.renorm["1"] == renormalize(s.logprobs)["1"]
    _Stub.renorm = {"1": 0.5, "2": 0.5}
    with pytest.raises(ValueError, match="renorm"):
        sc.score(msgs, "x [1].", [Site(3, ("1", "2"))])
