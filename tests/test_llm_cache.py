import pytest

from vizor.generate.llm import BudgetExceeded, CachedLLM, Completion, cost_usd


class Paid:
    model_id = "gpt-4o-mini"

    def __init__(self):
        self.calls = 0

    def complete(self, messages, *, temperature, seed, max_tokens):
        self.calls += 1
        return Completion(
            f"answer {seed}",
            "gpt-4o-mini-2024-07-18",
            {"prompt_tokens": 1_000_000, "completion_tokens": 0},
        )


def test_cache_hit_costs_nothing(tmp_path):
    inner = Paid()
    llm = CachedLLM(inner, tmp_path)
    msg = [{"role": "user", "content": "hi"}]
    a = llm.complete(msg, temperature=0.7, seed=1, max_tokens=10)
    b = llm.complete(msg, temperature=0.7, seed=1, max_tokens=10)
    assert inner.calls == 1 and b.cached and a.text == b.text
    assert llm.complete(msg, temperature=0.0, seed=1, max_tokens=10).cached is False


def test_ledger_cap_is_shared_across_instances(tmp_path):
    msg = [{"role": "user", "content": "x"}]
    first = CachedLLM(Paid(), tmp_path, max_cost_usd=0.2)
    first.complete(msg, temperature=0.7, seed=1, max_tokens=10)  # $0.15
    first.complete(msg, temperature=0.7, seed=2, max_tokens=10)  # $0.30 total
    second = CachedLLM(Paid(), tmp_path, max_cost_usd=0.2)
    with pytest.raises(BudgetExceeded):
        second.complete(msg, temperature=0.7, seed=3, max_tokens=10)
    # cached answers are still served past the cap
    assert second.complete(msg, temperature=0.7, seed=1, max_tokens=10).cached
    assert second.stats()["ledger_total_usd"] == pytest.approx(0.30)


def test_price_table():
    assert cost_usd(
        "gpt-4o-mini-2024-07-18", {"prompt_tokens": 2000, "completion_tokens": 300}
    ) == pytest.approx(0.00048)
    assert cost_usd("fake", {"prompt_tokens": 10}) == 0.0


def test_worst_case_reservation_blocks_before_calling(tmp_path):
    inner = Paid()
    llm = CachedLLM(inner, tmp_path, max_cost_usd=0.0001)
    big = [{"role": "user", "content": "x" * 20000}]
    with pytest.raises(BudgetExceeded):
        llm.complete(big, temperature=0.7, seed=1, max_tokens=450)
    assert inner.calls == 0 and llm.reserved == 0


def test_src_tree_marks_uncommitted_src_changes(tmp_path, monkeypatch):
    import subprocess

    from vizor.runstore import src_tree

    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    git("add", "src")
    git("-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "init")
    monkeypatch.chdir(tmp_path)
    clean = src_tree()
    assert len(clean) == 40 and not clean.endswith("+dirty")
    (tmp_path / "src" / "a.py").write_text("x = 2\n")
    assert src_tree() == clean + "+dirty"


def test_openai_compatible_local_server_needs_no_key(monkeypatch):
    from vizor.config import Config, make_llm

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = Config.model_validate(
        {
            "llm": {
                "backend": "openai_compat",
                "model": "local-preset",
                "base_url": "http://127.0.0.1:9",
                "server_meta": {"server": "x", "commit": "abc"},
            }
        }
    )
    llm = make_llm(cfg)
    assert llm.model_id == "local-preset" and llm.max_cost_usd is None
    assert cfg.model_dump()["llm"]["server_meta"]["commit"] == "abc"
    with pytest.raises(ValueError):
        make_llm(Config.model_validate({"llm": {"backend": "openai_compat"}}))
    # The server's URL and commit are part of the cache key; Ollama's key is unchanged.
    msgs = [{"role": "user", "content": "q"}]
    other = cfg.model_copy(deep=True)
    other.llm.server_meta["commit"] = "def"
    k = llm._key(msgs, 0.7, 1, 10)
    assert k != make_llm(other)._key(msgs, 0.7, 1, 10)
    assert llm.key_extra["base_url"] == "http://127.0.0.1:9"
    ollama = make_llm(Config.model_validate({"llm": {"backend": "ollama", "model": "m"}}))
    assert ollama.key_extra is None


def test_openai_compat_run_needs_server_commit(tmp_path):
    from vizor.config import Config
    from vizor.experiment import run_experiment

    cfg = Config.model_validate(
        {"llm": {"backend": "openai_compat", "model": "m", "base_url": "http://127.0.0.1:9"}}
    )
    with pytest.raises(ValueError, match="server-commit"):
        run_experiment(cfg, tmp_path / "out")


def test_request_extras_are_sent_and_keyed(tmp_path):
    import httpx

    from vizor.generate.llm import OpenAIChat

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "page",
                            "reasoning_content": "thought",
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 9,
                    "total_tokens": 12,
                    "thinking_tokens": 5,
                },
            },
        )

    extra = {"top_p": 0.95, "chat_template_kwargs": {"enable_thinking": True}}
    extra["max_thinking_tokens"] = 2048
    inner = OpenAIChat("m", base_url="http://x/v1", api_key="k", request_extra=extra)
    inner._client = inner._client.with_options(
        http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    llm = CachedLLM(inner, tmp_path, key_extra={"commit": "c"})
    msg = [{"role": "user", "content": "hi"}]
    c = llm.complete(msg, temperature=0.6, seed=7, max_tokens=4096)
    assert seen["top_p"] == 0.95 and seen["max_thinking_tokens"] == 2048
    assert seen["chat_template_kwargs"] == {"enable_thinking": True} and seen["seed"] == 7
    assert c.text == "page" and c.reasoning == "thought" and c.usage["thinking_tokens"] == 5
    again = llm.complete(msg, temperature=0.6, seed=7, max_tokens=4096)
    assert again.cached and again.reasoning == "thought"
    # the extras are part of the key; without any, the key is the old one
    plain = CachedLLM(
        OpenAIChat("m", base_url="http://x/v1", api_key="k"), tmp_path, key_extra={"commit": "c"}
    )
    assert plain._key(msg, 0.6, 7, 4096) != llm._key(msg, 0.6, 7, 4096)
    assert plain.request_extra == {}


def test_timeouts_and_dropped_connections_are_retried():
    import httpx
    from openai import APIConnectionError, APITimeoutError

    from vizor.generate.llm import _retryable

    req = httpx.Request("POST", "http://x")
    assert _retryable(APITimeoutError(req)) and _retryable(APIConnectionError(request=req))
    assert not _retryable(ValueError("no"))
