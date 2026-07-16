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
