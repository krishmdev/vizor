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
