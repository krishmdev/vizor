"""Pre-flight cost estimate for a real-model experiment, from the actual prompts it would send."""

from __future__ import annotations

from vizor.config import Config, build
from vizor.generate.llm import PRICES

CHARS_PER_TOKEN = 4.0  # rough for English prose; the real run records exact usage


def estimate_cost(cfg: Config, completion_tokens: int = 260) -> dict:
    cfg = cfg.model_copy(deep=True)
    backend = cfg.llm.backend
    cfg.llm.backend = "fake"  # build prompts only; never call a model here
    _, _, queries, engine = build(cfg)
    prompt_tokens = [len(engine.build_prompt(q)[0]) / CHARS_PER_TOKEN for q in queries]
    per_call_in = sum(prompt_tokens) / max(1, len(prompt_tokens))
    sweeps = len(cfg.sandbox.position_sweep) + len([b for b in cfg.sandbox.boost_sweep if b])
    arms = len([a for a in cfg.sandbox.arms if a != "noop"])
    if cfg.sandbox.llm_rewrites:
        arms += 4
    per_variant = len(queries) * cfg.samples
    greedy = cfg.bandit.greedy_steps * (len(queries) // 2) * cfg.samples
    calls = per_variant * (1 + arms + sweeps) + greedy
    pin, pout = PRICES.get(cfg.llm.model, (0.0, 0.0))
    usd = calls * (per_call_in * pin + completion_tokens * pout) / 1e6
    return {
        "backend": backend,
        "model": cfg.llm.model,
        "queries": len(queries),
        "samples": cfg.samples,
        "variants": 1 + arms + sweeps,
        "max_calls": calls,
        "avg_prompt_tokens": round(per_call_in),
        "assumed_completion_tokens": completion_tokens,
        "max_cost_usd_upper_bound": round(usd, 2),
        "note": "upper bound: unchanged prompts are cache hits and cost nothing",
    }
