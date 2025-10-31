"""Pre-flight cost bound for a real-model experiment, from the prompts it would actually send."""

from __future__ import annotations

from vizor.config import Config, build
from vizor.generate.llm import CHARS_PER_TOKEN, price_for
from vizor.optimize.transforms import LLM_REWRITES

REWRITE_MAX_TOKENS = 1500


def estimate_cost(cfg: Config) -> dict:
    """Upper bound: every variant is assumed to miss the cache and every answer to use the full
    max_tokens. Unchanged prompts are in fact cache hits and cost nothing."""
    cfg = cfg.model_copy(deep=True)
    backend, model = cfg.llm.backend, cfg.llm.model
    cfg.llm.backend = "fake"  # build prompts only; never call a model here
    _, docs, queries, engine = build(cfg)
    system = len(cfg.llm.system_prompt) / CHARS_PER_TOKEN
    prompt_tokens = [
        len(engine.build_prompt(q)[0]) / CHARS_PER_TOKEN + system + 16 for q in queries
    ]
    per_call_in = sum(prompt_tokens) / max(1, len(prompt_tokens))
    sweeps = len(cfg.sandbox.position_sweep) + len([b for b in cfg.sandbox.boost_sweep if b])
    arms = len([a for a in cfg.sandbox.arms if a != "noop"])
    rewrites = LLM_REWRITES if cfg.sandbox.llm_rewrites else []
    arms += len(rewrites)
    per_variant = len(queries) * cfg.samples
    greedy = cfg.bandit.greedy_steps * ((len(queries) + 1) // 2) * cfg.samples
    answer_calls = per_variant * (1 + arms + sweeps) + greedy
    targets = [d for d in docs if d.role == "target"]
    rewrite_in = sum(len(d.body) for d in targets) / CHARS_PER_TOKEN + 150 * len(targets)
    price = price_for(model)
    pin, pout = price if price else (0.0, 0.0)
    usd = (
        answer_calls * (per_call_in * pin + cfg.llm.max_tokens * pout)
        + len(rewrites) * (rewrite_in * pin + len(targets) * REWRITE_MAX_TOKENS * pout)
    ) / 1e6
    return {
        "backend": backend,
        "model": model,
        "priced": price is not None,
        "queries": len(queries),
        "samples": cfg.samples,
        "variants": 1 + arms + sweeps,
        "answer_calls": answer_calls,
        "rewrite_calls": len(rewrites) * len(targets),
        "avg_prompt_tokens": round(per_call_in),
        "assumed_completion_tokens": cfg.llm.max_tokens,
        "max_cost_usd_upper_bound": round(usd, 2),
        "note": "upper bound: unchanged prompts are cache hits and cost nothing",
    }
