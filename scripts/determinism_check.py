"""Byte-identity check for an OpenAI-compatible server, run before any answer is collected from it
(docs/bench-design.md, second model).

The same bench prompts, with the seed and temperature the run would use, are sent one at a time,
then again one at a time, then all at once in a concurrent batch. The check passes when every
batched answer is byte-identical to the one sent alone. If it fails, the mismatch rate goes into
the results directory and the run keeps `workers: 1` (one request at a time).

    uv run python scripts/determinism_check.py --config configs/bench_qwen9b_localhost_ai.yaml \
        --server-commit <localhost-ai HEAD> --out <results dir>

Exit code 0 when all answers match, 2 when any differ, 1 on errors. No answer is cached.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from vizor.config import Config, build


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--server-commit", required=True)
    ap.add_argument("--base-url", help="defaults to llm.base_url from the config")
    ap.add_argument("--n", type=int, default=10, help="prompts in the concurrent batch")
    ap.add_argument(
        "--server-batch",
        type=int,
        help="the server's fixed batch size, recorded in the result (1: concurrent requests are "
        "queued and run one at a time)",
    )
    args = ap.parse_args()

    cfg = Config.load(args.config)
    if cfg.llm.backend != "openai_compat":
        print("this check is for backend openai_compat", file=sys.stderr)
        return 1
    url = (args.base_url or cfg.llm.base_url or "").rstrip("/")
    # Build the retrieval side with the fake backend: only the prompts are needed here.
    fake = Config.model_validate({**cfg.model_dump(), "llm": {"backend": "fake"}})
    _, _, queries, engine = build(fake)
    step = max(1, len(queries) // args.n)
    picked = queries[::step][: args.n]
    jobs = []
    for q in picked:
        prompt, _ = engine.build_prompt(q)
        messages = [{"role": "user", "content": prompt}]
        if cfg.llm.system_prompt:
            messages.insert(0, {"role": "system", "content": cfg.llm.system_prompt})
        seed = engine.seed_for(q, 0, "")
        jobs.append({"query_id": q.query_id, "seed": seed, "messages": messages})

    client = httpx.Client(timeout=900)

    def call(job: dict) -> dict:
        t = time.perf_counter()
        r = client.post(
            url + "/chat/completions",
            json={
                "model": cfg.llm.model,
                "temperature": cfg.llm.temperature,
                "max_tokens": cfg.llm.max_tokens,
                "seed": job["seed"],
                "messages": job["messages"],
                **cfg.llm.request_extra(),
            },
        )
        r.raise_for_status()
        b = r.json()
        choice = b["choices"][0]
        return {
            "query_id": job["query_id"],
            "text": choice["message"]["content"],
            "finish_reason": choice.get("finish_reason"),
            "usage": b.get("usage"),
            "latency_s": round(time.perf_counter() - t, 2),
        }

    alone = [call(j) for j in jobs]
    again = [call(j) for j in jobs]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(len(jobs)) as pool:
        batch = list(pool.map(call, jobs))
    batch_wall = time.perf_counter() - t0

    same_batch = [a["text"] == b["text"] for a, b in zip(alone, batch, strict=True)]
    same_repeat = [a["text"] == b["text"] for a, b in zip(alone, again, strict=True)]
    ok = all(same_batch) and all(same_repeat)
    res = {
        "server_commit": args.server_commit,
        "base_url": url,
        "model": cfg.llm.model,
        "server_meta": cfg.llm.server_meta,
        "temperature": cfg.llm.temperature,
        "max_tokens": cfg.llm.max_tokens,
        "request_extra": cfg.llm.request_extra(),
        "server_batch": args.server_batch,
        "n_prompts": len(jobs),
        "alone_vs_alone_identical": sum(same_repeat),
        "alone_vs_batch_identical": sum(same_batch),
        "mismatch_rate_alone_vs_batch": 1 - sum(same_batch) / len(jobs),
        "byte_identical": ok,
        # with server_batch 1 concurrent requests are queued, not batched, so this says nothing
        # about answers computed inside a batch of two or more
        "decision": f"queued requests at batch {args.server_batch} are byte-identical"
        if ok
        else "answers differ; run with workers: 1 and report the mismatch rate",
        "batch_wall_s": round(batch_wall, 1),
    }
    print(json.dumps(res, indent=1))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "determinism.json").write_text(
        json.dumps({**res, "alone": alone, "alone_again": again, "batch": batch}, indent=1) + "\n"
    )
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
