"""Run 2026-09-18 against a gemma-4-e4b-mlx4 Localhost AI server (c61e05d, batch 1) with
mlx-lm 0.31.3 / mlx 0.32.2: /v1/score vs MLXScorer on one synthetic prompt padded to growing
lengths. Output in length_probe.jsonl. Usage: python length_probe.py <gemma snapshot dir>"""

import json, sys
from vizor.generate.scorer import LocalhostScorer, MLXScorer, Site
repo = sys.argv[1]
kw = {"enable_thinking": False}
srv = LocalhostScorer("gemma-4-e4b-mlx4", "http://127.0.0.1:8431", chat_template_kwargs=kw)
loc = MLXScorer(repo, None, chat_template_kwargs=kw)
filler = "The Harrow kettle holds 1.7 litres and ships in three colours. "
for reps in (2, 8, 30, 60, 120):
    msgs = [{"role": "user", "content": "Search results:\n[1] " + filler * reps + "\n[2] The Linden toaster has four slots.\nWhich kettle holds 1.7 litres?"}]
    cont = "The Harrow kettle holds 1.7 litres [1]. The toaster has four slots [2]."
    sites = [Site(cont.index("[1]") + 1, ("1", "2")), Site(cont.index("[2]") + 1, ("1", "2"))]
    a, b = srv.score(msgs, cont, sites), loc.score(msgs, cont, sites)
    n = len(loc.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, **kw))
    diff = max(abs(x.logprobs[c] - y.logprobs[c]) for x, y in zip(a.sites, b.sites) for c in ("1", "2"))
    print(json.dumps({"prompt_tokens": n, "max_abs_diff": diff}))
