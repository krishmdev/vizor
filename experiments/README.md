# Experiments

Each directory under `results/` contains one run of `vizor demo` or `vizor experiment`:

| File | Contents |
|---|---|
| `manifest.json` | effective config, git commit and `src_tree` (Git tree hash of `src/`), exact SHA-256 fingerprints of source files, effective config and corpus for new runs, embedder/reranker/LLM ids, model pins, seeds, prompt hash, system prompt, LLM usage and spend, host snapshot |
| `responses.jsonl.gz` | every answer: arm, query, sample, seed, sources with scores, raw model text, parsed sentences, usage |
| `prompts.jsonl.gz` | each distinct prompt once, keyed by the hash stored with each answer |
| `rows.csv.gz` | one row per answer x domain: retrieved, cited, markers, PAWC/word/position shares, sentiment |
| `baseline_domains.csv` | per-domain visibility at baseline |
| `deltas.csv` | per-arm paired comparison against baseline (CI, Wilcoxon, Holm, A/A band, uncited rate) |
| `position_sweep.csv`, `boost_sweep.csv` | engine-side sensitivity sweeps |
| `bandit_*.csv`, `trajectory.csv` | reward table, contexts, replay curves, held-out check, greedy loop |
| `queries.csv` | cross-fitting fold and focus page per query |
| `diffs.json` | before/after text of every edited page, by arm and fold |
| `summary.md` | the generated report for this directory |
| `run_manifest_*.json` | host state (chip, RAM, load, swap; whether the run was exclusive) around a real-model run |

`vizor recompute <dir>` re-parses each raw answer and recomputes the attribution rows without
calling a model. New runs should report `match`. The archived 2026-08-05 Qwen run reports
`MISMATCH`: the current citation parser changes 26 of its 640 stored answer parses, so its
original metrics are historical and should not be used as current evidence. The archived
FakeLLM run still reports `match`. The old files are retained for provenance. `vizor report`
rebuilds `RESULTS.md` and the README results block from the run directories; those older
generated tables still reflect the parser version used for the stored runs.

Reproduce:

```sh
vizor demo --config configs/demo.yaml --out experiments/results/<date>_fakellm      # keyless
vizor experiment --config configs/ollama.yaml --out experiments/results/<date>_qwen2.5-3b
vizor experiment --config configs/openai.yaml --out experiments/results/<date>_gpt-4o-mini
vizor report
```

Real-model responses are cached under `.cache/llm/` by model, messages, temperature, and seed.
With the same config and code, a rerun costs nothing and returns the same answers.
