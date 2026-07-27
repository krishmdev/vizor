# Archived runs

These runs are kept for provenance only. They are not current results, and `vizor report` does
not read this directory.

## 2026-08-05_qwen2.5-3b

qwen2.5:3b-instruct via Ollama, 20 queries x 2 samples, 22 pages, from before the bench corpus
and the pre-registered design. Its stored attribution rows come from an earlier citation parser.
With the current parser, `vizor recompute experiments/archive/2026-08-05_qwen2.5-3b` fails
(`0 rows recomputed; MISMATCH`): 26 of its 640 answer parses change. Its `summary.md`, including
the significance labels in it, is the output of that older code and should not be cited. The raw
responses and prompts are intact, so the run can be re-scored, but it would still be a reduced
design with too few page units to test any page edit.

The current real-model evidence is the bench run in
`experiments/results/2026-09-05_bench-qwen3b` (see `docs/bench-design.md`).
