# Verification log

Checks run on this machine (Apple M1 Pro, 16 GB, macOS), and how they were done.

## 2026-07-23

- **Pinned models**: `vizor models fetch --write-lock` downloaded bge-small-en-v1.5@5c38ec7,
  ms-marco-MiniLM-L-6-v2@233902d and twitter-roberta-base-sentiment-latest@3216a57 into `.models/`
  and recorded the sha256 of every file in `models.lock`. `vizor models verify` re-checks them.
- **Network test**: `pytest -m network tests/test_commoncrawl.py`. A live Common Crawl CDX lookup
  (index resolved through collinfo.json) and a WARC fetch both passed.
- **CI workflow**: linted with actionlint. It has not run on GitHub.

## 2026-08-05

- **Docker** (`docker build`, arm64, 4.0 GB image). `selfcheck egress` inside `--network none`
  reported blocked for every target, and the positive control on the default network reached all
  three. The keyless demo (8 queries x 2 samples, real retrieval models mounted read-only) ran
  inside `--network none`.
- **Sealed compose stack** (`make compose-offline`, run exclusively on the machine): api and
  dashboard on an `internal: true` network. The checker container saw egress blocked, `/health`
  reported `"egress": "blocked"` from the API process, `/answer` returned 5 sources, and the
  dashboard served HTTP 200.
- **Fresh clone**: `git clone` into an empty directory, then `make setup` (downloaded the three
  models and verified all 21 files against `models.lock`). `make offline-check` passed (canary
  blocked inside the sandbox profile, open outside it).

## 2026-08-20

- **Committed runs** (run exclusively on the machine, the FakeLLM one under
  `scripts/offline-run`):
  - `experiments/results/2026-08-05_qwen2.5-3b`: qwen2.5:3b-instruct via Ollama, 20 queries x
    2 samples. 83 new calls and 557 cache hits (answers from an interrupted run of the same code
    and config were reused), 357.5 s. The system message in `configs/ollama.yaml` is recorded in
    the manifest.
  - `experiments/results/2026-08-05_fakellm`: keyless demo, 40 queries x 5 samples, 48 s.
  - `vizor recompute` re-parses every raw answer and matches the stored rows for both runs.
  - Both manifests record a `git_commit` (38ef511) that no longer resolves, because history was
    rewritten after the runs. `src_tree` (7dce379c…, the git tree hash of `src/`) identifies the
    code, and the result files themselves are committed.
- **Fresh clone, offline half**: in the clone from 2026-08-05, `make demo` with
  `OFFLINE_WRAPPER` set to the sandbox profile completed (40 queries x 5 samples, FakeLLM).
- **Unit tests**: `make test` (`pytest -m "not slow and not network"`), 293 passed on the final
  commit. They include the hand-computed PAWC vector, parity with the GEO authors' impression
  functions on 200 random patterns, NumPy vs FAISS parity (in a torch-free subprocess), noop = 0
  under common random numbers, the cross-fitting leakage test, the exact Wilcoxon floor, and the
  API. `vizor demo --config configs/ci.yaml` under `scripts/offline-run` completed, and
  `vizor recompute` matched its rows.
- **Offline API and dashboard**: with `VIZOR_EGRESS_CANARY=1` under the sandbox profile, `/health`
  reported `"egress": "blocked"` and the dashboard showed the same in its sidebar.
- **gpt-4o-mini**: configured but not run yet (`configs/openai.yaml`, `make experiment-openai`).
  The pre-flight upper bound is 4,800 answer calls plus 20 rewrite calls, about $2.90, under the
  $3 cap. No result in this repo comes from an OpenAI model.

## 2026-09-05

- **Bench method repair before the pilot**: page-arm inference now clusters both cross-fit folds
  by the underlying page. The amendment is recorded in `docs/bench-design.md`. The run manifest
  fingerprints source files, effective config and corpus; the report suppresses inferential
  tables when input fingerprints changed during a run or stored attribution no longer recomputes.
- **Offline suite**: `pytest -m 'not slow and not network' -o addopts='' -q` passed 356 tests
  (one network test deselected). `ruff check .` passed.
- **Historical recomputation**: `2026-08-05_fakellm` matched 21,000 rows. The older
  `2026-08-05_qwen2.5-3b` run reported `0 rows recomputed; MISMATCH` because the current parser
  changes 26 of its 640 stored answer parses. Its raw responses and historical outputs remain
  intact; it is not current evidence for an effect.
- **Real bench**: the 72-query pilot is queued behind the shared compute lease. No pilot MDE or
  main-run result has been measured yet.

## 2026-09-18

- **Main bench run**: `experiments/results/2026-09-05_bench-qwen3b` (git 0444d1e, clean source
  tree, input fingerprints equal at start and end). `vizor recompute` matched all 28,800 rows;
  the pilot matched all 5,400.
- **Review fixes after the run**: planned-family MDE in the pilot summary, the run-time MDE on
  the primary metric, page means next to query-weighted means, brand-mention columns checked in
  recompute, the stricter bare citation form, and the OpenAI-compatible cache key and server
  commit. The pilot and main-run summaries and manifests were regenerated with `vizor report`;
  only the `sensitivity` and `planned_arms` keys changed.
- **Archive**: the 2026-08-05 qwen2.5:3b run moved to `experiments/archive/` as historical output.
- **Offline suite**: `pytest -m 'not slow and not network' -o addopts='' -q` passed 361 tests
  (one network test deselected). `ruff check .` passed.
- **Study 2 tooling (CPU only, no model called)**: `pytest -m 'not slow and not network' -o
  addopts='' -q` passed 402 tests (two deselected: the network test and the MLX scorer test,
  which needs local MLX weights). `ruff check .` passed. The Localhost AI scorer was tested
  against a stub HTTP server that follows the `/v1/score` contract, not against the real
  endpoint. With the fake scorer, `scripts/validation_gate.py` ran end to end on the Study 1 run
  (2,160 rows, 1,156 distinct scorer calls) and `configs/study2_dryrun.yaml` ran the whole
  Study 2 design (1,512 sampled answers, 2,016 scored rows, A/A and `noop` exactly 0). Those fake
  numbers check plumbing only.
- **Final review fixes (CPU only, no model called)**: `pytest -m 'not slow and not network' -o
  addopts='' -q` passed 423 tests (two deselected, as above), also under `scripts/offline-run`.
  `ruff check .` and `ruff format --check .` passed. `vizor recompute` matched every committed
  run (Study 2: 7,560 rows and 2,016 AP rows; gate: 2,160 AP rows). `vizor report` reproduces
  the README block unchanged. The pre-registered keys of `sampled_primary.json` are unchanged;
  the regenerated file only adds `p_perm_exact`, `exploratory_reference` and `gate`. The
  dashboard screenshots were taken headless (Streamlit, Playwright Chromium) at 1280 and 390 px
  against the API with a hashing embedder and FakeLLM config, so no model was loaded.
- **Study 3 (2026-09-28)**: `pytest -m 'not slow and not network' -o addopts='' -q` passed 440
  tests (two deselected, as above); `ruff check .` and `ruff format --check .` passed. The
  FakeLLM dry run (`configs/study3_dryrun.yaml`) ran the whole design before any model call.
  Under the compute lease: `/v1/score` matched in-process mlx-lm on `qwen2.5-3b-mlx4` (10
  prompts, 51 sites, largest difference 0.0); the 9B determinism check passed at batch 1 (10 of
  10); the pilot and the main run were written by `vizor experiment`. `vizor recompute` matched
  every committed run (Study 3 pilot 2,160 rows, main run 5,400 rows), and the main run's
  regenerated baseline and A/A answers equal the pilot's in 288 of 288 cases
  (`scripts/pilot_identity.py`, which writes `pilot_identity.json` and reproduces the committed
  file byte for byte). `vizor report`
  regenerates the README block; running it twice gives no further change.
