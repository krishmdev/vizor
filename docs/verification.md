# Verification log

What was checked on this machine (Apple M1 Pro, 16 GB, macOS), and how.

## 2026-09-23

- **Unit tests**: `pytest -m "not network"`, 284 passed. These include the hand-computed PAWC
  vector, parity with the GEO authors' impression functions on 200 random patterns, NumPy vs
  FAISS index parity (run in a torch-free subprocess), noop = 0 under common random numbers, the
  cross-fitting leakage test, and the API.
- **Network test**: `pytest -m network tests/test_commoncrawl.py`. A live Common Crawl CDX lookup
  (index resolved through collinfo.json) and a WARC fetch both passed.
- **Pinned models**: `vizor models fetch --write-lock` downloaded bge-small-en-v1.5@5c38ec7,
  ms-marco-MiniLM-L-6-v2@233902d and twitter-roberta-base-sentiment-latest@3216a57 into `.models/`
  and recorded the sha256 of every file in `models.lock`. `vizor models verify` re-checks them.
- **Offline runtime**: the keyless demo ran under a sandbox-exec wrapper (outbound
  network denied) with the real retrieval and sentiment models and FakeLLM answers, producing
  `experiments/results/2026-09-23_fakellm`. With `VIZOR_EGRESS_CANARY=1`, the API started under
  the same wrapper and reported `"egress": "blocked"` on `/health`, and the dashboard reported
  the same in its sidebar.
- **Docker** (`docker build`, arm64, 4.0 GB image). `selfcheck egress` inside
  `--network none` reported blocked for every target. The positive control on the default network
  reached all three. The keyless demo (8 queries x 2 samples, real retrieval models mounted
  read-only) ran inside `--network none`.
- **Sealed compose stack** (`make compose-offline`, run under the compute lease): api and
  dashboard on an `internal: true` network. The checker container saw egress blocked, `/health`
  reported `"egress": "blocked"` from the API process, `/answer` returned 5 sources, and the
  dashboard served HTTP 200.
- **CI workflow**: linted with actionlint. It has not run on GitHub.
- **qwen2.5:3b-instruct via Ollama** (local, under the compute lease): 20 queries x 2 samples,
  422 new calls, 218 cache hits, 31 minutes. Before the run, 8 test answers showed the model citing
  only 6 of 52 sentences with the shared prompt. A system message restating the citation rule (in
  `configs/ollama.yaml`, recorded in the manifest) raised that to 34 of 36. Results are in
  `experiments/results/2026-09-23_qwen2.5-3b`, and `vizor recompute` matches.
- **gpt-4o-mini**: not run. The run is wired (`configs/openai.yaml`, `make experiment-openai`),
  and the pre-flight estimate is up to 4,800 calls, about $2.13. Launching it needs Krish's go-ahead.
