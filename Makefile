# Two phases: `make setup` needs the network once (locked deps + pinned models); everything
# else runs offline. OFFLINE wraps a command so the whole process tree has no egress (macOS
# sandbox-exec; see README). Override it with OFFLINE= to run without the wrapper.
PY      := .venv/bin/python
VIZOR   := .venv/bin/vizor
OFFLINE ?= $(firstword $(wildcard ../.tools/offline-run $(HOME)/Developer/portfolio/.tools/offline-run))
DATE    := $(shell date +%F)

.PHONY: setup models verify-models lint test test-models demo demo-ci offline-check serve ui \
        report experiment-openai docker compose-offline clean

setup:
	uv sync --frozen --extra ml --extra ui
	$(MAKE) models

models:
	$(VIZOR) models fetch

verify-models:
	$(VIZOR) models verify

lint:
	.venv/bin/ruff check . && .venv/bin/ruff format --check .

test:
	$(PY) -m pytest -m "not slow and not network"

demo:
	$(OFFLINE) $(VIZOR) demo --config configs/demo.yaml --out runs/$(DATE)_demo

demo-ci:
	$(OFFLINE) $(VIZOR) demo --config configs/ci.yaml --out runs/ci-smoke

# The canary must fail to reach the internet from inside the wrapper, and succeed without it.
offline-check:
	$(OFFLINE) $(VIZOR) selfcheck egress --expect blocked
	$(VIZOR) selfcheck egress --expect open

serve:
	VIZOR_EGRESS_CANARY=1 $(OFFLINE) $(VIZOR) serve --port 8000

ui:
	VIZOR_EGRESS_CANARY=1 $(OFFLINE) .venv/bin/streamlit run src/vizor/dashboard/app.py --server.port 8501

report:
	$(VIZOR) report

# Real-model run. Needs OPENAI_API_KEY in the environment; stops at max_cost_usd (configs/openai.yaml).
experiment-openai:
	$(VIZOR) experiment --config configs/openai.yaml --out experiments/results/$(DATE)_gpt-4o-mini

docker:
	docker build -t vizor:dev .

# api + dashboard on an internal-only network: no egress from either container.
compose-offline:
	docker compose -f docker-compose.yml -f docker-compose.offline.yml up --build --abort-on-container-exit offline-check

clean:
	rm -rf runs/ci-smoke .pytest_cache .ruff_cache
