# RecLoop developer entry points. Run from the repo root; `make help` lists the targets.
SHELL := /bin/bash
.DEFAULT_GOAL := help

# Local overrides (copy .env.example to .env); the defaults below match .env.example.
-include .env
RECLOOP_PG_PORT ?= 15432
RECLOOP_REDIS_PORT ?= 16390
DATABASE_URL ?= postgres://recloop:recloop@localhost:$(RECLOOP_PG_PORT)/recloop?sslmode=disable
REDIS_URL ?= redis://localhost:$(RECLOOP_REDIS_PORT)/0
API_PORT ?= 18080
MODEL_SERVER_PORT ?= 18081
WEB_PORT ?= 13000
API_ADDR ?= 127.0.0.1:$(API_PORT)
MODEL_SERVER_ADDR ?= 127.0.0.1:$(MODEL_SERVER_PORT)
RECLOOP_API_URL ?= http://$(API_ADDR)
RECLOOP_BUNDLE ?= $(CURDIR)/artifacts/bundles/current
RECLOOP_TEXT_MODEL ?= $(CURDIR)/data/hf/bge-small-en-v1.5
# M2: event log, monitoring, tracing, the bench profile's api.
RECLOOP_KAFKA_PORT ?= 19092
KAFKA_BROKERS ?= localhost:$(RECLOOP_KAFKA_PORT)
RECLOOP_PROMETHEUS_PORT ?= 19090
PROMETHEUS_URL ?= http://localhost:$(RECLOOP_PROMETHEUS_PORT)
RECLOOP_BENCH_API_PORT ?= 18180
# Traces of `make serve-api` / `serve-model` go to the compose OTel collector. Not exported to
# every recipe: docker compose and other OTel-aware tools would send their own traces there.
OTEL_EXPORTER_OTLP_ENDPOINT ?= http://localhost:14317
TRACING := OTEL_EXPORTER_OTLP_ENDPOINT=$(OTEL_EXPORTER_OTLP_ENDPOINT) OTEL_EXPORTER_OTLP_INSECURE=true
AD_TOKEN_KEYS ?= dev1:recloop-dev-only-ad-token-key-not-a-secret-0001
export RECLOOP_PG_PORT RECLOOP_REDIS_PORT DATABASE_URL REDIS_URL API_ADDR MODEL_SERVER_PORT \
	MODEL_SERVER_ADDR RECLOOP_API_URL RECLOOP_BUNDLE RECLOOP_TEXT_MODEL RECLOOP_KAFKA_PORT \
	KAFKA_BROKERS PROMETHEUS_URL RECLOOP_BENCH_API_PORT AD_TOKEN_KEYS
# onnxruntime's telemetry uploader (macOS) can crash the interpreter at exit: off everywhere.
export ORT_DISABLE_TELEMETRY := 1

# Tool caches live in the repo (git-ignored) so the targets also work where ~/.cache is not
# writable, e.g. for sandboxed coding agents. Set them in the environment to use global caches.
UV_CACHE_DIR ?= $(CURDIR)/.cache/uv
GOMODCACHE ?= $(CURDIR)/.cache/go/mod
GOCACHE ?= $(CURDIR)/.cache/go/build
GOTOOLCHAIN ?= local
BUF_CACHE_DIR ?= $(CURDIR)/.cache/buf
npm_config_cache ?= $(CURDIR)/.cache/npm
export UV_CACHE_DIR GOMODCACHE GOCACHE GOTOOLCHAIN BUF_CACHE_DIR npm_config_cache

COMPOSE := docker compose -f deploy/docker-compose.yml
BENCH := $(COMPOSE) --profile bench
UV := uv run --project ml --frozen
PY := $(UV) python
MS := uv run --project model_server --frozen
MR := uv run --project minionerec --frozen
AG := uv run --project agent --frozen
# The agent's query encoder (sentence-transformers + torch) is an optional dependency group.
AGE := $(AG) --group encoder
# goose's CLI links every SQL driver it supports; build it with Postgres only.
GOOSE_TAGS := no_clickhouse,no_libsql,no_mssql,no_mysql,no_sqlite3,no_vertica,no_ydb
GOOSE := cd serving && GOFLAGS="-modcacherw -tags=$(GOOSE_TAGS)" \
	go tool goose -dir migrations postgres "$(DATABASE_URL)"
# Code generators (tools/go.mod), built once into .cache/bin.
BIN := $(CURDIR)/.cache/bin
TOOLS := buf protoc-gen-go protoc-gen-go-grpc sqlc
TOOL_PKGS := github.com/bufbuild/buf/cmd/buf google.golang.org/protobuf/cmd/protoc-gen-go \
	google.golang.org/grpc/cmd/protoc-gen-go-grpc github.com/sqlc-dev/sqlc/cmd/sqlc

.PHONY: help setup images up down down-volumes bench-up bench-down migrate migrate-status data \
	data-weights ingest train paper-sasrec eval ann bundle seed build-aggregator rebuild-redis tools \
	k6 proto sqlc generate build-api serve-model serve-api serve-web web-install test lint test-ml \
	test-model-server test-minionerec test-agent test-go test-web test-props e2e fmt sim-world sim \
	srm drill load load-smoke latency-report minionerec-gen minionerec-cost minionerec agent-cases \
	agent-eval agent-eval-llm clean-processed clean-artifacts

help: ## List targets
	@grep -E '^[a-z0-9-]+:.*## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*## "} {printf "  %-18s %s\n", $$1, $$2}'

setup: up data ingest train eval bundle seed web-install ## Everything from a clean checkout to a runnable demo
	@echo "ready: run make serve-model, make serve-api and make serve-web (three terminals)"

# The Go services as linux binaries for the containers (aggregator; api in the bench profile).
LINUX := $(CURDIR)/.cache/linux
images: ## Cross-compile the Go services into .cache/linux and build their image (recloop/go:dev)
	mkdir -p $(LINUX)
	cd serving && CGO_ENABLED=0 GOOS=linux GOARCH=$$(go env GOARCH) GOFLAGS=-modcacherw \
		go build -trimpath -o $(LINUX)/ ./cmd/api ./cmd/aggregator
	$(COMPOSE) build aggregator

up: images ## Start the stack (PostgreSQL, Redis, Redpanda, log-sink, aggregator, monitoring) + migrations
	$(COMPOSE) up -d --wait
	$(MAKE) migrate

down: ## Stop the stack (bench profile included), keep the volumes
	$(BENCH) down

down-volumes: ## Stop the stack and delete its volumes (database, event log, metrics)
	$(BENCH) down -v

bench-up: images ## Also start the bench profile: api + model server containers behind Toxiproxy
	$(BENCH) build model-server
	$(BENCH) up -d --wait

bench-down: ## Stop the bench profile's containers; the rest of the stack keeps running
	$(BENCH) stop api model-server toxiproxy

migrate: ## Apply database migrations (goose, serving/migrations)
	$(GOOSE) up

migrate-status: ## Show applied migrations
	$(GOOSE) status

data: ## Download + verify inputs, re-run MiniOneRec preprocessing, export catalog, verify, embed
	$(PY) -m ml.data.download
	$(PY) -m ml.data.preprocess
	$(PY) -m ml.data.catalog
	$(PY) -m ml.data.verify
	$(PY) -m ml.data.text_embed

data-weights: ## Also download the 3.56 GB MiniOneRec checkpoint weights (first needed in M7)
	$(PY) -m ml.data.download --with-weights

ingest: migrate ## Load items (with text embeddings and simulated stock) and users into PostgreSQL
	$(PY) -m ml.data.ingest

train: ## Train and tune SASRec on the train / valid splits -> artifacts/models/sasrec (~12 min)
	$(PY) -m ml.models.train_sasrec

paper-sasrec: ## Re-run MiniOneRec's own SASRec script on the split (~18 min; make eval reports it)
	$(PY) -m ml.eval.minionerec_sasrec

eval: ## Offline evaluation of Pop / ItemCF / SASRec -> docs/results/offline.md (no database)
	$(PY) -m ml.eval.run

ann: ## ANN benchmark + index updates -> docs/results/ann.md (~25 min; run on an idle machine)
	$(PY) -m ml.ann.run $(ANN_FLAGS)

# MiniOneRec (M7): generation runs natively on the Mac's GPU (MPS) in its own environment
# (minionerec/); scoring and the report use the ml environment. Needs `make data-weights`.
MINIONEREC_ARGS ?= --name test
minionerec-gen: ## MiniOneRec on MPS over the test split (~5 h; MINIONEREC_ARGS="--name trial --users 200")
	$(MR) python -m minionerec.generate $(MINIONEREC_ARGS)

minionerec-cost: ## MiniOneRec latency / memory at 10, 20, 50 beams (MPS) and SASRec on the CPU (~25 min, idle machine)
	$(MR) python -m minionerec.cost $(MINIONEREC_COST_ARGS)
	$(PY) -m ml.eval.sasrec_cost

minionerec: ## Score MiniOneRec's generations like make eval -> docs/results/minionerec.md (no GPU)
	$(PY) -m ml.eval.minionerec

bundle: ## Build artifacts/bundles/<version> and point artifacts/bundles/current at it
	$(PY) -m ml.bundle.build

seed: ## Flush Redis and load the dataset users' histories and the demo users
	$(PY) -m ml.data.seed_redis

build-aggregator:
	cd serving && GOFLAGS=-modcacherw go build -o $(BIN)/aggregator ./cmd/aggregator

# Redis holds derived state only (design §7.2). Rebuild = the seed state + every event since:
#   FROM=db  (default) replay events_raw (everything ever logged, M1's direct writes included)
#   FROM=log reset the aggregator's group to the start of the event log (7 days of retention)
# The aggregator is stopped while Redis is rebuilt; both paths use its dedup markers, so the
# events it applies after the restart that the replay already applied are skipped.
FROM ?= db
rebuild-redis: build-aggregator ## Rebuild Redis: seed, then replay events_raw (FROM=db) or the event log (FROM=log)
	$(COMPOSE) stop aggregator
	$(MAKE) seed
	@if [ "$(FROM)" = log ]; then \
		$(BIN)/aggregator reset-offsets; \
	else \
		$(BIN)/aggregator wait-lag log-sink && $(BIN)/aggregator rebuild-from-db; \
	fi
	$(COMPOSE) start aggregator
	$(BIN)/aggregator wait-lag aggregator

$(addprefix $(BIN)/,$(TOOLS)): tools/go.mod tools/go.sum
	cd tools && GOFLAGS=-modcacherw go build -o $(BIN)/ $(TOOL_PKGS)

tools: $(addprefix $(BIN)/,$(TOOLS)) ## Build the code generators (buf, protoc plugins, sqlc)

proto: tools ## Lint + format-check the protos, regenerate serving/gen and model_server/gen
	cd proto && PATH="$(BIN):$$PATH" buf lint && PATH="$(BIN):$$PATH" buf format -d --exit-code
	rm -rf serving/gen model_server/gen && mkdir -p model_server/gen
	cd proto && PATH="$(BIN):$$PATH" buf generate
	cd proto && PATH="$(BIN):$$PATH" buf build -o "$(CURDIR)/.cache/proto-image.binpb"
	$(MS) python -m grpc_tools.protoc --descriptor_set_in=.cache/proto-image.binpb \
		--python_out=model_server/gen --pyi_out=model_server/gen \
		--grpc_python_out=model_server/gen recloop/modelserver/v1/model_server.proto

sqlc: tools ## Regenerate serving/internal/store/db from the migrations and queries
	cd serving && $(BIN)/sqlc generate

generate: proto sqlc ## All generated code

build-api: ## Build the Go api into .cache/bin/api
	cd serving && GOFLAGS=-modcacherw go build -o $(BIN)/api ./cmd/api

serve-model: ## Model server (gRPC) on MODEL_SERVER_PORT, serving artifacts/bundles/current
	$(TRACING) $(MS) python -m model_server --bundle "$(RECLOOP_BUNDLE)" --port $(MODEL_SERVER_PORT)

serve-api: build-api ## Go api on API_ADDR
	$(TRACING) $(BIN)/api

serve-web: ## Next.js web app on WEB_PORT (proxies /api to the Go api)
	cd web && npm run dev -- --port $(WEB_PORT)

web-install: ## Install the web app's dependencies (npm ci)
	cd web && npm ci --no-audit --no-fund

# ---------------------------------------------------------------- M2: simulator and load tests

$(BIN)/k6: tools/k6/go.mod tools/k6/go.sum
	cd tools/k6 && GOFLAGS=-modcacherw go build -o $(BIN)/k6 go.k6.io/k6/v2

k6: $(BIN)/k6 ## Build k6 (pinned in tools/k6/go.mod) into .cache/bin

sim-world: ## Simulator world: test-period users, their targets, click model -> artifacts/sim/world.json
	$(PY) -m sim.world

SIM_FLAGS ?= --sessions 2000 --concurrency 16
sim: sim-world build-aggregator ## Simulated sessions against the api (SIM_FLAGS, e.g. --faults duplicate=0.1), then verify
	$(PY) -m sim.run $(SIM_FLAGS) --out artifacts/sim/run
	$(PY) -m sim.verify --run artifacts/sim/run

srm: sim-world build-aggregator ## Simulated traffic, then the SRM check of every experiment layer on events_raw
	$(PY) -m sim.run --sessions 8000 --concurrency 16 --out artifacts/sim/srm
	$(BIN)/aggregator wait-lag log-sink
	$(PY) -m analysis.srm --run artifacts/sim/srm

drill: sim-world build-aggregator ## Duplicate delivery + consumer kill -9 + broker restart under traffic, then verify (~2 min)
	$(PY) -m sim.drill $(DRILL_FLAGS)

load: bench-up k6 sim-world build-aggregator ## Latency experiments (bench profile) -> docs/results/latency.md (~90 min; idle machine)
	$(PY) -m loadtest.run --plan full $(LOAD_FLAGS)
	$(PY) -m loadtest.report

load-smoke: bench-up k6 sim-world build-aggregator ## Short make load (1 repeat, 15 s runs) -> artifacts/latency/smoke.md
	$(PY) -m loadtest.run --plan smoke --out artifacts/latency/smoke.json $(LOAD_FLAGS)
	$(PY) -m loadtest.report --results artifacts/latency/smoke.json --out artifacts/latency/smoke.md

latency-report: ## Re-render docs/results/latency.md from artifacts/latency/results.json
	$(PY) -m loadtest.report

# ---------------------------------------------------------------- M6: shopping assistant

agent-cases: ## Regenerate agent/eval/cases.jsonl from the catalog (reads items, read-only)
	$(AG) python -m agent.eval.build_cases

agent-eval: ## Shopping assistant: baseline + oracle agents -> docs/results/agent.md (no API calls)
	$(AGE) python -m agent.eval.run

# Paid: needs Anthropic credentials, MODELS (e.g. claude-opus-5-5) and a cap in MAX_COST_USD.
agent-eval-llm: ## Language-model agents (MODELS=..., REPEATS=3, MAX_COST_USD=... required)
	@test -n "$(MAX_COST_USD)" || { echo "set MAX_COST_USD (a spending cap per model run)"; exit 1; }
	@test -n "$(MODELS)" || { echo "set MODELS, e.g. MODELS=claude-opus-5-5"; exit 1; }
	$(AGE) python -m agent.eval.run --skip-offline $(foreach m,$(MODELS),--llm $(m)) \
		--repeats $(or $(REPEATS),3) --max-cost-usd $(MAX_COST_USD)

test: lint test-ml test-model-server test-minionerec test-agent test-go ## Lint and run the Python and Go tests (DB / Redis tests skip when the stack is down)

PYDIRS := ml sim loadtest analysis minionerec
lint: ## ruff check + format check (ml, sim, loadtest, analysis, minionerec, model_server, agent)
	$(UV) ruff check $(PYDIRS)
	$(UV) ruff format --check $(PYDIRS)
	$(MS) ruff check model_server
	$(MS) ruff format --check model_server
	$(AG) ruff check agent
	$(AG) ruff format --check agent

# test_ann.py loads faiss, so it gets a pytest process of its own: torch (test_sasrec.py) and
# faiss in one process abort on macOS (two OpenMP runtimes).
test-ml:
	$(UV) pytest ml/tests --ignore=ml/tests/test_ann.py
	$(UV) pytest ml/tests/test_ann.py

test-model-server:
	$(MS) pytest model_server/tests

# The minionerec environment is locked for Apple Silicon only (MPS); elsewhere its tests skip.
test-minionerec:
	@if [ "$$(uname -sm)" = "Darwin arm64" ]; then $(MR) pytest minionerec/tests; \
	else echo "skip minionerec tests: the environment is Apple Silicon only"; fi

# Scripted language model and an in-memory catalog; the database and encoder tests skip when the
# stack is down or the `encoder` group / model files are missing. Never calls a paid API.
test-agent:
	$(AG) pytest agent/tests

test-go:
	@cd serving && unformatted="$$(gofmt -l cmd internal)" && \
		if [ -n "$$unformatted" ]; then echo "gofmt needed: $$unformatted"; exit 1; fi
	cd serving && GOFLAGS=-modcacherw go vet ./... && GOFLAGS=-modcacherw go test ./...

# Property tests (rapid) run a moderate number of cases in `make test` (billing: 30 per
# property, against PostgreSQL); this runs many more.
CHECKS ?= 2000
test-props: ## Long run of the ads / ad_token / billing property tests (CHECKS=2000 each, ~4 min; needs PostgreSQL)
	cd serving && GOFLAGS=-modcacherw go test ./internal/ads/ ./internal/adtoken/ ./internal/billing/ \
		-count=1 -timeout 0 -rapid.checks=$(CHECKS)

test-web: ## Type-check and lint the web app (next typegen first: tsc needs the route types)
	cd web && npx next typegen && npx tsc --noEmit && npm run lint

e2e: ## Browser check (needs the three servers): click an item, reload, recommendations change
	cd web && RECLOOP_WEB_URL=http://localhost:$(WEB_PORT) npx playwright test

fmt: ## Format and auto-fix
	$(UV) ruff format $(PYDIRS)
	$(UV) ruff check --fix $(PYDIRS)
	$(MS) ruff format model_server
	$(MS) ruff check --fix model_server
	$(AG) ruff format agent
	$(AG) ruff check --fix agent
	cd serving && gofmt -w cmd internal

clean-processed: ## Delete data/processed (keeps downloads and third_party/)
	rm -rf data/processed

clean-artifacts: ## Delete trained models, bundles and evaluation outputs (artifacts/)
	rm -rf artifacts/models artifacts/bundles artifacts/eval
