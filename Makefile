# RecLoop developer entry points. Run from the repo root; `make help` lists the targets.
SHELL := /bin/bash
.DEFAULT_GOAL := help

# Local overrides (copy .env.example to .env); the defaults below match .env.example.
-include .env
RECLOOP_PG_PORT ?= 15432
RECLOOP_REDIS_PORT ?= 16390
RECLOOP_CACHE_REDIS_PORT ?= 16391
DATABASE_URL ?= postgres://recloop:recloop@localhost:$(RECLOOP_PG_PORT)/recloop?sslmode=disable
# Two Redis instances (design §19, 2026-11-08): state (user state, dedup markers, frequency caps,
# spend; never evicts) and cache (result pages, candidate cache; evicts the least recently used).
REDIS_URL ?= redis://localhost:$(RECLOOP_REDIS_PORT)/0
CACHE_REDIS_URL ?= redis://localhost:$(RECLOOP_CACHE_REDIS_PORT)/0
API_PORT ?= 18080
MODEL_SERVER_PORT ?= 18081
MODEL_SERVER_ADMIN_PORT ?= 18082
WEB_PORT ?= 13000
# M6: the shopping assistant's service (POST /v1/chat; the web app proxies /api/v1/chat to it).
AGENT_PORT ?= 18083
RECLOOP_AGENT_URL ?= http://127.0.0.1:$(AGENT_PORT)
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
RECLOOP_BENCH_MS_ADMIN_PORT ?= 18182
BENCH_MS_ADMIN_URL ?= http://127.0.0.1:$(RECLOOP_BENCH_MS_ADMIN_PORT)
RECLOOP_TOXIPROXY_PORT ?= 18474
TOXIPROXY_URL ?= http://127.0.0.1:$(RECLOOP_TOXIPROXY_PORT)
# Traces of `make serve-api` / `serve-model` go to the compose OTel collector. Not exported to
# every recipe: docker compose and other OTel-aware tools would send their own traces there.
OTEL_EXPORTER_OTLP_ENDPOINT ?= http://localhost:14317
TRACING := OTEL_EXPORTER_OTLP_ENDPOINT=$(OTEL_EXPORTER_OTLP_ENDPOINT) OTEL_EXPORTER_OTLP_INSECURE=true
AD_TOKEN_KEYS ?= dev1:recloop-dev-only-ad-token-key-not-a-secret-0001
# M3: the Toxiproxy API in front of billing's PostgreSQL connection (the billing drill).
RECLOOP_PG_PROXY_PORT ?= 18475
PG_PROXY_URL ?= http://localhost:$(RECLOOP_PG_PROXY_PORT)
# M4: releases load and switch bundle versions on the model server's admin port.
MODEL_SERVER_ADMIN_URL ?= http://127.0.0.1:$(MODEL_SERVER_ADMIN_PORT)
# The compose project: `recloop` is the shared stack (the compose file's name). A checkout with a
# private stack sets COMPOSE_PROJECT_NAME and its ports in .env; docker compose reads neither the
# repo's .env nor that name by itself here (the file is in deploy/), so $(COMPOSE) passes both
# and the Python tools get them from sim/compose.py.
COMPOSE_PROJECT_NAME ?= recloop
export RECLOOP_PG_PORT RECLOOP_REDIS_PORT DATABASE_URL REDIS_URL CACHE_REDIS_URL API_ADDR \
	MODEL_SERVER_PORT MODEL_SERVER_ADMIN_PORT MODEL_SERVER_ADDR RECLOOP_API_URL RECLOOP_BUNDLE \
	RECLOOP_TEXT_MODEL RECLOOP_KAFKA_PORT KAFKA_BROKERS PROMETHEUS_URL RECLOOP_BENCH_API_PORT \
	BENCH_MS_ADMIN_URL TOXIPROXY_URL AD_TOKEN_KEYS PG_PROXY_URL COMPOSE_PROJECT_NAME AGENT_PORT \
	RECLOOP_AGENT_URL MODEL_SERVER_ADMIN_URL
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

COMPOSE := docker compose -p $(COMPOSE_PROJECT_NAME) $(if $(wildcard .env),--env-file .env) \
	-f deploy/docker-compose.yml
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
	data-weights ingest train paper-sasrec eval ann bundle seed build-aggregator build-billing \
	rebuild-redis tools k6 proto sqlc generate build-api serve-model serve-api serve-web web-install \
	test lint test-ml test-model-server test-minionerec test-agent test-go test-web test-props e2e \
	fmt sim-world sim srm drill load load-smoke latency-report billing overdelivery billing-drill \
	billing-report minionerec-gen minionerec-cost minionerec agent-cases agent-eval agent-eval-llm \
	agent-eval-llm-smoke serve-agent agent-latency train-ranker spark-image samples test-samples \
	train-pctr collect release-drill release-report ab ab-report clean-processed clean-artifacts

help: ## List targets
	@grep -E '^[a-z0-9-]+:.*## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*## "} {printf "  %-18s %s\n", $$1, $$2}'

setup: up data ingest train train-ranker eval bundle seed web-install ## Everything from a clean checkout to a runnable demo
	@echo "ready: run make serve-model, make serve-api and make serve-web (three terminals)"

# The Go services as linux binaries for the containers (aggregator, billing; api in the bench
# profile).
LINUX := $(CURDIR)/.cache/linux
images: ## Cross-compile the Go services into .cache/linux and build their image (recloop/go:dev)
	mkdir -p $(LINUX)
	cd serving && CGO_ENABLED=0 GOOS=linux GOARCH=$$(go env GOARCH) GOFLAGS=-modcacherw \
		go build -trimpath -o $(LINUX)/ ./cmd/api ./cmd/aggregator ./cmd/billing
	$(COMPOSE) build aggregator

up: images ## Start the stack (PostgreSQL, Redis, Redpanda, log-sink, aggregator, billing, monitoring) + migrations
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

train-ranker: ## LightGBM LambdaRank on the validation split -> artifacts/models/ranker (~7 min; needs make train)
	$(PY) -m ml.models.train_ranker

eval: ## Offline evaluation of Pop / ItemCF / SASRec / RRF / LambdaRank -> docs/results/offline.md (no database)
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

bundle: ## Build artifacts/bundles/<version> (current -> it only if there is none: BUNDLE_FLAGS=--current)
	$(PY) -m ml.bundle.build $(BUNDLE_FLAGS)

seed: ## Flush Redis and load the dataset users' histories and the demo users
	$(PY) -m ml.data.seed_redis

build-aggregator:
	cd serving && GOFLAGS=-modcacherw go build -o $(BIN)/aggregator ./cmd/aggregator

build-billing:
	cd serving && GOFLAGS=-modcacherw go build -o $(BIN)/billing ./cmd/billing

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
	$(TRACING) $(MS) python -m model_server --bundle "$(RECLOOP_BUNDLE)" --port $(MODEL_SERVER_PORT) \
		--admin-port $(MODEL_SERVER_ADMIN_PORT)

serve-api: build-api ## Go api on API_ADDR
	$(TRACING) $(BIN)/api

# ADMIN=1 turns on the internal admin page (/admin): releases, experiments, faults (M4).
ADMIN ?= 0
serve-web: ## Next.js web app on WEB_PORT (proxies /api/v1 to the Go api; ADMIN=1: the /admin page)
	cd web && RECLOOP_ADMIN=$(ADMIN) npm run dev -- --port $(WEB_PORT)

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

# ---------------------------------------------------------------- M3: ads and billing

# Both runs need the stack (billing included), the model server and the api (`make serve-api`,
# its defaults). Each run reseeds Redis, creates campaigns of its own through the api and ends
# them afterwards. The state Redis must neither evict nor refuse a write during a run (the drill
# fails otherwise); the cache Redis's evictions are only recorded (sim/redis_stats.py).
billing: overdelivery billing-drill ## Over-delivery study + 30-minute billing drill -> docs/results/billing.md (~75 min)

overdelivery: sim-world build-aggregator build-billing ## Safety margins 0 / 5 / 20 clicks, with and without pacing (~35 min) -> artifacts/billing/overdelivery.json, billing.md
	$(PY) -m sim.overdelivery $(OVERDELIVERY_FLAGS)
	$(PY) -m analysis.billing_report

billing-drill: sim-world build-aggregator build-billing ## 30 min of traffic, kill -9 of billing, Redpanda restarts, PostgreSQL +200 ms; ledger vs truth -> billing.md
	$(PY) -m sim.billing_drill $(BILLING_DRILL_FLAGS)
	$(PY) -m analysis.billing_report

billing-report: ## Re-render docs/results/billing.md from artifacts/billing/
	$(PY) -m analysis.billing_report

# ---------------------------------------------------------------- M4: samples, ranking, release

# The training-sample job (design §7.5) is PySpark in local mode inside a container: HotSpot JVMs
# started from the Claude app's processes on the development Mac crash at startup, and the image
# pins the JVM and Python packages for every machine. The export runs on the host (psycopg).
SPARK_IMAGE ?= recloop/spark:dev
# It runs as the host user, so its outputs are the user's: Hadoop's login needs a name for that
# uid, given by a one-line passwd file mounted read-only.
SPARK_PASSWD := $(CURDIR)/.cache/spark-passwd
SPARK_RUN = docker run --rm --network none --hostname localhost --cpus 2 --memory 4g \
	--user $$(id -u):$$(id -g) -v "$(SPARK_PASSWD)":/etc/passwd:ro -v "$(CURDIR)":/repo $(SPARK_IMAGE)
DAY ?= $(shell date -u +%F)
spark-image: ## Image of the training-sample job: PySpark 4.2 (local mode) on a Temurin 21 JRE
	docker build -q -t $(SPARK_IMAGE) -f deploy/spark.Dockerfile deploy/spark
	@mkdir -p $(dir $(SPARK_PASSWD)) && printf 'spark:x:%s:%s:spark:/tmp:/bin/sh\n' "$$(id -u)" "$$(id -g)" > $(SPARK_PASSWD)

samples: spark-image ## Training samples of DAY (UTC, default today): export events_raw, PySpark join -> artifacts/samples
	$(PY) -m ml.samples.export --around $(DAY)
	$(SPARK_RUN) -m ml.samples.job --day $(DAY) $(SAMPLES_FLAGS)

test-samples: spark-image ## The sample job's tests, in its container (2 CPUs)
	$(SPARK_RUN) -m pytest -q -p no:cacheprovider ml/tests/test_samples.py

train-pctr: ## Logistic-regression pCTR on the samples (artifacts/samples/train) -> artifacts/models/pctr
	$(PY) -m ml.models.pctr

# The M4 runs need the stack, the model server and the api (`make serve-model`, `make serve-api`).
COLLECT_SECONDS ?= 900
collect: sim-world build-aggregator ## Traffic for the training samples: standing campaigns, COLLECT_SECONDS of sessions (~15 min)
	$(PY) -m sim.standing
	$(PY) -m sim.run --sessions 10000000 --max-seconds $(COLLECT_SECONDS) --concurrency 16 --out artifacts/sim/collect

release-drill: sim-world ## Canary -> full rollout of NEW, then BAD rolled back, under traffic (~35 min) -> docs/results/release.md
	$(PY) -m sim.release_drill --new $(NEW) --bad $(BAD) $(DRILL_FLAGS)
	$(PY) -m analysis.release_report

release-report: ## Re-render docs/results/release.md from artifacts/release/drill.json
	$(PY) -m analysis.release_report

ab: sim-world build-aggregator ## A/A, then the A/B of the ranker (RRF vs LightGBM) on simulated traffic (2 x 20 min); then make samples, ab-report
	$(PY) -m sim.ab $(AB_FLAGS)

ab-report: ## docs/results/ab.md from artifacts/ab/run.json and the sample job's impressions
	$(PY) -m analysis.ab_report

# ---------------------------------------------------------------- M6: shopping assistant

# Claude only where it is used, and only from the environment or .env (never on a command line):
# RECLOOP_AGENT_MODEL empty = the rules parser, no API calls.
serve-agent agent-eval-llm agent-eval-llm-smoke: export ANTHROPIC_API_KEY := $(ANTHROPIC_API_KEY)
serve-agent: export RECLOOP_AGENT_MODEL := $(RECLOOP_AGENT_MODEL)

serve-agent: ## Shopping assistant on AGENT_PORT: POST /v1/chat (SSE); needs the api (RECLOOP_API_URL)
	$(AGE) python -m agent.server --port $(AGENT_PORT)

agent-cases: ## Regenerate agent/eval/cases.jsonl from the catalog (reads items, read-only)
	$(AG) python -m agent.eval.build_cases

agent-eval: ## Shopping assistant: baseline + oracle agents -> docs/results/agent.md (no API calls)
	$(AGE) python -m agent.eval.run

# Needs the stack, the model server, the api and the agent service (make serve-agent); a quiet
# machine. LATENCY_FLAGS: --repeats N, --warmup N, --limit N.
agent-latency: ## Chat path (/v1/chat) latency through the running services -> agent.md (~1 min)
	$(AG) python -m agent.eval.chat_latency $(LATENCY_FLAGS)
	$(AG) python -m agent.eval.run --report-only

# Paid (design §19, 2026-11-08): needs the user's ANTHROPIC_API_KEY (in .env); without it the
# runner stops before any call. Defaults are the plan: three models, caps per run and in all,
# relevance graded by Sonnet. LLM_MODELS / LLM_CAPS / REPEATS / TOTAL_CAP_USD / LLM_JUDGE override.
LLM_MODELS ?= claude-opus-5-5 claude-sonnet-5-5 claude-haiku-5-5
LLM_CAPS ?= claude-opus-5-5=12,claude-sonnet-5-5=6,claude-haiku-5-5=1
LLM_JUDGE ?= claude-sonnet-5-5

agent-eval-llm-smoke: ## Paid smoke: 5 cases x the 3 models + judge, $1 in all (report untouched)
	$(AGE) python -m agent.eval.run --skip-offline --runs-dir artifacts/eval/agent/smoke --limit 5 \
		$(foreach m,$(LLM_MODELS),--llm $(m)) --repeats 1 --max-cost-usd $(LLM_CAPS) \
		--total-cap-usd 1 --judge $(LLM_JUDGE)

agent-eval-llm: ## Paid: 3 models x 3 runs (caps $12 / $6 / $1 per run) + Sonnet judge, $60 in all -> agent.md
	$(AGE) python -m agent.eval.run --skip-offline $(foreach m,$(LLM_MODELS),--llm $(m)) \
		--repeats $(or $(REPEATS),3) --max-cost-usd $(LLM_CAPS) \
		--total-cap-usd $(or $(TOTAL_CAP_USD),60) --judge $(LLM_JUDGE)

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
