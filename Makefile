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
AD_TOKEN_KEYS ?= dev1:recloop-dev-only-ad-token-key-not-a-secret-0001
export RECLOOP_PG_PORT RECLOOP_REDIS_PORT DATABASE_URL REDIS_URL API_ADDR MODEL_SERVER_PORT \
	MODEL_SERVER_ADDR RECLOOP_API_URL RECLOOP_BUNDLE AD_TOKEN_KEYS

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
UV := uv run --project ml --frozen
PY := $(UV) python
MS := uv run --project model_server --frozen
# goose's CLI links every SQL driver it supports; build it with Postgres only.
GOOSE_TAGS := no_clickhouse,no_libsql,no_mssql,no_mysql,no_sqlite3,no_vertica,no_ydb
GOOSE := cd serving && GOFLAGS="-modcacherw -tags=$(GOOSE_TAGS)" \
	go tool goose -dir migrations postgres "$(DATABASE_URL)"
# Code generators (tools/go.mod), built once into .cache/bin.
BIN := $(CURDIR)/.cache/bin
TOOLS := buf protoc-gen-go protoc-gen-go-grpc sqlc
TOOL_PKGS := github.com/bufbuild/buf/cmd/buf google.golang.org/protobuf/cmd/protoc-gen-go \
	google.golang.org/grpc/cmd/protoc-gen-go-grpc github.com/sqlc-dev/sqlc/cmd/sqlc

.PHONY: help setup up down down-volumes migrate migrate-status data data-weights ingest train \
	paper-sasrec eval ann bundle seed tools proto sqlc generate build-api serve-model serve-api \
	serve-web web-install test lint test-ml test-model-server test-go test-web test-props e2e fmt \
	clean-processed clean-artifacts

help: ## List targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*## "} {printf "  %-18s %s\n", $$1, $$2}'

setup: up data ingest train eval bundle seed web-install ## Everything from a clean checkout to a runnable demo
	@echo "ready: run make serve-model, make serve-api and make serve-web (three terminals)"

up: ## Start PostgreSQL + Redis (deploy/docker-compose.yml) and apply migrations
	$(COMPOSE) up -d --wait
	$(MAKE) migrate

down: ## Stop the stack, keep the database volume
	$(COMPOSE) down

down-volumes: ## Stop the stack and delete its volumes (all database contents)
	$(COMPOSE) down -v

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

bundle: ## Build artifacts/bundles/<version> and point artifacts/bundles/current at it
	$(PY) -m ml.bundle.build

seed: ## Flush Redis and load the dataset users' histories and the demo users
	$(PY) -m ml.data.seed_redis

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
	$(MS) python -m model_server --bundle "$(RECLOOP_BUNDLE)" --port $(MODEL_SERVER_PORT)

serve-api: build-api ## Go api on API_ADDR
	$(BIN)/api

serve-web: ## Next.js web app on WEB_PORT (proxies /api to the Go api)
	cd web && npm run dev -- --port $(WEB_PORT)

web-install: ## Install the web app's dependencies (npm ci)
	cd web && npm ci --no-audit --no-fund

test: lint test-ml test-model-server test-go ## Lint and run the Python and Go tests (DB / Redis tests skip when the stack is down)

lint: ## ruff check + format check (ml, model_server)
	$(UV) ruff check ml
	$(UV) ruff format --check ml
	$(MS) ruff check model_server
	$(MS) ruff format --check model_server

# test_ann.py loads faiss, so it gets a pytest process of its own: torch (test_sasrec.py) and
# faiss in one process abort on macOS (two OpenMP runtimes).
test-ml:
	$(UV) pytest ml/tests --ignore=ml/tests/test_ann.py
	$(UV) pytest ml/tests/test_ann.py

test-model-server:
	$(MS) pytest model_server/tests

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
	$(UV) ruff format ml
	$(UV) ruff check --fix ml
	$(MS) ruff format model_server
	$(MS) ruff check --fix model_server
	cd serving && gofmt -w cmd internal

clean-processed: ## Delete data/processed (keeps downloads and third_party/)
	rm -rf data/processed

clean-artifacts: ## Delete trained models, bundles and evaluation outputs (artifacts/)
	rm -rf artifacts/models artifacts/bundles artifacts/eval
