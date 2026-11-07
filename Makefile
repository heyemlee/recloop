# RecLoop developer entry points. Run from the repo root; `make help` lists the targets.
SHELL := /bin/bash
.DEFAULT_GOAL := help

# Local overrides (copy .env.example to .env); the defaults below match .env.example.
-include .env
RECLOOP_PG_PORT ?= 15432
RECLOOP_REDIS_PORT ?= 16390
DATABASE_URL ?= postgres://recloop:recloop@localhost:$(RECLOOP_PG_PORT)/recloop?sslmode=disable
REDIS_URL ?= redis://localhost:$(RECLOOP_REDIS_PORT)/0
export RECLOOP_PG_PORT RECLOOP_REDIS_PORT DATABASE_URL REDIS_URL

# Tool caches live in the repo (git-ignored) so the targets also work where ~/.cache is not
# writable, e.g. for sandboxed coding agents. Set them in the environment to use global caches.
UV_CACHE_DIR ?= $(CURDIR)/.cache/uv
GOMODCACHE ?= $(CURDIR)/.cache/go/mod
GOCACHE ?= $(CURDIR)/.cache/go/build
GOTOOLCHAIN ?= local
export UV_CACHE_DIR GOMODCACHE GOCACHE GOTOOLCHAIN

COMPOSE := docker compose -f deploy/docker-compose.yml
UV := uv run --project ml --frozen
PY := $(UV) python
# goose's CLI links every SQL driver it supports; build it with Postgres only.
GOOSE_TAGS := no_clickhouse,no_libsql,no_mssql,no_mysql,no_sqlite3,no_vertica,no_ydb
GOOSE := cd serving && GOFLAGS="-modcacherw -tags=$(GOOSE_TAGS)" \
	go tool goose -dir migrations postgres "$(DATABASE_URL)"

.PHONY: help up down down-volumes migrate migrate-status data data-weights ingest eval ann \
	test lint fmt clean-processed

help: ## List targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*## "} {printf "  %-16s %s\n", $$1, $$2}'

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

data: ## Download + verify inputs, re-run MiniOneRec preprocessing, export catalog, verify
	$(PY) -m ml.data.download
	$(PY) -m ml.data.preprocess
	$(PY) -m ml.data.catalog
	$(PY) -m ml.data.verify

data-weights: ## Also download the 3.56 GB MiniOneRec checkpoint weights (first needed in M7)
	$(PY) -m ml.data.download --with-weights

ingest: migrate ## Load items and users from data/processed into PostgreSQL
	$(PY) -m ml.data.ingest

eval: ## Offline evaluation of the baselines -> docs/results/offline.md (no database)
	$(PY) -m ml.eval.run

ann: ## ANN benchmark + index updates -> docs/results/ann.md (~30 min; run on an idle machine)
	$(PY) -m ml.ann.run $(ANN_FLAGS)

test: lint ## Lint and run the tests (data / database checks run when those are available)
	$(UV) pytest ml/tests

lint: ## ruff check + format check
	$(UV) ruff check ml
	$(UV) ruff format --check ml

fmt: ## Format and auto-fix
	$(UV) ruff format ml
	$(UV) ruff check --fix ml

clean-processed: ## Delete data/processed (keeps downloads and third_party/)
	rm -rf data/processed
