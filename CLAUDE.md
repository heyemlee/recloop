# RecLoop

A small office-supplies store with personalized recommendations, sponsored items in the feed and
a shopping assistant. Go for the online layer, Python for models and offline work, PostgreSQL +
pgvector, Redis, Redpanda.

The design lives in `docs/design.md` (Chinese). Read §0 (summary) and §16 (milestones) first;
decisions are logged in §19, ADRs in `docs/adr/`, experiment write-ups in `docs/results/`.

## Conventions

- Code, comments, this file, ADRs and `docs/results/` are in English. `docs/design.md` is Chinese.
- New design decisions go into `docs/design.md` §19 (date, decision, reason). Ask the user before
  changing the design itself.
- Datasets, checkpoints, model bundles and `third_party/` are never committed; only the scripts
  that fetch them. Secrets go in `.env` (git-ignored); `.env.example` holds the defaults.
- Milestone sessions don't commit or push. As the user asked, the coordinating session commits
  each milestone after verifying it and pushes to `origin` (github.com/heyemlee/recloop). Check
  §22.4 before anything is made public.
- `docs/results/*.md` are generated (`make eval` writes `offline.md`); change the generator, not
  the file.

## Layout

```
deploy/docker-compose.yml   PostgreSQL 18 + pgvector, Redis (ports 127.0.0.1:15432 / 16390)
serving/                    Go module github.com/heyemlee/recloop/serving; goose migrations in serving/migrations
ml/                         Python 3.12 uv project (own .venv); run modules from the repo root
  data/                     download, prefilter_meta, preprocess, catalog, verify, ingest
  models/                   pop, itemcf
  eval/                     splits, metrics, bootstrap, calc_check, report, run
  tests/
proto/ model_server/ agent/ web/ sim/ loadtest/ analysis/   later milestones
data/        (ignored) raw/ downloads, hf/ checkpoint files, processed/ outputs of `make data`
third_party/ (ignored) MiniOneRec clone, pinned
artifacts/   (ignored) eval outputs, model bundles
```

## Commands

```
make up            # compose up --wait + goose migrations
make data          # fetch + checksum inputs, re-run MiniOneRec preprocessing, export, verify
make ingest        # migrations + load items/users into PostgreSQL
make eval          # Pop / ItemCF -> docs/results/offline.md (no database)
make test          # ruff + pytest (data and DB tests skip with a reason when unavailable)
make down          # stop the stack; `make down-volumes` also deletes the database volume
make clean-processed   # rm -rf data/processed (downloads stay)
make data-weights  # 3.56 GB MiniOneRec weights, only needed from M7
```

Python: `uv run --project ml --frozen python -m ml.<package>.<module>` from the repo root (the
Makefile's `PY`). Go: `cd serving && go ...`; goose is a Go tool (`go tool goose`), built with
build tags that drop every driver but Postgres (`GOOSE_TAGS` in the Makefile).

Caches (`UV_CACHE_DIR`, `GOMODCACHE`, `GOCACHE`) default to `.cache/` in the repo. When running
inside the Claude Code sandbox: docker commands, connections to localhost:15432, `git clone` (it
writes `.git/config`), and Go module downloads (TLS verification fails) must run outside the
sandbox. Everything else (`make data` once inputs exist, `make eval`, `make test` minus the DB
tests) works inside it.

## Data (M0)

- Amazon Review Data (2018) Office Products, MiniOneRec preprocessing: 3,459 items, 8,328 users,
  56,984 interactions, 48,656 samples = 38,924 / 4,866 / 4,866 (chronological 8:1:1, history <= 10).
- `item_id` everywhere is MiniOneRec's item index (0..3458); `users.user_id` is its user index.
  `data/processed/items.parquet` maps item_id <-> ASIN; `interactions.parquet` / `samples.parquet`
  carry review timestamps.
- `make data` re-runs `amazon18_data_process.py` unmodified with `TZ=UTC` (it converts month
  bounds in local time) on metadata pre-filtered to the review ASINs, then rebuilds the
  published CSVs with the release-time converter and checks they are byte-identical. All pins
  (commits, SHA-256s) are in `ml/data/download.py`.
- 15 pairs of items share a semantic ID. 51.5% of test targets were reviewed on the same day as
  the last history item (order inside a day is file order); see `docs/results/offline.md`.

## Evaluation protocol

- Test = the published 4,866 samples; rank all 3,459 items; do not filter history items.
- HR / NDCG @ {3, 5, 10}; hit rule "item" (main) and "sid" (any item with the target's SID, what
  MiniOneRec's calc.py does). Ties: training popularity, then item_id.
- Fit on train rows only (`splits.train_sequences`), tune on valid, score test once.
- 95% CI: user-level bootstrap, B = 1000, seed 20261106 (`ml/eval/bootstrap.py`).
- To add a model: a class with `name`, `fit(sequences, num_items)`, `score(split) -> (n, 3459)`,
  `describe()`; add it to `models` in `ml/eval/run.py`. The calc.py cross-check runs for every model.
