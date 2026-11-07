# RecLoop

A small office-supplies store with personalized recommendations, sponsored items in the feed and
a shopping assistant. Go for the online layer, Python for models and offline work, PostgreSQL +
pgvector, Redis, Redpanda.

The design lives in `docs/design.md` (Chinese; kept locally, not in the repo). Read §0 (summary)
and §16 (milestones) first; decisions are logged in §19, ADRs in `docs/adr/`, experiment
write-ups in `docs/results/`.

## Conventions

- Code, comments, this file, ADRs and `docs/results/` are in English. `docs/design.md` is Chinese.
- New design decisions go into `docs/design.md` §19 (date, decision, reason). Ask the user before
  changing the design itself.
- Datasets, checkpoints, model bundles and `third_party/` are never committed; only the scripts
  that fetch them. Secrets go in `.env` (git-ignored); `.env.example` holds the defaults.
- Milestone sessions don't commit or push. As the user asked, the coordinating session commits
  each milestone after verifying it and pushes to `origin` (github.com/heyemlee/recloop). Check
  §22.4 before anything is made public.
- `docs/results/*.md` are generated (`make eval` writes `offline.md`, `make ann` writes
  `ann.md`, `make load` writes `latency.md`; `make latency-report` re-renders it from
  `artifacts/latency/results.json`); change the generator, not the file. The committed `offline.md` is `make setup`, then `make paper-sasrec`, then
  `make eval`. `make eval` re-runs give the same bytes; `make train` runs on the CPU with fixed
  seeds, and two runs with the same worker / thread counts gave identical weights.
- Generated code is committed: `serving/gen` and `model_server/gen` (`make proto`),
  `serving/internal/store/db` (`make sqlc`). Edit the `.proto` / SQL, then regenerate.
- Adapted third-party code keeps its licence: `licenses/` (SASRec.pytorch, Apache-2.0).

## Layout

```
deploy/docker-compose.yml   PostgreSQL 18 + pgvector, Redis, Redpanda (+ topic init), log-sink
                            (Redpanda Connect), aggregator, Prometheus, Grafana, OTel collector,
                            Jaeger; profile `bench`: Toxiproxy, api and model-server containers
deploy/connect/ prometheus/ grafana/ otel/ jaeger/ toxiproxy/   their configuration;
                            model-server.Dockerfile (bench image)
proto/                      buf workspace; recloop/modelserver/v1/model_server.proto (design §13.2)
serving/                    Go module github.com/heyemlee/recloop/serving
  cmd/api/                  HTTP api (:18080): /v1/recommendations, /v1/events, /v1/items/{id},
                            /v1/advertisers, /v1/campaigns[/{id}[/report]], ...
  cmd/aggregator/           events -> Redis; rebuild-from-db, reset-offsets, wait-lag
  internal/                 api, mixer (fan-out, RRF, filters), recall, experiment, events (record
                            format, Redpanda publisher, served writer, Direct = M1's sync path),
                            userstate (Redis + apply.lua), cache (pages, candidate cache, codec),
                            resilience (budgets, breakers), consumer (consume loop, lag),
                            aggregator, metrics (Prometheus), tracing (OTel), catalog, bundle,
                            store, ads (GSP auction, pCTR interface, campaign settings), adtoken
                            (signed ad_token), billing (charge transaction, Redis spend.lua)
  internal/store/db/        sqlc output; queries in internal/store/queries, schema = migrations/
  internal/testutil/        test DB per package (recloop_test_<package>) and Redis DB 15
  migrations/               goose (00001 catalog, 00002 events/orders/experiments/models,
                            00003 default experiment, 00004 text_emb stored inline,
                            00005 purchase -> order trigger on events_raw, 00006 advertisers,
                            campaigns, budget rows, ad_charges, ad_clicks)
  gen/                      protoc-gen-go output
model_server/               Python 3.12 uv project (own .venv): gRPC Retrieve over a bundle (:18081),
                            admin.py: fault switch + /metrics on :18082
ml/                         Python 3.12 uv project (own .venv); run modules from the repo root
  data/                     download, prefilter_meta, preprocess, catalog, verify, text_embed,
                            ingest, seed_redis
  models/                   pop, itemcf, sasrec (+ train_sasrec)
  eval/                     splits, metrics, bootstrap, calc_check, report, run, minionerec_sasrec,
                            minionerec (+ _report), sasrec_cost, footprint, machine
  bundle/                   build (+ faiss_index, run in a torch-free subprocess)
  ann/                      ANN benchmark (FAISS Flat / IVF / HNSW / IVF-PQ) and index updates
  experiment/               bucketing (Python twin of serving/internal/experiment)
  tests/
minionerec/                 Python 3.12 uv project (own .venv, Apple Silicon only): MiniOneRec's
                            evaluate.py + evaluate.py.patch on MPS (upstream, generate, cost)
web/                        Next.js 16 + shadcn/ui (:13000); /api/* is proxied to the Go api; e2e/;
                            /advertiser is the advertiser console
sim/                        traffic simulator (world, click model, sessions + fault switches, verify,
                            drill); runs in the ml environment (`$(PY) -m sim.<module>`)
loadtest/                   k6 scenario recs.js, experiment runner run.py, report.py (latency.md)
analysis/                   served_attribution.sql, srm.py
tools/                      go.mod pinning buf, protoc-gen-go(-grpc), sqlc (built into .cache/bin);
                            tools/k6/go.mod pins k6 (own module)
testdata/                   bucket_vectors.json, shared by the Go and Python bucketing tests
agent/                      later milestones
data/        (ignored) raw/ downloads, hf/ model files, processed/ outputs of `make data`
third_party/ (ignored) MiniOneRec and SASRec.pytorch clones
artifacts/   (ignored) models/sasrec, bundles/<version> + bundles/current, eval/, e2e/, logs/,
             sim/ (world.json, run ledgers), latency/ (results.json),
             minionerec/ (runs/<name>/, cost/, upstream/ = the staged, patched evaluate.py)
```

## Commands

```
make setup         # up, data, ingest, train, eval, bundle, seed, web-install: clean checkout -> demo
make up            # images + compose up --wait (event pipeline and monitoring too) + migrations
make images        # cross-compile api + aggregator into .cache/linux, build recloop/go:dev
make data          # fetch + checksum inputs, re-run MiniOneRec preprocessing, export, verify,
                   # text embeddings (bge-small-en-v1.5)
make ingest        # migrations + items (text_emb, simulated stock) / users into PostgreSQL
make train         # SASRec grid on train/valid (CPU, ~11 min) -> artifacts/models/sasrec
make paper-sasrec  # re-run MiniOneRec's sasrec.py on the split (~18 min; make eval picks it up)
make eval          # Pop / ItemCF / SASRec -> docs/results/offline.md (needs train; no database)
make ann           # ANN benchmark + index updates -> docs/results/ann.md (~25 min; idle machine:
                   # it marks the report preliminary when the load average is above 2 at the start)
make bundle        # artifacts/bundles/<version> (ONNX encoder, FAISS, ItemCF/Pop tables), current ->
make seed          # FLUSHDB Redis, load dataset users' train-period history + demo users
make rebuild-redis # seed + replay events_raw (FROM=db, default) or the event log (FROM=log)
make serve-model   # model server on :18081 (bundle: artifacts/bundles/current)
make serve-api     # Go api on :18080
make serve-web     # Next.js dev server on :13000
make test          # ruff + pytest (ml, model_server) + go vet/test; DB/Redis tests skip when down
make test-props    # long run of the ads / ad_token / billing property tests (CHECKS=2000, ~4 min)
make test-web      # tsc + eslint for web/
make e2e           # Playwright (local Chrome): click an item, reload, the feed follows (servers up)
make sim-world     # simulator world (users, targets, click model) -> artifacts/sim/world.json
make sim           # simulated sessions against the dev api (SIM_FLAGS, --faults ...) + verify
make srm           # simulated traffic, then the SRM check on events_raw (analysis/srm.py)
make drill         # duplicates + kill -9 of the consumers + broker restart, then verify (~2 min)
make bench-up      # bench profile: api :18180 + model server (admin :18182) behind Toxiproxy
make load          # latency experiments -> docs/results/latency.md (~90 min; idle machine; the
                   # report marks itself preliminary if the VM is not 8 CPUs or the machine is busy;
                   # LOAD_FLAGS='--resume "<reason>"' continues results.json, the report says so)
make load-smoke    # 1 repeat, 15 s runs -> artifacts/latency/smoke.md (development)
make latency-report  # re-render latency.md from artifacts/latency/results.json
make k6            # build k6 v2.3.0 (tools/k6) into .cache/bin
make proto / sqlc  # regenerate gRPC / SQL code (tools built once into .cache/bin)
make down          # stop the stack; `make down-volumes` also deletes its volumes
make clean-processed / clean-artifacts   # rm data/processed / artifacts/{models,bundles,eval}
make data-weights  # 3.56 GB MiniOneRec weights + the rest of its tokenizer (M7)
make minionerec-gen   # MiniOneRec on MPS over the test split, evaluate.sh's 8 shards one after
                      # another (~5 h; resumable); MINIONEREC_ARGS="--name trial --users 200"
make minionerec-cost  # latency / memory at 10, 20, 50 beams (MPS) + SASRec on the CPU (idle machine)
make minionerec       # score the runs like make eval -> docs/results/minionerec.md (no GPU)
```

Python: `uv run --project ml --frozen python -m ml.<package>.<module>` from the repo root (the
Makefile's `PY`); the model server is `uv run --project model_server ...` (`MS`), MiniOneRec
`uv run --project minionerec ...` (`MR`). Go:
`cd serving && go ...`; goose is a Go tool (`go tool goose`), built with tags that drop every
driver but Postgres (`GOOSE_TAGS`). `.claude/launch.json` starts the three servers for the
preview browser. Ports and URLs are in `.env.example`.

Go integration tests use `DATABASE_URL` / `REDIS_URL`: each test package gets its own database
on that server (`recloop_test_<package>`, created, migrated and emptied by `testutil.PG`) and
Redis DB 15. Several checkouts testing at once should each point them at a stack of their own
(e.g. `docker compose -p <name> -f deploy/docker-compose.yml up` with other `RECLOOP_*_PORT`s).

Caches (`UV_CACHE_DIR`, `GOMODCACHE`, `GOCACHE`, `BUF_CACHE_DIR`, npm) default to `.cache/` in
the repo. Inside the Claude Code sandbox these must run outside it: docker, anything that
connects to localhost (PostgreSQL, Redis, Redpanda, the servers, k6, the simulator), anything
that binds a local port (the servers, the model_server gRPC and admin tests, Go httptest,
miniredis), `git clone`, downloads (Go modules, npm, uv, Hugging Face), MPS (Metal) and
Playwright. `make eval`, `make train` and the non-network tests work inside it.

torch and faiss-cpu each ship an OpenMP runtime and crash when loaded together on macOS: keep
faiss out of torch processes (`ml/bundle/faiss_index.py` runs as a subprocess; the model server
has faiss + onnxruntime only; `make test-ml` runs `ml/tests/test_ann.py` in a pytest process of
its own). onnxruntime 1.31 on macOS starts a telemetry uploader at import that can segfault the
interpreter at exit: `ORT_DISABLE_TELEMETRY=1` (Makefile, model-server image, `ml/__init__.py`,
`model_server/__init__.py`).

## Data (M0)

- Amazon Review Data (2018) Office Products, MiniOneRec preprocessing: 3,459 items, 8,328 users,
  56,984 interactions, 48,656 samples = 38,924 / 4,866 / 4,866 (chronological 8:1:1, history <= 10).
- `item_id` everywhere is MiniOneRec's item index (0..3458); `users.user_id` is its user index.
  `data/processed/items.parquet` maps item_id <-> ASIN; `interactions.parquet` / `samples.parquet`
  carry review timestamps.
- `make data` re-runs `amazon18_data_process.py` unmodified with `TZ=UTC` (it converts month
  bounds in local time) on metadata pre-filtered to the review ASINs, then rebuilds the
  published CSVs with the release-time converter and checks they are byte-identical. All pins
  (commits, SHA-256s, the text model's revision) are in `ml/data/download.py`.
- 15 pairs of items share a semantic ID. 51.5% of test targets were reviewed on the same day as
  the last history item (order inside a day is file order); see `docs/results/offline.md`.

## Online loop (M1)

- `GET /v1/recommendations?user_id&surface=home|item&item_id&limit&cursor&debug=1`: Redis user
  context -> concurrent recall (home: sasrec via gRPC, itemcf, pop; item: itemcf, text via
  pgvector) with per-source timeouts and partial merge -> weighted RRF (k = 60, weights from the
  recall-layer experiment) -> filters (bought, seen in 24 h, sold out, current item) -> at most 2
  per brand in the top 10 -> pop backfill -> 200 items kept in `page:{request_id}` (10 min).
- Every response is logged as a `served` event (from M2 through Redpanda, see below); a CHECK
  constraint rejects served rows without `model_ver`, `exp.recall` and `config_version`.
- `POST /v1/events` ({"events": [...]}, <= 100): UUIDv7 `event_id`, `event_day` from its time
  (partition key). PostgreSQL dedups on (event_id, event_day); Redis applies each event once
  (`userstate/apply.lua`, `dedup:aggregator:{event_id}`; M1 used `dedup:api:`). Click /
  add_to_cart / purchase -> history + hv; purchase -> bought, order, stock - 1; all three and
  impressions -> seen.
- Experiments: `bucket = xxhash64(salt + ":" + user_id) % 1000` per layer (salts `<layer>-v1`);
  config in the `experiments` table, reloaded every 10 s.
- Bundle versions look like `2026-11-07-sasrec64-<content hash>`; the api registers the one it
  serves in `model_versions` (status active).

## Latency and events (M2)

- Events: the api validates and writes to the Redpanda topic `events` (6 partitions, key =
  user_id, 7 days; `KAFKA_BROKERS`, default localhost:19092; `EVENTS_MODE=direct` = M1's
  synchronous path) and answers 202. Record format `internal/events.Record` (JSON, header
  `type`, carries `impression_id` and `ad_token` for billing). Consumer groups: `aggregator`
  (Go, Redis; Lua dedup + state in one step, offsets committed after the batch, static
  membership) and `log-sink` (Redpanda Connect `deploy/connect/log-sink.yaml`: `sql_insert` ...
  `ON CONFLICT DO NOTHING`, unordered processing with 8 parallel 1,000-row inserts, ~16K rows/s;
  the purchase trigger creates orders and takes stock). The api and the
  aggregator keep the `events_raw` partitions of [today - 7, today + 1] in place; the api polls
  stock every 5 s. Served events: bounded buffer, batches retried for 30 s, dropped (counted) only
  when full.
- Request budget (`internal/resilience`, `BUDGET_*_MS`): 100 ms total; user context 5, candidate
  cache 3, recall 35, ranking reserve 25, tail 10. A stage gets min(its budget, left - later
  budgets); a remote source with < 5 ms left is skipped. Degradation reasons are
  `<dependency>_<timeout|error|circuit_open|skipped>`, `static_popular`, `pages_*`,
  `model_version_mismatch`; a degraded candidate list is never cached.
- Breakers (gobreaker) only in front of the model server and PostgreSQL text recall:
  `BREAKER_WINDOW=time:10s:20:10` (default, §5.1 since 2026-11-06: >= 50 % of the last 10 s with
  >= 20 calls, or 10 failures in a row; open 10 s, 5 half-open probes) or `calls:20` (the first
  rule, kept as a comparison row in `make load`). Redis has none: its calls are capped by their
  budgets and fail fast.
- Candidate cache: `cand:{uid}:home:{recall}.{rank variant}:{model_ver}:{hv}` and
  `cand:item:{seed}:...` (shared by users), 300 candidates, binary + S2, 5 min (`CANDIDATE_CACHE`).
- Naive mode for the fault experiments: `RECLOOP_RESILIENCE=naive` (no deadline, budgets,
  breakers or degradation; a failed dependency is a 503).
- Model server fault switch: `PUT :18082/fault {"delay_ms":200,"probability":0.3}` (`FAULT_*`
  env at start); the delay stops at the caller's deadline. 32 worker threads.
- Ports: Redpanda 19092 (admin 19644), log-sink 14195, aggregator metrics 18090, Prometheus
  19090, Grafana 13001 (dashboard "RecLoop"), Jaeger 16686, OTLP 14317 / 14318; bench api 18180,
  bench model-server admin 18182, Toxiproxy 18474. `make serve-api` / `serve-model` send traces
  to the collector (`OTEL_EXPORTER_OTLP_ENDPOINT`); the api serves `/metrics`.
- Load tests run on the bench profile: api and model server containers (2 CPU each); the fault
  experiment reaches the model server and Redis through Toxiproxy, the rest directly (the proxy
  adds its own tail); k6 runs on the host; before every run the consumers catch up and Redis is
  reseeded. Redis has `maxmemory 384mb` + `volatile-ttl` (pages and caches
  go first). Measurements need an idle machine and the coordinator's go-ahead.
- Simulator: `sim/world.py` (2,024 test-period users, targets = their test items, click model
  = examination x relevance by text similarity), `sim/run.py` (sessions, fault switches, ledger),
  `sim/verify.py` (events_raw / orders / Redis vs the ledger), `sim/drill.py`.

## Ads and billing (M3, logic part)

- Advertiser api: `GET/POST /v1/advertisers`; `GET/POST /v1/campaigns`, `GET/PATCH/DELETE
  /v1/campaigns/{id}`, `GET /v1/campaigns/{id}/report?from&to`. PATCH needs `If-Match` with the
  ETag (= `version`): 428 without it, 412 (with the current campaign) when it is stale. A new daily
  budget also updates today's budget row in the same transaction, never below today's spend.
  DELETE ends a campaign (terminal; rows stay for the ledger). Money is integer cents, days UTC.
- Auction (`ads.Run`): eCPM = bid x pCTR with pCTR quantised to ppm (exact integer ranking);
  GSP price = floor(eCPM(k+1) / pCTR(k)) + 1 cent, within [reserve 5 cents, own bid]; one slot per
  advertiser; quality floor `MinPCTR`. pCTR comes from `ads.PCTRModel` (`FixedPCTR` stub for now).
- `ad_token` (`adtoken`): HS256 JWT (golang-jwt) with rid, pos, cid, item, price, uid, iat, exp
  (iat + 30 min); keys from `AD_TOKEN_KEYS` (`kid:secret,...`, first signs, all verify).
- Billing (`billing.Service.Handle`, one transaction per click): claim the event_id in `ad_clicks`
  (technical dedup) -> lock the campaign's budget row for the day the impression was served ->
  billed = min(token price, budget - spent) -> insert into `ad_charges`, keyed by impression_id
  (business dedup) -> spent += billed; after COMMIT, Redis `spend:{cid}:{yyyymmdd}` once per
  event (`spend.lua`). Token expiry is judged at the click's receive time (stamped by the api),
  not at processing time. Not wired to the mixer or Redpanda yet.
- Property tests (rapid; billing ones against PostgreSQL): a sequential model with technical
  and business duplicates, late / forged / mismatched clicks, budget changes and crashes at every
  step must match the database exactly; concurrent workers must give the order-independent
  truth (one charge per clicked impression, spend = min(budget, prices), each outcome counted
  once). `make test` runs 30 cases each, `make test-props` 2,000.

## Evaluation protocol

- Test = the published 4,866 samples; rank all 3,459 items; do not filter history items.
- HR / NDCG @ {3, 5, 10}; hit rule "item" (main) and "sid" (any item with the target's SID, what
  MiniOneRec's calc.py does). Ties: training popularity, then item_id.
- Fit on train rows only (`splits.train_sequences`), tune on valid, score test once.
- Every model is also reported on two segments: target reviewed on the same day as the last
  history item, and later.
- 95% CI: user-level bootstrap, B = 1000, seed 20261106 (`ml/eval/bootstrap.py`).
- To add a model: a class with `name`, `fit(sequences, num_items)`, `score(split) -> (n, 3459)`,
  `describe()`; add it to `models` in `ml/eval/run.py`. The calc.py cross-check runs for every
  model. Models that are expensive to train (SASRec) are trained by their own make target and
  loaded by `make eval`.

## MiniOneRec (M7)

- Generation is MiniOneRec's own `evaluate.py`, staged from the pinned clone into
  `artifacts/minionerec/upstream/` with `minionerec/evaluate.py.patch` (mps device, no
  bitsandbytes import, `--dtype`, `--attn_implementation`, per-batch timing); prompts, SID
  prefix tree and constrained beam search are upstream's. `third_party/` stays untouched;
  `calc.py` and `merge.py` run unmodified. The environment pins MiniOneRec's versions (torch
  2.6.0, transformers 4.57.1).
- Attention is transformers' eager implementation (`upstream.ATTN_IMPLEMENTATION`): on MPS,
  torch 2.6's SDPA returns NaN for query rows that attend to nothing (a batch's left padding),
  and every padded sample then decodes token ids 0, 1, 2, ... instead of SIDs, in any dtype.
  `minionerec/tests/test_mps.py` pins this (it runs only where torch sees MPS).
- Scoring is ml/eval's metric code: item rule = each generated SID expands to its items in tie
  order; SID rule = beam position of the target's SID, which must equal calc.py at every K.
  Pop / ItemCF / SASRec are refitted as in `make eval` on the same samples; same bootstrap.
- Segments: history 1-3 / 4-10; target head (top 20% of items by training count) / tail;
  cross-category (level-2 category not in the history) / same; same day / later.
- Any MPS run (generation, cost, even a trial) waits for an idle machine: no compose stack, no
  other measurement. The full split peaks at 16.2 GiB of the 24, so quit Docker Desktop first:
  its VM keeps ~8 GB after the containers stop. The cost runs start each configuration only when
  the 1-minute load is below 2 and no Docker container runs, re-run it once if a container
  appears meanwhile, and stop if the machine is not quiet within 20 min
  (`ml.eval.machine.measure_quietly`; rerun the rest with `--only`). MPS is invisible inside the
  Claude Code sandbox (torch falls back to the CPU), so real runs go outside it. `--base-model`
  runs (tests, smoke checks with a tiny random Qwen2) are left out of the report.
