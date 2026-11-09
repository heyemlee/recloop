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
  `artifacts/latency/results.json`; `make billing` writes `billing.md`, `make billing-report`
  re-renders it from `artifacts/billing/`; `make agent-eval` writes `agent.md`, `make
  agent-latency` adds its chat-path section; `make release-drill` writes `release.md`, `make
  release-report` re-renders it from `artifacts/release/drill.json`; `make ab-report` writes
  `ab.md` from `artifacts/ab/` and the sample job's impressions; `make pctr-report` writes
  `pctr.md` from `artifacts/models/pctr/eval/`; `make deploy-runs` and `make deploy-drill` write
  `deploy.md`, `make deploy-report` re-renders it from `artifacts/deploy/`); change the
  generator, not the file. The committed `offline.md` is `make setup`, then `make paper-sasrec`,
  then `make eval` (`make setup` includes `make train-ranker`). `make eval` re-runs give the same
  bytes; `make train` runs on the CPU with fixed seeds, and two runs with the same worker / thread
  counts gave identical weights; `make train-ranker` (one thread) gives the same model. The committed
  `agent.md` is a fresh database (`make setup`: no orders, stock as ingested), then `make
  agent-eval`, then `make agent-latency` with the services up; re-runs differ only in their
  latency sections, and the report says when the database had orders (each one took a unit of
  stock).
- Generated code is committed: `serving/gen` and `model_server/gen` (`make proto`),
  `serving/internal/store/db` (`make sqlc`). Edit the `.proto` / SQL, then regenerate.
- Adapted third-party code keeps its licence: `licenses/` (SASRec.pytorch, Apache-2.0).

## Layout

```
deploy/docker-compose.yml   PostgreSQL 18 + pgvector, Redis (state + cache, redis_exporter), Redpanda
                            (+ topic init), log-sink (Redpanda Connect), aggregator, billing
                            (+ pg-proxy: Toxiproxy between billing and PostgreSQL), Prometheus,
                            Grafana, OTel collector, Jaeger; profile `bench`: Toxiproxy, api and
                            model-server containers
deploy/connect/ prometheus/ grafana/ otel/ jaeger/ toxiproxy/   their configuration (Grafana:
                            dashboard + provisioning/alerting/redis.yml); model-server.Dockerfile
                            (bench image); spark.Dockerfile + spark/ (the sample job's image:
                            PySpark 4.2.1, Temurin 21 JRE, its own uv lock); web.Dockerfile
                            (Next.js standalone build, for the cluster)
deploy/kind/                local Kubernetes (M5): cluster.yaml (one node, port mappings, bundles
                            mount), versions.lock (kind / kubectl / helm, node image and charts,
                            sha256-pinned), values/ (the official charts' values), cluster.sh
deploy/helm/recloop-infra/  CloudNativePG Cluster (pgvector), Redis StatefulSets, topic Job
deploy/helm/recloop/        api (HPA), model-server, aggregator + billing (StatefulSets, KEDA),
                            log-sink, web, pg-proxy, ServiceMonitors, Grafana dashboard + alerts
proto/                      buf workspace; recloop/modelserver/v1/model_server.proto (design §13.2)
serving/                    Go module github.com/heyemlee/recloop/serving
  cmd/api/                  HTTP api (:18080): /v1/recommendations, /v1/events, /v1/items/{id},
                            /v1/advertisers, /v1/campaigns[/{id}[/report]], ...,
                            POST /internal/ads/select, /internal/admin/* (M4: releases,
                            experiments, the model server's fault switch)
  cmd/aggregator/           events -> Redis (incl. frequency caps); rebuild-from-db, reset-offsets,
                            wait-lag
  cmd/billing/              sponsored clicks -> charge ledger + Redis spend; reconcile every minute;
                            `billing reconcile` = one pass
  internal/                 api, mixer (fan-out, RRF, filters), recall, experiment, events (record
                            format, Redpanda publisher, served writer, Direct = M1's sync path),
                            userstate (Redis + apply.lua, health: fullness, error kinds), cache
                            (pages, candidate cache, codec),
                            resilience (budgets, breakers), consumer (consume loop, lag),
                            aggregator, metrics (Prometheus), tracing (OTel), catalog, bundle,
                            store, ads (GSP auction, pCTR interface, campaign settings, serving:
                            campaign index, gates, pacing), adtoken (signed ad_token), billing
                            (charge transaction, Redis spend.lua, consumer, reconcile + raise.lua),
                            ranking (M4: the model server's Rank: features, LightGBM, pCTR),
                            itemctr (M4: the items' recent impression / click buckets, the api's
                            table of them: the pCTR's item CTR),
                            release (M4: model versions built -> ... -> retired, reconcile),
                            bundle (+ registry: several versions at once)
  internal/store/db/        sqlc output; queries in internal/store/queries, schema = migrations/
  internal/testutil/        test DB per package (recloop_test_<package>) and Redis DB 15
  migrations/               goose (00001 catalog, 00002 events/orders/experiments/models,
                            00003 default experiment, 00004 text_emb stored inline,
                            00005 purchase -> order trigger on events_raw, 00006 advertisers,
                            campaigns, budget rows, ad_charges, ad_clicks, 00007 release times +
                            one active version + model_version_events)
  gen/                      protoc-gen-go output
model_server/               Python 3.12 uv project (own .venv): gRPC Retrieve and Rank (:18081);
                            registry.py: several bundle versions, pin_version; ranking.py: the
                            features, the ONNX ranker, the pCTR; admin.py: fault switch, /metrics,
                            /bundles on :18082
ml/                         Python 3.12 uv project (own .venv); run modules from the repo root
  data/                     download, prefilter_meta, preprocess, catalog, verify, text_embed,
                            ingest, seed_redis
  models/                   pop, itemcf, sasrec (+ train_sasrec), rrf, ranker (+ train_ranker:
                            LightGBM LambdaRank), pctr (logistic regression)
  ranking/                  candidates (the api's recall + RRF replayed), features, onnx_trees
                            (LightGBM -> ONNX TreeEnsembleRegressor)
  samples/                  export (events_raw -> Parquet by day), job (PySpark: attribution,
                            labels, downsampled training samples; runs in its container)
  eval/                     splits, metrics, bootstrap, calc_check, report, run, minionerec_sasrec,
                            minionerec (+ _report), sasrec_cost, footprint, machine
  bundle/                   build (+ faiss_index, run in a torch-free subprocess)
  ann/                      ANN benchmark (FAISS Flat / IVF / HNSW / IVF-PQ) and index updates
  experiment/               bucketing (Python twin of serving/internal/experiment)
  nearline/                 keys, worker: the nearline generation worker (M7, Redis side)
  tests/
minionerec/                 Python 3.12 uv project (own .venv, Apple Silicon only): MiniOneRec's
                            evaluate.py + evaluate.py.patch on MPS (upstream, generate, cost);
                            engine (the same generation as a library, prompt encoded once),
                            replay (checks it against a run), serve_gen (the worker's generator)
web/                        Next.js 16 + shadcn/ui (:13000); /api/v1/* is proxied to the Go api
                            (/internal is not), /api/v1/chat to the agent service; e2e/;
                            /advertiser is the advertiser console, /chat the shopping assistant,
                            /admin the internal admin page (only with RECLOOP_ADMIN=1)
sim/                        traffic simulator (world, click model, sessions + fault switches, verify,
                            drill; ads, billing_drill, billing_verify, overdelivery; compose: this
                            checkout's compose project, redis_stats: both Redis instances; M4:
                            standing campaigns, traffic in the background, admin client,
                            release_drill, ab); runs in the ml environment (`$(PY) -m sim.<module>`)
loadtest/                   k6 scenario recs.js, experiment runner run.py, report.py (latency.md)
analysis/                   served_attribution.sql, srm.py, billing_report.py (billing.md),
                            ab_report.py (ab.md), release_report.py (release.md),
                            pctr_report.py (pctr.md)
kube/                       runs on the kind cluster: cluster (kubectl, Prometheus, watcher), run
                            (rollout, HPA, KEDA), drill (sim.billing_drill on pods), report
                            (deploy.md); ml environment
tools/                      go.mod pinning buf, protoc-gen-go(-grpc), sqlc (built into .cache/bin);
                            tools/k6/go.mod pins k6 (own module)
testdata/                   bucket_vectors.json, shared by the Go and Python bucketing tests;
                            item_ctr_vectors.json, by the Go item CTR and the sample job tests
agent/                      Python 3.12 uv project (own .venv): the shopping assistant (design §10)
  agent, parsing, schema    one turn; LLM / rule parsing into ShoppingRequest (Pydantic)
  catalog, tools, text      search_items (BM25 + pgvector, RRF, SQL filters), details, history,
                            get_recommendations
  history                   purchase history: orders + the dataset users' training period
  constraints, compat       code checks of every constraint; device matching with evidence
  llm, prompts, render      Claude Messages API adapter, prompts, replies rendered from rows
  store_api                 the Go api as a client: surface=chat recommendations, /internal/ads/select
  chat, server              ChatService (one turn -> events), FastAPI POST /v1/chat (SSE, :18083)
  eval/                     cases.jsonl (150, generated), harness, metrics, judge, report,
                            chat_latency (the chat path through the running services)
data/        (ignored) raw/ downloads, hf/ model files, processed/ outputs of `make data`
third_party/ (ignored) MiniOneRec and SASRec.pytorch clones
artifacts/   (ignored) models/sasrec, models/ranker, models/pctr, bundles/<version> +
             bundles/current, eval/, e2e/, logs/, samples/ (raw/, impressions/, train/, meta/),
             release/, ab/,
             sim/ (world.json, run ledgers), latency/ (results.json),
             minionerec/ (runs/<name>/, cost/, upstream/ = the staged, patched evaluate.py),
             nearline/ (ab/: run.json, worker/, traffic/, served_gen.jsonl),
             deploy/ (results.json, k6 summaries, drill/)
```

## Commands

```
make setup         # up, data, ingest, train, train-ranker, eval, bundle, seed, web-install:
                   # clean checkout -> demo
make up            # images + compose up --wait (event pipeline and monitoring too) + migrations
make images        # cross-compile api, aggregator, billing into .cache/linux, build recloop/go:dev
make data          # fetch + checksum inputs, re-run MiniOneRec preprocessing, export, verify,
                   # text embeddings (bge-small-en-v1.5)
make ingest        # migrations + items (text_emb, simulated stock) / users into PostgreSQL
make train         # SASRec grid on train/valid (CPU, ~11 min) -> artifacts/models/sasrec
make paper-sasrec  # re-run MiniOneRec's sasrec.py on the split (~18 min; make eval picks it up)
make train-ranker  # LightGBM LambdaRank on the validation split (~7 min) -> artifacts/models/ranker
make eval          # Pop / ItemCF / SASRec / RRF / LambdaRank -> docs/results/offline.md (needs
                   # train and train-ranker; no database)
make ann           # ANN benchmark + index updates -> docs/results/ann.md (~25 min; idle machine:
                   # it marks the report preliminary when the load average is above 2 at the start)
make bundle        # artifacts/bundles/<version> (ONNX encoder, FAISS, ItemCF/Pop tables, M4: feature
                   # tables, ONNX ranker, pCTR if trained); current -> it only when there is none
                   # (BUNDLE_FLAGS=--current forces it; --sasrec PATH, --no-ranker, --no-pctr)
make seed          # FLUSHDB both Redis instances (cache first), load dataset users' train-period
                   # history + demo users into the state Redis
make rebuild-redis # seed + replay events_raw (FROM=db, default) or the event log (FROM=log)
make serve-model   # model server on :18081 (bundle: artifacts/bundles/current)
make serve-api     # Go api on :18080
make serve-web     # Next.js dev server on :13000 (ADMIN=1: the /admin page works)
make test          # ruff + pytest (ml, model_server, minionerec, agent) + go vet/test; DB/Redis tests
                   # skip when down
make test-props    # long run of the ads / ad_token / billing property tests (CHECKS=2000, ~4 min)
make test-web      # tsc + eslint for web/
make e2e           # Playwright (local Chrome): click an item, reload, the feed follows (servers up)
make sim-world     # simulator world (users, targets, click model) -> artifacts/sim/world.json
make sim           # simulated sessions against the dev api (SIM_FLAGS, --faults ...) + verify
make srm           # simulated traffic, then the SRM check on events_raw (analysis/srm.py)
make drill         # duplicates + kill -9 of the consumers + broker restart, then verify (~2 min)
make bench-up      # bench profile: api :18180 + model server (admin :18182) behind Toxiproxy
make load          # latency experiments -> docs/results/latency.md (~2 h; idle machine; the
                   # report marks itself preliminary if the VM is not 8 CPUs or the machine is busy;
                   # LOAD_FLAGS='--resume "<reason>"' continues results.json, the report says so)
make load-smoke    # 1 repeat, 15 s runs -> artifacts/latency/smoke.md (development)
make latency-report  # re-render latency.md from artifacts/latency/results.json
make billing       # overdelivery + billing-drill -> docs/results/billing.md (~75 min; stack, model
                   # server and api up; reseeds Redis; refuses to run while other campaigns are
                   # active)
make overdelivery  # safety margin 0 / 5 / 20 clicks x pacing, 6 x ~5 min -> artifacts/billing/
make billing-drill # 30 min of traffic, kill -9 billing, Redpanda restarts, PostgreSQL +200 ms
                   # for billing; then the ledger against the simulator's truth, charge by charge
make billing-report  # re-render billing.md from artifacts/billing/
make k6            # build k6 v2.3.0 (tools/k6) into .cache/bin
make spark-image   # the sample job's image (recloop/spark:dev)
make samples       # DAY=YYYY-MM-DD (UTC, default today): export events_raw around it, then the
                   # PySpark job -> artifacts/samples/{impressions,train,meta}
make test-samples  # the sample job's tests, in its container
make train-pctr    # logistic-regression pCTR on artifacts/samples/train, with and without the
                   # item CTR columns -> artifacts/models/pctr (model.json, eval/<label>.json;
                   # PCTR_FLAGS="--days D --since T --label NAME")
make pctr-report   # docs/results/pctr.md from artifacts/models/pctr/eval
make collect       # standing campaigns + COLLECT_SECONDS (900) of simulated traffic for samples
make release-drill # NEW=<version> BAD=<version>: canary 5% -> full rollout, then BAD activated
                   # and rolled back, under traffic (~35 min) -> docs/results/release.md
make ab            # A/A, then RRF vs LightGBM, 2 x 20 min of traffic -> artifacts/ab/; then
                   # `make samples` and `make ab-report` -> docs/results/ab.md
make agent-cases   # regenerate agent/eval/cases.jsonl from the items table (read-only)
make agent-eval    # shopping assistant: baseline + oracle agents -> docs/results/agent.md (~1 min)
make serve-agent   # shopping assistant service on :18083 (POST /v1/chat; rules parser unless
                   # RECLOOP_AGENT_MODEL is set); needs the api
make agent-latency # chat path through the running services -> agent.md's chat section (~1 min;
                   # waits up to 20 min for a 1-minute load below 2; LATENCY_FLAGS)
make agent-eval-llm-smoke  # paid: 5 cases x 3 models + Sonnet judge, $1 in all (needs the key)
make agent-eval-llm  # paid: Opus / Sonnet / Haiku 5.5 x 3 runs, caps $12 / $6 / $1 per run, $60
                     # in all, Sonnet judge -> agent.md (needs the user's ANTHROPIC_API_KEY)
make proto / sqlc  # regenerate gRPC / SQL code (tools built once into .cache/bin)
make down          # stop the stack; `make down-volumes` also deletes its volumes
make clean-processed / clean-artifacts   # rm data/processed / artifacts/{models,bundles,eval}
make data-weights  # 3.56 GB MiniOneRec weights + the rest of its tokenizer (M7)
make minionerec-gen   # MiniOneRec on MPS over the test split, evaluate.sh's 8 shards one after
                      # another (~5 h; resumable); MINIONEREC_ARGS="--name trial --users 200"
make minionerec-cost  # latency / memory at 10, 20, 50 beams (MPS) + SASRec on the CPU (idle machine)
make minionerec       # score the runs like make eval -> docs/results/minionerec.md (no GPU)
make minionerec-replay   # the main run's batches with the prompt encoded once; every SID must match
                         # (MPS; REPLAY_ARGS="--name once-trial --batches 25" first)
make gen-worker       # nearline generation worker (MPS) on REDIS_URL; GEN_WORKER_ARGS="--out ..."
make gen-ab           # simulated A/B: 3 recall sources vs 3 + gen (starts the worker; stack up)
make nearline-report  # docs/results/nearline.md (after make gen-ab and make samples)
make kind-up       # kind cluster + infrastructure + RecLoop's images, data, services (~15 min;
                   # needs the machine: compose stopped, ~8 GB of Docker's VM)
make deploy        # rebuild + `kind load` RecLoop's images, helm upgrade its chart
make deploy-runs   # rolling update, HPA, KEDA (3 repeats each) -> docs/results/deploy.md; also
                   # deploy-rollout / deploy-hpa / deploy-keda; DEPLOY_FLAGS='--resume "<reason>"'
make deploy-drill  # 30-min billing drill with billing's pods deleted -> deploy.md
make deploy-report # re-render deploy.md from artifacts/deploy/
make deploy-smoke  # one short run of each kind + a 5-min drill -> artifacts/deploy/smoke.md
make kind-status / kind-down   # pods and autoscalers / delete the cluster
make kind-tools / lint-deploy  # pinned kind, kubectl, helm / helm lint + render, no cluster
```

Python: `uv run --project ml --frozen python -m ml.<package>.<module>` from the repo root (the
Makefile's `PY`); the model server is `uv run --project model_server ...` (`MS`), MiniOneRec
`uv run --project minionerec ...` (`MR`), the agent `uv run --project agent ...` (`AG`; `AGE`
adds the optional `encoder` group: torch + sentence-transformers for the query encoder, not
installed by tests or CI). Go:
`cd serving && go ...`; goose is a Go tool (`go tool goose`), built with tags that drop every
driver but Postgres (`GOOSE_TAGS`). `.claude/launch.json` starts the four servers for the
preview browser. Ports and URLs are in `.env.example`.

Go integration tests use `DATABASE_URL` / `REDIS_URL`: each test package gets its own database
on that server (`recloop_test_<package>`, created, migrated and emptied by `testutil.PG`) and
Redis DB 15 (the api tests' cache instance: DB 13). Several checkouts testing at once should each
point them at a stack of their own: `COMPOSE_PROJECT_NAME` and other `RECLOOP_*_PORT`s in the
checkout's `.env` (the Makefile's `$(COMPOSE)` passes `-p` and `--env-file .env`; the drills and
`loadtest/run.py` use `sim/compose.py`, never the shared `recloop` project by accident;
`RECLOOP_GO_IMAGE` / `RECLOOP_MS_IMAGE` keep the images private).

Caches (`UV_CACHE_DIR`, `GOMODCACHE`, `GOCACHE`, `BUF_CACHE_DIR`, npm) default to `.cache/` in
the repo. Inside the Claude Code sandbox these must run outside it: docker, anything that
connects to localhost (PostgreSQL, Redis, Redpanda, the servers, k6, the simulator), anything
that binds a local port (the servers, the model_server gRPC and admin tests, Go httptest,
miniredis), `git clone`, downloads (Go modules, npm, uv, Hugging Face), MPS (Metal) and
Playwright. `make eval`, `make train` and the non-network tests work inside it; `make
agent-cases` and `make agent-eval` read PostgreSQL, so they run outside.

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
  cache 3, recall 35, ranking reserve 25, ads 10 (M3), tail 10. A stage gets min(its budget, left - later
  budgets); a remote source with < 5 ms left is skipped. Degradation reasons are
  `<dependency>_<timeout|error|circuit_open|skipped>`, `static_popular`, `pages_*`,
  `model_version_mismatch`; a degraded candidate list is never cached.
- Breakers (gobreaker) only in front of the model server and PostgreSQL text recall:
  `BREAKER_WINDOW=time:10s:20:10` (default, §5.1 since 2026-11-06: >= 50 % of the last 10 s with
  >= 20 calls, or 10 failures in a row; open 10 s, 5 half-open probes) or `calls:20` (the first
  rule, kept as a comparison row in `make load`). A fifth field sets the half-open probes:
  `time:10s:20:25:2` (25 in a row, 2 probes) is `make load`'s candidate rule (`CANDIDATE_BREAKER`
  in `loadtest/run.py`: at the ladder's top rate and in the 30 %-slow fault), a comparison only.
  Redis has none: its calls are capped by their budgets and fail fast.
- Candidate cache: `cand:{uid}:home:{recall}.{rank variant}:{model_ver}:{hv}` and
  `cand:item:{seed}:...` (shared by users), 300 candidates, binary + S2, 5 min (`CANDIDATE_CACHE`).
- Naive mode for the fault experiments: `RECLOOP_RESILIENCE=naive` (no deadline, budgets,
  breakers or degradation; a failed dependency is a 503).
- Model server fault switch: `PUT :18082/fault {"delay_ms":200,"probability":0.3}` (`FAULT_*`
  env at start); the delay stops at the caller's deadline. 32 worker threads.
- Ports: Redpanda 19092 (admin 19644), log-sink 14195, aggregator metrics 18090, Prometheus
  19090, Grafana 13001 (dashboard "RecLoop"), Jaeger 16686, OTLP 14317 / 14318; bench api 18180,
  bench model-server admin 18182, Toxiproxy 18474; M3: billing metrics 18092, pg-proxy API 18475,
  model server admin `MODEL_SERVER_ADMIN_PORT` (18082); cache Redis 16391 (state Redis 16390).
  `make serve-api` / `serve-model` send traces to the collector (`OTEL_EXPORTER_OTLP_ENDPOINT`);
  the api serves `/metrics`.
- Load tests run on the bench profile: api and model server containers (2 CPU each); the fault
  experiment reaches the model server and Redis through Toxiproxy, the rest directly (the proxy
  adds its own tail); k6 runs on the host; before every run the consumers catch up and Redis is
  reseeded (both instances). Each repeat opens with the cache comparison (its three runs in a
  seeded shuffled order, `SHUFFLE_SEED`), then the ladder, the breaker comparison at the top rate
  (shuffled), the warm runs (1000 and 1500/s after 30 s at 400/s: a ladder run starts with the
  candidate cache empty), the faults (the 30 %-slow row's design variants shuffled) and the onsets;
  `results.json` records what was served (bundle, ranker, pCTR model, rank-layer experiments,
  both Redis policies). `latency.md` (2026-11-08 evening) was measured with both Redis
  instances and M4's bundle, so the model server's Rank (Python, 2 CPUs) is on every cache miss: the
  SLO holds up to 600/s; from 1000/s on, a run that starts with the cache empty never fills it (the
  breakers keep reopening and a degraded list is not cached); with the cache filled first, 1000/s
  meets it in 2 of 3 runs, 1500/s in none. Measurements need an idle machine and the coordinator's
  go-ahead.
- Simulator: `sim/world.py` (2,024 test-period users, targets = their test items, click model
  = examination x relevance by text similarity), `sim/run.py` (sessions, fault switches, ledger),
  `sim/verify.py` (events_raw / orders / Redis vs the ledger), `sim/drill.py`.

## Ads and billing (M3)

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
  not at processing time.
- Sponsored slots (`api/ads.go`, `ads/serving.go`): positions 3 and 9 of the home feed (the
  auction's first and second place, one slot per item; organic items move down; not on item
  pages). Active
  campaigns are kept in memory, reloaded every 5 s (a snapshot older than 1 min serves no ads:
  `ads_stale`). Per request: in the flight window -> items in stock, not bought, not in the
  request's organic list (all 200) -> one Redis MGET of `spend:{cid}:{yyyymmdd}` and
  `fc:{uid}:{cid}:{yyyymmdd}` (budget `BUDGET_ADS_MS`, 10) -> budget left > safety margin x bid
  (spend = max(Redis counter, the budget row of the last refresh): an evicted counter cannot
  make a spent campaign look new),
  frequency below the cap, pacing -> `FixedPCTR` (`ADS_FIXED_PCTR` 0.01, floor `ADS_MIN_PCTR`)
  -> `ads.Run` -> signed tokens. The page store keeps the slots (later pages show them); ads are
  never in the candidate cache. Failure: organic feed, `ads_<timeout|error|skipped>` /
  `pctr_error` in `degraded`; the §5.1 chain (no user state, SASRec or ranker) rules ads out, and
  so does a full state Redis (`ads_state_full`: caps and spend stop counting).
  The brand cap is an organic rule: ads are not counted in it.
- Serving policy: `ADS_SAFETY_MARGIN_CLICKS` (5), `ADS_PACING` (off): pacing serves a campaign
  with probability 1 while its spend is on or behind budget x elapsed share of today's flight
  window, falling to 0 at one bid ahead. An ads-layer experiment overrides both per user
  (params `safety_margin_clicks`, `pacing`, `min_pctr`); the billing runs set theirs that way.
- Served items of sponsored slots carry `campaign_id`, `price`, `pctr_ppm`, `ad_token`; the
  served event has `ads` (outcome, policy, gate counts). Responses: `slot: sponsored`,
  `label: Sponsored`, `campaign_id`, `ad_token`; the web app sends the token with the slot's
  impression and click.
- `POST /internal/ads/select` (for the shopping assistant, §10.3 (5)): `{"user_id",
  "candidate_item_ids" (items meeting the request's checked hard constraints),
  "exclude_item_ids" (the reply's organic items), "max_price_cents"?, "slots"? (1)}` -> at most
  one sponsored item (same gates and auction, positions from 1) with its token, a served event
  of surface `chat`; failures are 200 with no items and `degraded`. Not proxied by the web app.
- Frequency caps count impression events: a sponsored impression carries its slot's
  `ad_token`; the aggregator verifies it (`AD_TOKEN_KEYS`; expired tokens count, forged or
  misplaced ones do not) and `apply.lua` increments `fc:{uid}:{cid}:{day of serving}` (2 days)
  under the event's dedup marker. `rebuild-from-db` rebuilds them from the payload's token.
- Billing consumer (`cmd/billing`, group `billing`, static membership `billing-0`): only clicks
  with a token (header `type`), 8 at a time (`BILLING_WORKERS`), offsets committed after the
  whole batch; `ErrUnknownCampaign` and malformed records are logged and skipped. It reaches
  PostgreSQL through `pg-proxy`. Metrics on 18092 (`recloop_billing_*`, `recloop_reconcile_*`).
- Reconciliation (every minute in billing, `billing reconcile` once): per campaign-day of today
  and yesterday, ledger sum vs budget row (must be equal: reported) vs Redis. A counter behind
  the ledger is repaired by re-applying each charge with billing's own dedup marker
  (`dedup:billing:{click event_id}`), then raised to at least the ledger sum (`raise.lua`): it
  can never pass the ledger. A counter ahead is reported, not changed.
- Billing runs (`make billing`): `sim.overdelivery` and `sim.billing_drill` create their own
  campaigns through the api (and end them), refuse to start while other campaigns are active,
  reseed Redis before every configuration (`ml.data.seed_redis`, after the consumers caught
  up), and write `artifacts/billing/`. The state Redis must neither evict nor refuse a write
  during them (the drill fails otherwise); the cache Redis's evictions are recorded
  (`sim.redis_stats`). The api runs with its defaults (candidate cache on). The over-delivery
  study in `billing.md` is from 2026-11-08 01:20 UTC, before the split (one Redis, candidate cache
  off); the report names each section's setup. The truth (`sim.billing_verify`) comes from the simulator's
  ledger only: the tokens it was served (claims read, not verified), its clicks, which ones it
  forged (`forge` fault). `--dwell-ms` delays a page's events (time to click). The drills'
  docker commands go through `sim/compose.py`; `PG_PROXY_URL` reaches pg-proxy.
- Property tests (rapid; billing ones against PostgreSQL): a sequential model with technical
  and business duplicates, late / forged / mismatched clicks, budget changes and crashes at every
  step must match the database exactly; concurrent workers must give the order-independent
  truth (one charge per clicked impression, spend = min(budget, prices), each outcome counted
  once). `make test` runs 30 cases each, `make test-props` 2,000.

## Redis: state and cache (M3, 2026-11-08)

- Two instances (design §19, 2026-11-08), both without persistence (derived state, rebuilt from
  the log). State `redis` (`REDIS_URL`, 16390): `u:{uid}:*`, `dedup:*`, `fc:*`, `spend:*`,
  `demo:users`, `ictr:*` and `state:seeded_at` (M4, the item CTR buckets); `noeviction`, `RECLOOP_REDIS_MAXMEMORY` 1 GB (container 1.5 GB). Cache
  `redis-cache` (`CACHE_REDIS_URL`, 16391): `page:*`, `cand:*`; `allkeys-lru`, 384 MB. The api
  uses both (user context, ads MGET, demo users: state; candidate cache, pages: cache; the page +
  candidate write is still one pipeline). `CACHE_REDIS_URL` defaults to the cache instance and
  never falls back to `REDIS_URL`; the api warns at start when the state instance evicts, the
  cache does not, or both URLs reach one instance (same run_id). Aggregator, billing and
  reconciliation use the state instance only.
- Sizing (measured on redis 8.10, replaying the 30-minute billing drill with apply.lua): 124 B per
  event (7-day dedup marker) + 76 B per (user, item) seen in 24 h; the seed is 3 MB, the drill
  ~117 MB, one `make load` run ~150 MB (reseeded before every run); the item CTR buckets (M4)
  stay under 14 MB. 1 GB holds ~6.8 M events of a week. When the state instance is full it
  refuses writes (a script at its first write: nothing half applied): the aggregator retries its
  batch (offsets uncommitted, lag grows, no loss), billing still charges in PostgreSQL but its
  spend update and reconciliation's repairs fail, the api serves no ads (`userstate.Fullness`:
  INFO every 5 s, full at 98 % of maxmemory or on a new `errorstat_OOM`). Remedy: `make seed` /
  `make rebuild-redis`, or a larger maxmemory.
- Visible: redis_exporter v1.95.0 (`job="redis"`, label `redis="state"|"cache"`):
  `redis_memory_*`, `redis_evicted_keys_total`, `redis_errors_total{err="OOM"}`; the services'
  `recloop_redis_errors_total{component,op,kind}` and the api's `recloop_redis_state_full`.
  Grafana: the dashboard's Redis row and the provisioned alert rules (state evicted, state full,
  write errors, state above 80 %, reconciliation stale, aggregator lag, exporter missing); no
  contact point locally (the Alerting page shows them).

## Ranking and releases (M4)

- Offline ranker: `ml/ranking/candidates.py` replays the api's home feed on a split (SASRec,
  ItemCF (score > 0) and Pop top 200 each, by score then item id; weighted RRF, k = 60, summed in
  the sources' order, ties by best rank then item id; 300 candidates). `ml/ranking/features.py`:
  10 features (each source's score and rank if it returned the item, popularity, how many
  sources, price bucket, level-2 / leaf category overlap with the history; NaN = missing).
  `make train-ranker`: LightGBM LambdaRank on the validation split (recall models fitted on train
  rows; queries without the target among the candidates left out), grid by 4-fold CV over users,
  refit with the mean best rounds. LightGBM is built from source without OpenMP (ml/pyproject.toml
  `[tool.uv]`): one thread, deterministic, and it can share a process with torch. The item log's
  CTR features of design §4.3 are not used: the dataset has no impressions and simulated clicks
  follow the test targets (leakage); the user's decision is pending.
- Bundle (M4 files): `features.json` (item tables), `ranker.onnx` (ml/ranking/onnx_trees.py:
  thresholds rounded down to float32, LightGBM's missing-value rules, |x| <= 1e-35 read as 0;
  checked against LightGBM on every validation candidate), `ranker.txt`, `rank_golden.json`,
  `pctr.json` (when `make train-pctr` ran). A bundle without them still serves (RRF, fixed pCTR).
- Model server: `Rank(history, candidates + each source's (rank, score), ads + optional
  relevance, pin_version)` -> feature rows, LightGBM scores, pCTR of the ads (logistic regression
  in numpy at the reference position 3). Retrieve scores `ad_item_ids` with the same user vector.
  Several versions at once (`registry.py`): a request names its version (`pin_version`), else
  the default; versions are found by name under the startup bundle's parent directory; loading
  replays the golden data (encoder, ranker, pCTR). Admin: `GET/POST /bundles`, `PUT
  /bundles/default`, `DELETE /bundles/<version>`.
- api: a request reads its bundle once: the rank variant's `pin_version` if loaded, else the
  active one; recall and Rank pin the model server to it. On the home feed with an M4 bundle,
  Rank runs for every user (both rank variants: features for the served log, the LightGBM score)
  under the ranking budget (25 ms; skipped below 5 ms; breaker `rank` of its own); the rank
  variant's `{"ranker": "lightgbm"}` orders the list by it (ties: RRF's order), `rrf` (default,
  `RANKER`) keeps RRF's. A failed call: RRF's order, `rank_<status>` in `degraded`, no ads
  (§5.1); a degraded list is not cached. Served items log `f` (features, null = missing) and
  `lgbm`; the payload `features` (names), `ranker`, `pctr_model`. Candidate-cache hits carry no
  organic features. Ads: the in-flight campaigns' in-stock, unbought items go into Retrieve;
  pCTR comes from Rank (`ranking.Client` implements `ads.PCTRModel`), FixedPCTR when the bundle
  has no pCTR model (features are logged either way). `RANK=off`: RRF and FixedPCTR only.
- Releases (`internal/release`, design §6): `model_versions` built -> validated -> canary ->
  active -> retired, every step a row of `model_version_events` (migration 00007; at most one
  active version). validate = the api (checksums, catalog size, ItemCF golden queries) and the
  model server (its golden data) load the version while the active one serves; canary = the
  rank-layer experiment `release-canary` over buckets [0, 10 x percent) with `pin_version`;
  activate = one transaction retires the previous version, the api switches at once, the model
  server's default follows, the canary ends; rollback = back to the version the active one
  replaced, within 30 minutes (both sides keep it loaded that long); an aborted canary is
  retired without having been active. Every api instance reconciles every 2 s (loads the
  active, canary and recently retired versions, unloads the rest; under the release lock, so a
  pass never undoes a step it raced with); at startup the api takes
  model_versions' active version (the startup bundle only when there is none). Admin api:
  `/internal/admin/models` (GET, POST {version}), `.../models/{version}/validate|canary|activate`,
  `/internal/admin/canary/abort`, `/internal/admin/rollback`, `/internal/admin/experiments[/{id}
  [/stop]]` (GET / PUT / POST), `/internal/admin/fault` (GET / PUT).
- Admin page: `web/src/app/admin` (experiments and their split, versions with validate / canary /
  full rollout / abort / roll back, the release record, the fault switch); its server-side route
  `/admin/api/*` forwards an allowlist to the api's `/internal/admin/*` only when the web app runs
  with `RECLOOP_ADMIN=1`; otherwise 404.
- Samples (`ml/samples`, design §7.5): `export.py` writes events_raw by day and type to
  `artifacts/samples/raw/` on the host (`--around D`: D-2..D+1); `job.py` (PySpark, local mode)
  reads served of D-1..D, impressions of D-1..D, clicks / carts / purchases of D..D+1 (and the
  impressions and clicks of D-2..D+1 for the item CTR counts); one row per event_id, one
  impression per impression_id (a click without one stands for it), joined to its served record
  on (request_id, position) (no record or another item: dropped, counted); label = a click within
  30 minutes (event times = UUIDv7 times); carted / purchased within 30 minutes after the click;
  `final = false` until the window has passed (`--complete-through`, default the newest event);
  training samples = final home-feed impressions with served features, negatives kept by a hash
  of the impression id below the rate (0.1) with weight 1 / rate. The job runs in a container
  (`deploy/spark.Dockerfile`, `--network none --hostname localhost`, as the host user with a
  one-line passwd mounted): HotSpot JVMs started from the Claude app's processes on this Mac
  crash at startup (SIGBUS in CodeHeap::allocate, Temurin 17 and 21).
- pCTR (`ml/models/pctr.py`): logistic regression (scikit-learn) on the samples' features plus
  the slot and the position (one-hot 1..10, 11-20, 21+), weighted; C chosen on a time-ordered
  split, judged on the last 20 % (AUC, log loss, calibration by position); the model server sets
  the position to 3 and sponsored to 1. Variant `ctr` (shipped when the samples carry counts)
  adds the item CTR columns: per window (hour, day) the log-odds of (clicks + strength x prior)
  / (impressions + strength) (prior = the training rows' CTR, strength chosen with C from 10 /
  100 / 1000) and log(1 + impressions); missing counts = 0 = the prior. Both variants are judged
  on the same split, with a paired user bootstrap of the differences ->
  `artifacts/models/pctr/eval/<label>.json` -> `make pctr-report`.
- Item CTR counts (`internal/itemctr`, design §9.2 step 3, §19 2026-11-08): impression and click
  events with an impression_id, once per event_id, every surface, bucketed by the api's receive
  time (`Event.Received`, kept in microseconds = `events_raw.server_ts`; a client clock cannot
  reach a closed bucket). apply.lua counts them under the event's dedup marker into
  `ictr:5m:{yyyymmddHHMM}` / `ictr:1h:{yyyymmddHH}` (fields `i:{item}`, `c:{item}`; expire 90 min
  / 25 h 30 min after the bucket's start; an expired bucket of a replayed event is skipped). The
  api (when it serves ads) rebuilds an in-memory table 15 s after each 5-minute boundary E
  (`itemctr.Refresher`, checked every 10 s; one HGETALL at a time): hour = the 12 complete
  5-minute buckets before E, day = the 24 complete hours before E's hour, none that starts before
  `state:seeded_at` (written by `make seed`, deleted by `rebuild-from-db` / `reset-offsets`); a
  table older than 15 min is not used. The ads stage sends each candidate's counts in Rank
  (`AdCandidate.counts`); the served event logs `ictr` [hour impressions, hour clicks, day
  impressions, day clicks] next to every `f`, and `ictr_asof` / `ictr_since` (Unix ms). The sample
  job (step 8; the export covers D-2..D+1) recomputes the counts from the log for the logged E and
  seed (records from before: E = 15 s before serving, no seed); samples keep the logged counts
  (`ictr_source` served) or the recomputed ones (log); meta `item_ctr` compares them (the online /
  offline check). `sim.verify` (so `make drill`) checks the run's events were counted exactly
  once. Size (measured): ~43 B per field; at ~87 page views/s a 5-minute bucket holds ~4,200
  fields (~190 KB), an hourly one ~5,500 (~230 KB); at most 46 live hashes, under 14 MB with
  every item in each. Known limit: the counts are not validated (forged impressions / clicks
  raise an item's pCTR, and a higher pCTR lowers its GSP price).
- Runs (stack, model server and api up; one at a time on an idle machine): `make collect`, then
  `make samples` and `make train-pctr`, `make bundle` (the release), `make bundle
  BUNDLE_FLAGS="--sasrec artifacts/models/sasrec/trials/8.pt"` (a bad release: the BCE trial),
  `make release-drill NEW=... BAD=...`, `make ab`, `make samples`, `make ab-report`. The runs
  reseed Redis between phases (`make ab` does it itself). The drills' traffic (`sim.traffic`)
  pauses 50-300 ms between a session's page views (16 sessions: ~85 page views/s); unpaced, the
  simulator is a load test that fills the 1 GB state Redis within 15 minutes. The release drill's
  phases run from step to step by the steps' own times; `python -m sim.release_drill --analyse`
  re-derives its tables from `artifacts/release/` (then `make release-report`). After simulated
  traffic, `make samples` gets `SAMPLES_FLAGS="--complete-through <the traffic's end + 30 min,
  UTC>"`: the simulator has stopped, so nothing can arrive later and this reads what a run 30
  minutes later would (the default, the newest event, leaves the last 30 minutes not final).
- The committed `pctr.md`: `make samples` of the M4 runs' day, `make train-pctr PCTR_FLAGS="--days
  2026-11-08 --until 2026-11-08T15:00:00Z --label m4-runs"` (counts recomputed: those records
  logged none); then `make seed`, `COLLECT_SECONDS=1800 make collect` (+ `sim.verify --run
  artifacts/sim/collect`), `make samples` (the whole day again), `make train-pctr
  PCTR_FLAGS="--days 2026-11-08 --since <the collect's start> --label collect"` (counts logged);
  m4-runs once more (the same numbers from the new samples; the last run is the shipped model),
  `make pctr-report`.

## Local Kubernetes (M5)

- `make kind-up` (`deploy/kind/cluster.sh`): kind v0.33.0, one node `kindest/node:v1.36.4`
  (CloudNativePG 1.30 and KEDA 2.21 support 1.34-1.36) in Docker Desktop's VM, context
  `kind-recloop`. kind, kubectl, helm in `.cache/bin`; kubeconfig `.cache/kind/kubeconfig`, Helm's
  state `.cache/helm` (nothing goes to `~/.kube` or `~/.config`). Every tool, chart and the node
  image is pinned in `deploy/kind/versions.lock` and sha256-checked (a GitHub release download
  falls back to the API's asset endpoint, which some proxies need).
- Host access: NodePorts through kind's port mappings on 127.0.0.1: web 30000, api 30080,
  Prometheus 30090, Grafana 30001, PostgreSQL 30432, Redis 30379 (state) / 30380 (cache), Redpanda
  31092 (external listener advertised as localhost), pg-proxy's Toxiproxy API 30474.
  `kube.cluster.host_env()` points host tools (simulator, seed, `aggregator wait-lag`, `billing
  reconcile`) at them; `KIND_*` variables override.
- Infrastructure (official charts, `deploy/kind/values/`): kube-prometheus-stack 92.10.1 without
  Alertmanager, node-exporter and the control-plane scrapes (memory; kind binds those to the
  node's localhost), metrics-server 3.14.0 (`--kubelet-insecure-tls`), KEDA 2.21.0, CloudNativePG
  0.29.1 (operator 1.30.1), Redpanda chart 26.2.5 (v26.2.6: one broker, plaintext, no console or
  config watchers, fsync on, auto-create off). `deploy/helm/recloop-infra`: Cluster `recloop-pg`
  (CloudNativePG's standard 18.6 image = pgvector, digest-pinned; `vector` and `btree_gist` made
  at bootstrap, the app user is no superuser), Redis `redis` (state, noeviction, 1 GB) and
  `redis-cache` (allkeys-lru, 384 MB) StatefulSets with redis_exporter sidecars, the topic Job.
- RecLoop's images: compose's build definitions (`make images` with `RECLOOP_GO_IMAGE`,
  `deploy/model-server.Dockerfile`) plus `deploy/web.Dockerfile`, named `recloop-kind/*`, tagged
  with their image ID (unchanged code keeps its tag, so `make deploy` rolls only what changed),
  `kind load`ed, never pushed. The bundles: `artifacts/bundles` mounted read-only into the node
  (`/recloop/bundles`) and from there into the api and model-server pods.
- The chart (`deploy/helm/recloop`): every pod `enableServiceLinks: false` (a `model-server`
  Service would inject `MODEL_SERVER_PORT=tcp://...`, which the model server reads as its port);
  api readiness `/healthz`, not `/readyz` (a dependency outage must degrade, not empty the
  Service), startup probes on the api and the consumers; api CPU request = limit = 500m, HPA
  1-4 pods at 50 % (kube-proxy places connections: the oldest pod runs hotter than the average);
  model-server readiness = gRPC health (SERVING after the bundle checks), `maxSurge 1`,
  `maxUnavailable 0`, `minReadySeconds 5`, preStop sleep 5 s; aggregator and billing are
  StatefulSets (pod name = static `group.instance.id`), KEDA Kafka scaler 1-3 pods, 20,000
  records of lag per pod (above the 15 s stall a scale-in causes), aggregator CPU limit 250m;
  api `MODEL_SERVER_ADMIN_URL=http://model-server:18082` (M4 releases reconcile one address: with
  two model-server pods a release reaches one of them; the runs make none); no assistant (§23.2:
  the web image has an empty `RECLOOP_AGENT_URL`, no `/api/v1/chat` rewrite, the chat page says
  so); log-sink `Recreate` (its config's static id); billing through pg-proxy.
- Runs (`kube/`, the ml environment, k6 and checks on the host): `kube.run` = rollout (gated,
  no-probe, broken bundle: `item_ids.json` shuffled, golden queries fail), hpa (step in load),
  keda (billing behind PostgreSQL +200 ms; aggregator paused 90 s, with KEDA and capped at 1);
  `kube.drill` = `sim.billing_drill` with the `Kube` platform (billing pods deleted, alternately
  graceful and forced; `redpanda-0` deleted). `loadtest/recs.js` takes `PROFILE` (steps),
  `WINDOW_S` (per-window percentiles) and `CONN_CLOSE` (kube-proxy places connections, not
  requests); sponsored slots' events carry their `ad_token`. The cluster needs the machine to
  itself (compose stopped) and the coordinator's go-ahead; `make kind-down` deletes it with its
  volumes.

## Shopping assistant (M6)

- One turn (`agent/agent.py`): parse -> `get_purchase_history` (only when the request refers to
  an earlier purchase) -> `search_items` -> `get_item_details` -> code checks -> device
  matching -> personalisation -> reasons -> the sponsored item -> reply rendered from the item
  rows (titles, prices). The model never picks or orders items.
- Personalisation (store only): `get_recommendations` = `GET /v1/recommendations?surface=chat
  &seed_items=...` (the first 5 checked items as seeds). The chat surface runs the home sources
  over history + seeds (the first seed as the most recent item), no ads, no candidate cache, no
  paging, no seen-today filter. Its ranks re-order only the first 10 checked items: RRF of the
  checked order and the recommendation rank, the latter weighted 0.1 (its first pick can pass 6
  places, its 50th 3); it never adds or removes an item; confirmed-compatible items stay first.
- Sponsored item (store only): `POST /internal/ads/select` with the checked items the reply does
  not show (a named device: confirmed only) as candidates, the shown ones excluded, the budget
  as max price; at most one, dropped unless it is one of the candidates; "Sponsored" label and
  reason by code. Its card carries the ads module's impression_id and ad_token, which the chat
  page sends with its impression and click (billing, frequency caps). Failures of either call
  leave the reply and are listed in `degraded` (`recommendations_*`, `ads_*`).
- Service (`agent/server.py`, FastAPI; `make serve-agent`, :18083): `POST /v1/chat {user_id,
  message}` streams server-sent events: meta (reply_id), status per stage, reply (text, intro,
  item cards, sponsored card, footer, constraints, degraded), done (timings, LLM calls, cost).
  Organic cards' impression_id is `{reply_id}:{position}` (reply_id a UUIDv7; no served row).
  The web app proxies `/api/v1/chat` to it (`RECLOOP_AGENT_URL`); `/chat` is the page.
  `RECLOOP_AGENT_MODEL` empty = rules parser, no API calls.
- Parsing: `ShoppingRequest` (category key from `agent/taxonomy.py`, budget in cents, brands,
  excluded brands, genuine-only, device, last-purchase relation, search text). A reply that
  fails validation goes back once with the error; a second failure ends the turn.
- Hard constraints (price, stock, category) and brands are checked by code on the rows: an
  unknown price never meets a budget, an item without a category path never meets a category.
  Price, stock and category are also SQL filters of `search_items`.
- Devices (`agent/compat.py`): code match in title, features, description (the model as
  written, or a bare number in a list opened by the series prefix or a family word); the model
  judges the rest only when the code confirmed fewer than 3, and its "compatible" needs a quote
  that is in the named field and contains the model number. Otherwise: "not confirmed".
- Search: BM25 in-process on `agent/text.py` tokens (`WF-2760` = `WF2760` = 'wf2760'; there is
  no tsvector column, PostgreSQL's `english` parser splits model numbers), exact pgvector KNN of
  the bge query embedding (with bge's query instruction), RRF k = 60, 30 results.
  The query encoder is bge-small-en-v1.5 at ml/'s library versions, its files checked against
  `ml/data/download.py`; `RECLOOP_TEXT_MODEL` points at another copy of the model directory.
- Purchase history (`agent/history.py`): in the store it is what the feed counts as bought,
  `orders` plus the dataset users' training-period interactions (the items `make seed` puts in
  `u:{uid}:bought`), read once from `data/processed/interactions.parquet`
  (`RECLOOP_INTERACTIONS`). A store-facing agent passes `StorePurchases` as `purchases`; without
  it the agent reads `orders` only, and the evaluation passes each case's own list.
- Database access is a read-only session (`SET SESSION CHARACTERISTICS AS TRANSACTION READ
  ONLY`) on `items` and `orders`.
- Claude (`agent/llm.py`, anthropic 2.3.0): structured outputs, effort low, cached system
  prompt, `fallbacks="default"` on Opus 5.5 / Sonnet 5.5, cost per reply at the serving model's
  price. Nothing calls it except the paid targets and a service started with
  `RECLOOP_AGENT_MODEL`; tests use a scripted fake. Paid runs stop before a case (or judge call)
  that could pass their cap: twice the dearest so far.
- Evaluation: `agent/eval/cases.jsonl` (150 cases, 12 templates, gold constraints, purchase
  histories as fixtures, user ids from 900000); agents `baseline` (rules), `oracle` (gold
  constraints) and search ablations; LLM agents and the Sonnet judge (grades per (case, item),
  shared by every agent that showed it) wait for the user's key (design §19, 2026-11-08).
- Chat path latency (`make agent-latency`): the 150 messages x 3 through `/v1/chat`, one at a
  time, from a fixed sample of dataset users with history, after 30 warm-up replies; records
  load and the other projects' containers, and marks a busy start "indicative".

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
- Nearline (design §11 steps 5-6): apply.lua queues a user in `gen:queue` (state Redis) when
  their history changes, only while a worker holds `gen:worker`; due after 30 s of quiet, at
  most 120 s after the first unanswered change. `ml.nearline.worker` claims due users, reads
  the last 10 items and hv in one MULTI, has `minionerec.serve_gen` generate (minionerec.engine:
  evaluate.py's prompts, constraint and beam search, the prompt pass shared by the beams), and
  writes `gen:{uid}:{ver}` (JSON: hv, at, items, scores; testdata/gen_payload.json pins it).
  The api reads it in the user-context pipeline only for a recall variant with an `rrf_weights`
  entry `gen` > 0 and a `gen_version`; that arm's candidate cache key carries the result's hv.
- Prompt encoded once (`make minionerec-replay`, the main run's own batches): 4,861 of 4,866
  samples give the same 50 SIDs, 4,864 the same top 10, the same HR / NDCG at 3, 5, 10.
  `python -m minionerec.replay diff --name once` generates the batches of the differing samples
  again both ways with beam scores (diff.json; `make minionerec` reports it). Batch 8 x 50 beams
  peaks at 14.8 GiB even with the prompt once (the 400-row decoding steps), so the replay and the
  batch-8 cost row need Docker Desktop quit; the worker runs batches of 2 under its 8 GB cap (4
  run out of it).
- The committed `minionerec.md` / `nearline.md`: `make minionerec-replay REPLAY_ARGS="--name
  once-trial --batches 25 --mps-memory-gb 16"`, the cost rows (`minionerec.cost --only
  beam50_batch1_once beam50_batch8_once`), `REPLAY_ARGS="--name once --mps-memory-gb 16"` and
  `replay diff --name once` with Docker Desktop quit; then on a private stack from empty volumes
  `sim.gen_ab --seconds 3600 --concurrency 4` (worker `--max-batch 2`), `make samples
  SAMPLES_FLAGS="--complete-through <traffic end + 30 min>"`, `make nearline-report`, `make
  minionerec`.
