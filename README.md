<div align="center">

# RecLoop

**A recommender and ads serving stack that runs on one laptop: multi-source recall, ranking and a sponsored-slot auction inside a 100 ms budget in Go, an exactly-once event and billing pipeline on Redpanda, and the offline loop that trains, evaluates and releases the models.**

[![Go](https://img.shields.io/badge/Go-1.27-00ADD8?style=flat-square&logo=go&logoColor=white)](serving/go.mod)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](ml/pyproject.toml)
[![Next.js](https://img.shields.io/badge/Next.js-16-000000?style=flat-square&logo=nextdotjs&logoColor=white)](web/package.json)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-18_%2B_pgvector-4169E1?style=flat-square&logo=postgresql&logoColor=white)](deploy/docker-compose.yml)
[![Redis](https://img.shields.io/badge/Redis-8-FF4438?style=flat-square&logo=redis&logoColor=white)](deploy/docker-compose.yml)
[![Redpanda](https://img.shields.io/badge/Redpanda-Kafka_API-E2401B?style=flat-square)](deploy/docker-compose.yml)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-kind_%2B_Helm-326CE5?style=flat-square&logo=kubernetes&logoColor=white)](deploy/helm)
[![CI](https://github.com/heyemlee/recloop/actions/workflows/ci.yml/badge.svg)](https://github.com/heyemlee/recloop/actions/workflows/ci.yml)

[Highlights](#highlights) · [Architecture](#architecture) · [Tech stack](#tech-stack) · [Run it](#run-it) · [Docs](#docs)

</div>

<p align="center">
  <img src="docs/demo.webp" width="960" alt="The RecLoop store. The first items of a user's feed float above the wordmark and the search pill and take turns. Choosing another demo user changes the feed. With Debug on, a panel shows the request's recall sources, fusion weights, model version and experiment groups, and every item its ranks and scores. An item page shows similar items. A question typed into the search pill, ink cartridges for a Canon PIXMA MG3620 under 25 dollars, is answered by the shopping assistant with three cartridges whose compatibility is quoted from their listings" />
</p>

<p align="center"><sub>Office products from Amazon Review Data (2018). Shoppers, advertisers and budgets are simulated; every number below was measured on one laptop. The recording is sped up.</sub></p>

## Highlights

- **A 100 ms budget that degrades instead of failing.** Every request has one deadline that each stage draws from and that travels to the model server over gRPC; circuit breakers and a fallback chain answer with what is left. 600 requests/s meet the SLO (P99 ≤ 100 ms) with the ranker on the request path, and with the model server disconnected every request is still answered ([latency](docs/results/latency.md)).
- **Sponsored clicks billed exactly once.** Slots 3 and 9 of the feed go to a second-price auction over budgets and frequency caps, and each slot carries a signed token with its price. One PostgreSQL transaction charges a click, never above the day's budget. After a 30-minute chaos drill (8 × `kill -9` of billing, 3 broker restarts, forged and duplicate clicks) 487 of 487 charges equal the simulator's truth ([billing](docs/results/billing.md)).
- **An event pipeline that survives duplicates and crashes.** The api writes events to Redpanda; an aggregator (Redis, one Lua script per event), a log sink (PostgreSQL) and billing consume them idempotently and commit offsets after the batch, so each effect is applied once ([ADR-0003](docs/adr/0003-when-to-add-a-message-queue.md)).
- **Recall, fusion and ranking, with intervals.** SASRec (ONNX Runtime + FAISS), ItemCF and popularity are fused by weighted RRF and reordered by a LightGBM LambdaRank ranker: NDCG@10 0.137, against 0.119 for the fusion and 0.122 for SASRec alone, on the published test split with user-level bootstrap intervals ([offline](docs/results/offline.md)).
- **Releases with a canary and a rollback.** A model ships as a versioned bundle with golden queries and goes through validation, a 5 % canary and a full rollout; a rollback switches a pointer and took effect within 0.07 s under traffic. Layered experiments come with an A/A test and a sample-ratio check ([release](docs/results/release.md), [A/B](docs/results/ab.md)).
- **Deployed on Kubernetes.** Helm charts on a local kind cluster: 1 of 67,482 requests failed across three rolling updates of the model server, the api scales on CPU and the consumers on their Redpanda lag (KEDA), and the billing drill passes with pods deleted ([deploy](docs/results/deploy.md)).
- **A generative recommender, off the request path.** MiniOneRec (Qwen2.5-1.5B emitting semantic IDs) reproduces its paper's result on this machine, but takes seconds per request. It runs near-line as a fourth recall source and adds NDCG@10 +0.017 to the fused list offline ([MiniOneRec](docs/results/minionerec.md), [near-line](docs/results/nearline.md)).
- **A shopping assistant that cannot invent a price.** The language model only turns a request into constraints; code checks price, stock, category and device compatibility on the catalog rows and renders the reply. With Claude parsing, every item shown meets the gold constraints of 150 test requests ([agent](docs/results/agent.md)).

## Architecture

<p align="center">
  <img src="docs/architecture.svg" width="900" alt="System architecture of RecLoop. The store (Next.js) and the shopping assistant (FastAPI, Claude) call the Go api. A recommendation request runs five stages inside one 100 ms budget: user context from Redis; recall from SASRec through the gRPC model server, ItemCF, popularity and generated items; weighted RRF fusion and filters; LightGBM ranking through the model server; and a second-price ad auction with budgets and frequency caps. The api reads and writes PostgreSQL with pgvector, a Redis instance for state and one for cache, and publishes events to Redpanda. Stream consumers apply each event once: the aggregator updates the user state, the log sink stores events in PostgreSQL and billing writes the charge ledger. A near-line worker runs MiniOneRec and writes generated items to Redis. Offline, a Spark job joins training samples from the event log, training produces a versioned bundle, and a release goes through validation and a 5 percent canary before the model server serves it. A pulse follows a request, then a click event, then the offline loop" />
</p>

The pulse follows three paths in turn. **Blue, a request:** the api loads the user's context from Redis, recalls candidates from several sources at once (SASRec through the model server, ItemCF, popularity, generated items), fuses and filters them, has the model server rank them, and runs the ad auction, all inside one 100 ms budget. **Teal, an event:** a click goes to Redpanda, where the aggregator updates the user's state, the log sink stores it and billing charges a sponsored click; a changed history also queues the user for the near-line generative recommender. **Violet, the offline loop:** a Spark job joins the event log into training samples, training produces a versioned bundle, and a release moves it through validation and a canary to the model server.

Go serves the request path and Python the models; they meet at one gRPC interface ([ADR-0001](docs/adr/0001-go-online-layer.md)).

| Directory | What it is |
| --- | --- |
| `serving/` | Go: the api, the aggregator and billing |
| `model_server/` | Python gRPC server: Retrieve (SASRec, FAISS) and Rank (LightGBM, pCTR) |
| `ml/` · `minionerec/` | Data pipeline, models, offline evaluation, bundles, the Spark sample job; MiniOneRec |
| `agent/` | The shopping assistant |
| `web/` | The store, the advertiser console and the admin page |
| `sim/` · `loadtest/` · `kube/` | Traffic simulator and chaos drills, k6 load tests, runs on the cluster |
| `deploy/` | Docker Compose, Helm charts and the kind cluster |

## Tech stack

| Layer | Technology | Role |
| --- | --- | --- |
| Online serving | Go 1.27 · gRPC · `context` deadlines · gobreaker | Concurrent recall, fusion, filters and the ad auction under one budget |
| Models | PyTorch · ONNX Runtime · FAISS · LightGBM · scikit-learn | SASRec retrieval, LambdaRank ranking, pCTR; served from versioned bundles |
| Events | Redpanda (Kafka API) · Redpanda Connect · Redis Lua | The event log and its idempotent consumers |
| Storage | PostgreSQL 18 · pgvector · Redis 8 | Catalog, ledger and event store; user state and cache in separate Redis instances |
| Offline | PySpark 4 · bootstrap intervals | Training samples from the log; evaluation on the published split |
| Generative | MiniOneRec · Qwen2.5-1.5B · Apple MPS | Semantic-ID generation with constrained beam search, near-line |
| Assistant | FastAPI · Anthropic SDK · BM25 + pgvector | Constraint parsing, hybrid search, code-checked replies over SSE |
| Web | Next.js 16 · React 19 · Tailwind CSS 4 · shadcn/ui | Store with a debug panel, advertiser console, admin page |
| Observability | Prometheus · Grafana · OpenTelemetry · Jaeger | Per-stage latency, degradation, consumer lag, billing counters |
| Delivery | Docker Compose · kind · Helm · HPA · KEDA · GitHub Actions | The whole stack locally; rolling updates and autoscaling on Kubernetes |
| Testing | `go test` · rapid · pytest · Playwright · k6 · Toxiproxy | Unit, property and browser tests; load tests and chaos drills |

## Run it

Needs Docker, Go, Node and [uv](https://docs.astral.sh/uv/). Built and measured on one Mac (Apple M4 Pro, 24 GB); CI runs the lint and the tests on Linux. MiniOneRec's generation needs Apple Silicon.

```bash
make setup        # the stack, data, catalog, training, evaluation, model bundle, Redis seed
make serve-model  # gRPC model server   :18081
make serve-api    # Go api              :18080
make serve-web    # the store           http://localhost:13000
make serve-agent  # shopping assistant  :18083 (the store's search pill and /chat)
make test         # Python and Go tests; make e2e runs the browser checks
```

Every measurement has a target that writes its report: `make load`, `make billing`, `make release-drill`, `make ab`, `make kind-up` with `make deploy-runs`, and more in the [Makefile](Makefile). Grafana is at http://localhost:13001, Jaeger at http://localhost:16686.

## Docs

- **Design:** [how it works](docs/architecture.md) · [ADRs](docs/adr/) · [full design document with the decision log (Chinese)](docs/design.md)
- **Serving:** [latency and degradation](docs/results/latency.md) · [ANN benchmark](docs/results/ann.md) · [Kubernetes](docs/results/deploy.md)
- **Ads:** [billing and over-delivery](docs/results/billing.md) · [pCTR](docs/results/pctr.md)
- **Models:** [offline evaluation](docs/results/offline.md) · [releases](docs/results/release.md) · [A/B test](docs/results/ab.md) · [MiniOneRec](docs/results/minionerec.md) · [near-line generation](docs/results/nearline.md)
- **Assistant:** [evaluation](docs/results/agent.md)

## Data and license

The data is [Amazon Review Data (2018)](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/), Office Products, with the preprocessing and splits of [MiniOneRec](https://github.com/AkaliKong/MiniOneRec). It is for research use and is not redistributed here; `make data` downloads it and checks every file.

> Jianmo Ni, Jiacheng Li and Julian McAuley. Justifying Recommendations using Distantly-Labeled Reviews and Fine-Grained Aspects. EMNLP 2019.

The code is under the [MIT License](LICENSE). Code adapted from other projects keeps its license notice in [licenses/](licenses/): the SASRec model is adapted from [pmixer/SASRec.pytorch](https://github.com/pmixer/SASRec.pytorch) (Apache-2.0), and MiniOneRec (Apache-2.0) runs from an unmodified clone with [a patch for Apple Silicon](minionerec/evaluate.py.patch), while [minionerec/engine.py](minionerec/engine.py) copies its prefix tree and generation call. Its checkpoint (Apache-2.0, base model Qwen2.5-1.5B-Instruct) is downloaded by `make data-weights` and not redistributed. The text embeddings use [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) (MIT).
