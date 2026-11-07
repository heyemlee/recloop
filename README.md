# RecLoop

[![CI](https://github.com/heyemlee/recloop/actions/workflows/ci.yml/badge.svg)](https://github.com/heyemlee/recloop/actions/workflows/ci.yml)

RecLoop is a small office-supplies store with personalized recommendations, sponsored
products in the feed and a shopping assistant. The online layer is written in Go, models and
offline work in Python, and it runs on PostgreSQL + pgvector, Redis and Redpanda. User
behavior, advertisers and budgets are all simulated.

![The store with its debug panel: two users get different feeds, and a click on a binder changes the next feed](docs/demo.gif)

The store with the debug panel on: every request shows its recall sources, fusion scores,
model version and experiment groups. Two dataset users get different feeds; after a click on a
binder, the next feed includes items recalled via that binder (outlined in the GIF).

**Status: work in progress.** Milestones M0, M1 and M2 are done:

- a reproducible data pipeline, the catalog in PostgreSQL, and offline evaluation of Pop,
  ItemCF and SASRec with bootstrap intervals and a comparison with the published SASRec
  numbers ([results](docs/results/offline.md), [ADRs](docs/adr/));
- the online loop: a Python gRPC model server (SASRec user encoder in ONNX Runtime + FAISS,
  loaded from a versioned bundle), a Go API (concurrent multi-source recall, weighted RRF,
  filters, brand diversity, paging, layered experiment assignment, served-event logging,
  idempotent event ingestion into PostgreSQL and Redis) and a Next.js store with a debug panel.
  A click changes the next page's recommendations.
- latency and events: every recommendation request has a 100 ms budget that each stage draws
  from (the gRPC deadline travels to the model server), a degradation chain that answers with
  what is left instead of failing, a circuit breaker in front of the model server and a
  candidate cache keyed by the user's history version. Events go through a Redpanda log to an
  aggregator (Redis state) and a Redpanda Connect sink (PostgreSQL), each idempotent; a drill
  with duplicate deliveries, `kill -9` of the consumers and a broker restart ends with every
  event applied exactly once. A traffic simulator drives the load tests and an SRM check of the
  experiment split. Prometheus, Grafana and OpenTelemetry traces (Jaeger) show it per stage.
  On a laptop (8-vCPU Docker VM, api and model server limited to 2 CPUs each) every run met
  the SLO (P99 ≤ 100 ms, at most 1 % degraded) up to 1,200 requests/s, and the median P99 stays
  under 10 ms up to 1,500/s, the top of the test ladder; with the model server disconnected the
  API still answers every request (degraded), where the naive version fails 83 % of them
  ([results](docs/results/latency.md),
  [ADR-0003](docs/adr/0003-when-to-add-a-message-queue.md)).

Ahead of M4, an ANN benchmark compares exact search with IVF, HNSW and IVF-PQ on synthetic
catalogs of up to 1M items, and studies incremental index updates against full rebuilds. Exact
search is enough until it needs about 1 ms of CPU per query (about 250K items); at 1M items
IVF-Flat reaches recall@100 0.995 at a P99 of 152 µs ([results](docs/results/ann.md)).

Next up: sponsored items, the auction and correct billing (M3).

## Run

Needs Docker, Go, Node and [uv](https://docs.astral.sh/uv/).

```
make setup        # the stack (PostgreSQL, Redis, Redpanda, consumers, monitoring), data, catalog,
                  # SASRec training, eval, bundle, Redis seed
make serve-model  # gRPC model server   :18081
make serve-api    # Go API              :18080
make serve-web    # web store           http://localhost:13000
make test         # Python and Go tests; make e2e runs the browser check
make drill        # duplicates + consumer crashes under simulated traffic, then a consistency check
make load         # latency experiments on CPU-limited containers -> docs/results/latency.md
```

Grafana (dashboard "RecLoop") is at http://localhost:13001, Jaeger at http://localhost:16686,
Prometheus at http://localhost:19090.

## Data

[Amazon Review Data (2018)](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/), Office
Products, with the preprocessing and splits of
[MiniOneRec](https://github.com/AkaliKong/MiniOneRec). The data is for research use and is
not redistributed here; `make data` downloads it and checks every file.

> Jianmo Ni, Jiacheng Li and Julian McAuley. Justifying Recommendations using
> Distantly-Labeled Reviews and Fine-Grained Aspects. EMNLP 2019.

## License

The code is under the [MIT License](LICENSE). Code adapted from other projects keeps its
original license notice: the SASRec model is adapted from
[pmixer/SASRec.pytorch](https://github.com/pmixer/SASRec.pytorch) (Apache-2.0, see
[licenses/](licenses/)). The text embeddings use
[BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) (MIT). The dataset is
not part of this repository and has its own terms.
