# RecLoop

RecLoop is a small office-supplies store with personalized recommendations, sponsored
products in the feed and a shopping assistant. The online layer is written in Go, models and
offline work in Python, and it runs on PostgreSQL + pgvector, Redis and Redpanda. User
behavior, advertisers and budgets are all simulated.

**Status: work in progress.** Milestones M0 and M1 are done:

- a reproducible data pipeline, the catalog in PostgreSQL, and offline evaluation of Pop,
  ItemCF and SASRec with bootstrap intervals and a comparison with the published SASRec
  numbers ([results](docs/results/offline.md), [ADRs](docs/adr/));
- the online loop: a Python gRPC model server (SASRec user encoder in ONNX Runtime + FAISS,
  loaded from a versioned bundle), a Go API (concurrent multi-source recall, weighted RRF,
  filters, brand diversity, paging, layered experiment assignment, served-event logging,
  idempotent event ingestion into PostgreSQL and Redis) and a Next.js store with a debug panel.
  A click changes the next page's recommendations.

Ahead of M4, an ANN benchmark compares exact search with IVF, HNSW and IVF-PQ on synthetic
catalogs of up to 1M items, and studies incremental index updates against full rebuilds
([results](docs/results/ann.md); the latencies are preliminary until a run on an idle machine).

Next up: latency budgets, degradation and an event log with Redpanda (M2).

## Run

Needs Docker, Go, Node and [uv](https://docs.astral.sh/uv/).

```
make setup        # PostgreSQL + Redis, data, catalog, SASRec training, eval, bundle, Redis seed
make serve-model  # gRPC model server   :18081
make serve-api    # Go API              :18080
make serve-web    # web store           http://localhost:13000
make test         # Python and Go tests; make e2e runs the browser check
```

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
