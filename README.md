# RecLoop

RecLoop is a small office-supplies store with personalized recommendations, sponsored
products in the feed and a shopping assistant. The online layer is written in Go, models and
offline work in Python, and it runs on PostgreSQL + pgvector, Redis and Redpanda. User
behavior, advertisers and budgets are all simulated.

**Status: work in progress.** Milestone M0 is done: a reproducible data pipeline, the
product catalog in PostgreSQL, and offline evaluation with Pop and ItemCF baselines
([results](docs/results/offline.md), [ADRs](docs/adr/)). Next up: SASRec, the model
server, the Go API and the web store.

## Run

Needs Docker, Go and [uv](https://docs.astral.sh/uv/).

```
make up       # PostgreSQL + pgvector and Redis (docker compose)
make data     # download, preprocess and verify the dataset
make ingest   # load the catalog into PostgreSQL
make eval     # offline metrics -> docs/results/offline.md
make test
```

## Data

[Amazon Review Data (2018)](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/), Office
Products, with the preprocessing and splits of
[MiniOneRec](https://github.com/AkaliKong/MiniOneRec). The data is for research use and is
not redistributed here; `make data` downloads it and checks every file.

> Jianmo Ni, Jiacheng Li and Julian McAuley. Justifying Recommendations using
> Distantly-Labeled Reviews and Fine-Grained Aspects. EMNLP 2019.
