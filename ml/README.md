# ml

Offline side of RecLoop: dataset download and preprocessing, catalog export, database ingest,
models (Pop, ItemCF, SASRec), model bundles and the offline evaluation harness. Run everything
from the repo root through the Makefile (`make data`, `make ingest`, `make train`, `make eval`,
`make bundle`, `make test`); see `CLAUDE.md`.

`ann/` is the vector-index benchmark (Flat / IVF / HNSW / IVF-PQ on synthetic 64-d catalogs of
up to 1M items, and incremental index updates): `make ann` writes `docs/results/ann.md`.
