# ADR-0001: Go for the online layer

- Status: accepted
- Date: 2026-11-06
- Design: `docs/design.md` §0, §3, §5

## Context

RecLoop's hot path is `GET /v1/recommendations`: read user context from Redis, fan out to three
or four retrieval sources concurrently (in-process Pop and ItemCF, SASRec + FAISS in a Python
model server, near-line MiniOneRec results in Redis), merge, filter, rank, run the ad auction and
respond within a 100 ms deadline, degrading instead of waiting when a dependency is slow (§5).
Next to it run event consumers (aggregator, billing) whose counts and charges must stay correct
under redelivery, reordering and crashes (§9).

The models (SASRec, LightGBM, pCTR, MiniOneRec) need the Python ecosystem: PyTorch, FAISS,
LightGBM, transformers.

## Decision

- Write the online services in Go: `api` (recommendations, events, auction, experiment
  assignment, degradation, advertiser and admin APIs), `aggregator` and `billing`.
- Keep models in Python behind a gRPC model server. The Go client's deadline propagates to the
  server, which checks the remaining time before each step and stops early when the caller has
  already given up.
- `log-sink` is Redpanda Connect configuration, not code.
- Checkpoint: by the end of M1's first week the mixer skeleton (fan-out + per-source timeouts +
  partial merge) must work. If it does not, switch the online layer to FastAPI + asyncio with
  the same interfaces and design.

## Why Go

- Cancellation and deadlines are part of the standard library: a `context` deadline set at the
  edge reaches every Redis call, gRPC call and goroutine, and `errgroup` gives a bounded
  concurrent fan-out with partial results in a few lines. This maps directly onto the stage
  budgets of §4.1.
- Cheap goroutines and a low-pause GC keep tail latency predictable under open-loop load
  without a thread pool or event-loop tuning.
- Small static binaries with millisecond start-up and a small memory footprint, which matters
  with per-container CPU and memory limits on one 8 GB Docker VM and for rolling restarts on
  Kubernetes (§23).
- Mature clients for everything on the path: grpc-go, go-redis, pgx, franz-go (Kafka API),
  OpenTelemetry and Prometheus.

## Alternatives considered

- **FastAPI + asyncio for everything online**: one language, but deadline propagation and
  bounded fan-out need more care, CPU-bound merge/rank/auction work competes with the event loop
  (or needs a process pool), and per-request overhead is higher. Kept as the fallback above.
- **Java (Spring Boot)**: comparable concurrency tooling, but JIT warm-up and a larger memory
  footprint per container on a small VM, and more framework surface than these services need.
- **Model inference inside the Go process**: avoids a network hop, but means binding PyTorch and
  FAISS from Go and loses the clean model-version boundary of the model server (§6).

## Consequences

- Two languages and a gRPC contract (`proto/`) to keep in sync. The experiment-assignment hash
  must produce identical buckets in Go and Python, checked with shared test vectors (§8).
- The gRPC hop to the model server is on the hot path; it is budgeted (§4.1) and has a circuit
  breaker and fallbacks that stay inside the Go process (§5.1).
- Already in M0, the database schema is owned on the Go side: goose migrations in
  `serving/migrations`, run with `go tool goose` (module `github.com/heyemlee/recloop/serving`).
