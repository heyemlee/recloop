# RecLoop: how it works

RecLoop is a small store of 3,459 office products (Amazon Review Data 2018) with a personalized
feed, sponsored products in that feed and a shopping assistant, built to run on one laptop.
Shoppers, advertisers and budgets are simulated. This page describes the parts and why they are
built the way they are; the measurements are in [`docs/results/`](results/), the larger decisions
with their alternatives in [`docs/adr/`](adr/), and the commands in the [README](../README.md).

## 1. Parts

| Part | Language | What it does |
|---|---|---|
| web (`web/`) | Next.js, shadcn/ui | the store with a debug panel, the advertiser console, an internal admin page, the chat page |
| api (`serving/cmd/api`) | Go | recommendations, event ingestion, advertisers and campaigns, the sponsored-item endpoint for the assistant, admin endpoints for experiments and releases |
| model server (`model_server/`) | Python, gRPC | `Retrieve`: the SASRec user vector (ONNX Runtime) and FAISS search; `Rank`: ranking features, LightGBM scores and the pCTR of ad candidates; several model versions loaded at once |
| aggregator (`serving/cmd/aggregator`) | Go | events from Redpanda into the user state in Redis (history, purchases, items seen, frequency caps, item click counts) |
| log-sink (`deploy/connect/`) | Redpanda Connect | events from Redpanda into PostgreSQL (`events_raw`); a trigger turns purchases into orders |
| billing (`serving/cmd/billing`) | Go | sponsored clicks into the charge ledger and the spend counters; reconciliation every minute |
| shopping assistant (`agent/`) | Python, FastAPI | one turn per message: constraints, search, checks, personalisation, at most one sponsored item |
| offline (`ml/`, `ml/samples/`, `sim/`) | Python, PySpark | data pipeline, models and their evaluation, model bundles, training samples joined from the event log, the traffic simulator |

PostgreSQL 18 with pgvector holds the catalog (with text embeddings), events, orders,
experiments, model versions, campaigns and the ledger. Redis runs as two instances (section 8).
Redpanda carries the event log. Prometheus, Grafana and OpenTelemetry traces (Jaeger) cover every
stage. Everything runs in Docker Compose for development and measurements, and on a local
Kubernetes cluster for the deployment study (section 10).

Go serves the request path and Python the models
([ADR-0001](adr/0001-go-online-layer.md)): the online layer is mostly concurrency, deadlines
and I/O, the models need the Python ecosystem, and the two meet at one gRPC interface
(`proto/recloop/modelserver/v1/model_server.proto`).

## 2. Data and offline evaluation

The dataset is the Office Products category with MiniOneRec's preprocessing and split
([ADR-0002](adr/0002-dataset.md)): 8,328 users, 56,984 interactions, chronological 8:1:1 samples
with histories of up to 10 items. Using the published split makes the numbers comparable with
SASRec's and MiniOneRec's papers. `make data` re-runs the preprocessing and checks that its output
is byte-identical to the published files.

Every model ranks all 3,459 items for each of the 4,866 test samples and is scored once on the
test split, after fitting on train rows and tuning on validation rows. Metrics are HR and NDCG at
3, 5 and 10, with 95 % intervals from a user-level bootstrap, and every model is also reported on
two segments: targets reviewed on the same day as the last history item, and later ones. The
models: popularity, item-based CF, SASRec (adapted from SASRec.pytorch), the api's weighted
reciprocal rank fusion (RRF) of the three, and a LightGBM LambdaRank ranker over the fused
candidates ([offline.md](results/offline.md)).

## 3. A recommendation request

`GET /v1/recommendations?user_id&surface=home` runs these steps:

1. **User context** from the state Redis: recent history, a history version that changes with
   every click, cart or purchase, purchased items and items seen in the last 24 hours.
2. **Candidate cache** keyed by user, history version, model version and experiment variants: a
   hit skips recall and ranking.
3. **Recall** from three sources in parallel, each with its own timeout: SASRec (`Retrieve`),
   ItemCF and popularity. Item pages use ItemCF and text similarity (pgvector) instead. A source
   that fails or times out is left out and the others are merged.
4. **Fusion**: weighted RRF (k = 60) over the sources, 300 candidates.
5. **Ranking**: `Rank` on the model server computes 10 features per candidate (each source's rank
   and score, popularity, price bucket, category overlap with the history, ...) and the LightGBM
   score. The rank-layer experiment decides whether the list is ordered by that score or keeps
   RRF's order; the features are logged either way.
6. **Filters**: purchased items, items seen today, sold-out items and the current item go; at
   most 2 items of one brand in the top 10; popular items backfill a short list.
7. **Sponsored slots** at positions 3 and 9 (section 7).
8. The first 200 items are stored as the request's page (10 minutes) for the next pages, and the
   response is logged as a `served` event: items, positions, recall sources, scores, features,
   model version and experiment groups. Training samples are later joined against this log.

## 4. Latency budget and degradation

Each request has a 100 ms budget. Every stage gets the smaller of its own share and what is left
after the later stages' shares (user context 5 ms, candidate cache 3, recall 35, ranking 25, ads
10, the rest 10), and the deadline travels with the gRPC calls. A failed or slow dependency
degrades the answer instead of failing it: the response lists why in `degraded`
(`sasrec_timeout`, `rank_circuit_open`, `static_popular`, ...), and a degraded list is never
cached. Circuit breakers sit in front of the model server's two methods and PostgreSQL's text
recall; Redis has none, because its calls are bounded by their budgets and fail fast.

The load tests run the api and the model server in containers limited to 2 CPUs each, with k6
in an open loop ([latency.md](results/latency.md)). Every run met the SLO (P99 ≤ 100 ms, at most
1 % degraded) up to 600 requests/s. With the model server disconnected the api still answers
every request (degraded), where a naive version without budgets and breakers fails all of them.

Known limitation: a run that starts with an empty candidate cache at 1,000 requests/s or more never
fills it. Every miss calls the model server twice, those calls queue past their deadlines, the
breakers keep reopening, and degraded lists are not cached. With the cache filled first, 1,000/s
met the SLO in 2 of 3 runs. Admission control in front of the model server (no slot, no wait)
would break that cycle; it is left for after v1.

## 5. Events

The api validates events (`POST /v1/events`) and writes them to a Redpanda topic (6 partitions,
keyed by user) and answers 202, so a slow database never slows the store down
([ADR-0003](adr/0003-when-to-add-a-message-queue.md)). Two consumer groups read it:

- the **aggregator** applies each event to Redis in one Lua script that checks and sets the
  event's dedup marker in the same step, and commits offsets only after the batch;
- the **log-sink** inserts into PostgreSQL with `ON CONFLICT DO NOTHING` on the event id; a
  trigger creates the order of a purchase and takes the item's stock.

Both are idempotent, so a redelivered or retried event changes nothing. A drill with duplicate
deliveries, `kill -9` of both consumers and a broker restart under simulated traffic ends with
every event applied exactly once in both stores (`make drill`).

## 6. Experiments

Users are assigned per layer: `bucket = xxhash64(salt + ":" + user_id) % 1000`, with one salt per
layer (recall, rank, ads), so experiments in different layers are independent. The configuration
lives in PostgreSQL and is reloaded every 10 seconds, and every served event carries the user's
groups. A Python twin of the bucketing (checked against shared test vectors) recomputes the
assignment offline, and a sample ratio mismatch check runs on the logged traffic before any
comparison.

## 7. Sponsored products and billing

Advertisers create campaigns through the api (budgets and bids in integer cents, days in UTC,
updates guarded by an ETag). For each home request, the campaigns in their flight window pass
the gates: items in stock, not bought, not already in the organic list; budget left above a
safety margin of 5 clicks at the bid; frequency below the cap. Spend and frequency counters come
from the state Redis in one call, and a counter that Redis lost can never make a spent campaign
look new (the larger of the counter and the last budget row counts).

The auction ranks by eCPM = bid × pCTR (pCTR in parts per million, so the ranking is exact
integer arithmetic) and charges the generalized second price: the next eCPM divided by the
winner's pCTR, plus a cent, between the reserve and the winner's bid. Each sponsored slot carries
a signed `ad_token` (JWT) with its price, campaign, item, user and position; the store sends it
with the slot's impression and click.

Billing consumes clicks with a token and charges each in one PostgreSQL transaction: claim the
click's event id (technical dedup), lock the campaign's budget row for the day the impression
was served, charge min(price, budget left), insert the charge keyed by impression id (business
dedup), add to the day's spend. The Redis spend counter is updated after the commit. A
reconciliation every minute compares the ledger, the budget rows and Redis; it repairs a counter
that is behind and reports one that is ahead. Property tests drive a sequential model with
duplicates, forged and late clicks, budget changes and crashes at every step against the real
database.

After 30 minutes of simulated traffic with duplicate and forged clicks, 8 × `kill -9` of billing,
3 broker restarts and a slowed-down database, the ledger equals the simulator's truth charge by
charge. The over-delivery study measures what each safety margin costs in unspent budget
([billing.md](results/billing.md)). Pacing exists as an option and is off by default.

## 8. Redis: state and cache

Redis runs as two instances. The **state** instance holds user state, dedup markers, frequency
caps and spend counters; it never evicts (`noeviction`, 1 GB), because a lost dedup marker would
apply an event twice and a lost frequency counter would serve an ad past its cap. When it is full
it refuses writes: the aggregator retries its batch without losing events, billing keeps charging
in PostgreSQL, the feed serves no ads, and Grafana raises an alert. The **cache** instance holds
result pages and cached candidates and evicts the least recently used keys (384 MB). Both are
derived data, rebuilt from the event log (`make rebuild-redis`).

## 9. Ranking, pCTR and model releases

The ranker's training data replays the api's home feed offline on the validation split (the same
sources, fusion and candidate count), so the ranker sees the candidates it will rank online.
LightGBM LambdaRank is tuned by cross-validation over users, exported to ONNX and checked against
LightGBM on every validation candidate. Click-rate features from the logs are not used for it:
the dataset has no impressions, and the simulator's clicks follow the test targets, so they would
leak.

The sponsored slots' pCTR is a logistic regression trained on samples that a PySpark job joins
from the event log: each impression of a served item, labelled by a click within 30 minutes,
with the features the served event logged. It also uses the item's click rate over the last hour
and day, counted once per event by the aggregator, kept in a table the api rebuilds every five
minutes and logged with each impression; the Spark job's recount equals the logged counts for
every impression of a check run ([pctr.md](results/pctr.md)).

A model version is a bundle directory: the SASRec encoder (ONNX), the FAISS index, the ItemCF and
popularity tables, the feature tables, the ranker and the pCTR, with checksums and golden queries.
A version moves through built → validated → canary → active → retired, each step recorded in
PostgreSQL:

- **validate**: the api and the model server load it next to the active one and replay its
  golden data;
- **canary**: the rank-layer experiment pins a share of users to it (5 % in the drill);
- **activate**: one transaction retires the previous version, and the api and the model server
  switch;
- **rollback**: back to the version the active one replaced, within 30 minutes.

Every api instance reconciles with that table every 2 seconds. Under simulated traffic each
switch took effect within 0.07 s ([release.md](results/release.md)). The A/A test passes; the
ranker's CTR difference against RRF on simulated traffic is not significant, since the
simulator clicks by text relevance ([ab.md](results/ab.md)).

## 10. Deployment

Docker Compose runs everything for development; its bench profile runs the api and the model
server in CPU-limited containers for the load tests. The deployment study runs on a one-node
kind cluster: the infrastructure from official charts (CloudNativePG, Redpanda, KEDA,
metrics-server, kube-prometheus-stack) and Helm charts for RecLoop's services, with images built
locally and loaded into the node.

- The model server's readiness waits until its bundle has loaded and passed its golden checks,
  so a broken bundle never takes traffic.
- The api scales on CPU through a HorizontalPodAutoscaler, billing on its consumer lag through
  KEDA.
- The consumers are StatefulSets: their static group-member ids survive a restart.

Three rolling updates of the model server under load failed 1 of 67,482 requests, and the
billing drill with billing's pods deleted leaves the ledger equal to the truth
([deploy.md](results/deploy.md)).

## 11. Shopping assistant

A turn turns the message into a typed request (category, budget, brands, genuine parts only, a
device, a reference to the last purchase) with Claude or with rules. Then:

1. search: BM25 and vector search fused by RRF, with SQL filters for price, stock and category;
2. code checks of every constraint on the catalog rows (an unknown price never meets a budget);
3. printer models confirmed from quoted text in the item's fields;
4. the order personalized through the recommendation api;
5. at most one sponsored item from the same auction, labelled by the rendering code.

The language model never picks or orders items; the reply is rendered from the item rows. The
evaluation runs 150 generated requests with gold constraints and purchase histories, comparing
the rules parser, the gold constraints and the Claude models, with a Claude judge for relevance
([agent.md](results/agent.md)).

## 12. Generative retrieval: MiniOneRec

MiniOneRec fine-tunes Qwen2.5-1.5B to emit an item's semantic ID and decodes with beam search
constrained to the catalog. Its published Office checkpoint was re-run with its own evaluation
code on this laptop's GPU (Apple MPS): the result reproduces the paper within our intervals, and
on the same samples it ties SASRec while a 50-beam request costs about 3 seconds
([minionerec.md](results/minionerec.md)). Encoding each prompt once and sharing its attention
cache across the beams brings a request to 0.64 s with the same results (the same top 10 for
4,864 of the 4,866 test samples, the same HR and NDCG). That is still far over the feed's
budget, so it runs near-line. When a user's history changes, the event's Lua script also queues
the user in Redis (only while a worker is registered). A worker generates once the history has
been quiet for 30 s, at most 120 s after the first change, and writes the top 50 items to the
state Redis. The feed reads them in the same round trip as the rest of the user context and
fuses them as a fourth RRF source, only in the experiment arms that give it a weight; the
result's history version is part of the candidate cache key, so the next request uses a new
result. Offline the fourth source raises the fused list's NDCG@10 by +0.0174 [+0.0140,
+0.0209]. In a 60-minute simulated A/B a result was written 31 s (P50) after a change and the
feed's latency did not move ([nearline.md](results/nearline.md)).

## 13. Not in v1

- Real accounts, payment and shipping.
- Search ranking beyond what the assistant needs.
- Multi-node or managed Kubernetes.
- Hedged requests, an outbox, Bloom filters.
- The admission control of section 4.
- A ranker trained with the generative source's features: the near-line A/B orders both arms
  by RRF.
