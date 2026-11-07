# ADR-0003: When to add a message queue — the event log in M2

- Status: accepted
- Date: 2026-11-06
- Design: `docs/design.md` §7.2, §7.3, §12.3; results: `docs/results/latency.md`

## Context

In M1, `POST /v1/events` did all the work synchronously: insert into PostgreSQL (`events_raw`,
the order and the stock change of a purchase, one transaction), then apply the event to the
user's state in Redis, then answer. That was the right first version. There was one consumer of
the events (the Redis state), the client needed nothing back but "stored", nothing had to be
replayed, and every moving part we did not run was one we did not have to make correct.

M2 changes the requirements:

1. **Several consumers with different needs.** The aggregator keeps per-user state in Redis
   within milliseconds; the log-sink writes every raw event to PostgreSQL, best in large batches;
   billing (M3) charges sponsored clicks in its own ledger transactions. They fail and slow down
   independently.
2. **The intake must not depend on the slowest consumer.** Under load PostgreSQL is the slowest
   sink (wide JSON payloads, one row per impression). With the synchronous path every event
   request waits for it, and a PostgreSQL hiccup becomes failed `POST /v1/events` calls and
   lost served logs on the recommendation path.
3. **Replay.** Redis holds derived state only: when it is lost or its logic changes, it has to be
   rebuilt from the events. Later consumers (training-sample joins in M4, billing reconciliation)
   need to read history too.
4. **Order per user.** History is ordered by event time per user; consumers should see one user's
   events in order without a global order.

## Decision

Put an event log between the api and its consumers: Redpanda (Kafka API), topic `events`,
key = `user_id`, 6 partitions, 7 days of retention.

- The api validates a batch, writes it to the log (idempotent producer, acks from all in-sync
  replicas) and answers `202` once the log has it. It no longer touches PostgreSQL or Redis for
  events. Served events go to the same topic from a bounded in-process buffer that never blocks
  a request.
- Consumers, one group each: **aggregator** (Go) → Redis; **log-sink** (Redpanda Connect, only
  configuration) → `events_raw`; **billing** (M3) → the ledger. The record format
  (`serving/internal/events.Record`, header `type`) carries `impression_id` and `ad_token`, so
  billing can filter sponsored clicks without decoding every record.
- **At-least-once delivery + idempotent consumers = each effect exactly once.** Every consumer
  deduplicates by `event_id` in the same atomic step as its side effect (a Lua script that sets
  `dedup:aggregator:{event_id}` and updates the state; `ON CONFLICT (event_id, event_day) DO
  NOTHING` in PostgreSQL, where a trigger creates a purchase's order in the same transaction as
  its row) and commits offsets only after the effect. Kafka transactions are not used: they cover
  Kafka-to-Kafka, not writes to PostgreSQL or Redis.
- Derived state is rebuildable: `make rebuild-redis` = seed + replay of `events_raw` (everything
  ever logged), or `FROM=log` = seed + the consumer group reset to the start of the log.

## When a queue is worth it

The criteria we applied, in order of weight:

| Signal | M1 | M2 |
|---|---|---|
| More than one consumer of the same events, failing and lagging independently | no (one) | yes (three) |
| The producer must keep accepting while a consumer is slow or down | no | yes |
| Events must be replayable (rebuild state, backfill a new consumer) | no | yes |
| Bursts should be absorbed and turned into batches downstream | no | yes (log-sink batches of 500) |
| Per-key order is needed, global order is not | – | yes (per user) |
| The client needs the consumer's result in the response | – | no (`202` is enough) |

Not worth it yet when there is one consumer, the caller needs the result synchronously, volume
is low and nothing is replayed: a direct write (or the database's own log and an outbox) is
simpler and gives read-your-writes for free.

## Consequences

- **Eventual consistency.** A click is in Redis a few milliseconds after the `202`, not before
  it. The next recommendation request can, rarely, miss it; it shows up in the one after. The
  aggregator exports the event-time-to-applied delay (`recloop_event_to_state_seconds`).
- **Duplicates are normal**, so every consumer must be idempotent. The drill (`make drill`) runs
  the simulator with duplicate deliveries, whole-batch resends, reordering, late events and
  double clicks while it kills the aggregator and the log-sink with SIGKILL and restarts the
  broker: in one 60 s run, 83,203 unique events delivered 95,028 times were all in `events_raw`
  exactly once and applied to Redis exactly once (hv, history, purchases, dedup markers), with
  one order per purchase.
- **Restarts of a consumer** use static group membership, so a crashed consumer takes its old
  place at once instead of waiting for its session to time out.
- **A new stateful service** to run and watch: consumer lag (`recloop_consumer_lag`), retention,
  partitions. The dev broker is a single node in dev-container mode, which skips fsync; its
  durability is not production grade (a SIGKILL of the broker could lose acknowledged writes),
  so the drill restarts it gracefully.
- **Served logging is best-effort under a broker outage**: the buffer retries a failed batch for
  30 s and drops (counted) only when full; the recommendation path never waits.

## Alternatives considered

- **Keep the synchronous path, add an outbox table and change data capture.** PostgreSQL stays
  on the intake path of every event; the outbox is the queue, with more parts.
- **Redis Streams.** Already running, but Redis is our disposable derived-state store (no
  persistence); keeping 7 days of events in memory for replay does not fit it.
- **PostgreSQL as the queue (`SKIP LOCKED`).** Fine at low volume, but it couples intake to the
  slowest consumer's database and makes fan-out and replay awkward.
- **Apache Kafka.** The same API and semantics; Redpanda is one binary with no separate
  controller to run, which suits one laptop. The clients (franz-go, Redpanda Connect) would work
  against Kafka unchanged.
