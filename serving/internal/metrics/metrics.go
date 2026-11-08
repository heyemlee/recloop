// Package metrics defines the Prometheus series of the online services (design §5.2: one
// latency histogram per stage, so a P99 can be broken down by stage). Every process has its
// own registry and serves it on /metrics.
package metrics

import (
	"net/http"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/collectors"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// LatencyBuckets (seconds) are dense around the 100 ms SLO so histogram_quantile stays close to
// the true P99 there.
var LatencyBuckets = []float64{
	.0005, .001, .002, .003, .005, .0075, .01, .015, .02, .025, .03, .035, .04, .05, .06, .07,
	.08, .09, .1, .11, .125, .15, .2, .3, .5, 1, 2, 5,
}

// Registry of one process: the series below plus the Go runtime and process collectors.
var Registry = prometheus.NewRegistry()

func init() {
	Registry.MustRegister(collectors.NewGoCollector(), collectors.NewProcessCollector(collectors.ProcessCollectorOpts{}))
}

func reg[T prometheus.Collector](c T) T {
	Registry.MustRegister(c)
	return c
}

// Handler serves the registry.
func Handler() http.Handler {
	return promhttp.HandlerFor(Registry, promhttp.HandlerOpts{Registry: Registry})
}

// ---------------------------------------------------------------- api

var (
	// HTTPRequests by route pattern and status code.
	HTTPRequests = reg(prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "recloop_http_requests_total", Help: "HTTP requests by route and status code.",
	}, []string{"route", "code"}))
	// HTTPDuration by route pattern, measured in the handler.
	HTTPDuration = reg(prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name: "recloop_http_request_duration_seconds", Help: "HTTP request latency by route.",
		Buckets: LatencyBuckets,
	}, []string{"route"}))

	// StageDuration of the recommendation pipeline: user_context, cache_get, recall, rank,
	// filter, page_store, render and total.
	StageDuration = reg(prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name: "recloop_stage_duration_seconds", Help: "Recommendation latency by pipeline stage.",
		Buckets: LatencyBuckets,
	}, []string{"surface", "stage"}))
	// RecallDuration per source and outcome (ok, timeout, error, circuit_open, skipped).
	RecallDuration = reg(prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name: "recloop_recall_duration_seconds", Help: "Recall latency by source and outcome.",
		Buckets: LatencyBuckets,
	}, []string{"source", "outcome"}))
	// Recommendations served, by surface and whether the response was degraded.
	Recommendations = reg(prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "recloop_recommendations_total", Help: "Recommendation responses by surface and degradation.",
	}, []string{"surface", "degraded"}))
	// Degraded responses by reason (one response can have several).
	Degraded = reg(prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "recloop_degraded_total", Help: "Degradation reasons written into responses.",
	}, []string{"surface", "reason"}))
	// CandidateCache lookups: hit, miss, error, skip (cache off or no user state).
	CandidateCache = reg(prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "recloop_candidate_cache_total", Help: "Candidate cache lookups by result.",
	}, []string{"surface", "result"}))
	// BreakerState: 0 closed, 1 half-open, 2 open.
	BreakerState = reg(prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "recloop_breaker_state", Help: "Circuit breaker state: 0 closed, 1 half-open, 2 open.",
	}, []string{"dependency"}))
	BreakerTransitions = reg(prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "recloop_breaker_transitions_total", Help: "Circuit breaker transitions by target state.",
	}, []string{"dependency", "to"}))

	// EventsPublished by the api to the event log: type and result (ok, error).
	EventsPublished = reg(prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "recloop_events_published_total", Help: "Events written to the event log by type and result.",
	}, []string{"type", "result"}))
	// ServedEvents: produced, retried (one failed attempt), dropped (buffer full) or failed
	// (the log did not take them within the writer's retry time).
	ServedEvents = reg(prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "recloop_served_events_total", Help: "Served events by result.",
	}, []string{"result"}))
)

// ---------------------------------------------------------------- consumers

var (
	// ConsumerRecords handled, by group and result (applied, duplicate, ignored, invalid).
	ConsumerRecords = reg(prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "recloop_consumer_records_total", Help: "Records handled by consumer group and result.",
	}, []string{"group", "result"}))
	// ConsumerBatch duration (handle + commit) by group.
	ConsumerBatch = reg(prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name: "recloop_consumer_batch_duration_seconds", Help: "Time to apply and commit one polled batch.",
		Buckets: LatencyBuckets,
	}, []string{"group"}))
	// ConsumerRetries of a batch after a failed attempt.
	ConsumerRetries = reg(prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "recloop_consumer_retries_total", Help: "Batch retries after a failed attempt.",
	}, []string{"group"}))
	// EventToState: event time (UUIDv7) to the moment its effect is in Redis.
	EventToState = reg(prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name: "recloop_event_to_state_seconds", Help: "Event time to applied in Redis (aggregator).",
		Buckets: []float64{.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10, 30, 60, 300},
	}, []string{"group"}))
	// ConsumerLag (records) per group, summed over the topic's partitions.
	ConsumerLag = reg(prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "recloop_consumer_lag", Help: "Records not yet committed by a consumer group.",
	}, []string{"group", "topic"}))
	// FreqCapTokens: ad_tokens of sponsored impressions the aggregator counted towards a
	// frequency cap, or rejected (did not verify or named another impression).
	FreqCapTokens = reg(prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "recloop_freq_cap_tokens_total", Help: "Sponsored impressions counted towards a frequency cap, or rejected.",
	}, []string{"result"}))
)

// ---------------------------------------------------------------- ads and billing

var (
	// AdRequests by surface and outcome: filled (at least one sponsored slot), empty (no
	// eligible ad), not_allowed (the degradation chain rules ads out), stale (no fresh campaign
	// snapshot), timeout / error / skipped (the ads stage's Redis read).
	AdRequests = reg(prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "recloop_ad_requests_total", Help: "Ad requests by surface and outcome.",
	}, []string{"surface", "outcome"}))
	// AdSlots served, by surface and position.
	AdSlots = reg(prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "recloop_ad_slots_total", Help: "Sponsored slots served by surface and position.",
	}, []string{"surface", "position"}))
	// AdGates: campaigns held back by a serving gate (budget, freq_cap, pacing).
	AdGates = reg(prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "recloop_ad_gate_total", Help: "Campaigns held back on a request, by gate.",
	}, []string{"gate"}))

	// BillingClicks by outcome (charged, duplicate_impression, duplicate_event, expired,
	// mismatch, invalid, unknown_campaign).
	BillingClicks = reg(prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "recloop_billing_clicks_total", Help: "Sponsored clicks billing processed, by outcome.",
	}, []string{"outcome"}))
	// BillingCents of new charges: billed, or unbilled (price - billed: the budget had run out,
	// the overdelivery the platform absorbs, §9.4).
	BillingCents = reg(prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "recloop_billing_cents_total", Help: "Cents of new charges: billed, or unbilled because the budget ran out.",
	}, []string{"kind"}))
	// BillingHandle: one click's processing time (verification and the charge transaction).
	BillingHandle = reg(prometheus.NewHistogram(prometheus.HistogramOpts{
		Name: "recloop_billing_handle_seconds", Help: "Time to process one sponsored click.",
		Buckets: []float64{.001, .002, .005, .01, .025, .05, .1, .25, .5, 1, 2, 5, 10, 30},
	}))

	// ReconcileRuns by result (ok, error).
	ReconcileRuns = reg(prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "recloop_reconcile_runs_total", Help: "Reconciliation passes by result.",
	}, []string{"result"}))
	// ReconcileDays: campaign-days of the last pass by state: match, redis_behind (repaired),
	// redis_ahead (must not happen), row_mismatch (budget row != ledger: must not happen).
	ReconcileDays = reg(prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "recloop_reconcile_campaign_days", Help: "Campaign-days of the last reconciliation pass by state.",
	}, []string{"state"}))
	// ReconcileRepairedCents: spend the reconciler added to Redis counters that had lost it.
	ReconcileRepairedCents = reg(prometheus.NewCounter(prometheus.CounterOpts{
		Name: "recloop_reconcile_repaired_cents_total", Help: "Cents added to Redis spend counters by reconciliation.",
	}))
	// SpendCents of today (UTC) in the last pass: ledger (ad_charges) and redis.
	SpendCents = reg(prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "recloop_spend_cents", Help: "Today's ad spend by source, at the last reconciliation pass.",
	}, []string{"source"}))
	// ReconcileLastSuccess: unix time of the last pass without an error.
	ReconcileLastSuccess = reg(prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "recloop_reconcile_last_success_timestamp_seconds", Help: "Unix time of the last successful reconciliation pass.",
	}))
)
