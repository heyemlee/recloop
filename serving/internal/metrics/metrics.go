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
)
