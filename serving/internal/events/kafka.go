package events

import (
	"context"
	"strconv"

	"github.com/twmb/franz-go/pkg/kgo"

	"github.com/heyemlee/recloop/serving/internal/metrics"
)

// Topic is the event log (design §12.3: 6 partitions, key = user_id, 7 days retention).
const Topic = "events"

// Publisher hands accepted events to the event log. A nil error means every event is durable
// in the log (acks from all in-sync replicas); on error the caller answers 503 and the client
// retries the same event_ids, which every consumer deduplicates.
type Publisher interface {
	Publish(ctx context.Context, evs []Event) (Result, error)
}

// Result of one batch. Duplicates are only known on the synchronous path (Direct); with
// Redpanda deduplication happens in each consumer.
type Result struct {
	Accepted   int
	Duplicates int
}

// Kafka writes events to Redpanda (Kafka API, franz-go). The client should be idempotent with
// acks=all (franz-go's default).
type Kafka struct {
	Client *kgo.Client
	Topic  string
}

// KafkaRecord builds the topic record of an event.
func KafkaRecord(topic string, ev Event) (*kgo.Record, error) {
	value, err := EncodeRecord(ev)
	if err != nil {
		return nil, err
	}
	return &kgo.Record{
		Topic: topic,
		Key:   Key(ev.UserID),
		Value: value,
		Headers: []kgo.RecordHeader{
			{Key: HeaderType, Value: []byte(ev.Type)},
			{Key: HeaderVersion, Value: []byte(strconv.Itoa(RecordVersion))},
		},
	}, nil
}

func (k Kafka) produce(ctx context.Context, evs []Event) error {
	recs := make([]*kgo.Record, 0, len(evs))
	for _, ev := range evs {
		r, err := KafkaRecord(k.Topic, ev)
		if err != nil {
			return err
		}
		recs = append(recs, r)
	}
	return k.Client.ProduceSync(ctx, recs...).FirstErr()
}

// Publish writes a batch of client events and waits for the broker's acknowledgement.
func (k Kafka) Publish(ctx context.Context, evs []Event) (Result, error) {
	err := k.produce(ctx, evs)
	result := "ok"
	if err != nil {
		result = "error"
	}
	for _, ev := range evs {
		metrics.EventsPublished.WithLabelValues(ev.Type, result).Inc()
	}
	if err != nil {
		return Result{}, err
	}
	return Result{Accepted: len(evs)}, nil
}

// WriteServed implements ServedSink.
func (k Kafka) WriteServed(ctx context.Context, evs []Event) error { return k.produce(ctx, evs) }

// Header returns the value of a record header ("" if absent).
func Header(r *kgo.Record, key string) string {
	for _, h := range r.Headers {
		if h.Key == key {
			return string(h.Value)
		}
	}
	return ""
}
