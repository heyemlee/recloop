// Package tracing sets up OpenTelemetry tracing (design §5.2): spans go over OTLP/gRPC to the
// collector, which forwards them to Jaeger. Configuration is the standard OTEL_* environment:
// OTEL_EXPORTER_OTLP_ENDPOINT (tracing is off when it is unset), OTEL_TRACES_SAMPLER /
// OTEL_TRACES_SAMPLER_ARG (default parentbased_always_on), OTEL_SERVICE_NAME.
package tracing

import (
	"context"
	"os"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.43.0"
	"go.opentelemetry.io/otel/trace"
)

// Tracer of the online path.
var Tracer trace.Tracer = otel.Tracer("github.com/heyemlee/recloop/serving")

// Setup installs the global tracer provider and propagator; shutdown flushes buffered spans.
// Without OTEL_EXPORTER_OTLP_ENDPOINT it only installs the propagator (spans are no-ops).
func Setup(ctx context.Context, service string) (shutdown func(context.Context) error, enabled bool, err error) {
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(propagation.TraceContext{}, propagation.Baggage{}))
	if os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT") == "" && os.Getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") == "" {
		return func(context.Context) error { return nil }, false, nil
	}
	exp, err := otlptracegrpc.New(ctx)
	if err != nil {
		return nil, false, err
	}
	name := os.Getenv("OTEL_SERVICE_NAME")
	if name == "" {
		name = service
	}
	res, err := resource.Merge(resource.Default(), resource.NewSchemaless(semconv.ServiceName(name)))
	if err != nil {
		return nil, false, err
	}
	// The sampler comes from OTEL_TRACES_SAMPLER(_ARG) when set (the SDK reads them).
	tp := sdktrace.NewTracerProvider(sdktrace.WithBatcher(exp), sdktrace.WithResource(res))
	otel.SetTracerProvider(tp)
	Tracer = tp.Tracer("github.com/heyemlee/recloop/serving")
	return tp.Shutdown, true, nil
}
