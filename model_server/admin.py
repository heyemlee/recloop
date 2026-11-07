"""Fault injection, metrics and the admin HTTP port of the model server (design §5.2).

Fault injection: with probability `probability` a Retrieve call waits `delay_ms` before doing
its work. This is the "30 % of requests +200 ms" fault of the latency experiments: per request,
which a network proxy (Toxiproxy, per connection) cannot do on one long-lived gRPC connection.
The wait respects the caller's deadline: it ends when the deadline passes and the call is
aborted, so a delayed request the caller has already given up on does not hold a worker longer
than the caller waits (§5.1).

Admin HTTP (MODEL_SERVER_ADMIN_PORT, default 18082):
    GET  /metrics   Prometheus
    GET  /fault     current fault settings
    PUT  /fault     {"delay_ms": 200, "probability": 0.3}; {"delay_ms": 0} turns it off
    GET  /healthz
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

log = logging.getLogger("model_server.admin")

BUCKETS = (
    0.0001, 0.00025, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025,
    0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0,
)  # fmt: skip

RETRIEVE_SECONDS = Histogram(
    "recloop_ms_retrieve_seconds",
    "Retrieve latency inside the model server by stage (encode, search, total).",
    ["stage"],
    buckets=BUCKETS,
)
REQUESTS = Counter("recloop_ms_requests_total", "Retrieve calls by gRPC status code.", ["code"])
INJECTED = Counter("recloop_ms_injected_delays_total", "Retrieve calls the fault switch delayed.")
DEADLINE_ABORTS = Counter(
    "recloop_ms_deadline_aborts_total",
    "Calls given up because the caller's deadline had passed, by where it was noticed.",
    ["where"],
)
INFLIGHT = Gauge("recloop_ms_inflight", "Retrieve calls being handled.")


@dataclass
class FaultSettings:
    delay_ms: float = 0.0
    probability: float = 0.0


class Faults:
    """Thread-safe fault switch with a seeded random stream (reproducible injection rate)."""

    def __init__(self, settings: FaultSettings | None = None, seed: int = 20261106):
        self._lock = threading.Lock()
        self._settings = settings or FaultSettings()
        self._rng = random.Random(seed)

    @classmethod
    def from_env(cls) -> Faults:
        return cls(
            FaultSettings(
                delay_ms=float(os.environ.get("FAULT_DELAY_MS", "0")),
                probability=float(os.environ.get("FAULT_DELAY_PROB", "0")),
            ),
            seed=int(os.environ.get("FAULT_SEED", "20261106")),
        )

    def get(self) -> FaultSettings:
        with self._lock:
            return FaultSettings(**asdict(self._settings))

    def set(self, delay_ms: float, probability: float) -> FaultSettings:
        if delay_ms < 0 or not 0 <= probability <= 1:
            raise ValueError("delay_ms must be >= 0 and probability in [0, 1]")
        with self._lock:
            self._settings = FaultSettings(delay_ms=float(delay_ms), probability=float(probability))
            return FaultSettings(**asdict(self._settings))

    def draw(self) -> float:
        """Seconds this call should wait (0 most of the time)."""
        with self._lock:
            s = self._settings
            if s.delay_ms <= 0 or s.probability <= 0:
                return 0.0
            return s.delay_ms / 1000.0 if self._rng.random() < s.probability else 0.0


def _handler(faults: Faults):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 (http.server method name)
            if self.path == "/metrics":
                self._send(200, generate_latest(), CONTENT_TYPE_LATEST)
            elif self.path == "/fault":
                self._send(200, json.dumps(asdict(faults.get())).encode())
            elif self.path == "/healthz":
                self._send(200, b'{"status": "ok"}')
            else:
                self._send(404, b'{"error": "not found"}')

        def do_PUT(self):  # noqa: N802
            if self.path != "/fault":
                self._send(404, b'{"error": "not found"}')
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(n) or b"{}")
                s = faults.set(float(body.get("delay_ms", 0)), float(body.get("probability", 0)))
            except (ValueError, TypeError) as e:
                self._send(400, json.dumps({"error": str(e)}).encode())
                return
            log.warning(
                "fault injection set: delay %.0f ms, probability %.2f", s.delay_ms, s.probability
            )
            self._send(200, json.dumps(asdict(s)).encode())

        do_POST = do_PUT  # noqa: N815

        def log_message(self, *_args) -> None:  # keep the server log quiet
            return

    return Handler


def serve_admin(faults: Faults, host: str, port: int) -> ThreadingHTTPServer:
    """Start the admin HTTP server in a daemon thread; returns it (server_address has the port)."""
    httpd = ThreadingHTTPServer((host, port), _handler(faults))
    threading.Thread(target=httpd.serve_forever, name="admin-http", daemon=True).start()
    return httpd


def setup_tracing(service: str = "recloop-model-server"):
    """OTLP/gRPC tracing when OTEL_EXPORTER_OTLP_ENDPOINT is set; returns a server interceptor
    (or None). The sampler comes from OTEL_TRACES_SAMPLER(_ARG), parent-based by default, so the
    api's sampling decision carries over."""
    if not (
        os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    ):
        return None
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.grpc import server_interceptor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    name = os.environ.get("OTEL_SERVICE_NAME", service)
    provider = TracerProvider(resource=Resource.create({"service.name": name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    return server_interceptor()
