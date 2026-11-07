"""gRPC model server (design §3, §13.2): `Retrieve` = encode the history, search the item index.

Usage (from the repo root): python -m model_server [--bundle PATH] [--port 18081]
The bundle defaults to $RECLOOP_BUNDLE, else artifacts/bundles/current.

The gRPC health service reports SERVING only once the bundle has loaded and passed validation
(the readiness signal for M5's Kubernetes probes). The caller's deadline travels with the call:
before each step (the injected delay, encoding, the index search) a request checks the time
remaining and gives up once the caller has (§5.1), so no CPU goes into answers nobody waits
for. The admin port serves Prometheus metrics and the fault-injection switch (admin.py).
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
import time
from concurrent import futures
from pathlib import Path

import faiss
import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from model_server import admin
from model_server import bundle as bundle_mod
from model_server.rpc import SERVICE_NAME, pb, pb_grpc

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUNDLE = ROOT / "artifacts" / "bundles" / "current"
DEFAULT_K = 200
MAX_K = 1000
MIN_REMAINING_S = 0.001  # below this the caller will have given up before we answer
# Workers: an injected delay sleeps in a worker thread (no CPU, no GIL); with too few workers a
# 30 % x 200 ms fault would exhaust the pool and slow every call, a different fault.
DEFAULT_WORKERS = 32

log = logging.getLogger("model_server")


def _tracer():
    from opentelemetry import trace

    return trace.get_tracer("model_server")


class ModelServer(pb_grpc.ModelServerServicer):
    def __init__(self, bundle: bundle_mod.Bundle, faults: admin.Faults | None = None):
        # One reference, swapped atomically when hot reload arrives (M4).
        self.bundle = bundle
        self.faults = faults or admin.Faults()
        self.tracer = _tracer()

    @staticmethod
    def _expired(context, where: str) -> None:
        remaining = context.time_remaining()
        if remaining is not None and remaining < MIN_REMAINING_S:
            admin.DEADLINE_ABORTS.labels(where).inc()
            admin.REQUESTS.labels("DEADLINE_EXCEEDED").inc()
            context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, f"deadline passed ({where})")

    def Retrieve(self, request, context):  # noqa: N802 (gRPC method name)
        t0 = time.perf_counter()
        admin.INFLIGHT.inc()
        try:
            return self._retrieve(request, context, t0)
        finally:
            admin.INFLIGHT.dec()

    def _retrieve(self, request, context, t0: float):
        self._expired(context, "arrival")
        delay = self.faults.draw()
        if delay > 0:
            admin.INJECTED.inc()
            remaining = context.time_remaining()
            # Wait the injected delay, but not past the caller's deadline.
            time.sleep(delay if remaining is None else max(0.0, min(delay, remaining)))
            self._expired(context, "after_delay")
        b = self.bundle
        if request.pin_version and request.pin_version != b.version:
            admin.REQUESTS.labels("NOT_FOUND").inc()
            context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"bundle {request.pin_version!r} is not loaded (serving {b.version})",
            )
        if request.k < 0:
            admin.REQUESTS.labels("INVALID_ARGUMENT").inc()
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "k must be >= 0")
        k = min(request.k or DEFAULT_K, MAX_K)
        items, scores = [], []
        if b.window(request.history):
            t = time.perf_counter()
            with self.tracer.start_as_current_span("encode"):
                user = b.encode([request.history])
            admin.RETRIEVE_SECONDS.labels("encode").observe(time.perf_counter() - t)
            self._expired(context, "before_search")
            t = time.perf_counter()
            with self.tracer.start_as_current_span("search"):
                items, scores = b.search(user, k)
            admin.RETRIEVE_SECONDS.labels("search").observe(time.perf_counter() - t)
        resp = pb.RetrieveResponse(
            model_version=b.version,
            organic=[
                pb.Scored(item_id=int(i), score=float(s))
                for i, s in zip(items, scores, strict=True)
            ],
            # Ad scoring arrives with the ads pipeline (M3).
            ads=[],
        )
        admin.REQUESTS.labels("OK").inc()
        admin.RETRIEVE_SECONDS.labels("total").observe(time.perf_counter() - t0)
        return resp

    def Rank(self, request, context):  # noqa: N802
        context.abort(
            grpc.StatusCode.UNIMPLEMENTED,
            "Rank arrives in M3 (ads pCTR) and M4 (organic ranking)",
        )


def serve(
    b: bundle_mod.Bundle,
    host: str,
    port: int,
    workers: int = DEFAULT_WORKERS,
    faults: admin.Faults | None = None,
    interceptors=None,
):
    """Start the server; returns (server, bound port)."""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=workers),
        interceptors=interceptors or None,
        options=[("grpc.so_reuseport", 0)],
    )
    pb_grpc.add_ModelServerServicer_to_server(ModelServer(b, faults), server)
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    bound = server.add_insecure_port(f"{host}:{port}")
    if bound == 0:
        raise OSError(f"could not bind {host}:{port}")
    server.start()
    for name in ("", SERVICE_NAME):
        health_servicer.set(name, health_pb2.HealthCheckResponse.SERVING)
    return server, bound


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="RecLoop model server")
    parser.add_argument(
        "--bundle", type=Path, default=Path(os.environ.get("RECLOOP_BUNDLE", DEFAULT_BUNDLE))
    )
    parser.add_argument("--host", default=os.environ.get("MODEL_SERVER_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("MODEL_SERVER_PORT", "18081"))
    )
    parser.add_argument(
        "--admin-port", type=int, default=int(os.environ.get("MODEL_SERVER_ADMIN_PORT", "18082"))
    )
    parser.add_argument(
        "--workers", type=int, default=int(os.environ.get("MODEL_SERVER_WORKERS", DEFAULT_WORKERS))
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    faiss.omp_set_num_threads(1)

    t0 = time.monotonic()
    b = bundle_mod.load(args.bundle)
    log.info(
        "bundle %s loaded from %s in %.2fs: %d items, dim %d, golden queries passed",
        b.version,
        b.path,
        time.monotonic() - t0,
        b.num_items,
        b.dim,
    )
    faults = admin.Faults.from_env()
    interceptor = admin.setup_tracing()
    server, port = serve(
        b,
        args.host,
        args.port,
        args.workers,
        faults=faults,
        interceptors=[interceptor] if interceptor else None,
    )
    httpd = admin.serve_admin(faults, args.host, args.admin_port)
    s = faults.get()
    log.info(
        "serving recloop.modelserver.v1.ModelServer on %s:%d (%d workers); admin on :%d; "
        "fault: %.0f ms x %.2f; tracing %s",
        args.host,
        port,
        args.workers,
        httpd.server_address[1],
        s.delay_ms,
        s.probability,
        "on" if interceptor else "off",
    )

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    stop.wait()
    log.info("shutting down")
    httpd.shutdown()
    server.stop(grace=5).wait()


if __name__ == "__main__":
    main()
