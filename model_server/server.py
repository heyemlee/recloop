"""gRPC model server (design §3, §13.2): `Retrieve` = encode the history, search the item index.

Usage (from the repo root): python -m model_server [--bundle PATH] [--port 18081]
The bundle defaults to $RECLOOP_BUNDLE, else artifacts/bundles/current.

The gRPC health service reports SERVING only once the bundle has loaded and passed validation
(the readiness signal for M5's Kubernetes probes). Before doing any work a request checks the
deadline its caller propagated and gives up if it has already run out (§5.1).
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

from model_server import bundle as bundle_mod
from model_server.rpc import SERVICE_NAME, pb, pb_grpc

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUNDLE = ROOT / "artifacts" / "bundles" / "current"
DEFAULT_K = 200
MAX_K = 1000
MIN_REMAINING_S = 0.001  # below this the caller will have given up before we answer

log = logging.getLogger("model_server")


class ModelServer(pb_grpc.ModelServerServicer):
    def __init__(self, bundle: bundle_mod.Bundle):
        # One reference, swapped atomically when hot reload arrives (M4).
        self.bundle = bundle

    def Retrieve(self, request, context):  # noqa: N802 (gRPC method name)
        remaining = context.time_remaining()
        if remaining is not None and remaining < MIN_REMAINING_S:
            context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, "deadline already passed")
        b = self.bundle
        if request.pin_version and request.pin_version != b.version:
            context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"bundle {request.pin_version!r} is not loaded (serving {b.version})",
            )
        if request.k < 0:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "k must be >= 0")
        k = min(request.k or DEFAULT_K, MAX_K)
        items, scores = b.retrieve(request.history, k)
        return pb.RetrieveResponse(
            model_version=b.version,
            organic=[
                pb.Scored(item_id=int(i), score=float(s))
                for i, s in zip(items, scores, strict=True)
            ],
            # Ad scoring arrives with the ads pipeline (M3).
            ads=[],
        )

    def Rank(self, request, context):  # noqa: N802
        context.abort(
            grpc.StatusCode.UNIMPLEMENTED,
            "Rank arrives in M3 (ads pCTR) and M4 (organic ranking)",
        )


def serve(b: bundle_mod.Bundle, host: str, port: int, workers: int = 8):
    """Start the server; returns (server, bound port)."""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=workers),
        options=[("grpc.so_reuseport", 0)],
    )
    pb_grpc.add_ModelServerServicer_to_server(ModelServer(b), server)
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
    parser.add_argument("--workers", type=int, default=8)
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
    server, port = serve(b, args.host, args.port, args.workers)
    log.info("serving recloop.modelserver.v1.ModelServer on %s:%d", args.host, port)

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    stop.wait()
    log.info("shutting down")
    server.stop(grace=5).wait()


if __name__ == "__main__":
    main()
