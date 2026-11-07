"""Model server: input window, retrieval and the gRPC surface, on a small synthetic bundle.

The real bundle (artifacts/bundles/current) is exercised too when it exists: it must load,
pass its golden queries and answer Retrieve over gRPC.
"""

from pathlib import Path

import faiss
import grpc
import numpy as np
import pytest
from grpc_health.v1 import health_pb2, health_pb2_grpc

from model_server import bundle as bundle_mod
from model_server.rpc import SERVICE_NAME, pb, pb_grpc
from model_server.server import DEFAULT_BUNDLE, serve

N, DIM, L = 30, 4, 10


class SumEncoder:
    """Stand-in for the ONNX session: user vector = sum of the item vectors in the window."""

    def __init__(self, vectors: np.ndarray):
        self.table = np.vstack([np.zeros((1, vectors.shape[1]), np.float32), vectors])
        self.calls = []

    def run(self, output_names, feed):
        x = feed["history"]
        self.calls.append(x.copy())
        return [self.table[x].sum(axis=1)]


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(N, DIM)).astype(np.float32)
    index = faiss.IndexFlatIP(DIM)
    index.add(vectors)
    b = bundle_mod.Bundle(
        version="test-v1",
        path=Path("."),
        num_items=N,
        dim=DIM,
        history_len=L,
        encoder=SumEncoder(vectors),
        index=index,
        item_ids=np.arange(N),
    )
    return b, vectors


def test_window_keeps_last_known_items(toy):
    b, _ = toy
    history = [-1, 5, 99, *range(1, 13)]  # unknown ids dropped, then the last 10
    assert b.window(history) == list(range(3, 13))
    b.encode([[7, 8]])
    x = b.encoder.calls[-1]
    assert x.tolist() == [[0] * 8 + [8, 9]]  # left-padded with 0, ids shifted by +1


def test_retrieve_is_exact_top_k_by_inner_product(toy):
    b, vectors = toy
    items, scores = b.retrieve([3, 4], k=5)
    user = vectors[3] + vectors[4]
    expected = np.argsort(-(vectors @ user), kind="stable")[:5]
    assert items.tolist() == expected.tolist()
    np.testing.assert_allclose(scores, (vectors @ user)[expected], rtol=1e-5)


def test_retrieve_without_history_is_empty(toy):
    b, _ = toy
    for history in ([], [-3, N + 1]):
        items, scores = b.retrieve(history, k=5)
        assert len(items) == len(scores) == 0


def test_golden_check_rejects_a_drifted_index(toy):
    b, vectors = toy
    users = b.encode([[1, 2], [3]])
    scores, rows = b.index.search(users, 5)
    golden = {
        "top": 5,
        "min_recall": 0.99,
        "score_tolerance": 1e-3,
        "queries": [
            {"history": h, "items": r.tolist(), "scores": s.tolist()}
            for h, r, s in zip([[1, 2], [3]], rows, scores, strict=True)
        ],
    }
    assert bundle_mod.golden_check(b, golden)["recall"] == 1.0
    golden["queries"][0]["items"] = [N - 1] * 5
    with pytest.raises(bundle_mod.BundleError):
        bundle_mod.golden_check(b, golden)


def start(b):
    try:
        return serve(b, "127.0.0.1", 0, workers=2)
    except (RuntimeError, OSError) as e:  # e.g. a sandbox that forbids listening on local ports
        pytest.skip(f"cannot bind a local port: {e}")


@pytest.fixture
def stub(toy):
    server, port = start(toy[0])
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    yield pb_grpc.ModelServerStub(channel), channel
    channel.close()
    server.stop(None)


def test_grpc_retrieve(stub, toy):
    s, _ = stub
    resp = s.Retrieve(pb.RetrieveRequest(history=[3, 4], k=5), timeout=5)
    items, scores = toy[0].retrieve([3, 4], 5)
    assert resp.model_version == "test-v1"
    assert [x.item_id for x in resp.organic] == items.tolist()
    np.testing.assert_allclose([x.score for x in resp.organic], scores, rtol=1e-6)
    assert list(resp.ads) == []
    default_k = s.Retrieve(pb.RetrieveRequest(history=[3]), timeout=5)
    assert len(default_k.organic) == N  # default k (200) capped at the item count


def test_grpc_errors(stub):
    s, channel = stub
    with pytest.raises(grpc.RpcError) as e:
        s.Retrieve(pb.RetrieveRequest(history=[1], pin_version="other"), timeout=5)
    assert e.value.code() == grpc.StatusCode.NOT_FOUND
    with pytest.raises(grpc.RpcError) as e:
        s.Retrieve(pb.RetrieveRequest(history=[1], k=-1), timeout=5)
    assert e.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    with pytest.raises(grpc.RpcError) as e:
        s.Rank(pb.RankRequest(), timeout=5)
    assert e.value.code() == grpc.StatusCode.UNIMPLEMENTED
    health = health_pb2_grpc.HealthStub(channel)
    resp = health.Check(health_pb2.HealthCheckRequest(service=SERVICE_NAME), timeout=5)
    assert resp.status == health_pb2.HealthCheckResponse.SERVING


@pytest.mark.skipif(not DEFAULT_BUNDLE.exists(), reason="no bundle: run `make bundle`")
def test_real_bundle_loads_and_serves():
    b = bundle_mod.load(DEFAULT_BUNDLE)  # checksums + golden queries
    assert b.num_items == 3459
    server, port = start(b)
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            resp = pb_grpc.ModelServerStub(channel).Retrieve(
                pb.RetrieveRequest(history=[0, 1, 2], k=50), timeout=5
            )
        assert resp.model_version == b.version
        assert len(resp.organic) == 50
        scores = [x.score for x in resp.organic]
        assert scores == sorted(scores, reverse=True)
    finally:
        server.stop(None)
