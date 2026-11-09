"""Experiment bucketing, the bundle, text embeddings, Redis seeding and simulated stock."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from ml import paths
from ml.bundle import build
from ml.data import ingest, seed_redis, text_embed
from ml.data.download import sha256_file
from ml.experiment import bucketing

VECTORS = paths.ROOT / "testdata" / "bucket_vectors.json"


def test_bucketing_matches_shared_vectors():
    v = json.loads(VECTORS.read_text())
    assert v["salts"] == bucketing.SALTS and v["buckets"] == bucketing.BUCKETS
    for r in v["reference"]:
        assert f"{bucketing.xxh64(r['input']):016x}" == r["xxh64"]
    assert len(v["cases"]) >= 200
    for c in v["cases"]:
        assert f"{bucketing.xxh64(c['salt'] + ':' + c['unit']):016x}" == c["xxh64"]
        assert bucketing.bucket(c["salt"], c["unit"]) == c["bucket"]
    # The committed file is exactly what the generator writes.
    assert v == json.loads(json.dumps(bucketing.vectors()))


def test_initial_stock_is_seeded_with_some_sold_out():
    a, b = ingest.initial_stock(3459), ingest.initial_stock(3459)
    np.testing.assert_array_equal(a, b)
    assert a.min() == 0 and a.max() <= ingest.MAX_STOCK
    assert 0.02 < (a == 0).mean() < 0.04


def test_item_text():
    assert (
        text_embed.item_text("Pen", ["Blue ink", " ", "Pack of 12"]) == "Pen. Blue ink; Pack of 12"
    )
    assert text_embed.item_text("Pen", None) == "Pen"


@pytest.mark.skipif(not paths.TEXT_EMB_NPY.exists(), reason="no text embeddings: run `make data`")
def test_text_embeddings():
    emb = np.load(paths.TEXT_EMB_NPY)
    meta = json.loads(paths.TEXT_EMB_JSON.read_text())
    assert emb.shape == (3459, 384) and emb.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-4)
    assert hashlib.sha256(emb.tobytes()).hexdigest() == meta["sha256"]


def test_demo_users_pick_one_per_category_and_a_cold_user():
    items = pd.DataFrame(
        {
            "item_id": [0, 1, 2],
            "title": ["Pen", "Ink", "Paper"],
            "categories": [["O", "S", "Writing"], ["O", "S", "Ink"], ["O", "S", "Writing"]],
        }
    )
    train = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 3],
            "item_id": [0, 2, 0, 1, 1, 0],
            "seq_idx": [0, 1, 2, 0, 1, 0],
        }
    )
    demo = seed_redis.demo_users(train, items, pd.Series([1, 2, 3, 4]))
    assert [d["user_id"] for d in demo] == [1, 2, 4]
    assert demo[0]["history_len"] == 3 and demo[0]["recent"] == ["Pen", "Paper", "Pen"]
    assert demo[-1]["history_len"] == 0 and "cold start" in demo[-1]["label"]


BUNDLE = build.CURRENT


@pytest.mark.skipif(not BUNDLE.exists(), reason="no bundle: run `make bundle`")
def test_bundle_manifest_and_tables():
    m = json.loads((BUNDLE / "manifest.json").read_text())
    assert m["format"] == build.FORMAT and m["version"] == BUNDLE.resolve().name
    expected = {
        "sasrec.pt",
        "sasrec_encoder.onnx",
        "item_vectors.npy",
        "items.faiss",
        "item_ids.json",
        "itemcf.json",
        "pop.json",
        "golden.json",
    }
    # M4 bundles carry the ranking features' item tables (and the ranker when it was trained);
    # a bundle built before M4 has no "ranking" section and still serves (RRF order).
    ranking = m.get("ranking", {})
    if ranking:
        expected |= {"features.json"}
    if "ranker" in ranking:
        expected |= {"ranker.onnx", "ranker.txt", "rank_golden.json"}
    if "pctr" in m:
        expected |= {"pctr.json"}
    assert set(m["files"]) == expected
    if "ranker" in ranking:
        assert ranking["ranker"]["onnx_max_abs_diff"] < build.RANKER_TOLERANCE
    for name, f in m["files"].items():
        assert sha256_file(BUNDLE / name) == f["sha256"], name
    assert m["num_items"] == 3459 and m["index"]["ntotal"] == 3459
    assert (
        m["golden"]["recall"] >= 0.999 and m["encoder"]["onnx_max_abs_diff"] < build.ONNX_TOLERANCE
    )

    vectors = np.load(BUNDLE / "item_vectors.npy")
    assert vectors.shape == (3459, m["dim"])
    pop = json.loads((BUNDLE / "pop.json").read_text())
    assert sorted(pop["items"]) == list(range(3459))
    assert pop["counts"] == sorted(pop["counts"], reverse=True)
    cf = json.loads((BUNDLE / "itemcf.json").read_text())
    assert len(cf["neighbors"]) == 3459
    assert all(len(n) == len(s) <= 50 for n, s in zip(cf["neighbors"], cf["scores"], strict=True))
    assert all(s == sorted(s, reverse=True) for s in cf["scores"])
    golden = json.loads((BUNDLE / "golden.json").read_text())
    q = golden["queries"][0]
    assert len(q["items"]) == golden["top"] and q["scores"] == sorted(q["scores"], reverse=True)
