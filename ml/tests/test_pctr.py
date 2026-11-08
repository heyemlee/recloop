"""pCTR (ml/models/pctr.py) and its serving twin (model_server/ranking.py): same features, same
design matrix, same predictions; and the training recovers a known position effect."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ml.models import pctr
from ml.ranking import candidates as cm
from ml.ranking import features as feat
from model_server import ranking as serving


def tables(n: int = 30, seed: int = 0) -> feat.ItemTables:
    rng = np.random.default_rng(seed)
    cats = [
        [
            "Office Products",
            "Office & School Supplies",
            ["Paper", "Pens", "Ink"][i % 3],
            f"leaf{i % 5}",
        ]
        if i % 7
        else []
        for i in range(n)
    ]
    price = pd.array(
        [None if i % 6 == 0 else int(rng.integers(100, 30000)) for i in range(n)], dtype="Int64"
    )
    catalog = pd.DataFrame({"item_id": range(n), "categories": cats, "price_cents": price})
    return feat.ItemTables.build(rng.integers(0, 50, n).astype(float), catalog)


def test_serving_features_equal_the_offline_ones():
    t = tables()
    doc = json.loads(json.dumps(t.to_json()))  # what the bundle ships
    r = serving.Ranking(doc)
    rng = np.random.default_rng(1)
    for _ in range(20):
        hist = rng.choice(30, size=rng.integers(1, 10), replace=False)
        items = rng.choice(30, size=12, replace=False)
        ranks = {s: np.where(rng.random(12) < 0.5, rng.integers(1, 200, 12), 0) for s in cm.SOURCES}
        scores = {
            s: np.where(
                ranks[s] > 0, rng.normal(size=12).astype(np.float32).astype(np.float64), np.nan
            )
            for s in cm.SOURCES
        }
        c = cm.Candidates(
            items[None, :],
            np.zeros((1, 12)),
            {s: v[None, :] for s, v in ranks.items()},
            {s: v[None, :] for s, v in scores.items()},
        )
        want = feat.compute(c, [hist], t)[0]
        hits = [
            [(s, int(ranks[s][j]), float(scores[s][j])) for s in cm.SOURCES if ranks[s][j] > 0]
            for j in range(12)
        ]
        got = r.features(hist.tolist(), items.tolist(), hits)
        np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
        np.testing.assert_allclose(np.nan_to_num(got), np.nan_to_num(want), rtol=0, atol=1e-7)


def samples(n: int = 6000, seed: int = 2) -> pd.DataFrame:
    """Impressions whose clicks follow examination(pos) x relevance, the simulator's model."""
    rng = np.random.default_rng(seed)
    pos = rng.integers(1, 21, n)
    sponsored = rng.random(n) < 0.15
    rel = rng.normal(size=n)
    x = np.full((n, len(feat.FEATURES)), np.nan)
    col = {f: i for i, f in enumerate(feat.FEATURES)}
    x[:, col["sasrec_score"]] = np.where(rng.random(n) < 0.8, rel, np.nan)
    x[:, col["pop_log"]] = rng.random(n) * 4
    x[:, col["n_sources"]] = rng.integers(0, 4, n)
    x[:, col["price_bucket"]] = np.where(rng.random(n) < 0.1, np.nan, rng.integers(0, 6, n))
    x[:, col["cat2_overlap"]] = rng.random(n)
    p = np.clip(0.08 * pos**-0.7 * np.exp(0.6 * np.nan_to_num(x[:, col["sasrec_score"]])), 0, 1)
    y = (rng.random(n) < p).astype(int)
    keep = (y == 1) | (rng.random(n) < 0.5)  # negatives at rate 0.5, weight 2
    t0 = pd.Timestamp("2026-11-08T10:00:00Z")
    return pd.DataFrame(
        {
            "impression_id": [f"r{i}:{q}" for i, q in enumerate(pos)],
            "pos": pos,
            "slot": np.where(sponsored, "sponsored", "organic"),
            "t_impression": [t0 + pd.Timedelta(seconds=int(s)) for s in range(n)],
            "feature_names": [list(feat.FEATURES)] * n,
            "features": [[None if np.isnan(v) else float(v) for v in row] for row in x],
            "label": y,
            "weight": np.where(y == 1, 1.0, 2.0),
        }
    )[keep].reset_index(drop=True)


def test_training_recovers_the_position_effect_and_serving_agrees(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    df = samples()
    d = tmp_path / "train" / "day=2026-11-08"
    d.mkdir(parents=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), d / "part-0.parquet")
    monkeypatch.setattr(pctr, "SAMPLES", tmp_path / "train")
    monkeypatch.setattr(pctr, "OUT", tmp_path / "model")
    pctr.main()
    m = json.loads((tmp_path / "model" / "model.json").read_text())
    effect = m["training"]["position_log_odds_vs_pos1"]
    # The simulator's examination is pos^-0.7: log-odds fall with the position.
    assert (
        effect["pos_3"] < 0
        and effect["pos_10"] < effect["pos_3"]
        and effect["pos_11_20"] < effect["pos_10"]
    )
    assert abs(effect["pos_10"] - (-0.7 * np.log(10))) < 0.5
    held = m["training"]["held_out"]
    assert held["auc"] > held["auc_position_only"] > 0.5
    # The serving twin gives the same probabilities from the shipped JSON.
    x = np.asarray([[np.nan if v is None else v for v in row] for row in df.features[:200]])
    pos, sp = df.pos.to_numpy()[:200], (df.slot == "sponsored").to_numpy()[:200]
    want = pctr.predict_json(m, x, pos, sp)
    got = serving.PCTRModel.from_json(m).predict(x, pos, sp)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)
    # At serving, the position is the reference one.
    ref = serving.PCTRModel.from_json(m).predict(x)
    np.testing.assert_allclose(
        ref,
        pctr.predict_json(m, x, np.full(200, pctr.REFERENCE_POSITION), np.ones(200)),
        atol=1e-12,
    )
