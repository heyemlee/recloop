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
            "user_id": rng.integers(0, 300, n),
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
    pctr.main([])
    m = json.loads((tmp_path / "model" / "model.json").read_text())
    assert m["variant"] == "base" and "ctr" not in m  # samples without counts
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


def ctr_samples(n: int = 12000, seed: int = 3) -> pd.DataFrame:
    """Clicks follow examination(pos) x the item's own appeal; the item's past impressions and
    clicks (its counts) reveal the appeal, the other features do not."""
    rng = np.random.default_rng(seed)
    df = samples(n, seed)
    items = rng.integers(0, 150, len(df))
    appeal = rng.normal(0, 0.9, 150)
    p = np.clip(0.06 * df.pos.to_numpy() ** -0.7 * np.exp(appeal[items]), 0, 1)
    y = (rng.random(len(df)) < p).astype(int)
    imp_h = rng.poisson(80, len(df))
    imp_d = imp_h + rng.poisson(1500, len(df))
    rate = np.clip(0.03 * np.exp(appeal[items]), 0, 1)
    clk_h = rng.binomial(imp_h, rate)
    clk_d = clk_h + rng.binomial(imp_d - imp_h, rate)
    df = df.assign(
        item_id=items,
        label=y,
        weight=1.0,
        ictr=[list(map(int, r)) for r in zip(imp_h, clk_h, imp_d, clk_d, strict=True)],
        ictr_source="served",
    )
    return df


def test_ctr_columns_help_and_serving_agrees(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    df = ctr_samples()
    d = tmp_path / "train" / "day=2026-11-08"
    d.mkdir(parents=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), d / "part-0.parquet")
    monkeypatch.setattr(pctr, "SAMPLES", tmp_path / "train")
    monkeypatch.setattr(pctr, "OUT", tmp_path / "model")
    monkeypatch.setattr(pctr, "BOOTSTRAP", 50)
    pctr.main(["--label", "test"])
    m = json.loads((tmp_path / "model" / "model.json").read_text())
    ev = json.loads((tmp_path / "model" / "eval" / "test.json").read_text())
    assert m["variant"] == "ctr" and tuple(m["columns"]) == pctr.COLUMNS_CTR
    assert 0 < m["ctr"]["prior"] < 0.2 and m["ctr"]["strength"] in pctr.STRENGTH_GRID
    assert ev["shipped"] == "ctr" and ev["counts_source"] == {"served": len(df)}
    held = ev["variants"]
    assert held["ctr"]["held_out"]["auc"] > held["base"]["held_out"]["auc"] + 0.03
    c = ev["comparison"]["all"]
    assert c["auc"]["ci95"][0] > 0 and c["logloss"]["diff"] < 0
    assert set(held["ctr"]["held_out"]["by_decile"]) <= {f"d{k}" for k in range(1, 11)}

    # The serving twin: same probabilities from the shipped JSON, with and without counts.
    x = np.asarray([[np.nan if v is None else v for v in row] for row in df.features[:300]])
    pos, sp = df.pos.to_numpy()[:300], (df.slot == "sponsored").to_numpy()[:300]
    counts = np.asarray(df.ictr[:300].tolist(), dtype=np.float64)
    served = serving.PCTRModel.from_json(m)
    np.testing.assert_allclose(
        served.predict(x, pos, sp, counts), pctr.predict_json(m, x, pos, sp, counts), atol=1e-12
    )
    # Missing counts are the prior: the same as zero impressions and zero clicks.
    np.testing.assert_allclose(
        served.predict(x, pos, sp), served.predict(x, pos, sp, np.zeros((300, 4))), atol=1e-12
    )
    # More clicks of the day on the same impressions: a higher pCTR (the day's estimate of the
    # appeal is the precise one here; the hour's is collinear with it, its sign is not pinned).
    more = counts.copy()
    more[:, 3] += 30
    assert (served.predict(x, pos, sp, more) > served.predict(x, pos, sp, counts)).all()
    # Every row had counts in both windows; --since keeps the samples served from then on.
    assert ev["counts_coverage"]["all"] == {"hour": 1.0, "day": 1.0} and ev["since"] is None
    t = pd.to_datetime(df.t_impression, utc=True)
    cut = t.iloc[len(df) // 2]
    pctr.main(["--label", "late", "--since", cut.tz_localize(None).isoformat()])
    late = json.loads((tmp_path / "model" / "eval" / "late.json").read_text())
    assert late["rows"] == int((t >= cut).sum()) and late["since"] == cut.isoformat()
    assert json.loads((tmp_path / "model" / "model.json").read_text())["training"]["since"] == (
        cut.isoformat()
    )


def test_ctr_columns_are_finite_at_the_edges():
    c = np.asarray([[0, 0, 0, 0], [0, 7, 3, 9], [10, 10, 10, 10], [np.nan] * 4], dtype=float)
    for twin in (pctr.ctr_columns, serving.ctr_columns):
        out = twin(c, 0.01, 10.0)
        for v in out.values():
            assert np.isfinite(v).all()
        assert out["ictr_1h_logit"][0] == out["ictr_1h_logit"][3]  # missing = no evidence
        assert out["ictr_1h_logit"][1] > out["ictr_1h_logit"][0]
    a, b = pctr.ctr_columns(c, 0.01, 10.0), serving.ctr_columns(c, 0.01, 10.0)
    for k in pctr.CTR_COLUMNS:
        np.testing.assert_array_equal(a[k], b[k])
