"""pCTR of the sponsored slots: logistic regression on the logged impressions (design §9.2 step 3).

Training data is the sample job's output (ml.samples.job, artifacts/samples/train): home-feed
impressions whose window has closed, with the features the served event recorded and the
negatives downsampled (weight = 1 / rate restores them). The features of a row:

    sasrec_score (0 when missing) and a missing flag   SASRec's relevance of the item
    pop_log, n_sources                                 popularity, how many sources had it
    cat2_overlap, leaf_overlap (0 when missing)        category overlap with the history
    price bucket, one-hot (6 buckets + unknown)
    sponsored (0/1)                                    the slot type
    position, one-hot (1 .. 10, 11-20, 21+)

The position is a training feature only: at prediction it is fixed at the reference position
(REFERENCE_POSITION, the first sponsored slot), so every ad is scored as if shown at the same
place and the auction compares relevance, not where an ad happened to be shown before.

Model selection: the samples in time order are split 64 / 16 / 20; C (L2 strength) is chosen
on the middle part by weighted log loss, the model refitted on the first 80 % is reported on
the last 20 % (the future), and the shipped model is refitted on everything with that C.

Usage: python -m ml.models.pctr   (after `make samples`; writes artifacts/models/pctr/)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ml import paths
from ml.ranking import features as feat

SAMPLES = paths.ARTIFACTS / "samples" / "train"
OUT = paths.ARTIFACTS / "models" / "pctr"
REFERENCE_POSITION = 3
POSITION_BUCKETS = tuple(f"pos_{p}" for p in range(1, 11)) + ("pos_11_20", "pos_21_up")
PRICE_COLUMNS = tuple(f"price_{b}" for b in range(len(feat.PRICE_EDGES_CENTS) + 1)) + ("price_unknown",)
CONTINUOUS = ("sasrec_score", "pop_log", "n_sources", "cat2_overlap", "leaf_overlap")
COLUMNS = (*CONTINUOUS, "sasrec_missing", *PRICE_COLUMNS, "sponsored", *POSITION_BUCKETS)
C_GRID = (0.01, 0.1, 1.0, 10.0)
SEED = 20261108


def position_bucket(pos: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.int64)
    return np.where(pos <= 10, pos - 1, np.where(pos <= 20, 10, 11))


def design(x: np.ndarray, pos: np.ndarray, sponsored: np.ndarray) -> np.ndarray:
    """Rows of the ranker's feature vector (feat.FEATURES order) -> the model's columns, before
    standardisation. The model server builds the same matrix (model_server/ranking.py)."""
    x = np.asarray(x, dtype=np.float64)
    col = {f: x[:, i] for i, f in enumerate(feat.FEATURES)}
    n = len(x)
    out = np.zeros((n, len(COLUMNS)))
    c = {name: i for i, name in enumerate(COLUMNS)}
    for f in CONTINUOUS:
        out[:, c[f]] = np.nan_to_num(col[f], nan=0.0)
    out[:, c["sasrec_missing"]] = np.isnan(col["sasrec_score"])
    price = col["price_bucket"]
    known = ~np.isnan(price)
    out[np.flatnonzero(known), c["price_0"] + price[known].astype(np.int64)] = 1.0
    out[~known, c["price_unknown"]] = 1.0
    out[:, c["sponsored"]] = np.asarray(sponsored, dtype=np.float64)
    out[np.arange(n), c["pos_1"] + position_bucket(pos)] = 1.0
    return out


def load_samples(directory: Path = SAMPLES) -> pd.DataFrame:
    days = sorted(directory.glob("day=*"))
    if not days:
        raise SystemExit(f"no samples under {directory}: run `make samples`")
    frames = []
    for d in days:
        df = pd.read_parquet(d)
        df["day"] = d.name.removeprefix("day=")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    names = {tuple(n) for n in df.feature_names}
    if names != {feat.FEATURES}:
        raise SystemExit(f"samples carry features {names}, the ranker has {feat.FEATURES}")
    return df.sort_values(["t_impression", "impression_id"], kind="stable").reset_index(drop=True)


def matrices(df: pd.DataFrame):
    x = np.stack(df.features.map(lambda v: np.asarray([np.nan if f is None else f for f in v], dtype=np.float64)).to_numpy())
    sponsored = (df.slot == "sponsored").to_numpy()
    return design(x, df.pos.to_numpy(), sponsored), df.label.to_numpy().astype(np.int64), df.weight.to_numpy()


class Standardizer:
    def __init__(self, x: np.ndarray, w: np.ndarray):
        idx = [COLUMNS.index(f) for f in CONTINUOUS]
        self.mean = np.zeros(len(COLUMNS))
        self.scale = np.ones(len(COLUMNS))
        sub = x[:, idx]
        m = np.average(sub, axis=0, weights=w)
        s = np.sqrt(np.average((sub - m) ** 2, axis=0, weights=w))
        self.mean[idx] = m
        self.scale[idx] = np.where(s > 0, s, 1.0)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.scale


def fit(x, y, w, c: float):
    from sklearn.linear_model import LogisticRegression

    std = Standardizer(x, w)
    model = LogisticRegression(C=c, max_iter=2000, solver="lbfgs")
    model.fit(std(x), y, sample_weight=w)
    return model, std


def predict(model, std, x) -> np.ndarray:
    return model.predict_proba(std(x))[:, 1]


def weighted_logloss(y, p, w) -> float:
    from sklearn.metrics import log_loss

    return float(log_loss(y, np.clip(p, 1e-12, 1 - 1e-12), sample_weight=w, labels=[0, 1]))


def weighted_auc(y, p, w) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p, sample_weight=w)) if 0 < y.sum() < len(y) else float("nan")


def evaluate(model, std, x, y, w, df) -> dict:
    p = predict(model, std, x)
    base = np.average(y, weights=w)
    by_pos = {}
    buckets = position_bucket(df.pos.to_numpy())
    for b, name in enumerate(POSITION_BUCKETS):
        m = buckets == b
        if m.any():
            by_pos[name] = {
                "impressions": float(w[m].sum()),
                "observed": float(np.average(y[m], weights=w[m])),
                "predicted": float(np.average(p[m], weights=w[m])),
            }
    return {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "logloss": weighted_logloss(y, p, w),
        "logloss_constant": weighted_logloss(y, np.full(len(y), base), w),
        "auc": weighted_auc(y, p, w),
        "observed_ctr": float(base),
        "predicted_ctr": float(np.average(p, weights=w)),
        "by_position": by_pos,
    }


def to_json(model, std, c: float, info: dict) -> dict:
    return {
        "model": "logreg",
        "columns": list(COLUMNS),
        "features": list(feat.FEATURES),
        "continuous": list(CONTINUOUS),
        "reference_position": REFERENCE_POSITION,
        "position_buckets": list(POSITION_BUCKETS),
        "price_edges_cents": list(feat.PRICE_EDGES_CENTS),
        "mean": [float(v) for v in std.mean],
        "scale": [float(v) for v in std.scale],
        "coef": [float(v) for v in model.coef_[0]],
        "intercept": float(model.intercept_[0]),
        "C": c,
        "training": info,
    }


def predict_json(m: dict, x: np.ndarray, pos: np.ndarray, sponsored: np.ndarray) -> np.ndarray:
    """The shipped model from its JSON (what the model server evaluates)."""
    z = (design(x, pos, sponsored) - np.asarray(m["mean"])) / np.asarray(m["scale"])
    return 1.0 / (1.0 + np.exp(-(z @ np.asarray(m["coef"]) + m["intercept"])))


def main() -> None:
    t0 = time.monotonic()
    df = load_samples()
    x, y, w = matrices(df)
    n = len(df)
    a, b = int(n * 0.64), int(n * 0.80)
    trials = []
    for c in C_GRID:
        model, std = fit(x[:a], y[:a], w[:a], c)
        p = predict(model, std, x[a:b])
        trials.append({"C": c, "valid_logloss": weighted_logloss(y[a:b], p, w[a:b])})
    best = min(trials, key=lambda t: t["valid_logloss"])["C"]
    model, std = fit(x[:b], y[:b], w[:b], best)
    held_out = evaluate(model, std, x[b:], y[b:], w[b:], df.iloc[b:])
    # Position-only baseline on the same split: how much do the other features add?
    pos_cols = [COLUMNS.index(c) for c in ("sponsored", *POSITION_BUCKETS)]
    from sklearn.linear_model import LogisticRegression

    base = LogisticRegression(C=best, max_iter=2000).fit(x[:b][:, pos_cols], y[:b], sample_weight=w[:b])
    p_base = base.predict_proba(x[b:][:, pos_cols])[:, 1]
    held_out["logloss_position_only"] = weighted_logloss(y[b:], p_base, w[b:])
    held_out["auc_position_only"] = weighted_auc(y[b:], p_base, w[b:])

    final, final_std = fit(x, y, w, best)
    coef = dict(zip(COLUMNS, final.coef_[0] / final_std.scale, strict=True))  # per raw unit
    position_effect = {name: float(coef[name] - coef["pos_1"]) for name in POSITION_BUCKETS}
    info = {
        "days": sorted(df.day.unique().tolist()),
        "rows": int(n),
        "positives": int(y.sum()),
        "weighted_impressions": float(w.sum()),
        "split": {"train": a, "valid": b - a, "test": n - b, "by": "impression time"},
        "c_grid": trials,
        "selected_C": best,
        "held_out": held_out,
        "position_log_odds_vs_pos1": position_effect,
        "seconds": round(time.monotonic() - t0, 1),
    }
    doc = to_json(final, final_std, best, info)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model.json").write_text(json.dumps(doc, indent=2) + "\n")
    print(
        f"pctr     {n:,} rows ({int(y.sum()):,} clicks) from {', '.join(info['days'])}; C {best}; held-out "
        f"AUC {held_out['auc']:.3f} (position only {held_out['auc_position_only']:.3f}), log loss "
        f"{held_out['logloss']:.4f} vs constant {held_out['logloss_constant']:.4f}; predicted CTR "
        f"{held_out['predicted_ctr']:.4f} vs observed {held_out['observed_ctr']:.4f} -> {OUT.relative_to(paths.ROOT)}"
    )


if __name__ == "__main__":
    main()
