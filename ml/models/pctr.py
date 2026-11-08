"""pCTR of the sponsored slots: logistic regression on the logged impressions (design §9.2 step 3).

Training data is the sample job's output (ml.samples.job, artifacts/samples/train): home-feed
impressions whose window has closed, with the features the served event recorded and the
negatives downsampled (weight = 1 / rate restores them). The columns of a row:

    sasrec_score (0 when missing) and a missing flag   SASRec's relevance of the item
    pop_log, n_sources                                 popularity, how many sources had it
    cat2_overlap, leaf_overlap (0 when missing)        category overlap with the history
    price bucket, one-hot (6 buckets + unknown)
    sponsored (0/1)                                    the slot type
    position, one-hot (1 .. 10, 11-20, 21+)

and, in the "ctr" variant, the item's recent CTR: its impressions and clicks in the complete
buckets of the last hour and the last day before the api's table time (serving/internal/itemctr;
the values the served event logged, or the sample job's recomputation from the log for samples
served before the api logged them):

    ictr_1h_logit, ictr_24h_logit      (clicks + strength x prior) / (impressions + strength),
                                       in log-odds; prior = the training rows' CTR
    ictr_1h_imp_log, ictr_24h_imp_log  log(1 + impressions)

Missing counts are 0 impressions and 0 clicks: the prior. The position is a training feature
only: at prediction it is fixed at the reference position (REFERENCE_POSITION, the first
sponsored slot), so every ad is scored as if shown at the same place and the auction compares
relevance, not where an ad happened to be shown before.

Model selection: the samples in time order are split 64 / 16 / 20; C (L2 strength) and, with the
CTR columns, the prior's strength are chosen on the middle part by weighted log loss; the model
refitted on the first 80 % is reported on the last 20 % (the future); the shipped model is
refitted on everything with those settings. With counts in the samples both variants are judged
on the same split (AUC, weighted log loss, calibration by position and by decile of the
prediction; on every sample and on the sponsored slots alone; user-level paired bootstrap
intervals of the differences) and the "ctr" variant ships; without, "base" does.

Usage: python -m ml.models.pctr [--days D ...] [--since T] [--until T] [--label NAME]   (after
`make samples`; --since / --until keep the samples served in [since, until); writes
artifacts/models/pctr/model.json and artifacts/models/pctr/eval/<label>.json)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ml import paths
from ml.ranking import features as feat

SAMPLES = paths.ARTIFACTS / "samples" / "train"
OUT = paths.ARTIFACTS / "models" / "pctr"
REFERENCE_POSITION = 3
POSITION_BUCKETS = tuple(f"pos_{p}" for p in range(1, 11)) + ("pos_11_20", "pos_21_up")
PRICE_COLUMNS = tuple(f"price_{b}" for b in range(len(feat.PRICE_EDGES_CENTS) + 1)) + (
    "price_unknown",
)
CONTINUOUS = ("sasrec_score", "pop_log", "n_sources", "cat2_overlap", "leaf_overlap")
COLUMNS = (*CONTINUOUS, "sasrec_missing", *PRICE_COLUMNS, "sponsored", *POSITION_BUCKETS)
# The item CTR columns (model_server/ranking.py CTR_COLUMNS) from the counts [impressions, clicks]
# of the hour, then of the day (the served event's `ictr`).
CTR_COLUMNS = ("ictr_1h_logit", "ictr_1h_imp_log", "ictr_24h_logit", "ictr_24h_imp_log")
CTR_INPUTS = ("impressions_hour", "clicks_hour", "impressions_day", "clicks_day")
CTR_RATE_CLIP = 1e-6
COLUMNS_CTR = (
    *CONTINUOUS,
    *CTR_COLUMNS,
    "sasrec_missing",
    *PRICE_COLUMNS,
    "sponsored",
    *POSITION_BUCKETS,
)
C_GRID = (0.01, 0.1, 1.0, 10.0)
STRENGTH_GRID = (10.0, 100.0, 1000.0)
SEED = 20261108
BOOTSTRAP = 1000
BOOTSTRAP_SEED = 20261108
DECILES = 10


@dataclass(frozen=True)
class Variant:
    name: str
    columns: tuple[str, ...]
    continuous: tuple[str, ...]

    @property
    def uses_ctr(self) -> bool:
        return any(c in CTR_COLUMNS for c in self.columns)


BASE = Variant("base", COLUMNS, CONTINUOUS)
CTR = Variant("ctr", COLUMNS_CTR, (*CONTINUOUS, *CTR_COLUMNS))


def position_bucket(pos: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.int64)
    return np.where(pos <= 10, pos - 1, np.where(pos <= 20, 10, 11))


def ctr_columns(counts: np.ndarray, prior: float, strength: float) -> dict[str, np.ndarray]:
    """The CTR columns of counts (n, 4) = [impressions, clicks] of the hour and the day; NaN
    (missing) counts are 0. The model server computes the same (model_server/ranking.py)."""
    c = np.nan_to_num(np.asarray(counts, dtype=np.float64).reshape(-1, 4), nan=0.0)
    out = {}
    for name, i in (("1h", 0), ("24h", 2)):
        imp, clk = c[:, i], c[:, i + 1]
        rate = np.clip(
            (clk + strength * prior) / (imp + strength), CTR_RATE_CLIP, 1 - CTR_RATE_CLIP
        )
        out[f"ictr_{name}_logit"] = np.log(rate) - np.log1p(-rate)
        out[f"ictr_{name}_imp_log"] = np.log1p(imp)
    return out


def design(
    x: np.ndarray,
    pos: np.ndarray,
    sponsored: np.ndarray,
    counts: np.ndarray | None = None,
    ctr: dict | None = None,
    columns: tuple[str, ...] = COLUMNS,
) -> np.ndarray:
    """Rows of the ranker's feature vector (feat.FEATURES order), positions, slots and the items'
    counts -> the model's columns by name, before standardisation. The model server builds the
    same matrix (model_server/ranking.py PCTRModel.design)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    col = {f: x[:, i] for i, f in enumerate(feat.FEATURES)}
    if any(c in CTR_COLUMNS for c in columns):
        if ctr is None:
            raise ValueError("the CTR columns need the prior and its strength")
        c = np.full((n, 4), np.nan) if counts is None else counts
        col.update(ctr_columns(c, ctr["prior"], ctr["strength"]))
    bucket = position_bucket(pos)
    price = col["price_bucket"]
    out = np.zeros((n, len(columns)))
    for j, name in enumerate(columns):
        if name in CONTINUOUS or name in CTR_COLUMNS:
            out[:, j] = np.nan_to_num(col[name], nan=0.0)
        elif name == "sasrec_missing":
            out[:, j] = np.isnan(col["sasrec_score"])
        elif name == "price_unknown":
            out[:, j] = np.isnan(price)
        elif name.startswith("price_"):
            out[:, j] = price == int(name.removeprefix("price_"))
        elif name == "sponsored":
            out[:, j] = np.asarray(sponsored, dtype=np.float64)
        elif name == "pos_11_20":
            out[:, j] = bucket == 10
        elif name == "pos_21_up":
            out[:, j] = bucket == 11
        elif name.startswith("pos_"):
            out[:, j] = bucket == int(name.removeprefix("pos_")) - 1
        else:
            raise ValueError(f"unknown column {name!r}")
    return out


def utc(t: str) -> pd.Timestamp:
    """An ISO time; one without a zone is UTC."""
    ts = pd.Timestamp(t)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def load_samples(
    directory: Path | None = None,
    days: list[str] | None = None,
    since: str | None = None,
    until: str | None = None,
) -> pd.DataFrame:
    """The training samples of the given days (default: all), served in [since, until)."""
    directory = directory or SAMPLES
    found = sorted(directory.glob("day=*"))
    if days:
        found = [d for d in found if d.name.removeprefix("day=") in set(days)]
    if not found:
        raise SystemExit(
            f"no samples under {directory} for {days or 'any day'}: run `make samples`"
        )
    frames = []
    for d in found:
        df = pd.read_parquet(d)
        df["day"] = d.name.removeprefix("day=")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    t = pd.to_datetime(df.t_impression, utc=True)
    keep = np.ones(len(df), dtype=bool)
    if since:
        keep &= (t >= utc(since)).to_numpy()
    if until:
        keep &= (t < utc(until)).to_numpy()
    df = df[keep]
    if df.empty:
        raise SystemExit(f"no samples served in [{since}, {until})")
    names = {tuple(n) for n in df.feature_names}
    if names != {feat.FEATURES}:
        raise SystemExit(f"samples carry features {names}, the ranker has {feat.FEATURES}")
    return df.sort_values(["t_impression", "impression_id"], kind="stable").reset_index(drop=True)


@dataclass
class Data:
    x: np.ndarray  # (n, F) the ranker's features, NaN = missing
    pos: np.ndarray
    sponsored: np.ndarray
    counts: np.ndarray | None  # (n, 4), NaN = missing; None: the samples have none
    y: np.ndarray
    w: np.ndarray

    def part(self, s: slice | np.ndarray) -> Data:
        return Data(
            self.x[s],
            self.pos[s],
            self.sponsored[s],
            None if self.counts is None else self.counts[s],
            self.y[s],
            self.w[s],
        )


def has_counts(df: pd.DataFrame) -> bool:
    return "ictr" in df.columns and bool(df.ictr.notna().all())


def matrices(df: pd.DataFrame) -> Data:
    x = np.stack(
        df.features.map(
            lambda v: np.asarray([np.nan if f is None else f for f in v], dtype=np.float64)
        ).to_numpy()
    )
    counts = None
    if has_counts(df):
        counts = np.stack(df.ictr.map(lambda v: np.asarray(v, dtype=np.float64)).to_numpy())
    return Data(
        x,
        df.pos.to_numpy(),
        (df.slot == "sponsored").to_numpy(),
        counts,
        df.label.to_numpy().astype(np.int64),
        df.weight.to_numpy(),
    )


class Standardizer:
    def __init__(self, x: np.ndarray, w: np.ndarray, variant: Variant = BASE):
        idx = [variant.columns.index(f) for f in variant.continuous]
        self.mean = np.zeros(len(variant.columns))
        self.scale = np.ones(len(variant.columns))
        sub = x[:, idx]
        m = np.average(sub, axis=0, weights=w)
        s = np.sqrt(np.average((sub - m) ** 2, axis=0, weights=w))
        self.mean[idx] = m
        self.scale[idx] = np.where(s > 0, s, 1.0)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.scale


@dataclass
class Fitted:
    variant: Variant
    model: object
    std: Standardizer
    c: float
    ctr: dict | None  # {"prior", "strength"}

    def matrix(self, d: Data) -> np.ndarray:
        return design(d.x, d.pos, d.sponsored, d.counts, self.ctr, self.variant.columns)

    def predict(self, d: Data) -> np.ndarray:
        return self.model.predict_proba(self.std(self.matrix(d)))[:, 1]


def fit(d: Data, c: float, variant: Variant = BASE, strength: float | None = None) -> Fitted:
    from sklearn.linear_model import LogisticRegression

    ctr = None
    if variant.uses_ctr:
        ctr = {"prior": float(np.average(d.y, weights=d.w)), "strength": float(strength)}
    x = design(d.x, d.pos, d.sponsored, d.counts, ctr, variant.columns)
    std = Standardizer(x, d.w, variant)
    model = LogisticRegression(C=c, max_iter=2000, solver="lbfgs")
    model.fit(std(x), d.y, sample_weight=d.w)
    return Fitted(variant, model, std, c, ctr)


def weighted_logloss(y, p, w) -> float:
    from sklearn.metrics import log_loss

    return float(log_loss(y, np.clip(p, 1e-12, 1 - 1e-12), sample_weight=w, labels=[0, 1]))


def weighted_auc(y, p, w) -> float:
    from sklearn.metrics import roc_auc_score

    pos = (w * (y == 1)).sum()
    neg = (w * (y == 0)).sum()
    return float(roc_auc_score(y, p, sample_weight=w)) if pos > 0 and neg > 0 else float("nan")


def calibration(p, y, w, buckets: np.ndarray, names) -> dict:
    out = {}
    for b, name in enumerate(names):
        m = buckets == b
        if m.any() and w[m].sum() > 0:
            out[name] = {
                "impressions": float(w[m].sum()),
                "observed": float(np.average(y[m], weights=w[m])),
                "predicted": float(np.average(p[m], weights=w[m])),
            }
    return out


def deciles(p: np.ndarray) -> np.ndarray:
    edges = np.quantile(p, np.linspace(0, 1, DECILES + 1)[1:-1])
    return np.searchsorted(edges, p, side="right")


def evaluate(p: np.ndarray, d: Data) -> dict:
    y, w = d.y, d.w
    base = np.average(y, weights=w)
    by_decile = calibration(p, y, w, deciles(p), [f"d{k + 1}" for k in range(DECILES)])
    ece = sum(abs(v["predicted"] - v["observed"]) * v["impressions"] for v in by_decile.values())
    return {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "weighted_impressions": float(w.sum()),
        "logloss": weighted_logloss(y, p, w),
        "logloss_constant": weighted_logloss(y, np.full(len(y), base), w),
        "auc": weighted_auc(y, p, w),
        "observed_ctr": float(base),
        "predicted_ctr": float(np.average(p, weights=w)),
        "calibration_error": float(ece / w.sum()),
        "by_position": calibration(p, y, w, position_bucket(d.pos), POSITION_BUCKETS),
        "by_decile": by_decile,
    }


def paired_bootstrap(users, d: Data, p_a, p_b, mask=None) -> dict:
    """AUC and log loss of b minus a, resampling users (B = BOOTSTRAP, fixed seed): the user is
    the unit, a user's impressions are not independent."""
    m = np.ones(len(d.y), dtype=bool) if mask is None else mask
    codes, uniq = pd.factorize(np.asarray(users)[m])
    y, w, a, b = d.y[m], d.w[m], p_a[m], p_b[m]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    auc, ll = [], []
    for _ in range(BOOTSTRAP):
        mult = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        ww = w * mult[codes]
        auc.append(weighted_auc(y, b, ww) - weighted_auc(y, a, ww))
        ll.append(weighted_logloss(y, b, ww) - weighted_logloss(y, a, ww))

    def summary(point, draws):
        lo, hi = np.nanpercentile(draws, [2.5, 97.5])
        return {"diff": float(point), "ci95": [float(lo), float(hi)]}

    return {
        "users": int(len(uniq)),
        "rows": int(m.sum()),
        "resamples": BOOTSTRAP,
        "auc": summary(weighted_auc(y, b, w) - weighted_auc(y, a, w), auc),
        "logloss": summary(weighted_logloss(y, b, w) - weighted_logloss(y, a, w), ll),
    }


def select(d: Data, a: int, b: int, variant: Variant) -> tuple[dict, list[dict]]:
    """C (and the prior's strength) by weighted log loss on rows [a, b), fitted on [0, a)."""
    trials = []
    train, valid = d.part(slice(0, a)), d.part(slice(a, b))
    for strength in STRENGTH_GRID if variant.uses_ctr else (None,):
        for c in C_GRID:
            f = fit(train, c, variant, strength)
            trials.append(
                {
                    "C": c,
                    **({"strength": strength} if strength is not None else {}),
                    "valid_logloss": weighted_logloss(valid.y, f.predict(valid), valid.w),
                }
            )
    best = min(trials, key=lambda t: t["valid_logloss"])
    return best, trials


def position_only(d: Data, b: int, c: float) -> dict:
    """Position-only baseline on the same split: how much do the other columns add?"""
    from sklearn.linear_model import LogisticRegression

    cols = [COLUMNS.index(k) for k in ("sponsored", *POSITION_BUCKETS)]
    x = design(d.x, d.pos, d.sponsored)[:, cols]
    m = LogisticRegression(C=c, max_iter=2000).fit(x[:b], d.y[:b], sample_weight=d.w[:b])
    p = m.predict_proba(x[b:])[:, 1]
    return {
        "logloss_position_only": weighted_logloss(d.y[b:], p, d.w[b:]),
        "auc_position_only": weighted_auc(d.y[b:], p, d.w[b:]),
    }


def to_json(f: Fitted, info: dict) -> dict:
    out = {
        "model": "logreg",
        "variant": f.variant.name,
        "columns": list(f.variant.columns),
        "features": list(feat.FEATURES),
        "continuous": list(f.variant.continuous),
        "reference_position": REFERENCE_POSITION,
        "position_buckets": list(POSITION_BUCKETS),
        "price_edges_cents": list(feat.PRICE_EDGES_CENTS),
        "mean": [float(v) for v in f.std.mean],
        "scale": [float(v) for v in f.std.scale],
        "coef": [float(v) for v in f.model.coef_[0]],
        "intercept": float(f.model.intercept_[0]),
        "C": f.c,
        "training": info,
    }
    if f.ctr is not None:
        out["ctr"] = {
            "prior": f.ctr["prior"],
            "strength": f.ctr["strength"],
            "inputs": list(CTR_INPUTS),
            "columns": list(CTR_COLUMNS),
            "rate_clip": CTR_RATE_CLIP,
        }
    return out


def predict_json(
    m: dict, x: np.ndarray, pos: np.ndarray, sponsored: np.ndarray, counts: np.ndarray | None = None
) -> np.ndarray:
    """The shipped model from its JSON (what the model server evaluates)."""
    z = (
        design(x, pos, sponsored, counts, m.get("ctr"), tuple(m["columns"])) - np.asarray(m["mean"])
    ) / np.asarray(m["scale"])
    return 1.0 / (1.0 + np.exp(-(z @ np.asarray(m["coef"]) + m["intercept"])))


def run_variant(d: Data, a: int, b: int, variant: Variant):
    best, trials = select(d, a, b, variant)
    strength = best.get("strength")
    held = fit(d.part(slice(0, b)), best["C"], variant, strength)
    test = d.part(slice(b, len(d.y)))
    p = held.predict(test)
    res = evaluate(p, test)
    sp = test.sponsored.astype(bool)
    res["sponsored"] = evaluate(p[sp], test.part(sp)) if sp.any() else None
    res.update(position_only(d, b, best["C"]))
    final = fit(d, best["C"], variant, strength)
    return {"best": best, "trials": trials, "held_out": res, "p": p, "final": final}


def time_span(df: pd.DataFrame, s: slice) -> list[str]:
    t = pd.to_datetime(df.t_impression.iloc[s], utc=True)
    return [t.min().isoformat(), t.max().isoformat()] if len(t) else []


def coverage(d: Data) -> dict | None:
    """Share of the (weighted) impressions whose item had at least one impression in the hour
    and in the day window: a window with no counts leaves its columns at the prior."""
    if d.counts is None or not len(d.y):
        return None
    c = np.nan_to_num(d.counts, nan=0.0)
    return {
        "hour": float(np.average(c[:, 0] > 0, weights=d.w)),
        "day": float(np.average(c[:, 2] > 0, weights=d.w)),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--days", nargs="*", default=None, help="sample days (default: all)")
    p.add_argument("--label", default=None, help="name of the evaluation file (default: the days)")
    p.add_argument(
        "--since",
        default=None,
        help="only samples served at or after this ISO time (UTC unless it names a zone)",
    )
    p.add_argument("--until", default=None, help="only samples served before this ISO time")
    args = p.parse_args(argv)
    since = utc(args.since).isoformat() if args.since else None
    until = utc(args.until).isoformat() if args.until else None
    t0 = time.monotonic()
    df = load_samples(days=args.days, since=since, until=until)
    d = matrices(df)
    n = len(df)
    a, b = int(n * 0.64), int(n * 0.80)
    variants = [BASE, CTR] if d.counts is not None else [BASE]
    runs = {v.name: run_variant(d, a, b, v) for v in variants}
    shipped = runs[variants[-1].name]
    final = shipped["final"]

    coef = dict(zip(final.variant.columns, final.model.coef_[0] / final.std.scale, strict=True))
    position_effect = {name: float(coef[name] - coef["pos_1"]) for name in POSITION_BUCKETS}
    meta_dir = SAMPLES.parent / "meta"
    days = sorted(df.day.unique().tolist())
    samples_meta = {
        dd: json.loads((meta_dir / f"{dd}.json").read_text())
        for dd in days
        if (meta_dir / f"{dd}.json").exists()
    }
    info = {
        "days": days,
        "since": since,
        "until": until,
        # The sample job's counts for those days, as they were when this model was trained
        # (a later `make samples` of the same day replaces the files).
        "samples_meta": samples_meta,
        "rows": int(n),
        "positives": int(d.y.sum()),
        "weighted_impressions": float(d.w.sum()),
        "split": {"train": a, "valid": b - a, "test": n - b, "by": "impression time"},
        "c_grid": shipped["trials"],
        "selected_C": shipped["best"]["C"],
        "held_out": shipped["held_out"],
        "position_log_odds_vs_pos1": position_effect,
        "seconds": round(time.monotonic() - t0, 1),
    }
    if final.ctr is not None:
        info["selected_strength"] = final.ctr["strength"]
    doc = to_json(final, info)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model.json").write_text(json.dumps(doc, indent=2) + "\n")

    label = args.label or "_".join(days)
    ev = {
        "label": label,
        "days": days,
        "since": since,
        "until": until,
        "rows": int(n),
        "positives": int(d.y.sum()),
        "users": int(df.user_id.nunique()),
        "sponsored_rows": int(d.sponsored.sum()),
        "split": {
            "train": [0, a, *time_span(df, slice(0, a))],
            "valid": [a, b, *time_span(df, slice(a, b))],
            "test": [b, n, *time_span(df, slice(b, n))],
        },
        "counts_source": (
            df.ictr_source.value_counts().to_dict() if "ictr_source" in df.columns else {}
        ),
        "counts_coverage": {"all": coverage(d), "test": coverage(d.part(slice(b, n)))},
        "item_ctr_checks": {dd: m.get("item_ctr") for dd, m in samples_meta.items()},
        "variants": {
            k: {"selected": r["best"], "trials": r["trials"], "held_out": r["held_out"]}
            for k, r in runs.items()
        },
        "shipped": final.variant.name,
    }
    if "ctr" in runs:
        test = d.part(slice(b, n))
        users = df.user_id.to_numpy()[b:]
        pa, pb = runs["base"]["p"], runs["ctr"]["p"]
        ev["comparison"] = {
            "all": paired_bootstrap(users, test, pa, pb),
            "sponsored": paired_bootstrap(users, test, pa, pb, test.sponsored.astype(bool))
            if test.sponsored.any()
            else None,
        }
    (OUT / "eval").mkdir(parents=True, exist_ok=True)
    (OUT / "eval" / f"{label}.json").write_text(json.dumps(ev, indent=2) + "\n")

    held = shipped["held_out"]
    line = (
        f"pctr     {n:,} rows ({int(d.y.sum()):,} clicks) from {', '.join(days)}; {final.variant.name} "
        f"C {final.c}; held-out AUC {held['auc']:.3f} (position only {held['auc_position_only']:.3f}), "
        f"log loss {held['logloss']:.4f} vs constant {held['logloss_constant']:.4f}; predicted CTR "
        f"{held['predicted_ctr']:.4f} vs observed {held['observed_ctr']:.4f}"
    )
    if "comparison" in ev:
        c = ev["comparison"]["all"]
        line += (
            f"; with the CTR columns AUC {c['auc']['diff']:+.4f} [{c['auc']['ci95'][0]:+.4f}, "
            f"{c['auc']['ci95'][1]:+.4f}], log loss {c['logloss']['diff']:+.5f}"
        )
    print(line + f" -> {OUT}")


if __name__ == "__main__":
    main()
