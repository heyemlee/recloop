"""Train and tune the LightGBM ranker on the validation split: `make train-ranker`.

  1. The recall models are fitted on the train rows only: Pop, ItemCF (ml.eval.run's grid,
     chosen on the validation split exactly as `make eval` chooses it) and SASRec (the
     checkpoint of `make train`).
  2. RRF with the control weights fuses their top-200 lists into 300 candidates per validation
     sample (ml.ranking.candidates), and ml.ranking.features describes each candidate.
  3. Grid search with 4-fold cross-validation over the validation users (a user's samples stay
     in one fold). Each configuration trains on three folds with early stopping on the fourth
     (LightGBM's NDCG@10 over its queries that have the target among the candidates); its
     score is the mean held-out NDCG@10 under the evaluation's rule: every held-out sample,
     all items ranked, a target outside the candidates counts as a miss.
  4. The chosen configuration is refitted on all validation queries with the mean of its
     folds' best numbers of rounds.

The test split is never touched here. CPU, one thread (LightGBM is built without OpenMP),
fixed seed: re-runs give the same model. Writes artifacts/models/ranker/{model.txt,tuning.json}.

Usage: python -m ml.models.train_ranker
"""

from __future__ import annotations

import hashlib
import itertools
import json
import time

import numpy as np

from ml import paths
from ml.eval import metrics
from ml.eval.splits import NUM_ITEMS, load_split, train_sequences
from ml.models import train_sasrec
from ml.models.itemcf import ItemCF
from ml.models.pop import Pop
from ml.models.ranker import LambdaRank, RankerParams
from ml.models.rrf import RRF, dense
from ml.models.sasrec import SASRec
from ml.ranking import candidates as cand_mod
from ml.ranking import features as feat

OUT = paths.ARTIFACTS / "models" / "ranker"
FOLDS = 4
FOLD_SEED = 20261108
GRID = {"num_leaves": (7, 15, 31), "min_data_in_leaf": (100, 300), "feature_fraction": (0.7, 1.0)}
SELECT_BY = "NDCG@10"


def recall_models(valid, ev) -> tuple[dict, dict]:
    """Pop, ItemCF (tuned as in make eval) and SASRec, all fitted on the train rows."""
    from ml.eval import run

    sequences = train_sequences(load_split("train"))
    pop = Pop().fit(sequences, NUM_ITEMS)
    best, _ = run.tune_itemcf(sequences, valid, ev)
    itemcf = ItemCF(best).fit(sequences, NUM_ITEMS)
    sasrec = SASRec.load(train_sasrec.CHECKPOINT)
    return {"sasrec": sasrec, "itemcf": itemcf, "pop": pop}, {"itemcf_params": best.__dict__}


def fold_of_users(users: np.ndarray, folds: int = FOLDS, seed: int = FOLD_SEED) -> np.ndarray:
    """Fold of every sample; the users are shuffled once with a fixed seed and dealt out."""
    uniq = np.unique(users)
    perm = np.random.default_rng(seed).permutation(len(uniq))
    fold_of = np.empty(len(uniq), dtype=np.int64)
    fold_of[perm] = np.arange(len(uniq)) % folds
    return fold_of[np.searchsorted(uniq, users)]


def _dataset(lgb, x, y, groups):
    return lgb.Dataset(x, label=y, group=groups, feature_name=list(feat.FEATURES), free_raw_data=False)


def _held_out(booster, rounds, x_q, items, split_target, tie_pos) -> dict:
    """Item-rule HR@10 / NDCG@10 over every sample of a fold (misses included)."""
    n, depth, f = x_q.shape
    s = booster.predict(x_q.reshape(-1, f), raw_score=True, num_iteration=rounds).reshape(n, depth)
    scores = dense(items, np.where(items >= 0, s, np.nan))
    r = metrics.ranks_of(scores, split_target, tie_pos)
    return metrics.summarize(metrics.per_sample(r, (10,)))


def main() -> None:
    import lightgbm as lgb

    t0 = time.monotonic()
    if not train_sasrec.CHECKPOINT.exists():
        raise SystemExit("no SASRec checkpoint: run `make train` first")
    from ml.eval.run import Evaluator

    train = load_split("train")
    valid = load_split("valid")
    pop_counts = Pop().fit(train_sequences(train), NUM_ITEMS).counts
    tie_order = np.lexsort((np.arange(NUM_ITEMS), -pop_counts))
    ev = Evaluator(tie_order)
    sources, recall_info = recall_models(valid, ev)
    rrf = RRF(sources, cand_mod.CONTROL_WEIGHTS)
    tables = feat.ItemTables.build(pop_counts)

    c = rrf.candidates(valid)
    x_all = feat.compute(c, valid.history, tables)  # (n, depth, F)
    has_pos = ((c.items == valid.target[:, None]) & c.valid()).any(axis=1)
    fold = fold_of_users(valid.user)
    t_features = time.monotonic() - t0
    print(
        f"valid    {len(valid):,} samples, target among the candidates in {has_pos.mean():.1%}; "
        f"features in {t_features:.0f}s"
    )

    def subset(mask: np.ndarray):
        from ml.eval.splits import Split

        idx = np.flatnonzero(mask)
        sub = Split(valid.name, valid.user[idx], valid.target[idx], [valid.history[i] for i in idx])
        return idx, sub

    keys = list(GRID)
    trials = []
    for combo in itertools.product(*GRID.values()):
        params = RankerParams(**dict(zip(keys, combo, strict=True)))
        folds_out = []
        for k in range(FOLDS):
            tr_idx, tr = subset((fold != k) & has_pos)
            va_idx, va = subset((fold == k) & has_pos)
            all_idx, everyone = subset(fold == k)
            t = time.monotonic()
            xt, yt, gt, _, _ = _rows(x_all[tr_idx], c.items[tr_idx], tr.target)
            xv, yv, gv, _, _ = _rows(x_all[va_idx], c.items[va_idx], va.target)
            dtrain = _dataset(lgb, xt, yt, gt)
            dvalid = _dataset(lgb, xv, yv, gv)
            booster = lgb.train(
                params.lgb(),
                dtrain,
                num_boost_round=params.max_rounds,
                valid_sets=[dvalid],
                callbacks=[lgb.early_stopping(params.early_stopping, verbose=False)],
            )
            held = _held_out(booster, booster.best_iteration, x_all[all_idx], c.items[all_idx], everyone.target, ev.tie_pos)
            folds_out.append(
                {
                    "fold": k,
                    "best_iteration": int(booster.best_iteration),
                    "lgb_ndcg@10": float(booster.best_score["valid_0"]["ndcg@10"]),
                    "held_out": held,
                    "seconds": round(time.monotonic() - t, 1),
                }
            )
            print(
                f"  {params.label():45s} fold {k}: {booster.best_iteration:4d} rounds, "
                f"held-out NDCG@10 {held['NDCG@10']:.4f}  ({folds_out[-1]['seconds']:.0f}s)"
            )
        mean = {m: float(np.mean([f["held_out"][m] for f in folds_out])) for m in folds_out[0]["held_out"]}
        trials.append({"params": params.__dict__, "folds": folds_out, "cv": mean})
    best = max(trials, key=lambda t: (t["cv"][SELECT_BY], t["cv"]["HR@10"]))
    params = RankerParams(**best["params"])
    rounds = max(1, int(round(np.mean([f["best_iteration"] for f in best["folds"]]))))

    idx, sub = subset(has_pos)
    x, y, g, _, _ = _rows(x_all[idx], c.items[idx], sub.target)
    t = time.monotonic()
    booster = lgb.train(params.lgb(), _dataset(lgb, x, y, g), num_boost_round=rounds)
    gain = booster.feature_importance(importance_type="gain")
    split_counts = booster.feature_importance(importance_type="split")
    tuning = {
        "features": list(feat.FEATURES),
        "params": params.__dict__,
        "rounds": rounds,
        "grid": {k: list(v) for k, v in GRID.items()},
        "folds": FOLDS,
        "fold_seed": FOLD_SEED,
        "select_by": SELECT_BY,
        "trials": trials,
        "selected": trials.index(best),
        "train": {
            "samples": len(valid),
            "queries": int(has_pos.sum()),
            "target_in_candidates": float(has_pos.mean()),
            "rows": int(len(y)),
            "seconds": round(time.monotonic() - t, 1),
        },
        "importance": {
            f: {"gain": float(gv), "splits": int(sv)}
            for f, gv, sv in zip(feat.FEATURES, gain, split_counts, strict=True)
        },
        "recall": recall_info | {"sasrec_sha256": _sha256(train_sasrec.CHECKPOINT), "weights": cand_mod.CONTROL_WEIGHTS},
        "lightgbm": lgb.__version__,
        "seconds": round(time.monotonic() - t0, 1),
    }
    model = LambdaRank(rrf, tables, booster, params, rounds)
    model.save(OUT, tuning)
    print(
        f"ranker   {params.label()}, {rounds} trees, CV NDCG@10 {best['cv']['NDCG@10']:.4f} "
        f"(HR@10 {best['cv']['HR@10']:.4f}); {tuning['seconds']:.0f}s -> {OUT.relative_to(paths.ROOT)}"
    )


def _rows(x_q: np.ndarray, items: np.ndarray, target: np.ndarray):
    """Flatten (queries, depth, F) to LightGBM rows, dropping padded slots."""
    valid = items >= 0
    rows, slots = np.nonzero(valid)
    y = (items == target[:, None])[rows, slots].astype(np.float32)
    return x_q[rows, slots], y, valid.sum(axis=1), rows, slots


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


if __name__ == "__main__":
    main()
