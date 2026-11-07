"""Offline evaluation: fit baselines on the train split, tune on valid, score test once.

  * Pop and ItemCF are fitted on the training rows only (see splits.train_sequences).
  * ItemCF's hyper-parameters are chosen on the validation split by NDCG@10.
  * The test split (4,866 samples) is ranked once per model, over all 3,459 items.
  * HR / NDCG at K = 3, 5, 10 under the item rule (main) and the SID rule; 95% intervals from
    a user-level bootstrap (B = 1000, fixed seed); paired ItemCF - Pop difference.
  * The SID-rule numbers are cross-checked against MiniOneRec's calc.py.

Writes docs/results/offline.md (rendered by ml.eval.report) and artifacts/eval/offline.json.
Does not touch the database.

Usage: python -m ml.eval.run
"""

from __future__ import annotations

import itertools
import json
import time

import numpy as np
import pandas as pd

from ml import paths
from ml.data import preprocess
from ml.eval import bootstrap, calc_check, metrics, report
from ml.eval.splits import NUM_ITEMS, Split, item_sids, load_split, sid_codes, train_sequences
from ml.models.itemcf import ItemCF, ItemCFParams, cooccurrence
from ml.models.pop import Pop

EVAL_DIR = paths.ARTIFACTS / "eval"

ITEMCF_GRID = {
    "beta": (0.0, 0.1, 0.2, 0.3, 0.5, 0.7),
    "history_decay": (1.0, 0.8, 0.6, 0.4),
    "proximity_decay": (1.0, 0.7, 0.4),
    "iuf": (True, False),
}
SELECT_BY = "NDCG@10"


class Evaluator:
    """Scores a model on a split under both hit rules, with a fixed tie order."""

    def __init__(self, tie_order: np.ndarray):
        self.tie_order = tie_order
        self.tie_pos = np.empty_like(tie_order)
        self.tie_pos[tie_order] = np.arange(len(tie_order))
        self.groups = metrics.sid_group_matrix(sid_codes())

    def ranks(self, scores: np.ndarray, split: Split) -> dict[str, np.ndarray]:
        return {
            "item": metrics.ranks_of(scores, split.target, self.tie_pos),
            "sid": metrics.sid_ranks(scores, split.target, self.tie_pos, self.groups),
        }

    def per_sample(self, scores: np.ndarray, split: Split, ks=metrics.KS) -> dict:
        return {rule: metrics.per_sample(r, ks) for rule, r in self.ranks(scores, split).items()}


def _point(values: dict[str, np.ndarray]) -> dict[str, float]:
    return metrics.summarize(values)


def _with_ci(values: dict[str, np.ndarray], resampler: bootstrap.Resampler) -> dict:
    return {
        name: {"value": float(v.mean()), "ci": resampler.interval(v)} for name, v in values.items()
    }


def same_day_as_last(split: Split) -> np.ndarray:
    """True where the target was reviewed on the same day as the last history item.

    Uses the re-run's timestamps (data/processed); row i of the published CSV is sample i of
    the split there (verified by `make data`). Review times are day-granular, and inside a day
    the upstream sequence order is the order of the raw file, not a real time order.
    """
    samples = pd.read_parquet(paths.SAMPLES_PARQUET)
    samples = samples[samples.split == split.name].sort_values("row")
    assert np.array_equal(samples.item_id.to_numpy(), split.target)
    inter = pd.read_parquet(paths.INTERACTIONS_PARQUET, columns=["user_id", "seq_idx", "ts"])
    ts = dict(zip(zip(inter.user_id, inter.seq_idx, strict=True), inter.ts, strict=True))
    last = np.array([ts[(u, q - 1)] for u, q in zip(samples.user_id, samples.seq_idx, strict=True)])
    return last == samples.ts.to_numpy()


def segment_results(split: Split, test_values: dict) -> dict:
    mask = same_day_as_last(split)
    out = {}
    for label, m in (("same_day", mask), ("later_day", ~mask)):
        resampler = bootstrap.Resampler.build(split.user[m])
        out[label] = {
            "samples": int(m.sum()),
            "share": float(m.mean()),
            "models": {
                name: _with_ci({k: values["item"][k][m] for k in ("HR@10", "NDCG@10")}, resampler)
                for name, values in test_values.items()
            },
        }
    return out


def tune_itemcf(sequences, valid: Split, ev: Evaluator) -> tuple[ItemCFParams, list[dict]]:
    cooc_cache = {}
    trials = []
    keys = list(ITEMCF_GRID)
    for combo in itertools.product(*ITEMCF_GRID.values()):
        params = ItemCFParams(**dict(zip(keys, combo, strict=True)))
        ck = (params.proximity_decay, params.iuf)
        if ck not in cooc_cache:
            cooc_cache[ck] = cooccurrence(sequences, NUM_ITEMS, *ck)
        model = ItemCF(params).fit(sequences, NUM_ITEMS, cooc=cooc_cache[ck])
        values = metrics.per_sample(metrics.ranks_of(model.score(valid), valid.target, ev.tie_pos))
        trials.append({"params": params.__dict__, "valid": _point(values)})
    best = max(trials, key=lambda t: (t["valid"][SELECT_BY], t["valid"]["HR@10"]))
    return ItemCFParams(**best["params"]), trials


def main() -> None:
    t0 = time.monotonic()
    train, valid, test = (load_split(s) for s in paths.SPLITS)
    sequences = train_sequences(train)
    sids = item_sids()

    pop = Pop().fit(sequences, NUM_ITEMS)
    # Ties: training popularity, then item_id.
    tie_order = np.lexsort((np.arange(NUM_ITEMS), -pop.counts))
    ev = Evaluator(tie_order)

    best_params, trials = tune_itemcf(sequences, valid, ev)
    itemcf = ItemCF(best_params).fit(sequences, NUM_ITEMS)
    models = [pop, itemcf]

    resampler = bootstrap.Resampler.build(test.user)
    results, test_values, calc_checks = {}, {}, []
    for model in models:
        valid_scores = model.score(valid)
        valid_values = ev.per_sample(valid_scores, valid)
        scores = model.score(test)
        values = ev.per_sample(scores, test)
        test_values[model.name] = values
        results[model.name] = {
            "description": model.describe(),
            "valid": {rule: _point(v) for rule, v in valid_values.items()},
            "test": {rule: _with_ci(v, resampler) for rule, v in values.items()},
        }
        # calc.py cross-check on the SID rule, K = 1..50.
        top = metrics.rank_items(np.asarray(scores), tie_order)[:, : calc_check.TOP_N]
        item_rank = metrics.target_ranks(top, test.target)
        full_rank = metrics.ranks_of(scores, test.target, ev.tie_pos)
        assert np.array_equal(item_rank, np.minimum(full_rank, calc_check.TOP_N + 1))
        sid_values = metrics.summarize(
            metrics.per_sample(ev.ranks(scores, test)["sid"], metrics.CALC_KS)
        )
        calc_checks.append(calc_check.check(model.name, test, top, sid_values, sids, EVAL_DIR))

    base = test_values[pop.name]
    diff = {
        rule: _with_ci(
            {k: test_values[itemcf.name][rule][k] - base[rule][k] for k in base[rule]}, resampler
        )
        for rule in base
    }

    verify = json.loads(paths.VERIFY_JSON.read_text())
    window = json.loads(preprocess.RUN_JSON.read_text())
    out = {
        "setup": {
            "num_items": NUM_ITEMS,
            "window": {k: window[k] for k in ("tz", "final_start", "final_end")},
            "samples": {s: len(load_split(s)) for s in paths.SPLITS},
            "test_users": int(np.unique(test.user).size),
            "train_users": len(sequences),
            "train_interactions": int(sum(len(s) for s in sequences.values())),
            "ks": list(metrics.KS),
            "bootstrap": {"B": bootstrap.B, "seed": bootstrap.SEED, "alpha": bootstrap.ALPHA},
            "itemcf_grid": {k: list(v) for k, v in ITEMCF_GRID.items()},
            "select_by": SELECT_BY,
        },
        "models": results,
        "itemcf_minus_pop": diff,
        "segments": segment_results(test, test_values),
        "itemcf_trials": trials,
        "calc_py": [{k: v for k, v in c.items() if k != "stdout"} for c in calc_checks],
        "verify": verify,
    }
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "offline.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    (paths.RESULTS / "offline.md").write_text(report.render(out))

    for name, r in results.items():
        t = r["test"]["item"]
        print(
            f"{name:7s} HR@10 {t['HR@10']['value']:.4f} "
            f"[{t['HR@10']['ci'][0]:.4f}, {t['HR@10']['ci'][1]:.4f}]  "
            f"NDCG@10 {t['NDCG@10']['value']:.4f} "
            f"[{t['NDCG@10']['ci'][0]:.4f}, {t['NDCG@10']['ci'][1]:.4f}]  ({r['description']})"
        )
    for c in calc_checks:
        print(
            f"calc.py {c['model']}: {'match' if c['ok'] else 'MISMATCH'} "
            f"(max |diff| {c['max_abs_diff']:.1e}, "
            f"invalid predictions {c['invalid_predictions']})"
        )
    print(f"wrote docs/results/offline.md in {time.monotonic() - t0:.0f}s")
    if not all(c["ok"] for c in calc_checks):
        raise SystemExit("calc.py cross-check FAILED")


if __name__ == "__main__":
    main()
