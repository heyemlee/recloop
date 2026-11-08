"""Offline evaluation: fit baselines on the train split, tune on valid, score test once.

  * Pop and ItemCF are fitted on the training rows only (see splits.train_sequences); ItemCF's
    hyper-parameters are chosen on the validation split by NDCG@10.
  * SASRec is trained and tuned by `make train` (ml.models.train_sasrec: train rows only,
    validation NDCG@10 for early stopping and model selection); here the selected checkpoint is
    loaded and scored, together with the BCE ablation trained alongside it.
  * The test split (4,866 samples) is ranked once per model, over all 3,459 items.
  * HR / NDCG at K = 3, 5, 10 under the item rule (main) and the SID rule; 95% intervals from
    a user-level bootstrap (B = 1000, fixed seed); paired differences between models.
  * Every model is also reported separately on test targets reviewed on the same day as the
    last history item and on the rest (51.5% / 48.5%).
  * The SID-rule numbers are cross-checked against MiniOneRec's calc.py for every model.
  * SASRec is compared with the paper's SASRec (MiniOneRec, Table 1) and, when
    `make paper-sasrec` has run, with MiniOneRec's own SASRec script re-run on this split.
  * RRF is the api's ranking (weighted reciprocal rank fusion of the three sources' top 200,
    300 candidates; both variants of the recall experiment). LambdaRank is the LightGBM ranker
    over RRF's candidates, trained on the validation split by `make train-ranker` and loaded
    here (M4).

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
from ml.eval import bootstrap, calc_check, metrics, minionerec_sasrec, report
from ml.eval.splits import NUM_ITEMS, Split, item_sids, load_split, sid_codes, train_sequences
from ml.models import train_sasrec
from ml.models import train_ranker
from ml.models.itemcf import ItemCF, ItemCFParams, cooccurrence
from ml.models.pop import Pop
from ml.models.ranker import LambdaRank
from ml.models.rrf import RRF
from ml.models.sasrec import SASRec
from ml.ranking import candidates as cand_mod
from ml.ranking import features as feat

EVAL_DIR = paths.ARTIFACTS / "eval"

ITEMCF_GRID = {
    "beta": (0.0, 0.1, 0.2, 0.3, 0.5, 0.7),
    "history_decay": (1.0, 0.8, 0.6, 0.4),
    "proximity_decay": (1.0, 0.7, 0.4),
    "iuf": (True, False),
}
SELECT_BY = "NDCG@10"
SEGMENT_METRICS = ("HR@5", "HR@10", "NDCG@5", "NDCG@10")

# MiniOneRec (arXiv:2510.24431v1), Table 1, Office; read from the PDF, page 9.
PAPER = {
    "source": "MiniOneRec, arXiv:2510.24431v1, Table 1 (Office)",
    "rows": {
        "SASRec": {
            "HR@3": 0.0861,
            "NDCG@3": 0.0769,
            "HR@5": 0.0949,
            "NDCG@5": 0.0805,
            "HR@10": 0.1120,
            "NDCG@10": 0.0858,
        },
        "MiniOneRec": {
            "HR@3": 0.1217,
            "NDCG@3": 0.1088,
            "HR@5": 0.1420,
            "NDCG@5": 0.1172,
            "HR@10": 0.1634,
            "NDCG@10": 0.1242,
        },
    },
    "baseline_training": "binary cross-entropy, Adam, lr in {1e-2, 1e-3, 1e-4}, weight decay "
    "in {1e-2, ..., 1e-6}, batch 1024 (paper, Section 4)",
}


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


PAIRS = (
    ("ItemCF", "Pop"),
    ("SASRec", "Pop"),
    ("SASRec", "ItemCF"),
    ("RRF", "SASRec"),
    ("LambdaRank", "RRF"),
    ("LambdaRank", "SASRec"),
)


def segment_results(split: Split, test_values: dict) -> dict:
    mask = same_day_as_last(split)
    out = {}
    for label, m in (("same_day", mask), ("later_day", ~mask)):
        resampler = bootstrap.Resampler.build(split.user[m])
        out[label] = {
            "samples": int(m.sum()),
            "share": float(m.mean()),
            "models": {
                name: _with_ci({k: values["item"][k][m] for k in SEGMENT_METRICS}, resampler)
                for name, values in test_values.items()
            },
            "pairs": {
                f"{a} - {b}": _with_ci(
                    {
                        k: test_values[a]["item"][k][m] - test_values[b]["item"][k][m]
                        for k in ("HR@10", "NDCG@10")
                    },
                    resampler,
                )
                for a, b in PAIRS
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


def load_sasrec() -> tuple[SASRec, dict]:
    if not train_sasrec.CHECKPOINT.exists() or not train_sasrec.TUNING_JSON.exists():
        raise SystemExit("no SASRec checkpoint: run `make train` first")
    return SASRec.load(train_sasrec.CHECKPOINT), json.loads(train_sasrec.TUNING_JSON.read_text())


def sasrec_variants(tuning: dict, test: Split, ev: Evaluator, resampler) -> dict:
    """Test metrics (item rule) of the extra configurations trained next to the grid (the
    upstream BCE loss) and of their CE twins in the grid (same settings, only the loss differs),
    for the comparison with the paper. Model selection never sees these numbers."""
    trials = tuning["trials"]
    n_grid = len(trials) - len(tuning["extra"])

    def scored(i: int) -> dict:
        model = SASRec.load(train_sasrec.trial_checkpoint(i))
        values = ev.per_sample(model.score(test), test)["item"]
        return {
            "trial": i,
            "config": trials[i]["config"],
            "description": model.describe(),
            "valid": trials[i]["valid"],
            "test": _with_ci(values, resampler),
        }

    out = {}
    for j, cfg in enumerate(tuning["extra"]):
        out[f"loss={cfg['loss']}"] = scored(n_grid + j)
        twin = {**cfg, "loss": "ce"}
        k = next((t for t in range(n_grid) if trials[t]["config"] == twin), None)
        if k is not None:
            out[f"loss=ce twin of loss={cfg['loss']}"] = scored(k)
    return out


def load_ranker(rrf: RRF, pop_counts: np.ndarray) -> tuple[LambdaRank, dict]:
    if not (train_ranker.OUT / "model.txt").exists():
        raise SystemExit("no ranker: run `make train-ranker` first")
    return LambdaRank.load(train_ranker.OUT, rrf, feat.ItemTables.build(pop_counts))


def ranking_summary(rrf: RRF, test: Split, tuning: dict) -> dict:
    """What the ranker could reach (target among the candidates) and how it was trained."""
    c = rrf.candidates(test)
    found = ((c.items == test.target[:, None]) & c.valid()).any(axis=1)
    return {
        "candidates": cand_mod.CANDIDATE_DEPTH,
        "source_k": cand_mod.SOURCE_K,
        "rrf_k": cand_mod.RRF_K,
        "weights": {"control": cand_mod.CONTROL_WEIGHTS, "sasrec_heavy": cand_mod.SASREC_HEAVY_WEIGHTS},
        "test_target_in_candidates": float(found.mean()),
        "features": list(feat.FEATURES),
        "tuning": {k: v for k, v in tuning.items() if k != "trials"}
        | {
            "trials": [
                {k: v for k, v in t.items() if k != "folds"}
                | {
                    "rounds": [f["best_iteration"] for f in t["folds"]],
                    "fold_ndcg10": [f["held_out"]["NDCG@10"] for f in t["folds"]],
                }
                for t in tuning["trials"]
            ]
        },
    }


def paper_comparison(test_values: dict, variants: dict, resampler) -> dict:
    out = {"paper": PAPER, "ours": {}, "variants": variants, "minionerec_script": None}
    for name in ("SASRec", "ItemCF"):
        out["ours"][name] = _with_ci(test_values[name]["item"], resampler)
    if minionerec_sasrec.OUT_JSON.exists():
        run = json.loads(minionerec_sasrec.OUT_JSON.read_text())
        run.pop("per_sample", None)
        out["minionerec_script"] = run
    return out


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
    sasrec, tuning = load_sasrec()
    sources = {"sasrec": sasrec, "itemcf": itemcf, "pop": pop}
    rrf = RRF(sources, cand_mod.CONTROL_WEIGHTS)
    rrf_heavy = RRF(sources, cand_mod.SASREC_HEAVY_WEIGHTS, name="RRF-heavy", variant="sasrec_heavy")
    ranker, ranker_tuning = load_ranker(rrf, pop.counts)
    if ranker_tuning["recall"]["itemcf_params"] != best_params.__dict__:
        raise SystemExit("the ranker was trained on another ItemCF: run `make train-ranker` again")
    models = [pop, itemcf, sasrec, rrf, rrf_heavy, ranker]

    resampler = bootstrap.Resampler.build(test.user)
    results, test_values, calc_checks = {}, {}, []
    for model in models:
        if model is ranker:
            # Trained on the validation split: its in-sample numbers say nothing; report the
            # cross-validated ones of the selected configuration instead.
            valid_point = {"item": ranker_tuning["trials"][ranker_tuning["selected"]]["cv"]}
        else:
            valid_values = ev.per_sample(model.score(valid), valid)
            valid_point = {rule: _point(v) for rule, v in valid_values.items()}
        scores = model.score(test)
        values = ev.per_sample(scores, test)
        test_values[model.name] = values
        results[model.name] = {
            "description": model.describe(),
            "valid": valid_point,
            "valid_kind": "cross-validated" if model is ranker else "held out",
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

    pairs = {
        f"{a} - {b}": {
            rule: _with_ci(
                {
                    k: test_values[a][rule][k] - test_values[b][rule][k]
                    for k in test_values[b][rule]
                },
                resampler,
            )
            for rule in ("item", "sid")
        }
        for a, b in PAIRS
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
        "pairs": pairs,
        "segments": segment_results(test, test_values),
        "itemcf_params": best_params.__dict__,
        "itemcf_trials": trials,
        "sasrec_tuning": {k: v for k, v in tuning.items() if k != "trials"}
        | {"trials": [{k: v for k, v in t.items() if k != "curve"} for t in tuning["trials"]]},
        "ranking": ranking_summary(rrf, test, ranker_tuning),
        "sasrec_vs_paper": paper_comparison(
            test_values, sasrec_variants(tuning, test, ev, resampler), resampler
        ),
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
