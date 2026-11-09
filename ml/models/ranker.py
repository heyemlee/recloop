"""LightGBM LambdaRank over the fused candidates (docs/design.md §4.3, M4).

The ranker re-orders RRF's 300 candidates (`ml.ranking.candidates`) with the features of
`ml.ranking.features`. It is trained on the validation period: the recall models (Pop, ItemCF,
SASRec) are fitted on the train rows, so their scores on validation samples are what they look
like on data they have not seen, which is what they look like online; the ranker never learns
the inflated scores a recall model gives its own training data. Each validation sample is a
query; its candidates are the documents; the label is 1 for the true next item. Queries whose
target is not among the candidates carry no signal for a pairwise loss and are left out of
training. Items outside the candidates score -inf, as for RRF.

Trained by `make train-ranker` (ml.models.train_ranker), loaded by `make eval` and the bundle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ml.eval.splits import Split
from ml.models.rrf import RRF, dense
from ml.ranking import candidates as cand_mod
from ml.ranking import features as feat


@dataclass(frozen=True)
class RankerParams:
    num_leaves: int = 31
    min_data_in_leaf: int = 100
    learning_rate: float = 0.05
    feature_fraction: float = 1.0
    lambda_l2: float = 1.0
    max_rounds: int = 2000
    early_stopping: int = 100
    truncation_level: int = 30  # lambdarank_truncation_level: pairs that involve the top 30
    seed: int = 20261106

    def lgb(self) -> dict:
        return {
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": [10],
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "learning_rate": self.learning_rate,
            "feature_fraction": self.feature_fraction,
            "lambda_l2": self.lambda_l2,
            "lambdarank_truncation_level": self.truncation_level,
            "seed": self.seed,
            "deterministic": True,
            "force_row_wise": True,
            "num_threads": 1,
            "verbose": -1,
        }

    def label(self) -> str:
        return (
            f"{self.num_leaves} leaves, min {self.min_data_in_leaf} per leaf, "
            f"feature fraction {self.feature_fraction}, lr {self.learning_rate}, L2 {self.lambda_l2}"
        )


class LambdaRank:
    name = "LambdaRank"

    def __init__(
        self,
        rrf: RRF,
        tables: feat.ItemTables,
        booster=None,
        params: RankerParams | None = None,
        rounds: int | None = None,
    ):
        self.rrf, self.tables = rrf, tables
        self.booster, self.params, self.rounds = booster, params or RankerParams(), rounds

    def fit(self, sequences, num_items: int) -> LambdaRank:
        return self  # trained by train_ranker on the validation split; see module docstring

    def score(self, split: Split) -> np.ndarray:
        assert self.booster is not None, "load() or train first"
        c = self.rrf.candidates(split)
        x = feat.compute(c, split.history, self.tables)
        n, depth, f = x.shape
        s = self.booster.predict(x.reshape(-1, f), raw_score=True).reshape(n, depth)
        return dense(c.items, np.where(c.valid(), s, np.nan))

    def describe(self) -> str:
        return (
            f"LightGBM LambdaRank re-ranking RRF's {cand_mod.CANDIDATE_DEPTH} candidates, "
            f"{len(feat.FEATURES)} features, {self.params.label()}, {self.rounds} trees; "
            "trained on the validation split"
        )

    def save(self, directory: Path, tuning: dict) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(directory / "model.txt"))
        (directory / "tuning.json").write_text(json.dumps(tuning, indent=2) + "\n")

    @classmethod
    def load(cls, directory: Path, rrf: RRF, tables: feat.ItemTables) -> tuple[LambdaRank, dict]:
        import lightgbm as lgb

        tuning = json.loads((directory / "tuning.json").read_text())
        if tuple(tuning["features"]) != feat.FEATURES:
            raise SystemExit(f"{directory}: trained on other features; run `make train-ranker`")
        booster = lgb.Booster(model_file=str(directory / "model.txt"))
        params = RankerParams(**tuning["params"])
        return cls(rrf, tables, booster, params, tuning["rounds"]), tuning
