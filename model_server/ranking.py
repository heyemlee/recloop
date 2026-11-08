"""The ranking part of a bundle (M4): features, the LightGBM ranker (ONNX) and the pCTR model.

The bundle's features.json holds the item tables; this module computes the same features as
ml/ranking/features.py (one row per candidate, float32, NaN = missing):

    sasrec_score, sasrec_rank, itemcf_score, itemcf_rank, pop_rank   from the recall sources' hits
    pop_log, n_sources, price_bucket, cat2_overlap, leaf_overlap      from the tables + history

ranker.onnx scores organic candidates; pctr.json is a logistic regression over a fixed design of
those features plus the slot and the position (ml/models/pctr.py). Both are checked at load:
rank_golden.json replays validation queries (features from the hits, then scores) and pctr.json
carries golden rows; a bundle that disagrees is refused.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FEATURES = (
    "sasrec_score",
    "sasrec_rank",
    "itemcf_score",
    "itemcf_rank",
    "pop_rank",
    "pop_log",
    "n_sources",
    "price_bucket",
    "cat2_overlap",
    "leaf_overlap",
)
SOURCES = ("sasrec", "itemcf", "pop")  # the home feed's sources; n_sources counts these
ZERO_THRESHOLD = 1e-35  # LightGBM reads |x| <= 1e-35 as 0 (ml/ranking/onnx_trees.py)


class RankingError(Exception):
    """The ranking files are missing, inconsistent or disagree with their golden data."""


@dataclass(frozen=True)
class PCTRModel:
    columns: tuple[str, ...]
    continuous: frozenset[str]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    reference_position: int
    name: str

    @classmethod
    def from_json(cls, m: dict) -> PCTRModel:
        if tuple(m["features"]) != FEATURES:
            raise RankingError(
                f"pctr.json is over {m['features']}, this server computes {FEATURES}"
            )
        return cls(
            tuple(m["columns"]),
            frozenset(m["continuous"]),
            np.asarray(m["mean"], dtype=np.float64),
            np.asarray(m["scale"], dtype=np.float64),
            np.asarray(m["coef"], dtype=np.float64),
            float(m["intercept"]),
            int(m["reference_position"]),
            str(m["model"]),
        )

    def design(self, x: np.ndarray, pos: np.ndarray, sponsored: np.ndarray) -> np.ndarray:
        """The model's columns from feature rows (ml.models.pctr.design, by column name)."""
        x = np.asarray(x, dtype=np.float64)
        col = {f: x[:, i] for i, f in enumerate(FEATURES)}
        pos = np.asarray(pos, dtype=np.int64)
        bucket = np.where(pos <= 10, pos - 1, np.where(pos <= 20, 10, 11))
        price = col["price_bucket"]
        out = np.zeros((len(x), len(self.columns)))
        for j, name in enumerate(self.columns):
            if name in self.continuous:
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
                raise RankingError(f"pctr.json: unknown column {name!r}")
        return out

    def predict(self, x: np.ndarray, pos=None, sponsored=None) -> np.ndarray:
        n = len(x)
        pos = np.full(n, self.reference_position) if pos is None else pos
        sponsored = np.ones(n) if sponsored is None else sponsored
        z = (self.design(x, pos, sponsored) - self.mean) / self.scale
        return 1.0 / (1.0 + np.exp(-(z @ self.coef + self.intercept)))


class Ranking:
    def __init__(self, tables: dict, ranker=None, pctr: PCTRModel | None = None):
        if tuple(tables["features"]) != FEATURES:
            raise RankingError(
                f"features.json lists {tables['features']}, this server computes {FEATURES}"
            )
        self.pop_log = np.log1p(np.asarray(tables["pop_count"], dtype=np.float64))
        self.price_bucket = np.asarray(
            [math.nan if v is None else float(v) for v in tables["price_bucket"]]
        )
        self.cat2 = np.asarray(tables["cat2"], dtype=np.int64)
        self.leaf = np.asarray(tables["leaf"], dtype=np.int64)
        self.num_items = len(self.cat2)
        self.ranker = ranker  # onnxruntime session (or None)
        self.pctr = pctr

    def _overlaps(self, history: list[int], items: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = np.asarray([i for i in history if 0 <= i < self.num_items], dtype=np.int64)
        out = []
        for table in (self.cat2, self.leaf):
            codes = table[h] if len(h) else np.empty(0, dtype=np.int64)
            codes = codes[codes >= 0]
            cand = table[items]
            same = (cand[:, None] == codes[None, :]).sum(axis=1).astype(np.float64)
            share = same / len(h) if len(h) else np.full(len(items), np.nan)
            out.append(np.where(cand < 0, np.nan, share))
        return out[0], out[1]

    def features(
        self, history: list[int], items: list[int], hits: list[list[tuple[str, int, float]]]
    ) -> np.ndarray:
        """(n, F) float32 for organic candidates with their sources' (source, rank, score)."""
        ids = np.asarray(items, dtype=np.int64)
        n = len(ids)
        if n and (ids.min() < 0 or ids.max() >= self.num_items):
            raise ValueError("candidate item outside the catalog")
        x = np.full((n, len(FEATURES)), np.nan)
        col = {f: i for i, f in enumerate(FEATURES)}
        n_src = np.zeros(n)
        for j, item_hits in enumerate(hits):
            for source, rank, score in item_hits:
                if source in SOURCES:
                    n_src[j] += 1
                    if source != "pop":
                        x[j, col[f"{source}_score"]] = score
                    x[j, col[f"{source}_rank"]] = rank
        self._fill_item_columns(x, col, history, ids, n_src)
        return x.astype(np.float32)

    def ad_features(
        self, history: list[int], items: list[int], relevance: np.ndarray
    ) -> np.ndarray:
        """(n, F) float32 for ad candidates: no recall hits, SASRec relevance as sasrec_score."""
        ids = np.asarray(items, dtype=np.int64)
        x = np.full((len(ids), len(FEATURES)), np.nan)
        col = {f: i for i, f in enumerate(FEATURES)}
        x[:, col["sasrec_score"]] = np.asarray(relevance, dtype=np.float32)
        self._fill_item_columns(x, col, history, ids, np.zeros(len(ids)))
        return x.astype(np.float32)

    def _fill_item_columns(self, x, col, history, ids, n_src) -> None:
        x[:, col["pop_log"]] = self.pop_log[ids]
        x[:, col["n_sources"]] = n_src
        x[:, col["price_bucket"]] = self.price_bucket[ids]
        cat2, leaf = self._overlaps(history, ids)
        x[:, col["cat2_overlap"]] = cat2
        x[:, col["leaf_overlap"]] = leaf

    def scores(self, x: np.ndarray) -> np.ndarray:
        if self.ranker is None or len(x) == 0:
            return np.empty(0, dtype=np.float32)
        inputs = np.where(np.abs(x) <= ZERO_THRESHOLD, np.float32(0), x).astype(np.float32)
        return self.ranker.run(None, {"features": np.ascontiguousarray(inputs)})[0][:, 0]


def _onnx_ranker(path: Path):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def load(path: Path, manifest: dict, check_golden: bool = True) -> Ranking | None:
    """The bundle's ranking part, or None for a bundle from before M4 (no features.json)."""
    spec = manifest.get("ranking")
    if spec is None:
        return None
    tables = json.loads((path / spec["file"]).read_text())
    ranker = None
    if "ranker" in spec:
        ranker = _onnx_ranker(path / spec["ranker"]["file"])
    pctr = None
    if "pctr" in manifest:
        pctr_doc = json.loads((path / manifest["pctr"]["file"]).read_text())
        pctr = PCTRModel.from_json(pctr_doc)
    r = Ranking(tables, ranker, pctr)
    if r.num_items != int(manifest["num_items"]):
        raise RankingError(
            f"features.json has {r.num_items} items, the manifest {manifest['num_items']}"
        )
    if check_golden:
        if "ranker" in spec:
            golden_ranker(r, json.loads((path / spec["ranker"]["golden_file"]).read_text()))
        if pctr is not None:
            golden_pctr(pctr, pctr_doc["golden"])
    return r


def _nan(values) -> np.ndarray:
    return np.asarray([math.nan if v is None else v for v in values], dtype=np.float64)


def golden_ranker(r: Ranking, golden: dict) -> dict:
    worst_f, worst_s = 0.0, 0.0
    for q in golden["queries"]:
        hits = [[(s, int(k), float(v)) for s, k, v in h] for h in q["hits"]]
        x = r.features(q["history"], q["items"], hits)
        want = np.asarray([_nan(row) for row in q["features"]])
        if not np.array_equal(np.isnan(x), np.isnan(want)):
            raise RankingError(
                f"rank_golden.json: missing features differ (valid row {q['valid_row']})"
            )
        diff = np.nan_to_num(np.abs(x - want), nan=0.0)
        worst_f = max(worst_f, float(diff.max(initial=0.0)))
        worst_s = max(
            worst_s, float(np.abs(r.scores(x) - np.asarray(q["scores"])).max(initial=0.0))
        )
    out = {
        "queries": len(golden["queries"]),
        "max_feature_diff": worst_f,
        "max_score_diff": worst_s,
    }
    if worst_f > golden["feature_tolerance"] or worst_s > golden["score_tolerance"]:
        raise RankingError(f"rank_golden.json disagrees: {out}")
    return out


def golden_pctr(m: PCTRModel, golden: dict) -> dict:
    rows = golden["rows"]
    x = np.asarray([_nan(g["features"]) for g in rows])
    got = m.predict(
        x, np.asarray([g["pos"] for g in rows]), np.asarray([g["sponsored"] for g in rows])
    )
    worst = float(np.abs(got - np.asarray([g["pctr"] for g in rows])).max(initial=0.0))
    if worst > golden["tolerance"]:
        raise RankingError(f"pctr.json golden rows disagree: max |diff| {worst:.2e}")
    return {"rows": len(rows), "max_diff": worst}
