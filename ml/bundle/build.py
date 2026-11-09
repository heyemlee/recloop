"""Build a versioned model bundle (design §6, minimal M1 version): `make bundle`.

Writes artifacts/bundles/{version}/ and points artifacts/bundles/current at it:

  manifest.json          version, build time, data snapshot, models (settings + offline metrics),
                         sha256 and size of every other file, dimensions, item count
  sasrec.pt              SASRec checkpoint (config, state dict, training log)
  sasrec_encoder.onnx    user encoder: history (B, 10) int64 of item_id + 1, left-padded with 0,
                         most recent last -> user vector (B, 64) float32
  item_vectors.npy       (3459, 64) float32; row i is item_id i (SASRec's shared embeddings)
  items.faiss            faiss.IndexFlatIP over item_vectors (exact inner product)
  item_ids.json          FAISS row -> item_id
  itemcf.json            ItemCF top-50 neighbours per item with similarities, plus the scoring
                         parameters (the api scores ItemCF in process)
  pop.json               items ordered by training interactions, with counts (api, in process)
  golden.json            validation histories with the PyTorch model's top-20: the model server
                         re-runs them on load and refuses a bundle that disagrees
  features.json          M4: the item tables of the ranking features (popularity, price bucket,
                         category codes), from which the model server computes them
  ranker.onnx            M4: the LightGBM ranker as an ONNX tree ensemble (raw scores); the
  ranker.txt             LightGBM model it came from (`make train-ranker`)
  rank_golden.json       M4: validation queries (history, candidates with their sources' ranks
                         and scores) with the features and scores Python computed for them
  pctr.json              M4, when `make train-pctr` has run: the logistic-regression pCTR and
                         golden rows

The builder checks its own output before publishing it: the ONNX encoder must reproduce the
PyTorch user vectors, the golden histories run through ONNX + FAISS (in a process without
PyTorch, ml.bundle.faiss_index) must return the PyTorch top-20, and the ONNX ranker must give
LightGBM's scores on every validation candidate.

A new bundle is published next to the others; `current` moves to it only when there is none
yet or with --current. Releasing it (canary, activation, rollback) goes through the api's
release endpoints (design §6), which serve whatever version model_versions says is active.

Needs `make train`, `make train-ranker` and `make eval` first (the SASRec checkpoint, the
ranker, and offline.json for the tuned ItemCF parameters and the offline metrics).

Usage: python -m ml.bundle.build [--current] [--no-ranker] [--no-pctr]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from ml import paths
from ml.data import download
from ml.eval.splits import HISTORY_LEN, NUM_ITEMS, Split, load_split, train_sequences
from ml.models import pctr as pctr_mod
from ml.models import train_ranker, train_sasrec
from ml.models.itemcf import ItemCF, ItemCFParams
from ml.models.pop import Pop
from ml.models.ranker import LambdaRank
from ml.models.rrf import RRF
from ml.models.sasrec import SASRec, encode_histories
from ml.ranking import candidates as cand_mod
from ml.ranking import features as feat
from ml.ranking import onnx_trees

BUNDLES = paths.ARTIFACTS / "bundles"
CURRENT = BUNDLES / "current"
OFFLINE_JSON = paths.ARTIFACTS / "eval" / "offline.json"

GOLDEN_QUERIES = 256
GOLDEN_TOP = 20
GOLDEN_SEED = 20261106
ONNX_TOLERANCE = 1e-4
RANK_GOLDEN_QUERIES = 32
RANKER_TOLERANCE = 1e-4
FORMAT = 1


def sha256_file(path: Path) -> str:
    return download.sha256_file(path)


def export_encoder(model: SASRec, path: Path) -> None:
    net = model.net.cpu().eval()
    example = encode_histories(load_split("valid").history_matrix()[:8])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            net,
            (example,),
            str(path),
            input_names=["history"],
            output_names=["user_vector"],
            dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=4096)},),
            opset_version=18,
            dynamo=True,
            external_data=False,
            verbose=False,
        )


def check_encoder(model: SASRec, onnx_path: Path, history: np.ndarray) -> float:
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ids = encode_histories(history)
    want = model.user_vectors(history)
    worst = 0.0
    for lo in range(0, len(ids), 1024):
        got = sess.run(None, {"history": ids[lo : lo + 1024].numpy()})[0]
        worst = max(worst, float(np.abs(got - want[lo : lo + 1024]).max()))
    if worst > ONNX_TOLERANCE:
        raise SystemExit(f"ONNX encoder differs from PyTorch: max |diff| {worst:.2e}")
    return worst


def golden_queries(model: SASRec) -> dict:
    """Validation histories with the PyTorch model's top-k (dense scores over all items)."""
    valid = load_split("valid")
    rng = np.random.default_rng(GOLDEN_SEED)
    rows = np.sort(rng.choice(len(valid), size=GOLDEN_QUERIES, replace=False))
    history = valid.history_matrix()[rows]
    dense = model.user_vectors(history) @ model.item_vectors().T
    top = np.argsort(-dense, axis=1, kind="stable")[:, :GOLDEN_TOP]
    return {
        "top": GOLDEN_TOP,
        "history_len": HISTORY_LEN,
        "min_recall": 0.99,
        "score_tolerance": 1e-3,
        "queries": [
            {
                "valid_row": int(r),
                "history": [int(i) for i in h if i >= 0],
                "items": [int(i) for i in t],
                "scores": [round(float(s), 6) for s in dense[q, t]],
            }
            for q, (r, h, t) in enumerate(zip(rows, history, top, strict=True))
        ],
    }


def build_index(stage: Path) -> dict:
    """FAISS index + golden-query replay in a torch-free subprocess (see ml.bundle.faiss_index)."""
    out = subprocess.run(
        [sys.executable, "-m", "ml.bundle.faiss_index", str(stage)],
        cwd=paths.ROOT,
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise SystemExit(f"building the FAISS index failed:\n{out.stderr}")
    result = json.loads(out.stdout.strip().splitlines()[-1])
    if result["golden_recall"] < 0.999 or result["golden_max_score_diff"] > 1e-3:
        raise SystemExit(f"golden queries through ONNX + FAISS disagree with PyTorch: {result}")
    return result


def itemcf_table(model: ItemCF, golden_rows: int = 64) -> dict:
    """Neighbour lists (best first) plus golden queries: validation histories with the Python
    model's top-20 and scores, which the Go scorer must reproduce (serving/internal/recall)."""
    sim = model.sim.tocsr()
    neighbors, scores = [], []
    for i in range(NUM_ITEMS):
        lo, hi = sim.indptr[i], sim.indptr[i + 1]
        cols, vals = sim.indices[lo:hi], sim.data[lo:hi]
        order = np.lexsort((cols, -vals))
        neighbors.append([int(c) for c in cols[order]])
        scores.append([round(float(v), 8) for v in vals[order]])
    valid = load_split("valid")
    rng = np.random.default_rng(GOLDEN_SEED + 1)
    rows = np.sort(rng.choice(len(valid), size=golden_rows, replace=False))
    sub = Split("golden", valid.user[rows], valid.target[rows], [valid.history[r] for r in rows])
    dense = model.score(sub)
    golden = []
    for h, sc in zip(sub.history, dense, strict=True):
        order = np.lexsort((np.arange(NUM_ITEMS), -sc))[:GOLDEN_TOP]
        order = [int(j) for j in order if sc[j] > 0]
        golden.append(
            {
                "history": [int(i) for i in h],
                "items": order,
                "scores": [round(float(sc[j]), 8) for j in order],
            }
        )
    return {
        "params": model.params.__dict__,
        "scoring": "score(j) = sum_k history_decay^(L - k) * sim(h_k -> j), most recent h_L",
        "neighbors": neighbors,
        "scores": scores,
        "golden": golden,
    }


def _nan_to_none(values) -> list:
    return [None if math.isnan(v) else round(float(v), 7) for v in values]


def ranker_files(stage: Path, ranker: LambdaRank, valid: Split) -> dict:
    """ranker.onnx + ranker.txt, checked against LightGBM on every validation candidate, and
    the golden queries the model server replays: features from the hits, then scores."""
    c = ranker.rrf.candidates(valid)
    x = feat.compute(c, valid.history, ranker.tables)
    rows = x[c.valid()]
    model = onnx_trees.booster_to_onnx(ranker.booster, len(feat.FEATURES))
    (stage / "ranker.onnx").write_bytes(model.SerializeToString())
    ranker.booster.save_model(str(stage / "ranker.txt"))
    want = ranker.booster.predict(rows, raw_score=True)
    got = onnx_trees.onnx_scores(model.SerializeToString(), rows)
    worst = float(np.abs(got - want).max())
    if worst > RANKER_TOLERANCE:
        raise SystemExit(f"ONNX ranker differs from LightGBM: max |diff| {worst:.2e}")

    rng = np.random.default_rng(GOLDEN_SEED + 2)
    picks = np.sort(rng.choice(len(valid), size=RANK_GOLDEN_QUERIES, replace=False))
    queries = []
    for i in picks:
        row = c.row(int(i))
        # Scores travel as float32 (proto); the expected features are computed from those.
        hits = [[[src, rk, float(np.float32(sc))] for src, rk, sc in h] for h in row["hits"]]
        one = cand_mod.Candidates(
            items=c.items[i : i + 1],
            rrf=c.rrf[i : i + 1],
            ranks={s: c.ranks[s][i : i + 1] for s in c.ranks},
            scores={
                s: c.scores[s][i : i + 1].astype(np.float32).astype(np.float64) for s in c.scores
            },
        )
        xf = feat.compute(one, [valid.history[i]], ranker.tables)[0][c.valid()[i]]
        queries.append(
            {
                "valid_row": int(i),
                "history": [int(h) for h in valid.history[i]],
                "items": row["items"],
                "hits": hits,
                "features": [_nan_to_none(r) for r in xf],
                "scores": [round(float(v), 6) for v in ranker.booster.predict(xf, raw_score=True)],
            }
        )
    _write_json(
        stage / "rank_golden.json",
        {
            "features": list(feat.FEATURES),
            "feature_tolerance": 1e-5,
            "score_tolerance": 1e-3,
            "queries": queries,
        },
    )
    return {
        "onnx_max_abs_diff": worst,
        "checked_rows": int(len(rows)),
        "golden_queries": len(queries),
    }


def pctr_file(stage: Path, valid_x: np.ndarray) -> dict | None:
    """pctr.json (the model of `make train-pctr`) with golden rows at several positions."""
    path = pctr_mod.OUT / "model.json"
    if not path.exists():
        return None
    m = json.loads(path.read_text())
    if tuple(m["features"]) != feat.FEATURES or tuple(m["columns"]) not in (
        pctr_mod.COLUMNS,
        pctr_mod.COLUMNS_CTR,
    ):
        raise SystemExit(f"{path}: other features than the ranker's; run `make train-pctr` again")
    rng = np.random.default_rng(GOLDEN_SEED + 3)
    rows = valid_x[rng.choice(len(valid_x), size=48, replace=False)]
    pos = np.tile(np.asarray([1, 3, 9, 15, 25, 40], dtype=np.int64), 8)
    sponsored = np.tile(np.asarray([0, 1], dtype=np.int64), 24)
    # Item CTR counts [impressions, clicks] of the hour and the day; every sixth row has none
    # (the api had no table: the prior), and some have more clicks than impressions (clicks of
    # impressions counted in an earlier bucket).
    imp_h = rng.integers(0, 400, 48)
    imp_d = imp_h + rng.integers(0, 6000, 48)
    clk_h = rng.binomial(imp_h + 2, 0.02)
    clk_d = clk_h + rng.binomial(imp_d - imp_h + 2, 0.02)
    counts = np.stack([imp_h, clk_h, imp_d, clk_d], axis=1).astype(np.float64)
    counts[::6] = np.nan
    p = pctr_mod.predict_json(m, rows, pos, sponsored, counts)
    m["golden"] = {
        "tolerance": 1e-6,
        "rows": [
            {
                "features": _nan_to_none(r),
                "pos": int(q),
                "sponsored": int(sp),
                "counts": None if np.isnan(c).any() else [int(v) for v in c],
                "pctr": float(v),
            }
            for r, q, sp, c, v in zip(rows, pos, sponsored, counts, p, strict=True)
        ],
    }
    _write_json(stage / "pctr.json", m)
    return {
        "file": "pctr.json",
        "model": m["model"],
        "variant": m.get("variant", "base"),
        **({"ctr": {k: m["ctr"][k] for k in ("prior", "strength")}} if "ctr" in m else {}),
        "reference_position": m["reference_position"],
        "C": m["C"],
        "days": m["training"]["days"],
        "rows": m["training"]["rows"],
        "held_out": {
            k: m["training"]["held_out"][k]
            for k in ("auc", "logloss", "logloss_constant", "observed_ctr", "predicted_ctr")
        },
    }


def _git_commit() -> dict:
    try:
        head = subprocess.run(
            ["git", "-C", str(paths.ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(paths.ROOT), "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": head, "dirty": dirty}


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, separators=(",", ":")) + "\n")


def build(
    set_current: bool = False,
    with_ranker: bool = True,
    with_pctr: bool = True,
    sasrec_path: Path | None = None,
) -> Path:
    if not train_sasrec.CHECKPOINT.exists():
        raise SystemExit("no SASRec checkpoint: run `make train` first")
    if not OFFLINE_JSON.exists():
        raise SystemExit("no artifacts/eval/offline.json: run `make eval` first")
    offline = json.loads(OFFLINE_JSON.read_text())

    checkpoint = Path(sasrec_path) if sasrec_path else train_sasrec.CHECKPOINT
    sasrec = SASRec.load(checkpoint)
    sequences = train_sequences(load_split("train"))
    pop = Pop().fit(sequences, NUM_ITEMS)
    itemcf = ItemCF(ItemCFParams(**offline["itemcf_params"])).fit(sequences, NUM_ITEMS)
    tables = feat.ItemTables.build(pop.counts)
    ranker = None
    if with_ranker:
        if not (train_ranker.OUT / "model.txt").exists():
            raise SystemExit("no ranker: run `make train-ranker` first (or --no-ranker)")
        rrf = RRF({"sasrec": sasrec, "itemcf": itemcf, "pop": pop}, cand_mod.CONTROL_WEIGHTS)
        ranker, ranker_tuning = LambdaRank.load(train_ranker.OUT, rrf, tables)

    BUNDLES.mkdir(parents=True, exist_ok=True)
    stage = BUNDLES / f".staging-{os.getpid()}"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir()

    # Model files
    shutil.copyfile(checkpoint, stage / "sasrec.pt")
    export_encoder(sasrec, stage / "sasrec_encoder.onnx")
    onnx_diff = check_encoder(
        sasrec, stage / "sasrec_encoder.onnx", load_split("valid").history_matrix()
    )
    vectors = np.ascontiguousarray(sasrec.item_vectors(), dtype=np.float32)
    np.save(stage / "item_vectors.npy", vectors)
    _write_json(stage / "item_ids.json", list(range(NUM_ITEMS)))
    _write_json(stage / "itemcf.json", itemcf_table(itemcf))
    order = np.lexsort((np.arange(NUM_ITEMS), -pop.counts))
    _write_json(
        stage / "pop.json",
        {"items": [int(i) for i in order], "counts": [int(pop.counts[i]) for i in order]},
    )
    golden = golden_queries(sasrec)
    _write_json(stage / "golden.json", golden)
    index = build_index(stage)
    _write_json(stage / "features.json", tables.to_json())
    ranking = {"features": list(feat.FEATURES), "file": "features.json"}
    if ranker is not None:
        valid = load_split("valid")
        checks = ranker_files(stage, ranker, valid)
        test = offline["models"].get("LambdaRank")
        ranking["ranker"] = {
            "model": "lightgbm",
            "file": "ranker.onnx",
            "lightgbm_file": "ranker.txt",
            "params": ranker_tuning["params"],
            "rounds": ranker_tuning["rounds"],
            "cv": ranker_tuning["trials"][ranker_tuning["selected"]]["cv"],
            "test": _test_summary(test),
            "golden_file": "rank_golden.json",
            **checks,
        }
    pctr_info = None
    if with_pctr:
        c = (
            ranker.rrf
            if ranker is not None
            else RRF({"sasrec": sasrec, "itemcf": itemcf, "pop": pop}, cand_mod.CONTROL_WEIGHTS)
        ).candidates(load_split("valid"))
        vx = feat.compute(c, load_split("valid").history, tables)[c.valid()]
        pctr_info = pctr_file(stage, vx)

    files = {
        p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in sorted(stage.iterdir())
    }
    content = hashlib.sha256(
        "".join(f"{n}:{f['sha256']}\n" for n, f in files.items() if n != "sasrec.pt").encode()
    ).hexdigest()
    built = dt.datetime.now(dt.UTC)
    version = f"{built:%Y-%m-%d}-sasrec{sasrec.cfg.hidden}-{content[:8]}"

    models = offline["models"]
    manifest = {
        "format": FORMAT,
        "version": version,
        "built_at": built.isoformat(timespec="seconds"),
        "source": _git_commit(),
        "num_items": NUM_ITEMS,
        "history_len": HISTORY_LEN,
        "dim": int(vectors.shape[1]),
        "index": {"type": "IndexFlatIP", "metric": "inner_product", "ntotal": index["ntotal"]},
        "encoder": {
            "file": "sasrec_encoder.onnx",
            "input": "history: int64 (B, 10), item_id + 1, left-padded with 0, most recent last",
            "output": "user_vector: float32 (B, dim)",
            "onnx_max_abs_diff": onnx_diff,
            "torch": torch.__version__,
            "onnxruntime": ort.__version__,
        },
        "data": {
            "dataset": paths.SPLIT_STEM,
            "train_csv_sha256": download.MINIONEREC_FILES[
                f"data/Amazon/train/{paths.SPLIT_STEM}.csv"
            ],
            "train_users": len(sequences),
            "train_interactions": int(sum(len(s) for s in sequences.values())),
        },
        "models": {
            "sasrec": {
                "config": asdict(sasrec.cfg),
                "checkpoint": str(checkpoint.relative_to(paths.ROOT))
                if checkpoint.is_relative_to(paths.ROOT)
                else str(checkpoint),
                "best_epoch": sasrec.log.best_epoch,
                "valid": sasrec.log.best_valid,
                # make eval's test numbers are the selected checkpoint's
                "test": _test_summary(models.get("SASRec"))
                if checkpoint == train_sasrec.CHECKPOINT
                else None,
            },
            "itemcf": {
                "params": offline["itemcf_params"],
                "test": _test_summary(models.get("ItemCF")),
            },
            "pop": {"test": _test_summary(models.get("Pop"))},
        },
        "golden": {
            "queries": len(golden["queries"]),
            "top": golden["top"],
            "recall": index["golden_recall"],
            "max_score_diff": index["golden_max_score_diff"],
        },
        "ranking": ranking,
        "files": files,
    }
    if pctr_info is not None:
        manifest["pctr"] = pctr_info
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    final = BUNDLES / version
    if final.exists():
        shutil.rmtree(final)
    stage.rename(final)
    if set_current or not CURRENT.exists():
        tmp_link = BUNDLES / f".current-{os.getpid()}"
        tmp_link.unlink(missing_ok=True)
        tmp_link.symlink_to(version)
        tmp_link.replace(CURRENT)  # atomic switch of the symlink
    return final


def _test_summary(model_results: dict | None) -> dict | None:
    if model_results is None:
        return None
    item = model_results["test"]["item"]
    return {k: round(item[k]["value"], 6) for k in ("HR@10", "NDCG@10")}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Build a model bundle")
    p.add_argument("--current", action="store_true", help="point artifacts/bundles/current at it")
    p.add_argument(
        "--no-ranker", action="store_true", help="feature tables only, no LightGBM ranker"
    )
    p.add_argument("--no-pctr", action="store_true", help="leave the pCTR model out")
    p.add_argument(
        "--sasrec",
        type=Path,
        help="another SASRec checkpoint (e.g. a weaker trial, for the release drill)",
    )
    args = p.parse_args(argv)
    final = build(
        args.current,
        with_ranker=not args.no_ranker,
        with_pctr=not args.no_pctr,
        sasrec_path=args.sasrec,
    )
    manifest = json.loads((final / "manifest.json").read_text())
    print(f"bundle   {manifest['version']}  ({len(manifest['files'])} files)")
    print(f"encoder  ONNX vs PyTorch max |diff| {manifest['encoder']['onnx_max_abs_diff']:.1e}")
    g = manifest["golden"]
    print(
        f"golden   {g['queries']} validation histories, top-{g['top']} through ONNX + FAISS: "
        f"recall {g['recall']:.4f}, max score diff {g['max_score_diff']:.1e}"
    )
    r = manifest["ranking"].get("ranker")
    if r:
        print(
            f"ranker   {r['rounds']} trees as ONNX, max |diff| to LightGBM {r['onnx_max_abs_diff']:.1e} "
            f"over {r['checked_rows']:,} validation candidates; {r['golden_queries']} golden queries"
        )
    pc = manifest.get("pctr")
    print(
        f"pctr     {pc['model'] + ' from ' + ', '.join(pc['days']) if pc else 'none (FixedPCTR stays)'}"
    )
    print(
        f"current  -> {os.readlink(CURRENT)}"
        + (
            ""
            if os.readlink(CURRENT) == final.name
            else f" (this bundle: {final.name}; release it through the api)"
        )
    )


if __name__ == "__main__":
    main()
