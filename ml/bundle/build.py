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

The builder checks its own output before publishing it: the ONNX encoder must reproduce the
PyTorch user vectors, and the golden histories run through ONNX + FAISS (in a process without
PyTorch, ml.bundle.faiss_index) must return the PyTorch top-20.
Full activation, canary and rollback come in M4; until then `current` is the active bundle.

Needs `make train` and `make eval` first (the SASRec checkpoint, and offline.json for the tuned
ItemCF parameters and the offline metrics recorded in the manifest).

Usage: python -m ml.bundle.build
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
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
from ml.models import train_sasrec
from ml.models.itemcf import ItemCF, ItemCFParams
from ml.models.pop import Pop
from ml.models.sasrec import SASRec, encode_histories

BUNDLES = paths.ARTIFACTS / "bundles"
CURRENT = BUNDLES / "current"
OFFLINE_JSON = paths.ARTIFACTS / "eval" / "offline.json"

GOLDEN_QUERIES = 256
GOLDEN_TOP = 20
GOLDEN_SEED = 20261106
ONNX_TOLERANCE = 1e-4
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


def build() -> Path:
    if not train_sasrec.CHECKPOINT.exists():
        raise SystemExit("no SASRec checkpoint: run `make train` first")
    if not OFFLINE_JSON.exists():
        raise SystemExit("no artifacts/eval/offline.json: run `make eval` first")
    offline = json.loads(OFFLINE_JSON.read_text())

    sasrec = SASRec.load(train_sasrec.CHECKPOINT)
    sequences = train_sequences(load_split("train"))
    pop = Pop().fit(sequences, NUM_ITEMS)
    itemcf = ItemCF(ItemCFParams(**offline["itemcf_params"])).fit(sequences, NUM_ITEMS)

    BUNDLES.mkdir(parents=True, exist_ok=True)
    stage = BUNDLES / f".staging-{os.getpid()}"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir()

    # Model files
    shutil.copyfile(train_sasrec.CHECKPOINT, stage / "sasrec.pt")
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
                "best_epoch": sasrec.log.best_epoch,
                "valid": sasrec.log.best_valid,
                "test": _test_summary(models.get("SASRec")),
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
        "files": files,
    }
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    final = BUNDLES / version
    if final.exists():
        shutil.rmtree(final)
    stage.rename(final)
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


def main() -> None:
    final = build()
    manifest = json.loads((final / "manifest.json").read_text())
    print(f"bundle   {manifest['version']}  ({len(manifest['files'])} files)")
    print(f"encoder  ONNX vs PyTorch max |diff| {manifest['encoder']['onnx_max_abs_diff']:.1e}")
    g = manifest["golden"]
    print(
        f"golden   {g['queries']} validation histories, top-{g['top']} through ONNX + FAISS: "
        f"recall {g['recall']:.4f}, max score diff {g['max_score_diff']:.1e}"
    )
    print(f"current  -> {os.readlink(CURRENT)}")


if __name__ == "__main__":
    main()
