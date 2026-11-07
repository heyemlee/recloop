"""Load and validate a bundle written by ml/bundle/build.py (design §6).

A bundle is accepted only if every file matches the sha256 in its manifest, the index, the item
mapping and the encoder agree on the item count and the vector dimension, and the golden
queries (validation histories with the training-time model's top-20) come back the same through
the ONNX encoder and the FAISS index. The user encoder and the item index always come from the
same bundle, and every response carries the bundle's version.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np

FORMAT = 1


class BundleError(Exception):
    """The bundle is incomplete, corrupted or inconsistent; it must not be served."""


class Encoder(Protocol):
    def run(self, output_names, input_feed: dict) -> list[np.ndarray]: ...


@dataclass(frozen=True)
class Bundle:
    version: str
    path: Path
    num_items: int
    dim: int
    history_len: int
    encoder: Encoder
    index: faiss.Index
    item_ids: np.ndarray  # FAISS row -> item_id
    manifest: dict = field(default_factory=dict, repr=False)

    def window(self, history: Sequence[int]) -> list[int]:
        """The encoder's input: the last `history_len` known item ids, oldest first."""
        known = [int(i) for i in history if 0 <= i < self.num_items]
        return known[-self.history_len :]

    def encode(self, histories: Sequence[Sequence[int]]) -> np.ndarray:
        """(B, dim) user vectors. Ids are shifted by +1 and left-padded with 0, as in training."""
        x = np.zeros((len(histories), self.history_len), dtype=np.int64)
        for row, history in enumerate(histories):
            w = self.window(history)
            if w:
                x[row, self.history_len - len(w) :] = np.asarray(w, dtype=np.int64) + 1
        return np.ascontiguousarray(self.encoder.run(None, {"history": x})[0], dtype=np.float32)

    def retrieve(self, history: Sequence[int], k: int) -> tuple[np.ndarray, np.ndarray]:
        """Top-k item ids and inner-product scores for one history; empty without history."""
        if k <= 0 or not self.window(history):
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        user = self.encode([history])
        scores, rows = self.index.search(user, min(k, self.num_items))
        keep = rows[0] >= 0
        return self.item_ids[rows[0][keep]], scores[0][keep]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def _onnx_session(path: Path):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    # One request is a (1, 10) matrix: parallelism inside the op only adds overhead; the gRPC
    # thread pool provides concurrency across requests.
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def load(path: Path, check_golden: bool = True) -> Bundle:
    real = Path(path).resolve()
    manifest_path = real / "manifest.json"
    if not manifest_path.exists():
        raise BundleError(f"{real}: no manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != FORMAT:
        raise BundleError(f"{real}: bundle format {manifest.get('format')}, expected {FORMAT}")

    for name, f in manifest["files"].items():
        p = real / name
        if not p.exists():
            raise BundleError(f"{real}: missing {name}")
        if p.stat().st_size != f["bytes"] or sha256_file(p) != f["sha256"]:
            raise BundleError(f"{real}: {name} does not match its checksum in the manifest")

    num_items, dim = int(manifest["num_items"]), int(manifest["dim"])
    index = faiss.read_index(str(real / "items.faiss"))
    item_ids = np.asarray(json.loads((real / "item_ids.json").read_text()), dtype=np.int64)
    if index.ntotal != num_items or len(item_ids) != num_items:
        raise BundleError(
            f"{real}: index has {index.ntotal} vectors, mapping {len(item_ids)} ids, "
            f"manifest {num_items} items"
        )
    if index.d != dim:
        raise BundleError(f"{real}: index dimension {index.d}, manifest {dim}")

    bundle = Bundle(
        version=manifest["version"],
        path=real,
        num_items=num_items,
        dim=dim,
        history_len=int(manifest["history_len"]),
        encoder=_onnx_session(real / manifest["encoder"]["file"]),
        index=index,
        item_ids=item_ids,
        manifest=manifest,
    )
    probe = bundle.encode([[0]])
    if probe.shape != (1, dim) or not np.isfinite(probe).all():
        raise BundleError(f"{real}: encoder output {probe.shape}, expected (1, {dim})")
    if check_golden:
        golden_check(bundle, json.loads((real / "golden.json").read_text()))
    return bundle


def golden_check(bundle: Bundle, golden: dict) -> dict:
    """Re-run the golden queries; raise BundleError if recall or scores drift."""
    queries = golden["queries"]
    top = int(golden["top"])
    users = bundle.encode([q["history"] for q in queries])
    scores, rows = bundle.index.search(users, top)
    recall, worst = [], 0.0
    for q, s, r in zip(queries, scores, rows, strict=True):
        got = bundle.item_ids[r]
        recall.append(len(set(got.tolist()) & set(q["items"])) / top)
        worst = max(worst, float(np.abs(np.sort(s)[::-1] - np.asarray(q["scores"])).max()))
    result = {"queries": len(queries), "recall": float(np.mean(recall)), "max_score_diff": worst}
    if result["recall"] < golden["min_recall"] or worst > golden["score_tolerance"]:
        raise BundleError(f"{bundle.path}: golden queries failed: {result}")
    return result
