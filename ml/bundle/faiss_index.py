"""Build and check a bundle's FAISS index, in a process without PyTorch.

torch and faiss-cpu each ship their own OpenMP runtime; on macOS, faiss crashes once both are
loaded in one process. ml.bundle.build therefore runs this module as a subprocess: it builds the
IndexFlatIP from item_vectors.npy and replays the golden queries the way the model server does
(ONNX encoder -> FAISS search), so the bundle is checked through the serving path itself.

Do not import torch here, directly or through other ml modules.

Usage: python -m ml.bundle.faiss_index <bundle staging dir>   (prints a JSON summary)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import faiss
import numpy as np
import onnxruntime as ort


def encode(sess: ort.InferenceSession, histories: list[list[int]], length: int) -> np.ndarray:
    x = np.zeros((len(histories), length), dtype=np.int64)
    for row, h in enumerate(histories):
        h = h[-length:]
        if h:
            x[row, length - len(h) :] = np.asarray(h, dtype=np.int64) + 1
    return np.ascontiguousarray(sess.run(None, {"history": x})[0], dtype=np.float32)


def main(stage: Path) -> dict:
    vectors = np.load(stage / "item_vectors.npy")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    faiss.write_index(index, str(stage / "items.faiss"))

    golden = json.loads((stage / "golden.json").read_text())
    sess = ort.InferenceSession(
        str(stage / "sasrec_encoder.onnx"), providers=["CPUExecutionProvider"]
    )
    users = encode(sess, [q["history"] for q in golden["queries"]], golden["history_len"])
    scores, rows = index.search(users, golden["top"])
    recall, worst = [], 0.0
    for q, s, r in zip(golden["queries"], scores, rows, strict=True):
        recall.append(len(set(r.tolist()) & set(q["items"])) / golden["top"])
        worst = max(worst, float(np.abs(s - np.asarray(q["scores"])).max()))
    return {
        "ntotal": int(index.ntotal),
        "dim": int(index.d),
        "golden_recall": float(np.mean(recall)),
        "golden_max_score_diff": worst,
    }


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1]))))
