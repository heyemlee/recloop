"""SASRec on the CPU, timed on MiniOneRec's cost workload (second half of `make minionerec-cost`).

The requests are the ones minionerec.cost generated for (artifacts/minionerec/cost/manifest.json:
one test sample per user). A request is what MiniOneRec's 50 beams answer: encode the history
with the selected SASRec checkpoint (PyTorch, CPU), score all 3,459 items, keep the top 50.
Latency is per request (batch 1), once on one thread and once on all of PyTorch's default
threads; throughput is at batch 8 and 256 on the default threads. Every setting is warmed up
first. (Online, the model server runs the same encoder in ONNX Runtime and searches FAISS; one
Retrieve took 1-3 ms including the local gRPC round trip in M1, docs/design.md §19.)

Writes artifacts/minionerec/sasrec_cost.json. Usage: python -m ml.eval.sasrec_cost
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from ml import paths
from ml.eval import footprint, machine
from ml.eval.splits import load_split
from ml.models import train_sasrec
from ml.models.sasrec import SASRec, encode_histories

MANIFEST = paths.ARTIFACTS / "minionerec" / "cost" / "manifest.json"
OUT_JSON = paths.ARTIFACTS / "minionerec" / "sasrec_cost.json"
TOP_K = 50
REPEATS = 5  # passes over the workload per latency setting
WARMUP = 20
BATCH_SIZES = (8, 256)


@torch.inference_mode()
def recommend(model: SASRec, items: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Top-50 item ids per row of `ids` (shifted ids, 0 = padding)."""
    return torch.topk(model.net(ids) @ items.T, TOP_K, dim=1).indices


def latency(model: SASRec, items: torch.Tensor, ids: torch.Tensor, threads: int) -> dict:
    torch.set_num_threads(threads)
    for i in range(WARMUP):
        recommend(model, items, ids[i % len(ids)][None])
    seconds = []
    for _ in range(REPEATS):
        for row in ids:
            t0 = time.perf_counter()
            recommend(model, items, row[None])
            seconds.append(time.perf_counter() - t0)
    return {"threads": threads, "batch_size": 1, "seconds": seconds}


def throughput(model: SASRec, items: torch.Tensor, ids: torch.Tensor, batch: int) -> dict:
    reps = max(1, int(np.ceil(batch / len(ids))))
    work = ids.repeat(reps, 1)[: max(batch, len(ids))]
    batches = [work[i : i + batch] for i in range(0, len(work) - batch + 1, batch)]
    for b in batches[:3]:
        recommend(model, items, b)
    seconds = []
    for _ in range(REPEATS):
        for b in batches:
            t0 = time.perf_counter()
            recommend(model, items, b)
            seconds.append(time.perf_counter() - t0)
    return {"threads": torch.get_num_threads(), "batch_size": batch, "seconds": seconds}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    args = ap.parse_args(argv)
    if not args.manifest.exists():
        raise SystemExit("no MiniOneRec cost workload: run python -m minionerec.cost first")
    rows = np.asarray(json.loads(args.manifest.read_text())["rows"])
    test = load_split("test")
    model = SASRec.load(train_sasrec.CHECKPOINT)
    model.net.eval()
    items = model.net.item_vectors().detach()
    ids = encode_histories(test.history_matrix()[rows])
    default_threads = torch.get_num_threads()

    load_start = os.getloadavg()
    runs = [latency(model, items, ids, 1), latency(model, items, ids, default_threads)]
    torch.set_num_threads(default_threads)
    runs += [throughput(model, items, ids, b) for b in BATCH_SIZES]
    fp = footprint.read()
    out = {
        "checkpoint": str(train_sasrec.CHECKPOINT.relative_to(paths.ROOT)),
        "description": model.describe(),
        "parameters": int(sum(p.numel() for p in model.net.parameters())),
        "parameter_bytes": int(sum(p.numel() * p.element_size() for p in model.net.parameters())),
        "requests": len(rows),
        "top_k": TOP_K,
        "repeats": REPEATS,
        "torch": torch.__version__,
        "default_threads": default_threads,
        "machine": machine.info(),
        "loadavg_start": list(load_start),
        "loadavg_end": list(os.getloadavg()),
        "footprint": None if fp is None else {"current": fp.current, "peak": fp.lifetime_max},
        "runs": runs,
    }
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    for r in runs:
        s = np.asarray(r["seconds"])
        print(
            f"SASRec batch {r['batch_size']:>3}, {r['threads']:>2} threads: median "
            f"{np.median(s) * 1e3:.3f} ms per call, P99 {np.quantile(s, 0.99) * 1e3:.3f} ms"
        )


if __name__ == "__main__":
    main()
