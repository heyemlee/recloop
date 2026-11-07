"""Train SASRec on the train split, tune on the validation split (`make train`).

Every configuration of GRID is trained with early stopping on validation NDCG@10 (patience in
SASRecConfig); the configuration with the best validation NDCG@10 is kept. The test split is not
touched here: `make eval` scores the saved model once.

Training runs on the CPU with a fixed seed and a fixed number of threads per worker, so a re-run
reproduces the same weights (and `make eval` the same offline.md). Configurations run in
parallel worker processes.

Writes artifacts/models/sasrec/model.pt (best checkpoint) and tuning.json (every trial with its
validation curve).

Usage: python -m ml.models.train_sasrec [--workers 4] [--threads 3] [--grid none]
"""

from __future__ import annotations

import argparse
import io
import itertools
import json
import multiprocessing as mp
import time
from dataclasses import asdict, replace

import numpy as np
import torch

from ml import paths
from ml.eval.splits import NUM_ITEMS, load_split, train_sequences
from ml.models.pop import Pop
from ml.models.sasrec import SASRec, SASRecConfig, SASRecNet, TrainLog, train

MODEL_DIR = paths.ARTIFACTS / "models" / "sasrec"
CHECKPOINT = MODEL_DIR / "model.pt"  # the selected configuration
TUNING_JSON = MODEL_DIR / "tuning.json"
TRIALS = MODEL_DIR / "trials"  # every trial's best checkpoint, trials/{i}.pt (i = trial index)


def trial_checkpoint(i: int):
    return TRIALS / f"{i}.pt"


BASE = SASRecConfig()
GRID = {
    "dropout": (0.2, 0.5),
    "norm_first": (False, True),
    "loss_positions": ("all", "last"),
}
# Trained alongside the grid: the upstream loss (BCE against one sampled negative), the setup of
# the published SASRec baselines; it shows how much of the gap to the paper is the loss.
EXTRA = [replace(BASE, loss="bce")]


def grid_configs(grid: dict | None = None) -> list[SASRecConfig]:
    grid = GRID if grid is None else grid
    keys = list(grid)
    return [
        replace(BASE, **dict(zip(keys, c, strict=True))) for c in itertools.product(*grid.values())
    ]


def popularity_tie_pos(sequences) -> np.ndarray:
    """Position of every item in the evaluation's tie order (training popularity, then id)."""
    counts = Pop().fit(sequences, NUM_ITEMS).counts
    order = np.lexsort((np.arange(NUM_ITEMS), -counts))
    pos = np.empty_like(order)
    pos[order] = np.arange(NUM_ITEMS)
    return pos


def _run(args: tuple[dict, int]) -> tuple[dict, dict, bytes]:
    cfg_dict, threads = args
    torch.set_num_threads(threads)
    cfg = SASRecConfig(**cfg_dict)
    sequences = train_sequences(load_split("train"))
    net, log = train(
        cfg, sequences, NUM_ITEMS, load_split("valid"), popularity_tie_pos(sequences), "cpu"
    )
    # Send the weights back as bytes: tensors would go through shared memory.
    buf = io.BytesIO()
    torch.save({k: v.cpu() for k, v in net.state_dict().items()}, buf)
    return cfg_dict, asdict(log), buf.getvalue()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", type=int, default=4, help="parallel training processes")
    parser.add_argument("--threads", type=int, default=3, help="torch threads per process")
    parser.add_argument(
        "--grid", choices=("default", "none"), default="default", help="'none': BASE only"
    )
    args = parser.parse_args(argv)

    configs = grid_configs() + EXTRA if args.grid == "default" else [BASE]
    t0 = time.monotonic()
    print(f"training {len(configs)} SASRec configuration(s), {args.workers} workers", flush=True)
    jobs = [(asdict(c), args.threads) for c in configs]
    ctx = mp.get_context("spawn")
    results = []
    with ctx.Pool(min(args.workers, len(jobs))) as pool:
        for cfg_dict, log, state in pool.imap(_run, jobs):
            cfg = SASRecConfig(**cfg_dict)
            v = log["best_valid"]
            print(
                f"  valid HR@10 {v['HR@10']:.4f} NDCG@10 {v['NDCG@10']:.4f} "
                f"(epoch {log['best_epoch']}/{len(log['epochs'])}, {log['seconds']:.0f}s)  "
                f"{cfg.label()}",
                flush=True,
            )
            results.append((cfg, log, state))

    # Best validation NDCG@10; ties: HR@10, then grid order.
    best_i = max(
        range(len(results)),
        key=lambda i: (
            results[i][1]["best_valid"]["NDCG@10"],
            results[i][1]["best_valid"]["HR@10"],
            -i,
        ),
    )
    for i, (cfg, log, state) in enumerate(results):
        model = SASRec(cfg)
        model.net = SASRecNet(NUM_ITEMS, cfg)
        model.net.load_state_dict(torch.load(io.BytesIO(state), weights_only=True))
        model.net.eval()
        model.log = TrainLog(**log)
        model.save(trial_checkpoint(i))
        if i == best_i:
            model.save(CHECKPOINT)
    cfg, log, _ = results[best_i]

    tuning = {
        "base": asdict(BASE),
        "grid": {k: list(v) for k, v in GRID.items()} if args.grid == "default" else {},
        "extra": [asdict(c) for c in EXTRA] if args.grid == "default" else [],
        "select_by": "NDCG@10",
        "selected": best_i,
        "threads_per_worker": args.threads,
        "torch": torch.__version__,
        "trials": [
            {
                "config": asdict(c),
                "best_epoch": lg["best_epoch"],
                "epochs_run": len(lg["epochs"]),
                "valid": lg["best_valid"],
                "seconds": lg["seconds"],
                "curve": lg["epochs"],
            }
            for c, lg, _ in results
        ],
    }
    TUNING_JSON.write_text(json.dumps(tuning, indent=2) + "\n")
    v = log["best_valid"]
    print(
        f"selected: {cfg.label()} -- valid HR@10 {v['HR@10']:.4f} NDCG@10 {v['NDCG@10']:.4f}, "
        f"epoch {log['best_epoch']}"
    )
    print(f"wrote {CHECKPOINT.relative_to(paths.ROOT)} in {time.monotonic() - t0:.0f}s")


if __name__ == "__main__":
    main()
