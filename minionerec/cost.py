"""Latency, throughput and memory of MiniOneRec generation on the Mac's GPU.

`make minionerec-cost` runs this, then ml.eval.sasrec_cost for the SASRec side. Run it on an
idle machine (no compose stack, no other measurement). Each configuration starts only when the
1-minute load average is below 2 and no Docker container runs (ml.eval.machine.measure_quietly
waits up to 20 min for that); `docker ps` is checked every 10 s while it runs, and a
configuration during which a container ran is run once more. The manifest keeps the load and
the containers at the start and the end of every attempt; the report marks what stayed
disturbed or started busy.

The workload is one test sample per user for USERS random test users (seeded; each user's
sample is drawn at random too), so every request is a different user. Every configuration runs
in a fresh process of MiniOneRec's own evaluate.py (patched for Apple Silicon, see
upstream.py), in bfloat16 unless --dtype says otherwise:

  * beams 10, 20 and 50 at batch 1: one user per generate() call, i.e. the latency of one
    recommendation request;
  * the same beams at batch 8 (evaluate.sh's batch size): throughput;
  * a prefill-only run: 50 beams, batch 1, one new token. Hugging Face's beam search copies the
    prompt once per beam before the first forward pass, so this is the cost of encoding the same
    prompt 50 times; the rest of a request is the 4 remaining decoding steps (an SID is three
    tokens, then a newline and the end-of-sequence token).

Each configuration's first calls include Metal kernel compilation; the report leaves them out.
Writes artifacts/minionerec/cost/<config>/ (timing.json, result.json, evaluate.log, input.csv)
and artifacts/minionerec/cost/manifest.json.

Usage: python -m minionerec.cost [--users 96] [--dtype bfloat16] [--only beam50_batch1 ...]
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from minionerec import generate, upstream
from ml import paths
from ml.eval import machine

COST = paths.ARTIFACTS / "minionerec" / "cost"
USERS = 96
PREFILL_USERS = 32
SEED = 20261107
BEAMS = (10, 20, 50)
BATCH_SIZES = (1, 8)


def configs() -> list[dict]:
    out = [
        {"name": f"beam{b}_batch{s}", "num_beams": b, "batch_size": s, "max_new_tokens": 256}
        for b in BEAMS
        for s in BATCH_SIZES
    ]
    out.append(
        {
            "name": "beam50_batch1_prefill",
            "num_beams": 50,
            "batch_size": 1,
            "max_new_tokens": 1,
            "limit": PREFILL_USERS,
        }
    )
    return out


def one_sample_per_user(test_csv, users: int, seed: int = SEED) -> np.ndarray:
    """Row numbers of one random sample of each of `users` random users, in file order."""
    user_col = pd.read_csv(test_csv, usecols=["user_id"]).user_id.to_numpy()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.unique(user_col), size=users, replace=False)
    rows = [int(rng.choice(np.flatnonzero(user_col == u))) for u in chosen]
    return np.sort(np.asarray(rows))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--users", type=int, default=USERS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dtype", default="bfloat16", choices=generate.DTYPES)
    ap.add_argument("--only", nargs="*", default=None, help="configuration names to (re)run")
    ap.add_argument(
        "--out", type=Path, default=COST, help="output directory (default: %(default)s)"
    )
    ap.add_argument(
        "--base-model", type=Path, default=paths.HF_CKPT, help="model directory (tests only)"
    )
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    script = upstream.stage(generate.STAGE)
    test_csv = paths.public_split_csv("test")
    rows = one_sample_per_user(test_csv, args.users, args.seed)
    manifest_path = args.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest and manifest.get("rows") != rows.tolist():
        raise SystemExit(f"{manifest_path} was written for other rows; move it away first")
    manifest.update(
        {
            "users": args.users,
            "seed": args.seed,
            "rows": rows.tolist(),
            "dtype": args.dtype,
            "attn_implementation": upstream.ATTN_IMPLEMENTATION,
            "checkpoint": (
                "data/hf/Office_ckpt (ml/data/download.py pins revision and SHA-256)"
                if args.base_model == paths.HF_CKPT
                else str(args.base_model)
            ),
            "versions": generate.versions(),
            "machine": machine.info(),
        }
    )
    manifest.setdefault("configs", {})

    for cfg in configs():
        if args.only is not None and cfg["name"] not in args.only:
            continue
        out_dir = args.out / cfg["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg_rows = rows[: cfg.get("limit", len(rows))]
        input_csv = generate.write_subset(test_csv, cfg_rows, out_dir / "input.csv")
        cmd = upstream.evaluate_command(
            script,
            input_csv,
            out_dir / "result.json",
            out_dir / "timing.json",
            num_beams=cfg["num_beams"],
            batch_size=cfg["batch_size"],
            max_new_tokens=cfg["max_new_tokens"],
            length_penalty=upstream.EVALUATE_SH["length_penalty"],
            dtype=args.dtype,
            base_model=args.base_model,
        )
        print(f"[{time.strftime('%H:%M:%S')}] {cfg['name']}: {len(cfg_rows)} users", flush=True)
        res, quiet = machine.measure_quietly(
            functools.partial(upstream.run_logged, cmd, script.parent, out_dir / "evaluate.log")
        )
        if res["returncode"] != 0:
            raise SystemExit(f"{cfg['name']} failed; see {out_dir / 'evaluate.log'}")
        valid, invalid = generate.predictions_valid(out_dir / "result.json")
        timing = json.loads((out_dir / "timing.json").read_text())
        manifest["configs"][cfg["name"]] = {
            **cfg,
            "samples": len(cfg_rows),
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "wall_seconds": res["wall_seconds"],
            "device": timing["device"],
            "setup_seconds": timing["setup_seconds"],
            "loadavg_start": res["loadavg_start"],
            "loadavg_end": res["loadavg_end"],
            "memory": res["memory"],
            "swapped_out_bytes": res["swapped_out_bytes"],
            "invalid_predictions": invalid if cfg["max_new_tokens"] > 1 else None,
            "quiet": quiet,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        seconds = [b["seconds"] for b in timing["batches"]]
        print(
            f"  median {np.median(seconds):.2f} s per call, "
            f"peak footprint {res['memory']['lifetime_max_footprint'] / 2**30:.1f} GiB",
            flush=True,
        )
        if cfg["max_new_tokens"] > 1 and not valid:
            raise SystemExit(f"{cfg['name']}: {invalid} predictions are not catalog SIDs")


if __name__ == "__main__":
    main()
