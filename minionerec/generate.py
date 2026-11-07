"""MiniOneRec on the published test split, on the Mac's GPU (`make minionerec-gen`).

Runs MiniOneRec's evaluate.py (patched for Apple Silicon, see minionerec/upstream.py) over the
published test CSV, or over the samples of a seeded subset of test users (every sample of each
chosen user, in file order, so the user-level bootstrap still applies), then MiniOneRec's
calc.py on the result. Defaults are evaluate.sh's: 50 beams, batch 8, at most 256 new tokens,
length penalty 0, bfloat16.

The full split runs as evaluate.sh's 8 shards (split.py's boundaries), one process after the
other on the one GPU, then merge.py; a subset runs as one shard. Shards make a night-long run
resumable -- rerunning the same command skips the shards that finished -- and keep every batch
of 8 exactly as evaluate.sh forms it (batches never cross a shard boundary).

--dtype auto runs one batch in bfloat16 first, then float16, then float32, and keeps the first
dtype whose batch finishes and yields only SIDs that exist in the catalog (MPS lacks some
bfloat16 kernels on older macOS versions).

Writes artifacts/minionerec/runs/<name>/: rows.json (row numbers of the evaluated samples in the
published CSV, in order), result.json (evaluate.py's output, merged: input, output, predict),
timing.json (per batch), run.json (settings, versions, machine, wall time, memory, calc.py's
numbers), calc.txt, and shards/ (each shard's input, result, timing and log). Scoring with
RecLoop's metrics is `make minionerec` (ml environment).

Usage:
  python -m minionerec.generate --name test                  # full test split (~5 h at 50 beams)
  python -m minionerec.generate --name trial-200 --users 200 # 200 users (~480 samples)
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from minionerec import upstream
from ml import paths
from ml.eval import machine

RUNS = paths.ARTIFACTS / "minionerec" / "runs"
STAGE = paths.ARTIFACTS / "minionerec" / "upstream"
SEED = 20261106
DTYPES = ("bfloat16", "float16", "float32")
EVALUATE_SH_SHARDS = 8


def select_rows(test_csv: Path, users: int | None, seed: int = SEED) -> np.ndarray:
    """Row numbers of the samples to evaluate: all rows, or every row of `users` random users.

    The users are a prefix of one seeded permutation, so with the same seed a smaller subset is
    inside a larger one and runs with other settings can be compared sample by sample.
    """
    user_col = pd.read_csv(test_csv, usecols=["user_id"]).user_id.to_numpy()
    if users is None:
        return np.arange(len(user_col))
    uniq = np.unique(user_col)
    if users > len(uniq):
        raise SystemExit(f"--users {users}: the split has only {len(uniq)} users")
    chosen = np.random.default_rng(seed).permutation(uniq)[:users]
    return np.flatnonzero(np.isin(user_col, chosen))


def write_subset(test_csv: Path, rows: np.ndarray, out_csv: Path) -> Path:
    """The rows of the published CSV, same columns and values, for evaluate.py to read."""
    df = pd.read_csv(test_csv, dtype=str, keep_default_na=False)
    if len(rows) == len(df) and np.array_equal(rows, np.arange(len(df))):
        return test_csv  # the published file itself
    df.iloc[rows].to_csv(out_csv, index=False)
    return out_csv


def versions() -> dict:
    import tokenizers
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "mps_available": torch.backends.mps.is_available(),
    }


def predictions_valid(result_json: Path) -> tuple[bool, int]:
    """(all predictions are catalog SIDs, number that are not)."""
    sids = {ln.split("\t")[0].strip() for ln in paths.PUBLIC_INFO.read_text().splitlines()}
    rows = json.loads(result_json.read_text())
    bad = sum(p.strip('"\n').strip() not in sids for r in rows for p in r["predict"])
    return bad == 0, bad


def _command(script: Path, csv: Path, result: Path, timing: Path, args, dtype: str) -> list[str]:
    return upstream.evaluate_command(
        script,
        csv,
        result,
        timing,
        num_beams=args.num_beams,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        length_penalty=args.length_penalty,
        dtype=dtype,
        base_model=args.base_model,
    )


def probe_dtype(script: Path, test_csv: Path, workdir: Path, args) -> tuple[str, list[dict]]:
    """First dtype in DTYPES whose single batch runs and yields valid SIDs."""
    probe_csv = write_subset(
        test_csv, np.arange(min(args.batch_size, 8)), workdir / "probe_input.csv"
    )
    tried = []
    for dtype in DTYPES:
        out = workdir / f"probe_{dtype}.json"
        cmd = _command(script, probe_csv, out, workdir / f"probe_{dtype}_timing.json", args, dtype)
        res = upstream.run_logged(cmd, script.parent, workdir / f"probe_{dtype}.log")
        ok, bad = (False, None) if res["returncode"] != 0 else predictions_valid(out)
        tried.append({"dtype": dtype, "returncode": res["returncode"], "invalid": bad, "ok": ok})
        print(f"dtype probe {dtype}: {'ok' if ok else 'failed'} ({tried[-1]})", flush=True)
        if ok:
            return dtype, tried
    raise SystemExit(f"no dtype works on this machine: {tried}")


def run_shards(
    script: Path, test_csv: Path, rows: np.ndarray, out_dir: Path, args, dtype: str
) -> tuple[list[dict], dict]:
    """Run (or resume) every shard; merge results with merge.py; return shard info, timing."""
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(exist_ok=True)
    bounds = upstream.shard_bounds(len(rows), args.shards)
    infos = []
    for i, (lo, hi) in enumerate(bounds):
        done = shard_dir / f"{i}_run.json"
        if done.exists():
            infos.append(json.loads(done.read_text()))
            print(f"shard {i}: done earlier, skipped", flush=True)
            continue
        csv = write_subset(test_csv, rows[lo:hi], shard_dir / f"{i}.csv")
        result, timing = shard_dir / f"{i}.json", shard_dir / f"{i}_timing.json"
        print(
            f"[{time.strftime('%H:%M:%S')}] shard {i + 1} of {len(bounds)}: rows {lo}-{hi - 1}",
            flush=True,
        )
        res = upstream.run_logged(
            _command(script, csv, result, timing, args, dtype),
            script.parent,
            shard_dir / f"{i}.log",
        )
        if res["returncode"] != 0:
            raise SystemExit(
                f"evaluate.py failed on shard {i} ({res['returncode']}); see "
                f"{shard_dir / f'{i}.log'}. Rerun the same command to resume."
            )
        if len(json.loads(result.read_text())) != hi - lo:
            raise SystemExit(f"shard {i}: {result} does not have {hi - lo} results")
        info = {
            "shard": i,
            "rows": [lo, hi],
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            **res,
        }
        done.write_text(json.dumps(info, indent=1) + "\n")
        infos.append(info)
    upstream.run_merge(shard_dir, len(bounds), out_dir / "result.json")
    parts = [json.loads((shard_dir / f"{i}_timing.json").read_text()) for i in range(len(bounds))]
    timing = {
        "device": parts[0]["device"],
        "dtype": parts[0]["dtype"],
        "setup_seconds": sum(p["setup_seconds"] for p in parts),
        "shards": [
            {"setup_seconds": p["setup_seconds"], "batches": len(p["batches"])} for p in parts
        ],
        "batches": [b for p in parts for b in p["batches"]],
    }
    (out_dir / "timing.json").write_text(json.dumps(timing, indent=1) + "\n")
    return infos, timing


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True, help="run directory under artifacts/minionerec/runs")
    ap.add_argument("--users", type=int, default=None, help="random test users (default: all)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--num-beams", type=int, default=upstream.EVALUATE_SH["num_beams"])
    ap.add_argument("--batch-size", type=int, default=upstream.EVALUATE_SH["batch_size"])
    ap.add_argument("--max-new-tokens", type=int, default=upstream.EVALUATE_SH["max_new_tokens"])
    ap.add_argument("--length-penalty", type=float, default=upstream.EVALUATE_SH["length_penalty"])
    ap.add_argument("--dtype", default="bfloat16", choices=("auto", *DTYPES))
    ap.add_argument(
        "--shards",
        type=int,
        default=None,
        help=f"default: {EVALUATE_SH_SHARDS} (evaluate.sh's) for the full split, 1 for a subset",
    )
    ap.add_argument(
        "--base-model", type=Path, default=paths.HF_CKPT, help="model directory (tests only)"
    )
    args = ap.parse_args(argv)
    if args.num_beams < 10:
        raise SystemExit("--num-beams must be at least 10: the report's main metric is HR@10")
    if args.shards is None:
        args.shards = EVALUATE_SH_SHARDS if args.users is None else 1

    out_dir = RUNS / args.name
    if (out_dir / "run.json").exists():
        raise SystemExit(f"{out_dir} already has a finished run; pick another --name")
    out_dir.mkdir(parents=True, exist_ok=True)
    script = upstream.stage(STAGE)
    test_csv = paths.public_split_csv("test")
    rows = select_rows(test_csv, args.users, args.seed)
    settings = {
        "num_beams": args.num_beams,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "length_penalty": args.length_penalty,
        "dtype_requested": args.dtype,
        "shards": args.shards,
        "base_model": str(args.base_model),
    }
    start_file = out_dir / "start.json"  # what a resumed run must match
    if start_file.exists():
        start = json.loads(start_file.read_text())
        if start["rows"] != rows.tolist() or start["settings"] != settings:
            raise SystemExit(f"{out_dir} holds a run with other rows or settings")
        print(f"{args.name}: resuming the run started {start['started']}", flush=True)
    else:
        dtype, probes = args.dtype, None
        if dtype == "auto":
            dtype, probes = probe_dtype(script, test_csv, out_dir, args)
        start = {
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "rows": rows.tolist(),
            "settings": settings,
            "dtype": dtype,
            "dtype_probes": probes,
        }
        start_file.write_text(json.dumps(start) + "\n")
        (out_dir / "rows.json").write_text(json.dumps(rows.tolist()) + "\n")
    dtype = start["dtype"]

    host = machine.info()
    print(
        f"{args.name}: {len(rows):,} samples, {args.num_beams} beams, batch {args.batch_size}, "
        f"{dtype}, {args.shards} shard(s); logs in {out_dir / 'shards'}",
        flush=True,
    )
    infos, timing = run_shards(script, test_csv, rows, out_dir, args, dtype)
    calc = upstream.run_calc(out_dir / "result.json", out_dir / "calc.txt")

    peaks = [(i["memory"] or {}).get("lifetime_max_footprint", 0) for i in infos]
    run = {
        "name": args.name,
        "split": "test",
        "samples": len(rows),
        "selection": "all" if args.users is None else {"users": args.users, "seed": args.seed},
        "settings": {
            **{k: v for k, v in settings.items() if k != "base_model"},
            "dtype": dtype,
            "dtype_probes": start["dtype_probes"],
            "shard_bounds": upstream.shard_bounds(len(rows), args.shards),
        },
        "command": [
            Path(c).name if c.startswith("/") else c
            for c in _command(
                script, Path("shard.csv"), Path("r.json"), Path("t.json"), args, dtype
            )
        ],
        "upstream": {
            "clone": "third_party/MiniOneRec",
            "patched_evaluate_sha256": upstream.PATCHED_EVALUATE_SHA256,
            "files_sha256": upstream.UPSTREAM_SHA256,
            "tools_sha256": upstream.TOOLS_SHA256,
        },
        "checkpoint": (
            "data/hf/Office_ckpt (ml/data/download.py pins revision and SHA-256)"
            if args.base_model == paths.HF_CKPT
            else str(args.base_model)
        ),
        "versions": versions(),
        "machine": host,
        "started": start["started"],
        "device": timing["device"],
        "wall_seconds": sum(i["wall_seconds"] for i in infos),
        "setup_seconds": timing["setup_seconds"],
        "generate_seconds": sum(b["seconds"] for b in timing["batches"]),
        "loadavg_start": infos[0]["loadavg_start"],
        "loadavg_end": infos[-1]["loadavg_end"],
        "loadavg_max_start": max(i["loadavg_start"][0] for i in infos),
        "memory": {"lifetime_max_footprint": max(peaks), "per_shard": peaks},
        "calc_py": calc,
    }
    (out_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    ks = [k for k in (1, 3, 5, 10, 20, 50) if k <= args.num_beams]
    hr = dict(zip(ks, calc["HR"] or [], strict=False))
    nd = dict(zip(ks, calc["NDCG"] or [], strict=False))
    per = run["generate_seconds"] / len(rows)
    print(
        f"{args.name}: calc.py HR@10 {hr.get(10, float('nan')):.4f} NDCG@10 "
        f"{nd.get(10, float('nan')):.4f}, invalid predictions {calc['invalid_predictions']}; "
        f"{run['generate_seconds']:.0f} s generating ({per:.2f} s per sample), "
        f"{run['wall_seconds']:.0f} s in all; peak footprint {max(peaks) / 2**30:.1f} GiB",
        flush=True,
    )


if __name__ == "__main__":
    main()
