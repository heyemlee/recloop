"""Replay a finished run's batches through minionerec.engine and compare every SID sequence.

The offline check of "encode the prompt once" (docs/design.md §11 step 5): the engine, with the
prompt pass shared across the beams (engine.share_prompt_pass), reads the reference run's own
shard files (runs/<reference>/shards/<i>.csv) and forms its batches exactly as evaluate.py did
(the same prompts in the same order, batch_size at a time, never across a shard), so the only
difference left is the first forward pass computing one copy of each prompt instead of
num_beams. For every sample the 50 predicted SIDs, in order, are compared with the reference's
(runs/<reference>/shards/<i>.json). The gate is all of them identical.

Writes artifacts/minionerec/runs/<name>/ like minionerec.generate (rows.json, result.json,
timing.json, run.json with settings["prefill"], calc.txt from MiniOneRec's calc.py) plus
compare.json, so `make minionerec` reports the run next to the others ("Other settings on the
same samples": identical top-10, overlap, HR@10).

  python -m minionerec.replay --name once-trial --batches 25   # the first 25 batches (200 samples)
  python -m minionerec.replay --name once-test                 # every shard of the main run
  python -m minionerec.replay --name plain-trial --batches 25 --per-beam   # the unchanged path

`csv` mode is evaluate.py's interface on one CSV (the cost harness runs it like evaluate.py):

  python -m minionerec.replay csv --test-data-path in.csv --result-json r.json --timing-json t.json \\
      --batch-size 8 --num-beams 50 --max-new-tokens 256 --dtype bfloat16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from minionerec import engine, generate, upstream
from ml import paths
from ml.eval import footprint, machine

RUNS = generate.RUNS


def evaluate_csv(gen: engine.Generator, csv: Path, batch_size: int, max_batches: int | None = None):
    """evaluate.py's loop over one CSV: (result rows, per-batch timing). With max_batches, only
    that many batches (the first ones) run and only their samples are returned."""
    ds = gen.dataset(csv)
    encodings = [ds[i] for i in range(len(ds))]
    test_data = ds.get_all()
    n_batches = (len(encodings) + batch_size - 1) // batch_size
    if max_batches is not None:
        n_batches = min(n_batches, max_batches)
    predict, timing = [], []
    for b in range(n_batches):
        out = gen.generate_encoded(encodings[b * batch_size : (b + 1) * batch_size])
        predict.extend(out.sids)
        entry = {"size": len(out.sids), "seconds": out.seconds}
        if out.mps_allocated is not None:
            entry["mps_allocated"] = out.mps_allocated
            entry["mps_driver_allocated"] = out.mps_driver_allocated
        timing.append(entry)
        print(
            f"[{time.strftime('%H:%M:%S')}] batch {b + 1}/{n_batches}: {out.seconds:.2f} s",
            flush=True,
        )
    rows = []
    for i, p in enumerate(predict):
        r = dict(test_data[i])
        r.pop("dedup", None)  # as evaluate.py
        r["predict"] = p
        rows.append(r)
    return rows, timing


def compare(ours: list[dict], reference: list[dict]) -> dict:
    """Per sample: the whole list of SIDs equal (same order), the top 10 equal."""
    same = [a["predict"] == b["predict"] for a, b in zip(ours, reference, strict=False)]
    top10 = [a["predict"][:10] == b["predict"][:10] for a, b in zip(ours, reference, strict=False)]
    first = next((i for i, s in enumerate(same) if not s), None)
    out = {
        "samples": len(same),
        "identical_sequences": int(sum(same)),
        "identical_top10": int(sum(top10)),
        "all_identical": bool(same) and all(same),
    }
    if first is not None:
        out["first_difference"] = {
            "index": first,
            "ours": ours[first]["predict"][:10],
            "reference": reference[first]["predict"][:10],
        }
    return out


def _replay(args) -> None:
    ref_dir = RUNS / args.reference
    ref = json.loads((ref_dir / "run.json").read_text())
    ref_rows = json.loads((ref_dir / "rows.json").read_text())
    s = ref["settings"]
    out_dir = RUNS / args.name
    if (out_dir / "run.json").exists():
        raise SystemExit(f"{out_dir} already has a finished run; pick another --name")
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    load_start, vm_start = os.getloadavg(), machine.vm_counters()
    t_start = time.monotonic()
    sampler = footprint.Sampler(os.getpid()).start()
    gen = engine.Generator(
        args.base_model,
        num_beams=s["num_beams"],
        max_new_tokens=s["max_new_tokens"],
        length_penalty=s["length_penalty"],
        dtype=s["dtype"],
        prefill_once=not args.per_beam,
    )
    results, batches, rows, compared = [], [], [], []
    left = args.batches
    for i, (lo, hi) in enumerate(s["shard_bounds"]):
        if left is not None and left <= 0:
            break
        shard_rows, shard_timing = evaluate_csv(
            gen, ref_dir / "shards" / f"{i}.csv", s["batch_size"], left
        )
        reference = json.loads((ref_dir / "shards" / f"{i}.json").read_text())
        compared.append(compare(shard_rows, reference))
        print(f"shard {i}: {compared[-1]}", flush=True)
        results.extend(shard_rows)
        batches.extend(shard_timing)
        rows.extend(ref_rows[lo : lo + len(shard_rows)])
        if left is not None:
            left -= len(shard_timing)
    memory = sampler.stop()
    summary = {
        "samples": sum(c["samples"] for c in compared),
        "identical_sequences": sum(c["identical_sequences"] for c in compared),
        "identical_top10": sum(c["identical_top10"] for c in compared),
        "per_shard": compared,
    }
    summary["all_identical"] = summary["identical_sequences"] == summary["samples"] > 0
    (out_dir / "rows.json").write_text(json.dumps(rows) + "\n")
    (out_dir / "result.json").write_text(json.dumps(results, indent=4))
    (out_dir / "compare.json").write_text(json.dumps(summary, indent=2) + "\n")
    timing = {
        "device": gen.device,
        "dtype": gen.dtype,
        "attn_implementation": gen.attn_implementation,
        "setup_seconds": gen.setup_seconds,
        "batches": batches,
    }
    (out_dir / "timing.json").write_text(json.dumps(timing, indent=1) + "\n")
    calc = upstream.run_calc(out_dir / "result.json", out_dir / "calc.txt")
    run = {
        "name": args.name,
        "split": "test",
        "samples": len(rows),
        "selection": {"replay_of": args.reference, "batches": args.batches or "all"},
        "settings": {
            **{k: s[k] for k in ("num_beams", "batch_size", "max_new_tokens", "length_penalty")},
            "dtype": gen.dtype,
            "attn_implementation": gen.attn_implementation,
            "shards": s.get("shards"),
            "shard_bounds": s["shard_bounds"],
            "prefill": "per_beam" if args.per_beam else "once",
            "engine": "minionerec.engine",
        },
        "command": ["python", "-m", "minionerec.replay", *sys.argv[1:]],
        "upstream": {
            "clone": "third_party/MiniOneRec",
            "files_sha256": upstream.UPSTREAM_SHA256,
            "tools_sha256": upstream.TOOLS_SHA256,
        },
        "checkpoint": (
            "data/hf/Office_ckpt (ml/data/download.py pins revision and SHA-256)"
            if args.base_model == paths.HF_CKPT
            else str(args.base_model)
        ),
        "versions": generate.versions(),
        "machine": machine.info(),
        "started": started,
        "device": gen.device,
        "wall_seconds": time.monotonic() - t_start,
        "setup_seconds": gen.setup_seconds,
        "generate_seconds": sum(b["seconds"] for b in batches),
        "loadavg_start": list(load_start),
        "loadavg_end": list(os.getloadavg()),
        "memory": memory,
        "swapped_out_bytes": machine.swapped_out_bytes(vm_start, machine.vm_counters()),
        "containers_end": machine.containers(),
        "calc_py": calc,
        "replay": {k: v for k, v in summary.items() if k != "per_shard"},
    }
    (out_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    per = run["generate_seconds"] / max(1, len(rows))
    print(
        f"{args.name}: {summary['identical_sequences']:,} of {summary['samples']:,} samples have "
        f"the reference's 50 SIDs in the same order ({summary['identical_top10']:,} the same top "
        f"10); {per:.2f} s per sample; peak footprint "
        f"{memory['lifetime_max_footprint'] / 2**30:.1f} GiB",
        flush=True,
    )
    if not summary["all_identical"]:
        raise SystemExit(f"{args.name}: not identical to {args.reference}; see compare.json")


def _csv(args) -> None:
    """evaluate.py's command line on one CSV (result + timing files in its formats)."""
    gen = engine.Generator(
        args.base_model,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        length_penalty=args.length_penalty,
        dtype=args.dtype,
        prefill_once=not args.per_beam,
    )
    rows, batches = evaluate_csv(gen, args.test_data_path, args.batch_size)
    Path(args.result_json).write_text(json.dumps(rows, indent=4))
    Path(args.timing_json).write_text(
        json.dumps(
            {
                "device": gen.device,
                "dtype": gen.dtype,
                "attn_implementation": gen.attn_implementation,
                "setup_seconds": gen.setup_seconds,
                "prefill": "per_beam" if args.per_beam else "once",
                "batches": batches,
            },
            indent=1,
        )
    )


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["csv"]:
        ap = argparse.ArgumentParser(prog="minionerec.replay csv")
        ap.add_argument("--test-data-path", type=Path, required=True)
        ap.add_argument("--result-json", type=Path, required=True)
        ap.add_argument("--timing-json", type=Path, required=True)
        ap.add_argument("--batch-size", type=int, default=upstream.EVALUATE_SH["batch_size"])
        ap.add_argument("--num-beams", type=int, default=upstream.EVALUATE_SH["num_beams"])
        ap.add_argument(
            "--max-new-tokens", type=int, default=upstream.EVALUATE_SH["max_new_tokens"]
        )
        ap.add_argument(
            "--length-penalty", type=float, default=upstream.EVALUATE_SH["length_penalty"]
        )
        ap.add_argument("--dtype", default="bfloat16", choices=generate.DTYPES)
        ap.add_argument("--per-beam", action="store_true", help="the unchanged first pass")
        ap.add_argument("--base-model", type=Path, default=paths.HF_CKPT)
        _csv(ap.parse_args(argv[1:]))
        return
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True, help="run directory under artifacts/minionerec/runs")
    ap.add_argument("--reference", default="test", help="the finished run to replay")
    ap.add_argument("--batches", type=int, default=None, help="only the first N batches")
    ap.add_argument("--per-beam", action="store_true", help="the unchanged first pass")
    ap.add_argument(
        "--base-model", type=Path, default=paths.HF_CKPT, help="model directory (tests only)"
    )
    _replay(ap.parse_args(argv))


if __name__ == "__main__":
    main()
