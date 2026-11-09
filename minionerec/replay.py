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

`diff` mode looks into a finished replay's differences. Neither evaluate.py's result file nor the
replay keeps the beam scores, so it generates the batches that hold a differing sample once more
both ways, per beam (the reference's path) and with the prompt encoded once, with their scores.
It checks that each path gives its own run's SIDs again, and writes runs/<name>/diff.json: per
differing sample the first beam position where the two lists part, the two SIDs there and the
score gap between them on each path, and the largest score difference of one SID between the two
paths over every beam of those batches (how far the two paths' rounding moves a score):

  python -m minionerec.replay diff --name once

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
    cap = engine.cap_mps_memory(args.mps_memory_gb) if gen.device == "mps" else None
    results, batches, rows, compared = [], [], [], []
    left = args.batches
    for i, (lo, _) in enumerate(s["shard_bounds"]):
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
        "mps_cap": cap,
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


def differing(ours: list[dict], ref_dir: Path, shard_bounds: list) -> dict[int, list[int]]:
    """Per shard of the reference, the indices (within the shard) of the replayed samples whose
    SID lists differ from the reference's. The replay's results are the shards in order (the last
    one possibly cut short)."""
    out: dict[int, list[int]] = {}
    off = 0
    for i in range(len(shard_bounds)):
        if off >= len(ours):
            break
        ref_rows = json.loads((ref_dir / "shards" / f"{i}.json").read_text())
        n = min(len(ref_rows), len(ours) - off)
        for j in range(n):
            if ours[off + j]["input"] != ref_rows[j]["input"]:
                raise SystemExit(f"shard {i} sample {j}: not the reference's prompt")
            if ours[off + j]["predict"] != ref_rows[j]["predict"]:
                out.setdefault(i, []).append(j)
        off += n
    return out


def swap(ref: tuple[list[str], list[float]], ours: tuple[list[str], list[float]]) -> dict:
    """Where two beam lists of one sample part: the first position p whose SIDs differ, the SID
    each list has there (a above b in the reference, b above a in ours), and the score gap between
    them on each path (None when the other SID is not among that path's beams). Position None:
    the two lists are the same."""
    (rs, rsc), (os_, osc) = ref, ours
    p = next((k for k, (x, y) in enumerate(zip(rs, os_, strict=True)) if x != y), None)
    if p is None:
        return {"position": None}
    a, b = rs[p], os_[p]
    r, o = dict(zip(rs, rsc, strict=True)), dict(zip(os_, osc, strict=True))
    return {
        "position": p + 1,
        "reference_sid": a,
        "replay_sid": b,
        "gap_reference": r[a] - r[b] if b in r else None,
        "gap_replay": o[b] - o[a] if a in o else None,
        "reference_position_of_replay_sid": rs.index(b) + 1 if b in r else None,
        "replay_position_of_reference_sid": os_.index(a) + 1 if a in o else None,
    }


def score_differences(
    ref: tuple[list[str], list[float]], ours: tuple[list[str], list[float]]
) -> list[float]:
    """|score on one path - score on the other| of each SID both lists have."""
    r, o = dict(zip(*ref, strict=True)), dict(zip(*ours, strict=True))
    return [abs(r[k] - o[k]) for k in sorted(r.keys() & o.keys())]


def score_difference(ref: tuple[list[str], list[float]], ours: tuple[list[str], list[float]]):
    """The largest of score_differences (0 when the lists share no SID)."""
    return max(score_differences(ref, ours), default=0.0)


def near_tie(rec: dict, moves: list[float]) -> bool:
    """A swap the two paths' rounding can explain: both gaps measured, and neither larger than
    the largest move of one shared SID's score between the paths in the same sample."""
    gaps = (rec.get("gap_reference"), rec.get("gap_replay"))
    return all(g is not None for g in gaps) and max(map(abs, gaps)) <= max(moves, default=0.0)


def _quantiles(values: list[float]) -> dict:
    v = sorted(values)
    at = lambda q: v[min(len(v) - 1, int(q * len(v)))]  # noqa: E731
    return {"count": len(v), "median": at(0.5), "p90": at(0.9), "p99": at(0.99), "max": v[-1]}


def _diff(args) -> None:
    import gc

    import torch

    run_dir = RUNS / args.name
    run = json.loads((run_dir / "run.json").read_text())
    ref_name = run["selection"]["replay_of"]
    ref_dir = RUNS / ref_name
    s = json.loads((ref_dir / "run.json").read_text())["settings"]
    bs = s["batch_size"]
    ours = json.loads((run_dir / "result.json").read_text())
    diffs = differing(ours, ref_dir, s["shard_bounds"])
    batches = sorted({(i, j // bs) for i, js in diffs.items() for j in js})
    if not batches:
        raise SystemExit(f"{args.name}: every sample has {ref_name}'s SIDs; nothing to look into")
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    vm_start, t_start = machine.vm_counters(), time.monotonic()
    regen: dict[str, dict] = {}
    for mode in ("per_beam", "once"):
        gen = engine.Generator(
            args.base_model,
            num_beams=s["num_beams"],
            max_new_tokens=s["max_new_tokens"],
            length_penalty=s["length_penalty"],
            dtype=s["dtype"],
            prefill_once=mode == "once",
        )
        if gen.device == "mps":
            engine.cap_mps_memory(args.mps_memory_gb)
        out, encodings = {}, {}
        for i, b in batches:
            if i not in encodings:
                ds = gen.dataset(ref_dir / "shards" / f"{i}.csv")
                encodings[i] = [ds[k] for k in range(len(ds))]
            batch = gen.generate_encoded(encodings[i][b * bs : (b + 1) * bs])
            for k, (sids, scores) in enumerate(zip(batch.sids, batch.scores, strict=True)):
                out[(i, b * bs + k)] = (sids, scores)
            print(f"[{time.strftime('%H:%M:%S')}] {mode} shard {i} batch {b}", flush=True)
        regen[mode] = out
        device = gen.device
        del gen
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    # Each path must give its own run's SIDs again, or its scores say nothing about that run.
    ref_sids = {i: json.loads((ref_dir / "shards" / f"{i}.json").read_text()) for i, _ in batches}
    offsets = [0]
    for i in range(len(s["shard_bounds"])):
        offsets.append(
            offsets[-1] + len(json.loads((ref_dir / "shards" / f"{i}.json").read_text()))
        )
    keys = sorted(regen["once"])
    same_ref = sum(regen["per_beam"][k][0] == ref_sids[k[0]][k[1]]["predict"] for k in keys)
    same_ours = sum(regen["once"][k][0] == ours[offsets[k[0]] + k[1]]["predict"] for k in keys)
    samples = []
    for i, js in sorted(diffs.items()):
        for j in js:
            ref_beams, our_beams = regen["per_beam"][(i, j)], regen["once"][(i, j)]
            b = j // bs
            n_batch = sum(1 for k in keys if k[0] == i and k[1] // bs == b)
            rec = swap(ref_beams, our_beams)
            moves = score_differences(ref_beams, our_beams)
            samples.append(
                {
                    "shard": i,
                    "index": j,
                    "sample": offsets[i] + j,
                    "batch_samples": n_batch,
                    **rec,
                    # how far the two paths move this sample's shared SIDs' scores
                    "score_moves": _quantiles(moves) if moves else None,
                    "near_tie": near_tie(rec, moves),
                    "beams": {
                        "reference": [list(x) for x in zip(*ref_beams, strict=True)],
                        "replay": [list(x) for x in zip(*our_beams, strict=True)],
                    },
                }
            )
    gaps = [
        abs(g)
        for r in samples
        for g in (r.get("gap_reference"), r.get("gap_replay"))
        if g is not None
    ]
    moves = [d for k in keys for d in score_differences(regen["per_beam"][k], regen["once"][k])]
    noise = max(moves)
    summary = {
        "name": args.name,
        "reference": ref_name,
        "started": started,
        "device": device,
        "dtype": s["dtype"],
        "batches": [list(x) for x in batches],
        "samples_regenerated": len(keys),
        "per_beam_equals_reference": same_ref,
        "once_equals_replay": same_ours,
        "differing": samples,
        "largest_gap": max(gaps, default=None),
        "largest_score_difference": noise,
        # how far the two paths move one SID's beam score, over every beam of those batches
        "score_differences": _quantiles(moves),
        "unmeasured": sum(
            r.get("gap_reference") is None or r.get("gap_replay") is None for r in samples
        ),
        "near_ties": sum(r["near_tie"] for r in samples),
        "wall_seconds": time.monotonic() - t_start,
        "swapped_out_bytes": machine.swapped_out_bytes(vm_start, machine.vm_counters()),
    }
    (run_dir / "diff.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"{args.name}: {len(samples)} differing samples in {len(batches)} batches; per beam "
        f"reproduces {same_ref}/{len(keys)}, once {same_ours}/{len(keys)}; largest gap "
        f"{summary['largest_gap']}, largest score difference {noise}",
        flush=True,
    )


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
    if argv[:1] == ["diff"]:
        ap = argparse.ArgumentParser(prog="minionerec.replay diff")
        ap.add_argument("--name", required=True, help="the finished replay to look into")
        ap.add_argument("--mps-memory-gb", type=float, default=0.0)
        ap.add_argument("--base-model", type=Path, default=paths.HF_CKPT)
        _diff(ap.parse_args(argv[1:]))
        return
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True, help="run directory under artifacts/minionerec/runs")
    ap.add_argument("--reference", default="test", help="the finished run to replay")
    ap.add_argument("--batches", type=int, default=None, help="only the first N batches")
    ap.add_argument("--per-beam", action="store_true", help="the unchanged first pass")
    ap.add_argument(
        "--mps-memory-gb", type=float, default=0.0, help="cap MPS allocations (0: no cap)"
    )
    ap.add_argument(
        "--base-model", type=Path, default=paths.HF_CKPT, help="model directory (tests only)"
    )
    _replay(ap.parse_args(argv))


if __name__ == "__main__":
    main()
