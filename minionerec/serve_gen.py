"""The nearline worker's generator: MiniOneRec on the Mac's GPU behind a pipe (design §11 step 5).

ml.nearline.worker starts this process (minionerec environment: the pinned torch and
transformers) and talks JSON lines over stdin / stdout; logs go to stderr.

  first line out: {"ready": true, "pid": ..., "version": ..., "settings": {...}, "device": "mps",
                  "setup_seconds": ...}
  in:  {"id": 7, "histories": [["<a_1><b_2><c_3>", ...], ...]}      SID histories, oldest first
  out: {"id": 7, "sids": [[num_beams SIDs, best first], ...], "scores": [[...]], "seconds": 1.9,
        "batches": [{"size": 8, "seconds": 1.9, "mps_driver_allocated": ...}]}
  out on failure: {"id": 7, "error": "..."}

Generation is minionerec.engine (evaluate.py's prompts, constraint and beam search; the prompt
pass shared across the beams unless --per-beam). Prompts are sorted by length before batching
(less left padding) and the answers put back in request order. --mps-memory-gb caps what MPS may
allocate (torch.mps.set_per_process_memory_fraction): an allocation past it raises instead of
pushing the machine into swap, and a batch that runs out is split in halves and retried.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from minionerec import engine, generate, upstream
from ml import paths
from ml.data import download

VERSION_PREFIX = "minionerec"


def version(num_beams: int, prefill_once: bool) -> str:
    """The generator's version, the {ver} of gen:{uid}:{ver}: checkpoint revision, beams, path."""
    return f"{VERSION_PREFIX}-{download.HF_REVISION[:8]}-b{num_beams}" + (
        "" if prefill_once else "-perbeam"
    )


def _send(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _is_oom(e: BaseException) -> bool:
    text = str(e).lower()
    return "out of memory" in text or "mps backend out of memory" in text


def generate_all(gen: engine.Generator, histories: list[list[str]], max_batch: int):
    """(sids, scores, batch records) for the histories, in their order."""
    import torch

    encodings = gen.encode_histories(histories)
    order = sorted(range(len(encodings)), key=lambda i: len(encodings[i]["input_ids"]))
    sids: list = [None] * len(order)
    scores: list = [None] * len(order)
    records = []
    queue = [order[i : i + max_batch] for i in range(0, len(order), max_batch)]
    while queue:
        idx = queue.pop(0)
        try:
            out = gen.generate_encoded([encodings[i] for i in idx])
        except RuntimeError as e:
            if not _is_oom(e) or len(idx) == 1:
                raise
            if gen.device == "mps":
                torch.mps.empty_cache()
            half = len(idx) // 2
            _log(f"out of memory at batch {len(idx)}: retrying as {half} + {len(idx) - half}")
            queue[:0] = [idx[:half], idx[half:]]
            records.append({"size": len(idx), "oom": True})
            continue
        for j, i in enumerate(idx):
            sids[i], scores[i] = out.sids[j], out.scores[j]
        rec = {"size": len(idx), "seconds": out.seconds}
        if out.mps_driver_allocated is not None:
            rec["mps_driver_allocated"] = out.mps_driver_allocated
        records.append(rec)
    return sids, scores, records


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--num-beams", type=int, default=upstream.EVALUATE_SH["num_beams"])
    ap.add_argument("--max-batch", type=int, default=upstream.EVALUATE_SH["batch_size"])
    ap.add_argument("--dtype", default="bfloat16", choices=generate.DTYPES)
    ap.add_argument("--per-beam", action="store_true", help="the unchanged first pass")
    ap.add_argument("--mps-memory-gb", type=float, default=8.0, help="0: no cap")
    ap.add_argument(
        "--base-model", type=Path, default=paths.HF_CKPT, help="model directory (tests only)"
    )
    ap.add_argument(
        "--device", choices=("mps", "cpu"), default=None, help="default: mps when available"
    )
    args = ap.parse_args(argv)

    gen = engine.Generator(
        args.base_model,
        num_beams=args.num_beams,
        dtype=args.dtype,
        prefill_once=not args.per_beam,
        device=args.device,
    )
    ver = version(args.num_beams, not args.per_beam)
    if args.base_model != paths.HF_CKPT:
        ver += "-test"  # never mistaken for the checkpoint's results
    cap = engine.cap_mps_memory(args.mps_memory_gb) if gen.device == "mps" else None
    _send(
        {
            "ready": True,
            "pid": os.getpid(),
            "version": ver,
            "settings": {**gen.settings(), "max_batch": args.max_batch},
            "device": gen.device,
            "setup_seconds": gen.setup_seconds,
            "mps_cap": cap,
            "versions": generate.versions(),
        }
    )
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        try:
            sids, scores, records = generate_all(gen, req["histories"], args.max_batch)
        except Exception as e:  # noqa: BLE001 -- the worker decides; the process keeps serving
            _send({"id": req.get("id"), "error": f"{type(e).__name__}: {e}"})
            continue
        _send(
            {
                "id": req.get("id"),
                "sids": sids,
                "scores": scores,
                "seconds": sum(r.get("seconds", 0.0) for r in records),
                "batches": records,
            }
        )


if __name__ == "__main__":
    main()
