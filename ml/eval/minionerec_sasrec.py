"""Re-run MiniOneRec's own SASRec baseline on the same split (`make paper-sasrec`).

The paper's SASRec row (Table 1, Office: HR@10 0.1120, NDCG@10 0.0858) comes from
third_party/MiniOneRec/sasrec.py. Running that script unmodified, with its default
hyper-parameters, separates two explanations for a gap between our SASRec and the paper:
  * if this re-run also lands far from the paper, the data or the evaluation differ;
  * if it reproduces the paper, the difference is in the model and its training (loss,
    output layer, sizes), which our own ablations then isolate.

The script runs as is, from a scratch directory under artifacts/ (it writes ./result_temp/ and
reads ./data/Amazon/, linked to the pinned clone), with a stub `wandb` module: `--debug True`
skips wandb.init, but the script still calls wandb.log. A `sitecustomize` next to the stub sets
the multiprocessing start method to fork: the script's DataLoader starts 8 workers every epoch,
and with macOS's default (spawn) each one re-imports torch, which made an epoch take ~15 s.
Workers only index the training DataFrame; the batch order comes from the parent process, so the
start method does not change the run (checked: 3-epoch runs with spawn and with fork print
identical metrics; fork was ~10x faster). Its own printed test metrics are kept, and its saved
test logits are scored again with our metrics, tie order and bootstrap.

Writes artifacts/eval/minionerec_sasrec.json. Usage: python -m ml.eval.minionerec_sasrec
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

import numpy as np

from ml import paths
from ml.eval import bootstrap, metrics
from ml.eval.splits import NUM_ITEMS, load_split, train_sequences
from ml.models.pop import Pop

SCRIPT = paths.MINIONEREC / "sasrec.py"
WORK = paths.ARTIFACTS / "eval" / "minionerec_sasrec"
OUT_JSON = paths.ARTIFACTS / "eval" / "minionerec_sasrec.json"
# The script's defaults, passed explicitly so the run is documented in one place.
ARGS = {
    "--data": "Office_Products",
    "--model": "SASRec",
    "--loss_type": "bce",
    "--hidden_factor": "32",
    "--batch_size": "1024",
    "--lr": "0.001",
    "--l2_decay": "1e-05",
    "--dropout_rate": "0.3",
    "--epoch": "500",
    "--early_stop": "20",
    "--seed": "1",
    "--cuda": "0",
}
SITECUSTOMIZE = '''"""Fork DataLoader workers instead of spawning them (see ml/eval/minionerec_sasrec.py)."""
import multiprocessing

multiprocessing.set_start_method("fork", force=True)
'''
WANDB_STUB = '''"""Stub of the wandb API used by MiniOneRec's sasrec.py: logging is dropped."""
run = None
def init(*args, **kwargs):
    return None
def log(*args, **kwargs):
    return None
'''


def logits_path() -> os.PathLike:
    a = ARGS
    name = (
        f"{a['--data']}_{a['--model']}_emb{a['--hidden_factor']}_bs{a['--batch_size']}"
        f"_lr{float(a['--lr'])}_decay{float(a['--l2_decay'])}_seed{a['--seed']}"
        f"_loss_{a['--loss_type']}_dropout{float(a['--dropout_rate'])}_logits.npy"
    )
    return WORK / "result_temp" / name


def parse_test_metrics(stdout: str) -> dict[str, float]:
    """The last 'hr@1 ndcg@1 ...' block the script prints (the best model on test)."""
    blocks = re.findall(r"^(hr@1\t.*)\n([\d.\t]+)$", stdout, re.M)
    if not blocks:
        raise SystemExit("could not find test metrics in sasrec.py output")
    header, values = blocks[-1]
    names = header.split()
    vals = [float(v) for v in values.split()]
    return {n.upper(): v for n, v in zip(names, vals, strict=True)}


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "result_temp").mkdir(exist_ok=True)
    link = WORK / "data"
    if not link.exists():
        link.symlink_to(paths.MINIONEREC / "data")
    stub_dir = WORK / "stubs"
    stub_dir.mkdir(exist_ok=True)
    (stub_dir / "wandb.py").write_text(WANDB_STUB)
    (stub_dir / "sitecustomize.py").write_text(SITECUSTOMIZE)

    cmd = [sys.executable, str(SCRIPT), "--debug", "True"]
    for k, v in ARGS.items():
        cmd += [k, v]
    cmd += ["--result_json_path", str(WORK / "result_temp" / "result.json")]
    env = dict(
        os.environ,
        PYTHONPATH=str(stub_dir),
        WANDB_MODE="disabled",
        # fork after the Objective-C runtime is up is refused on macOS without this
        OBJC_DISABLE_INITIALIZE_FORK_SAFETY="YES",
    )
    t0 = time.monotonic()
    print("running " + " ".join(cmd[1:]), flush=True)
    proc = subprocess.run(cmd, cwd=WORK, env=env, capture_output=True, text=True)
    (WORK / "stdout.log").write_text(proc.stdout)
    (WORK / "stderr.log").write_text(proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(f"sasrec.py failed ({proc.returncode}); see {WORK / 'stderr.log'}")
    seconds = time.monotonic() - t0
    own = parse_test_metrics(proc.stdout)
    best_epoch = re.findall(r"^BEST EPOCH:(\d+)$", proc.stdout, re.M)

    # Score the saved test logits with our metrics (item rule, popularity tie order).
    test = load_split("test")
    logits = np.load(logits_path()).astype(np.float64)
    assert logits.shape == (len(test), NUM_ITEMS), logits.shape
    counts = Pop().fit(train_sequences(load_split("train")), NUM_ITEMS).counts
    order = np.lexsort((np.arange(NUM_ITEMS), -counts))
    tie_pos = np.empty_like(order)
    tie_pos[order] = np.arange(NUM_ITEMS)
    values = metrics.per_sample(metrics.ranks_of(logits, test.target, tie_pos))
    resampler = bootstrap.Resampler.build(test.user)
    ours = {k: {"value": float(v.mean()), "ci": resampler.interval(v)} for k, v in values.items()}

    out = {
        "script": "third_party/MiniOneRec/sasrec.py",
        "args": ARGS,
        "seconds": seconds,
        "best_epoch": int(best_epoch[-1]) if best_epoch else None,
        "own_metrics": own,
        "our_metrics": ours,
        "per_sample": {k: v.tolist() for k, v in values.items() if k in ("HR@10", "NDCG@10")},
    }
    OUT_JSON.write_text(json.dumps(out, indent=2) + "\n")
    print(
        f"MiniOneRec sasrec.py: its metrics HR@10 {own.get('HR@10'):.4f} NDCG@10 "
        f"{own.get('NDCG@10'):.4f}; ours HR@10 {ours['HR@10']['value']:.4f} NDCG@10 "
        f"{ours['NDCG@10']['value']:.4f} ({seconds:.0f}s)"
    )


if __name__ == "__main__":
    main()
