"""Run MiniOneRec's own evaluation script on Apple Silicon.

The pinned clone (third_party/MiniOneRec @ 0cbd94de, Apache-2.0) is never modified. stage()
copies evaluate.py and the three modules it imports into a work directory, checks every file
against its SHA-256 at the pinned commit, and applies evaluate.py.patch (next to this file).
The patch changes only what a Mac needs, plus timing:

  * `import bitsandbytes` is removed: bitsandbytes is CUDA-only and the evaluation never uses it;
  * the device is "mps" when CUDA is absent and the Apple GPU is available (upstream: cuda, else
    cpu);
  * the model is loaded on the CPU in the dtype given by --dtype (default bfloat16, the
    checkpoint's dtype) instead of device_map="auto"; the script itself then moves it to the
    device, as before;
  * --attn_implementation picks transformers' attention code (empty: transformers' choice,
    sdpa); RecLoop passes eager, see ATTN_IMPLEMENTATION below;
  * the seconds and the MPS memory of every generate() batch, and the setup time, go to the JSON
    file given by --timing_json.

Prompts (data.EvalSidDataset), the SID prefix tree, the constrained beam search
(LogitProcessor.ConstrainedLogitsProcessor) and the generation settings run as published.
evaluate.sh splits the test file into 8 contiguous shards (split.py), runs one process per GPU
and concatenates the results (merge.py). On one GPU, generate.py runs the same shards one after
another and then merge.py, unmodified; a subset of users runs as one shard.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ml import paths
from ml.eval import footprint

PATCH = Path(__file__).with_name("evaluate.py.patch")
# The files evaluate.py needs from the clone, at MINIONEREC_COMMIT (ml/data/download.py).
UPSTREAM_SHA256 = {
    "evaluate.py": "904675387c08bd127c56272e968313449d02280c0f59ab7617f7e8e151363003",
    "data.py": "85edcb6b1e7d1124c204bbdcb312d70903920b3cb5f00bc3fc1414f3df03b555",
    "LogitProcessor.py": "13ab33ff0ed3f769370478a86624499ccc0f8813de9d63e3dc4e2fa25ec59eb8",
    "npu_support.py": "3202621f37b5e3c96bb147eaeddb16102009f398d47617091a313d67777328b0",
}
PATCHED_EVALUATE_SHA256 = "3afc92329b0a6dc19b93c2f63e780966071f70f1b742aa26ade2dc83317c34df"
# On MPS, torch 2.6.0's scaled_dot_product_attention returns NaN for a query row that attends
# to no key, in every dtype (the CPU and CUDA's kernels return finite values, and transformers
# relies on that from torch 2.5 on). Left padding makes such rows: each prompt shorter than the
# longest in its batch starts with pad positions that attend to nothing. Their NaN reaches the
# real positions through the values of the next layer, the sample's logits become NaN, and
# beam search emits token ids 0, 1, 2, ... instead of SIDs (the first trial: 7 of the 8
# samples of a batch, in bfloat16, float16 and float32 alike). Eager attention adds a large
# negative number for masked keys and takes the softmax in float32, so padded rows stay
# finite and the real rows never see them.
ATTN_IMPLEMENTATION = "eager"
# Scripts run unmodified from the clone.
TOOLS_SHA256 = {
    "calc.py": "3e5dc6bf10c2e23f50180f520993462c7756b717dd60d646d8b540274a3fce82",
    "merge.py": "065463581feb7b894249dfb9ac11c8af8370fa73c8725be7d48cdfd44ae31b48",
}
CALC_PY = paths.MINIONEREC / "calc.py"
MERGE_PY = paths.MINIONEREC / "merge.py"

# evaluate.sh's settings for every category.
EVALUATE_SH = {"batch_size": 8, "num_beams": 50, "max_new_tokens": 256, "length_penalty": 0.0}
CATEGORY = "Office_Products"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_pinned(name: str, expected: str) -> Path:
    src = paths.MINIONEREC / name
    got = sha256(src)
    if got != expected:
        raise SystemExit(f"{src}: sha256 {got}, expected {expected} (pinned commit?)")
    return src


def shard_bounds(n: int, shards: int) -> list[tuple[int, int]]:
    """split.py's boundaries: shard i holds rows [i * n // shards, (i + 1) * n // shards)."""
    return [(i * n // shards, (i + 1) * n // shards) for i in range(shards)]


def stage(workdir: Path) -> Path:
    """Copy the pinned upstream files into `workdir`, apply the patch; return evaluate.py."""
    workdir.mkdir(parents=True, exist_ok=True)
    for name, expected in UPSTREAM_SHA256.items():
        shutil.copyfile(_check_pinned(name, expected), workdir / name)
    proc = subprocess.run(
        ["patch", "-p1", "--forward", "--no-backup-if-mismatch", "-d", str(workdir)],
        stdin=PATCH.open("rb"),
        capture_output=True,
        text=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"patch failed:\n{proc.stdout.decode()}\n{proc.stderr.decode()}")
    script = workdir / "evaluate.py"
    got = sha256(script)
    if got != PATCHED_EVALUATE_SHA256:
        raise SystemExit(f"patched evaluate.py: sha256 {got}, expected {PATCHED_EVALUATE_SHA256}")
    return script


def evaluate_command(
    script: Path,
    test_csv: Path,
    result_json: Path,
    timing_json: Path,
    *,
    num_beams: int,
    batch_size: int,
    max_new_tokens: int,
    length_penalty: float,
    dtype: str,
    base_model: Path = paths.HF_CKPT,
    attn_implementation: str = ATTN_IMPLEMENTATION,
) -> list[str]:
    """The evaluate.sh invocation for one process (no CUDA_VISIBLE_DEVICES, no split)."""
    return [
        sys.executable,
        "-u",
        str(script),
        "--base_model",
        str(base_model),
        "--info_file",
        str(paths.PUBLIC_INFO),
        "--category",
        CATEGORY,
        "--test_data_path",
        str(test_csv),
        "--result_json_data",
        str(result_json),
        "--batch_size",
        str(batch_size),
        "--num_beams",
        str(num_beams),
        "--max_new_tokens",
        str(max_new_tokens),
        "--length_penalty",
        str(length_penalty),
        "--dtype",
        dtype,
        "--attn_implementation",
        attn_implementation,
        "--timing_json",
        str(timing_json),
    ]


def run_logged(cmd: list[str], cwd: Path, log: Path, *, sample_memory: bool = True) -> dict:
    """Run `cmd`, stdout + stderr to `log`; watch the child's memory footprint while it runs."""
    env = dict(
        os.environ,
        TOKENIZERS_PARALLELISM="false",  # silences the tokenizers fork warning
        PYTHONUNBUFFERED="1",
    )
    # An op MPS does not implement must fail, not fall back to the CPU unnoticed.
    env.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    start = time.monotonic()
    load_start = os.getloadavg()
    with log.open("w") as out:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT)
        sampler = footprint.Sampler(proc.pid).start() if sample_memory else None
        returncode = proc.wait()
        memory = sampler.stop() if sampler else None
    return {
        "returncode": returncode,
        "wall_seconds": time.monotonic() - start,
        "loadavg_start": list(load_start),
        "loadavg_end": list(os.getloadavg()),
        "memory": memory,
    }


_CALC_LINE = re.compile(r"^(NDCG|HR):?\s*\[([^\]]*)\]", re.M | re.S)


def run_merge(shard_dir: Path, shards: int, out_json: Path) -> None:
    """MiniOneRec's merge.py, unmodified: concatenate shard_dir/0.json ... in order."""
    _check_pinned("merge.py", TOOLS_SHA256["merge.py"])
    proc = subprocess.run(
        [
            sys.executable,
            str(MERGE_PY),
            "--input_path",
            str(shard_dir),
            "--output_path",
            str(out_json),
            "--cuda_list",
            ",".join(str(i) for i in range(shards)),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(f"merge.py failed:\n{proc.stdout}\n{proc.stderr}")


def run_calc(result_json: Path, out_txt: Path) -> dict:
    """MiniOneRec's calc.py, unmodified, on a result file; its printed HR / NDCG and CC."""
    _check_pinned("calc.py", TOOLS_SHA256["calc.py"])
    proc = subprocess.run(
        [
            sys.executable,
            str(CALC_PY),
            "--path",
            str(result_json),
            "--item_path",
            str(paths.PUBLIC_INFO),
        ],
        capture_output=True,
        text=True,
    )
    out_txt.write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(f"calc.py failed; see {out_txt}")
    parsed = {k: [float(x) for x in v.split()] for k, v in _CALC_LINE.findall(proc.stdout)}
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    return {
        "HR": parsed.get("HR"),
        "NDCG": parsed.get("NDCG"),
        "invalid_predictions": int(lines[-1]),
    }
