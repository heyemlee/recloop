"""Cross-check our SID-rule metrics against MiniOneRec's own calc.py.

A model's top-50 items per test sample are written as SID strings in the result format of
MiniOneRec's evaluate.py (a JSON list of {"input", "output": target SID + "\\n", "predict": [50
SIDs]}), then third_party/MiniOneRec/calc.py is run on that file unmodified and its printed
HR / NDCG at K = 1, 3, 5, 10, 20, 50 are parsed back. They must equal our SID-rule numbers to
the printed precision (numpy's default 8 decimals).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from ml import paths
from ml.eval import metrics
from ml.eval.splits import Split

CALC_PY = paths.MINIONEREC / "calc.py"
TOP_N = 50
PROMPT = (
    "Can you predict the next possible item the user may expect, given the following "
    "chronological interaction history: "
)


def write_results(path: Path, split: Split, top: np.ndarray, sids: list[str]) -> None:
    rows = []
    for hist, target, predicted in zip(split.history, split.target, top, strict=True):
        rows.append(
            {
                "input": PROMPT + ", ".join(sids[i] for i in hist),
                "output": sids[target] + "\n",
                "predict": [sids[i] for i in predicted],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=4))


def _floats(label: str, text: str) -> list[float]:
    m = re.search(rf"^{label}:?\s*\[([^\]]*)\]", text, re.M | re.S)
    if m is None:
        raise ValueError(f"calc.py output has no {label} line:\n{text}")
    return [float(x) for x in m.group(1).split()]


def run_calc_py(results_json: Path) -> dict:
    out = subprocess.run(
        [
            sys.executable,
            str(CALC_PY),
            "--path",
            str(results_json),
            "--item_path",
            str(paths.PUBLIC_INFO),
        ],
        cwd=paths.ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    lines = [ln for ln in out.strip().splitlines() if ln.strip()]
    return {
        "NDCG": _floats("NDCG", out),
        "HR": _floats("HR", out),
        "invalid_predictions": int(lines[-1]),  # calc.py's CC: predictions not in the info file
        "stdout": out,
    }


def check(
    name: str,
    split: Split,
    top: np.ndarray,
    ours_sid: dict[str, float],
    sids: list[str],
    workdir: Path,
) -> dict:
    """Run calc.py on `top` and compare with our SID-rule metrics `ours_sid` (K in CALC_KS)."""
    results_json = workdir / f"{name.lower()}_{split.name}_top{TOP_N}.json"
    write_results(results_json, split, top, sids)
    calc = run_calc_py(results_json)
    rows, max_diff, printed_equal = [], 0.0, True
    for i, k in enumerate(metrics.CALC_KS):
        for metric in ("HR", "NDCG"):
            ours = ours_sid[f"{metric}@{k}"]
            theirs = calc[metric][i]
            diff = abs(ours - theirs)
            max_diff = max(max_diff, diff)
            # calc.py prints a numpy array (8 decimals); print ours the same way.
            same = np.format_float_positional(
                ours, precision=8, trim="-"
            ) == np.format_float_positional(theirs, precision=8, trim="-")
            printed_equal &= same
            rows.append(
                {
                    "k": k,
                    "metric": metric,
                    "ours": ours,
                    "calc_py": theirs,
                    "printed_equal": bool(same),
                }
            )
    return {
        "model": name,
        "results_json": str(
            results_json.relative_to(paths.ROOT)
            if results_json.is_relative_to(paths.ROOT)
            else results_json
        ),
        "samples": len(split),
        "invalid_predictions": calc["invalid_predictions"],
        "rows": rows,
        "max_abs_diff": max_diff,
        "ok": bool(printed_equal and max_diff <= 5e-9 and calc["invalid_predictions"] == 0),
        "stdout": calc["stdout"],
    }
