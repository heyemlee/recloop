"""Re-run MiniOneRec's preprocessing on the raw Amazon 2018 files.

Steps (each is skipped when its output is already there):
  1. decompress the 5-core reviews (the upstream script needs a plain file);
  2. pre-filter the metadata to ASINs that occur in the reviews (see prefilter_meta.py);
  3. run third_party/MiniOneRec/data/amazon18_data_process.py, unmodified, with TZ=UTC and the
     README parameters (5-core, 2017-10 .. 2018-11; the script itself widens the start year
     while fewer than 3,000 items survive, which lands on 2016-10);
  4. copy the published index.json (semantic IDs) next to the re-run output and run the
     release-time convert_dataset.py on it, producing the train/valid/test CSVs and the info
     file in MiniOneRec's published format.

Why TZ=UTC: get_timestamp_start() turns the start month into a timestamp with a naive
datetime, i.e. in the machine's local time zone. Review timestamps are UTC midnights, so on a
machine west of UTC the reviews of the first day (2016-10-01) fall outside the window.

Usage: python -m ml.data.preprocess [--force]
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ml import paths
from ml.data import prefilter_meta

REVIEWS_GZ = paths.UCSD / "Office_Products_5.json.gz"
META_GZ = paths.UCSD / "meta_Office_Products.json.gz"
REVIEWS_JSON = paths.INTERIM / "Office_Products_5.json"
META_FILTERED = paths.INTERIM / "meta_Office_Products.filtered.json"

UPSTREAM_SCRIPT = paths.MINIONEREC / "data" / "amazon18_data_process.py"
RELEASE_CONVERTER = paths.MINIONEREC_RELEASE / "convert_dataset.py"
UPSTREAM_LOG = paths.RERUN / "amazon18_data_process.log"
RUN_JSON = paths.RERUN / "run.json"  # TZ, arguments and the final time window of the re-run
CONVERTER_LOG = paths.CONVERTED / "convert_dataset.log"

# README parameters for Office_Products.
UPSTREAM_ARGS = {
    "user_k": 5,
    "item_k": 5,
    "st_year": 2017,
    "st_month": 10,
    "ed_year": 2018,
    "ed_month": 11,
}

RERUN_OUTPUTS = [
    paths.RERUN_DATASET_DIR / f"{paths.DATASET}.{name}"
    for name in (
        "train.inter",
        "valid.inter",
        "test.inter",
        "inter.json",
        "item.json",
        "user2id",
        "item2id",
    )
]


def _decompress(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    with gzip.open(src, "rb") as fin, tmp.open("wb") as fout:
        shutil.copyfileobj(fin, fout, length=1 << 22)
    tmp.replace(dst)


def _run(cmd: list[str], log: Path, env: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("run      ", " ".join(str(c) for c in cmd), f"(log: {log.relative_to(paths.ROOT)})")
    t0 = time.monotonic()
    with log.open("w") as out:
        proc = subprocess.run(cmd, cwd=paths.ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        tail = "".join(log.read_text().splitlines(keepends=True)[-30:])
        raise SystemExit(f"command failed ({proc.returncode}); last lines of {log}:\n{tail}")
    print(f"          done in {time.monotonic() - t0:.0f}s")


def rerun_upstream(tz: str, force: bool) -> bool:
    """Run the upstream script unless its outputs for this TZ exist. True if it ran."""
    done = RUN_JSON.exists() and json.loads(RUN_JSON.read_text()).get("tz") == tz
    if not force and done and all(p.exists() for p in RERUN_OUTPUTS):
        print(f"ok        {paths.RERUN_DATASET_DIR.relative_to(paths.ROOT)} (exists)")
        return False
    shutil.rmtree(paths.RERUN, ignore_errors=True)
    cmd = [sys.executable, str(UPSTREAM_SCRIPT), "--dataset", paths.DATASET]
    for key, value in UPSTREAM_ARGS.items():
        cmd += [f"--{key}", str(value)]
    cmd += [
        "--reviews_file",
        str(REVIEWS_JSON),
        "--metadata_file",
        str(META_FILTERED),
        "--output_path",
        str(paths.RERUN),
    ]
    env = {**os.environ, "TZ": tz, "PYTHONHASHSEED": "0"}
    _run(cmd, UPSTREAM_LOG, env)
    start, end = final_window(UPSTREAM_LOG.read_text())
    RUN_JSON.write_text(
        json.dumps(
            {"tz": tz, "args": UPSTREAM_ARGS, "final_start": start, "final_end": end}, indent=2
        )
        + "\n"
    )
    print(f"          final window {start} .. {end} (TZ={tz})")
    return True


def final_window(log_text: str) -> tuple[str, str]:
    """The (start, end) months the upstream script settled on, e.g. ("2016-10", "2018-11").

    The script prints "Initial time range: Y-M to Y-M" and, each time it widens the window,
    "New time range: Y-M to Y-M"; the last such line wins.
    """
    found = re.findall(
        r"(?:Initial|New) time range: (\d{4})-(\d{1,2}) to (\d{4})-(\d{1,2})", log_text
    )
    if not found:
        raise ValueError("no time range in the upstream log")
    sy, sm, ey, em = found[-1]
    return f"{int(sy):04d}-{int(sm):02d}", f"{int(ey):04d}-{int(em):02d}"


def convert(force: bool) -> None:
    index_dst = paths.RERUN_DATASET_DIR / f"{paths.DATASET}.index.json"
    shutil.copyfile(paths.PUBLIC_INDEX_JSON, index_dst)
    outputs = [paths.CONVERTED / s / f"{paths.SPLIT_STEM}.csv" for s in paths.SPLITS]
    outputs.append(paths.CONVERTED / "info" / f"{paths.SPLIT_STEM}.txt")
    if not force and all(p.exists() for p in outputs):
        print(f"ok        {paths.CONVERTED.relative_to(paths.ROOT)} (exists)")
        return
    shutil.rmtree(paths.CONVERTED, ignore_errors=True)
    cmd = [
        sys.executable,
        str(RELEASE_CONVERTER),
        "--data_dir",
        str(paths.RERUN_DATASET_DIR),
        "--dataset_name",
        paths.DATASET,
        "--output_dir",
        str(paths.CONVERTED),
        "--category",
        paths.DATASET,
        "--seed",
        "42",
    ]
    _run(cmd, CONVERTER_LOG, dict(os.environ))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true", help="redo every step")
    parser.add_argument("--tz", default="UTC", help="TZ for the upstream script (default: UTC)")
    args = parser.parse_args(argv)

    if args.force or not REVIEWS_JSON.exists():
        print(f"unzip     {REVIEWS_GZ.relative_to(paths.ROOT)}")
        _decompress(REVIEWS_GZ, REVIEWS_JSON)
    if args.force or not META_FILTERED.exists():
        print(f"filter    {META_GZ.relative_to(paths.ROOT)}")
        prefilter_meta.main(
            ["--reviews", str(REVIEWS_GZ), "--meta", str(META_GZ), "--out", str(META_FILTERED)]
        )
    ran = rerun_upstream(args.tz, args.force)
    convert(args.force or ran)


if __name__ == "__main__":
    main()
