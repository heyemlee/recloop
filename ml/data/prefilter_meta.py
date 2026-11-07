"""Shrink the product metadata to the ASINs that occur in the 5-core reviews.

MiniOneRec's amazon18_data_process.py loads the whole metadata file into memory (about 5 GB
decompressed for Office Products). It only ever looks up ASINs that occur in the reviews:
  * k-core filtering keeps a review only if its ASIN has a valid title in `id_title`;
  * item features are built for ASINs of the filtered reviews, taking the *last* metadata line
    of each ASIN.
Keeping exactly the lines whose ASIN occurs in the reviews -- copied byte for byte, in the
original order, duplicates included -- therefore leaves every output of the script unchanged.
The metadata is streamed straight from the .gz file; the decompressed file never hits the disk.

Usage: python -m ml.data.prefilter_meta --reviews R.json.gz --meta M.json.gz --out OUT.json
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections.abc import Iterable, Iterator
from pathlib import Path


def review_asins(lines: Iterable[bytes]) -> set[str]:
    return {json.loads(line)["asin"] for line in lines}


def filter_lines(
    lines: Iterable[bytes], keep: set[str], stats: dict | None = None
) -> Iterator[bytes]:
    """Yield the metadata lines whose top-level "asin" is in `keep`, unchanged and in order."""
    total = kept = 0
    for line in lines:
        total += 1
        if json.loads(line)["asin"] in keep:
            kept += 1
            yield line
    if stats is not None:
        stats.update(total=total, kept=kept)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reviews", type=Path, required=True, help="5-core reviews (.json.gz)")
    parser.add_argument("--meta", type=Path, required=True, help="product metadata (.json.gz)")
    parser.add_argument("--out", type=Path, required=True, help="filtered metadata (.json)")
    args = parser.parse_args(argv)

    with gzip.open(args.reviews, "rb") as f:
        keep = review_asins(f)
    print(f"{len(keep):,} distinct ASINs in {args.reviews.name}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".part")
    stats: dict = {}
    with gzip.open(args.meta, "rb") as src, tmp.open("wb") as dst:
        for line in filter_lines(src, keep, stats):
            dst.write(line)
    tmp.replace(args.out)
    print(f"kept {stats['kept']:,} of {stats['total']:,} metadata lines -> {args.out}")


if __name__ == "__main__":
    main()
