"""Step 1 of the sample job (design §7.5): events_raw, one day at a time, to Parquet.

    artifacts/samples/raw/event_day=YYYY-MM-DD/type=<type>/part-0.parquet

for the types the job reads (served, impression, click, add_to_cart, purchase). A partition is
rewritten as a whole (written next to it, then renamed), so exporting a day again after late
events arrived gives the complete day. Columns as in events_raw; payload stays a JSON string,
the timestamps are UTC.

Usage: python -m ml.samples.export --around 2026-11-08   (or --day D [--day ...])
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ml import paths

TYPES = ("served", "impression", "click", "add_to_cart", "purchase")
RAW = paths.ARTIFACTS / "samples" / "raw"
SCHEMA = pa.schema(
    [
        ("event_id", pa.string()),
        ("user_id", pa.int64()),
        ("request_id", pa.string()),
        ("impression_id", pa.string()),
        ("item_id", pa.int32()),
        ("payload", pa.string()),
        ("client_ts", pa.timestamp("us", tz="UTC")),
        ("server_ts", pa.timestamp("us", tz="UTC")),
    ]
)
QUERY = """
SELECT event_id::text, user_id, request_id::text, impression_id, item_id, payload::text,
       client_ts, server_ts
FROM events_raw
WHERE event_day = %s AND type = %s
ORDER BY event_id
"""


def database_url() -> str:
    return os.environ.get("DATABASE_URL", "postgres://recloop:recloop@localhost:15432/recloop?sslmode=disable")


def export_day(conn, day: dt.date, raw: Path = RAW, batch: int = 50_000) -> dict[str, int]:
    """Write every type's partition of `day`; returns rows per type."""
    counts = {}
    for typ in TYPES:
        target = raw / f"event_day={day.isoformat()}" / f"type={typ}"
        staging = target.with_name(f".{target.name}.{os.getpid()}")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        writer = pq.ParquetWriter(staging / "part-0.parquet", SCHEMA)
        n = 0
        with conn.cursor(name=f"export_{typ}") as cur:
            cur.itersize = batch
            cur.execute(QUERY, (day, typ))
            while rows := cur.fetchmany(batch):
                cols = list(zip(*rows, strict=True))
                writer.write_table(pa.Table.from_arrays([pa.array(c, type=f.type) for c, f in zip(cols, SCHEMA, strict=True)], schema=SCHEMA))
                n += len(rows)
        writer.close()
        shutil.rmtree(target, ignore_errors=True)
        staging.rename(target)
        counts[typ] = n
    return counts


def main(argv: list[str] | None = None) -> None:
    import psycopg

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--day", action="append", default=[], type=dt.date.fromisoformat)
    p.add_argument(
        "--around", type=dt.date.fromisoformat, help="the day before, the day and the day after (what the job reads)"
    )
    p.add_argument("--out", type=Path, default=RAW)
    args = p.parse_args(argv)
    days = list(args.day)
    if args.around:
        one = dt.timedelta(days=1)
        days += [args.around - one, args.around, args.around + one]
    if not days:
        p.error("give --day or --around")
    with psycopg.connect(database_url()) as conn:
        for day in sorted(set(days)):
            t0 = time.monotonic()
            counts = export_day(conn, day, args.out)
            total = sum(counts.values())
            print(
                f"export   {day}: {total:,} rows ("
                + ", ".join(f"{k} {v:,}" for k, v in counts.items())
                + f") in {time.monotonic() - t0:.1f}s"
            )


if __name__ == "__main__":
    main()
