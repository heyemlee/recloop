"""Export the catalog, users, timestamped interactions and samples of the re-run.

The published MiniOneRec data only has internal ids, titles and semantic IDs. The re-run
(preprocess.py) adds the mappings ASIN -> item_id and reviewerID -> user_id. This module joins
them back to the raw files and writes, under data/processed/:

  items.parquet         item_id, asin, title, brand, categories, price_raw, price_cents,
                        description, features, image_url, sid
  users.parquet         user_id, external_id (reviewerID)
  interactions.parquet  user_id, item_id, ts, rating, seq_idx, split
  samples.parquet       split, row, user_id, item_id, ts, seq_idx, history

Interactions are rebuilt from the raw reviews with the upstream script's own rules (time window,
then users/items that survived k-core; per-user stable sort by time; sliding windows of 10;
global stable sort by target time; 8:1:1 split). The rebuild is checked against the script's
outputs -- user2id/item2id order, inter.json and the three .inter files -- so every sample's
timestamp is the right one.

Item fields follow the upstream choice of the *last* metadata line per ASIN. The title is
MiniOneRec's cleaned title (identical to the info file). Prices are parsed to cents; ranges such
as "$5.99 - $19.99" and anything that is not a single dollar amount become NULL.

Usage: python -m ml.data.catalog
"""

from __future__ import annotations

import datetime as dt
import html
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from ml import paths
from ml.data import preprocess

HISTORY_LEN = 10  # sliding-window length used by amazon18_data_process.py


def clean_text(text) -> str:
    """Same cleaning as amazon18_data_process.clean_text (tags, entities, whitespace)."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", str(text))
    text = html.unescape(text)
    text = text.replace("&quot;", '"').replace("&amp;", "&")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


_PRICE = re.compile(r"\$(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?")


def parse_price_cents(raw: str | None) -> int | None:
    """ "$1,299.99" -> 129999. Ranges, empty strings and anything else -> None (unknown)."""
    if raw is None:
        return None
    m = _PRICE.fullmatch(raw.strip())
    if m is None:
        return None
    dollars = int(m.group(1).replace(",", ""))
    cents = int((m.group(2) or "").ljust(2, "0"))
    return dollars * 100 + cents


def read_remap(path: Path) -> dict[str, int]:
    """Read an upstream .user2id / .item2id file (original<TAB>index), keeping file order."""
    out: dict[str, int] = {}
    with path.open() as f:
        for line in f:
            key, idx = line.rstrip("\n").split("\t")
            out[key] = int(idx)
    return out


def month_start_ts(month: str, tz: str) -> int:
    """Unix time of the first instant of `month` ("YYYY-MM") in time zone `tz`."""
    year, mon = (int(x) for x in month.split("-"))
    return int(dt.datetime(year, mon, 1, tzinfo=ZoneInfo(tz)).timestamp())


# --------------------------------------------------------------------------- interactions


@dataclass
class Rebuild:
    interactions: pd.DataFrame
    samples: pd.DataFrame
    checks: dict


def _read_inter(path: Path) -> list[tuple[int, list[int], int]]:
    rows = []
    with path.open() as f:
        next(f)  # header
        for line in f:
            user, hist, target = line.rstrip("\n").split("\t")
            rows.append((int(user), [int(x) for x in hist.split()] if hist else [], int(target)))
    return rows


def rebuild_interactions(
    reviews_json: Path, user2id: dict[str, int], item2id: dict[str, int], start: int, end: int
) -> Rebuild:
    # 1. The reviews that survived: inside the window, user and item kept by k-core. File order.
    kept = []
    with reviews_json.open() as f:
        for line in f:
            r = json.loads(line)
            ts = int(r["unixReviewTime"])
            if ts < start or ts > end:
                continue
            if r["reviewerID"] in user2id and r["asin"] in item2id:
                kept.append((r["reviewerID"], r["asin"], ts, float(r["overall"])))

    # 2. Per-user sequences: users in order of first appearance, reviews stably sorted by time.
    by_user: dict[str, list[tuple[str, int, float]]] = defaultdict(list)
    for user, asin, ts, rating in kept:
        by_user[user].append((asin, ts, rating))
    for seq in by_user.values():
        seq.sort(key=lambda x: x[1])

    users_in_order = list(by_user)
    item_order: dict[str, int] = {}
    for user in users_in_order:
        for asin, _, _ in by_user[user]:
            item_order.setdefault(asin, len(item_order))

    checks = {
        "kept_reviews": len(kept),
        "user2id_order": users_in_order == list(user2id)
        and all(user2id[u] == i for i, u in enumerate(users_in_order)),
        "item2id_order": item_order == item2id,
    }
    inter_json = json.loads((paths.RERUN_DATASET_DIR / f"{paths.DATASET}.inter.json").read_text())
    checks["sequences_match_inter_json"] = len(inter_json) == len(users_in_order) and all(
        inter_json[str(user2id[u])] == [item2id[a] for a, _, _ in by_user[u]]
        for u in users_in_order
    )

    # 3. Samples: sliding windows, global stable sort by target time, 8:1:1 split.
    inter_rows, samples = [], []
    for user in users_in_order:
        uid = user2id[user]
        seq = by_user[user]
        items = [item2id[a] for a, _, _ in seq]
        for i, (asin, ts, rating) in enumerate(seq):
            inter_rows.append((uid, item2id[asin], ts, rating, i))
            if i >= 1:
                samples.append((uid, items[max(0, i - HISTORY_LEN) : i], items[i], ts, i))
    samples.sort(key=lambda s: s[3])
    n = len(samples)
    train_end, valid_end = int(n * 0.8), int(n * 0.9)
    bounds = {"train": (0, train_end), "valid": (train_end, valid_end), "test": (valid_end, n)}

    sample_rows = []
    for split, (lo, hi) in bounds.items():
        for row, (uid, hist, target, ts, seq_idx) in enumerate(samples[lo:hi]):
            sample_rows.append((split, row, uid, target, ts, seq_idx, hist))
    samples_df = pd.DataFrame(
        sample_rows, columns=["split", "row", "user_id", "item_id", "ts", "seq_idx", "history"]
    )

    match = True
    for split in paths.SPLITS:
        expected = _read_inter(paths.RERUN_DATASET_DIR / f"{paths.DATASET}.{split}.inter")
        got = samples_df[samples_df.split == split]
        match &= expected == list(zip(got.user_id, got.history, got.item_id, strict=True))
    checks["samples_match_inter_files"] = bool(match)

    inter_df = pd.DataFrame(inter_rows, columns=["user_id", "item_id", "ts", "rating", "seq_idx"])
    # Period of each interaction: the split of the sample that targets it; a user's first
    # interaction goes with the user's first sample (it first shows up as that sample's history).
    split_of = {
        (u, s): sp
        for sp, u, s in zip(samples_df.split, samples_df.user_id, samples_df.seq_idx, strict=True)
    }
    inter_df["split"] = [
        split_of[(u, max(s, 1))] for u, s in zip(inter_df.user_id, inter_df.seq_idx, strict=True)
    ]
    return Rebuild(inter_df, samples_df, checks)


# --------------------------------------------------------------------------- items


def last_meta_by_asin(meta_json: Path, asins: set[str]) -> dict[str, dict]:
    """The last metadata line of each ASIN (what the upstream item features use)."""
    last: dict[str, dict] = {}
    with meta_json.open("rb") as f:
        for line in f:
            meta = json.loads(line)
            if meta["asin"] in asins:
                last[meta["asin"]] = meta
    return last


def _clean_list(values) -> list[str]:
    if isinstance(values, str):
        values = [values]
    out = []
    for v in values or []:
        if "</span>" in str(v):
            continue
        c = clean_text(v)
        if c:
            out.append(c)
    return out


def build_items(item2id: dict[str, int], meta: dict[str, dict]) -> pd.DataFrame:
    titles = json.loads(paths.PUBLIC_ITEM_JSON.read_text())
    index = json.loads(paths.PUBLIC_INDEX_JSON.read_text())
    rows = []
    for asin, item_id in sorted(item2id.items(), key=lambda kv: kv[1]):
        m = meta[asin]
        images = m.get("imageURLHighRes") or m.get("imageURL") or []
        brand = clean_text(str(m.get("brand", "")).replace("by\n", ""))
        description = "\n".join(_clean_list(m.get("description")))
        rows.append(
            {
                "item_id": item_id,
                "asin": asin,
                "title": titles[str(item_id)]["title"],
                "brand": brand or None,
                "categories": _clean_list(m.get("category")),
                "price_raw": m.get("price"),
                "price_cents": parse_price_cents(m.get("price")),
                "description": description or None,
                "features": _clean_list(m.get("feature")),
                "image_url": images[0] if images else None,
                "sid": "".join(index[str(item_id)]),
            }
        )
    df = pd.DataFrame(rows)
    df["price_cents"] = df["price_cents"].astype("Int64")
    return df


def main() -> None:
    run = json.loads(preprocess.RUN_JSON.read_text())
    start = month_start_ts(run["final_start"], run["tz"])
    end = month_start_ts(run["final_end"], run["tz"])
    user2id = read_remap(paths.RERUN_DATASET_DIR / f"{paths.DATASET}.user2id")
    item2id = read_remap(paths.RERUN_DATASET_DIR / f"{paths.DATASET}.item2id")

    rebuild = rebuild_interactions(preprocess.REVIEWS_JSON, user2id, item2id, start, end)
    failed = [k for k, v in rebuild.checks.items() if v is False]
    if failed:
        raise SystemExit(f"rebuild of the upstream output failed: {failed} {rebuild.checks}")

    meta = last_meta_by_asin(preprocess.META_FILTERED, set(item2id))
    missing = set(item2id) - set(meta)
    if missing:
        raise SystemExit(f"{len(missing)} items have no metadata line, e.g. {sorted(missing)[:5]}")
    items = build_items(item2id, meta)
    users = pd.DataFrame({"user_id": list(user2id.values()), "external_id": list(user2id.keys())})

    paths.PROCESSED.mkdir(parents=True, exist_ok=True)
    items.to_parquet(paths.CATALOG_PARQUET, index=False)
    users.to_parquet(paths.USERS_PARQUET, index=False)
    rebuild.interactions.to_parquet(paths.INTERACTIONS_PARQUET, index=False)
    rebuild.samples.to_parquet(paths.SAMPLES_PARQUET, index=False)
    (paths.PROCESSED / "rebuild_checks.json").write_text(json.dumps(rebuild.checks, indent=2))

    priced = int(items.price_cents.notna().sum())
    print(f"items        {len(items):,}  (priced {priced:,} = {priced / len(items):.1%})")
    print(f"users        {len(users):,}")
    print(f"interactions {len(rebuild.interactions):,}")
    print(f"samples      {rebuild.samples.split.value_counts().reindex(paths.SPLITS).to_dict()}")
    print(f"rebuild      {rebuild.checks}")


if __name__ == "__main__":
    main()
