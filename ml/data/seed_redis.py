"""Load the dataset users' state into Redis (`make seed`).

Redis holds only derived state (design §7.2), so this flushes the database and rebuilds it:

  * u:{uid}:hist   ZSET item -> time (ms): each user's training-period interactions are treated
                   as having happened (design §2.2); the most recent 50 are kept. Review times are
                   day-granular, so the score is review time (ms) + position in the sequence, which
                   keeps the dataset's order inside a day.
  * u:{uid}:bought SET: the same items (a review means the item was bought).
  * demo:users     a short list of dataset users for the web app's user switcher: the user with
                   the longest training history in each of the largest categories, plus one user
                   with no training history (cold start).

hv, seen and the dedup markers start empty. Events stored in PostgreSQL are not replayed here;
rebuilding Redis from the event log comes with the aggregator in M2.

Usage: REDIS_URL=redis://... python -m ml.data.seed_redis
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter

import pandas as pd
import redis

from ml import paths

DEFAULT_REDIS_URL = "redis://localhost:16390/0"
HISTORY_MAX = 50  # serving/internal/userstate.HistoryMax
DEMO_CATEGORIES = 8
CHUNK = 500


def category(cats) -> str:
    """Third level of the category path (e.g. 'Printer Ink & Toner'), else the deepest one."""
    cats = list(cats) if cats is not None else []
    if len(cats) >= 3:
        return cats[2]
    return cats[-1] if cats else "Other"


def demo_users(train: pd.DataFrame, items: pd.DataFrame, all_users: pd.Series) -> list[dict]:
    cat_of = dict(zip(items.item_id, (category(c) for c in items.categories), strict=True))
    title_of = dict(zip(items.item_id, items.title, strict=True))
    lengths = train.groupby("user_id").size()
    dominant = train.groupby("user_id").item_id.agg(
        lambda s: Counter(cat_of[i] for i in s).most_common(1)[0][0]
    )
    users = pd.DataFrame({"n": lengths, "category": dominant}).reset_index()
    top_categories = Counter(cat_of.values()).most_common(DEMO_CATEGORIES)
    out = []
    for cat, _ in top_categories:
        cands = users[users.category == cat].sort_values(["n", "user_id"], ascending=[False, True])
        if cands.empty:
            continue
        uid = int(cands.iloc[0].user_id)
        seq = train[train.user_id == uid].sort_values("seq_idx").item_id.tolist()
        out.append(
            {
                "user_id": uid,
                "label": f"User {uid} · {cat}",
                "category": cat,
                "history_len": len(seq),
                "recent": [title_of[i] for i in seq[-3:][::-1]],
            }
        )
    cold = sorted(set(all_users) - set(train.user_id))
    if cold:
        out.append(
            {
                "user_id": int(cold[0]),
                "label": f"User {cold[0]} · no history (cold start)",
                "category": None,
                "history_len": 0,
                "recent": [],
            }
        )
    return out


def main() -> None:
    url = os.environ.get("REDIS_URL", DEFAULT_REDIS_URL)
    t0 = time.monotonic()
    inter = pd.read_parquet(paths.INTERACTIONS_PARQUET)
    items = pd.read_parquet(paths.CATALOG_PARQUET, columns=["item_id", "title", "categories"])
    users = pd.read_parquet(paths.USERS_PARQUET, columns=["user_id"])
    train = inter[inter.split == "train"].sort_values(["user_id", "seq_idx"])

    r = redis.Redis.from_url(url)
    r.ping()
    r.flushdb()
    pipe = r.pipeline(transaction=False)
    n_users = n_hist = 0
    for uid, g in train.groupby("user_id", sort=True):
        scores: dict[str, int] = {}
        for item, ts, idx in zip(g.item_id, g.ts, g.seq_idx, strict=True):
            scores[str(int(item))] = int(ts) * 1000 + int(idx)  # a repeat keeps its latest time
        recent = dict(sorted(scores.items(), key=lambda kv: kv[1])[-HISTORY_MAX:])
        pipe.zadd(f"u:{uid}:hist", recent)
        pipe.sadd(f"u:{uid}:bought", *scores.keys())
        n_users += 1
        n_hist += len(recent)
        if n_users % CHUNK == 0:
            pipe.execute()
    demo = demo_users(train, items, users.user_id)
    pipe.set("demo:users", json.dumps(demo))
    pipe.execute()
    print(
        f"redis    {n_users:,} users with history ({n_hist:,} items), "
        f"{len(demo)} demo users, {time.monotonic() - t0:.1f}s"
    )
    for d in demo:
        print(f"  {d['label']} ({d['history_len']} items)")


if __name__ == "__main__":
    main()
