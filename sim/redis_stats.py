"""The two Redis instances as the runs see them (design §19, 2026-11-08).

    state  REDIS_URL        user state, dedup markers, frequency caps, spend; noeviction. A run
                            fails if it evicted a key or refused a write (it is full).
    cache  CACHE_REDIS_URL  result pages, candidate cache; allkeys-lru. Its evictions are
                            recorded, not a failure: an evicted page is a 410, an evicted
                            candidate list is computed again.

Counters come from INFO (stats, errorstats, memory) and are cumulative since the instance
started: a run records them before and after (delta()).
"""

from __future__ import annotations

import os

import redis

INSTANCES = {
    "state": ("REDIS_URL", "redis://localhost:16390/0"),
    "cache": ("CACHE_REDIS_URL", "redis://localhost:16391/0"),
}


def url(name: str) -> str:
    env, default = INSTANCES[name]
    return os.environ.get(env, default)


def client(name: str) -> redis.Redis:
    return redis.Redis.from_url(url(name))


def snapshot(name: str) -> dict:
    """Setup and counters of one instance."""
    info = client(name).info("all")
    return {
        "maxmemory_mb": round(int(info.get("maxmemory", 0)) / (1 << 20)),
        "policy": info.get("maxmemory_policy", "?"),
        "evicted_keys": int(info.get("evicted_keys", 0)),
        "oom_errors": int((info.get("errorstat_OOM") or {}).get("count", 0)),
        "used_memory_mb": round(int(info.get("used_memory", 0)) / (1 << 20), 1),
    }


def counters() -> dict[str, dict]:
    return {name: snapshot(name) for name in INSTANCES}


def delta(before: dict[str, dict], after: dict[str, dict]) -> dict[str, dict]:
    """What a run did to each instance: evicted keys and refused writes between two readings,
    the setup and memory of the second."""
    return {
        name: {
            "maxmemory_mb": after[name]["maxmemory_mb"],
            "policy": after[name]["policy"],
            "evicted_keys": after[name]["evicted_keys"] - before[name]["evicted_keys"],
            "oom_errors": after[name]["oom_errors"] - before[name]["oom_errors"],
            "used_memory_mb": after[name]["used_memory_mb"],
        }
        for name in INSTANCES
    }


def state_ok(d: dict[str, dict]) -> bool:
    """The state instance neither evicted nor refused anything."""
    return d["state"]["evicted_keys"] == 0 and d["state"]["oom_errors"] == 0


def describe(d: dict[str, dict]) -> str:
    """One line for the logs: setup and what the run did to each instance."""
    return "; ".join(
        f"{name} Redis ({v['policy']}, {v['maxmemory_mb']} MB): evicted {v['evicted_keys']}, "
        f"refused {v['oom_errors']}, {v['used_memory_mb']} MB used"
        for name, v in d.items()
    )
