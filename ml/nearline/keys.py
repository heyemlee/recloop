"""Keys, payload and timing of the nearline generative recall (design §11 step 5, §12.2).

All in the state Redis (REDIS_URL; noeviction):

  gen:worker           STRING (JSON), PX 30 s   a worker is alive (refreshed every 10 s); apply.lua
                                               queues history changes only while it exists
  gen:queue            ZSET uid -> due (ms)     users whose history changed; due = min(change +
                                               30 s, first unhandled change + 120 s)
  gen:pending:{uid}    LIST "<event ms>:<applied ms>"   the changes not picked up yet
  gen:{uid}:{ver}      STRING (JSON), PX 7 d    {"hv", "at", "items", "scores"}: the top items
                                               generated from the history at version hv

The Go side is serving/internal/userstate (apply.lua queues, gen.go reads); the payload format is
pinned for both by testdata/gen_payload.json.
"""

from __future__ import annotations

import json

WORKER_KEY = "gen:worker"
QUEUE_KEY = "gen:queue"
# Go: userstate.GenDebounce / GenMaxWait (apply.lua gets them as arguments).
DEBOUNCE_MS = 30_000
MAX_WAIT_MS = 120_000
HEARTBEAT_TTL_MS = 30_000
HEARTBEAT_EVERY_S = 10.0
RESULT_TTL_MS = 7 * 24 * 3600 * 1000  # the dedup markers' 7 days: a user-state time scale
TOP_K = 50
HISTORY_WINDOW = 10  # recall.HistoryWindow: the items SASRec and ItemCF look at, too


def pending_key(uid: int) -> str:
    return f"gen:pending:{uid}"


def result_key(uid: int, version: str) -> str:
    return f"gen:{uid}:{version}"


def hist_key(uid: int) -> str:
    return f"u:{uid}:hist"


def hv_key(uid: int) -> str:
    return f"u:{uid}:hv"


def encode(hv: int, at_ms: int, items: list[int], scores: list[float]) -> str:
    """The value of gen:{uid}:{ver}: compact JSON, scores to 4 decimals."""
    if len(items) != len(scores):
        raise ValueError("items and scores differ in length")
    return json.dumps(
        {
            "hv": int(hv),
            "at": int(at_ms),
            "items": [int(i) for i in items],
            "scores": [round(float(s), 4) for s in scores],
        },
        separators=(",", ":"),
    )


def decode(raw: str | bytes) -> dict:
    return json.loads(raw)


def parse_changes(text: str) -> list[tuple[int, int]]:
    """The claim script's "<event ms>:<applied ms>,..." -> [(event ms, applied ms), ...]."""
    out = []
    for part in filter(None, text.split(",")):
        event, applied = part.split(":")
        out.append((int(event), int(applied)))
    return out
