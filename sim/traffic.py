"""Simulated traffic in the background, and what its ledger says per time window and arm (M4).

The drills keep the simulator (sim.run's sessions, no faults) running while they change the
release state, then read its ledger: page views with the bundle version and experiment variants
that served them, the client-side latency, failed requests, and the impression and click events.
CTR here is the simulator's own record (each click joined to its page by impression id), not
the event pipeline's: it is the truth the guardrails are judged on.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from sim import run as run_mod
from sim import world as world_mod
from sim.client import uuid7_ms

# Pauses between a session's page views, as in the billing drill: 16 sessions then make ~80 page
# views/s. Without them the simulator is a load test (~1,200/s on a laptop), which also fills the
# 1 GB state Redis within 15 minutes (a 7-day dedup marker per event).
THINK_MS = (50.0, 300.0)


def think_arg(text: str) -> tuple[float, float]:
    """`min,max` in ms (a command-line flag)."""
    lo, hi = (float(x) for x in text.split(","))
    return lo, hi


class Traffic:
    """sim.run's sessions in a background thread, one ledger per run."""

    def __init__(
        self,
        out: Path,
        api_url: str,
        concurrency: int = 16,
        seed: int = world_mod.SEED,
        world: Path = world_mod.DEFAULT_WORLD,
        dwell_ms=(0.0, 0.0),
        think_ms=THINK_MS,
    ):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.ledger = run_mod.Ledger(self.out / "ledger.jsonl")
        w = world_mod.load(world)
        self.sim = run_mod.Simulator(
            w,
            api_url,
            run_mod.Faults(),
            self.ledger,
            think_ms=tuple(think_ms),
            seed=seed,
            dwell_ms=dwell_ms,
        )
        self.concurrency = concurrency
        self.thread: threading.Thread | None = None
        self.stats = None

    def start(self, seconds: float) -> float:
        t0 = time.time()

        def go():
            self.stats = self.sim.run(10**9, self.concurrency, deadline=t0 + seconds)

        self.thread = threading.Thread(target=go, name="traffic", daemon=True)
        self.thread.start()
        return t0

    def extend(self, seconds: float) -> None:
        self.sim.deadline = time.time() + seconds

    def stop(self) -> None:
        self.sim.deadline = time.time()
        if self.thread is not None:
            self.thread.join()
        self.ledger.close()


def snapshot(ledger: run_mod.Ledger) -> tuple[dict, list, list]:
    """read_ledger() of a ledger that may still be written (flushed first) or already closed
    (Traffic.stop() closes it)."""
    with ledger.lock:
        if not ledger.f.closed:
            ledger.f.flush()
    return read_ledger(Path(ledger.f.name))


def read_ledger(path: Path) -> tuple[dict, list, list]:
    """pages by request_id (first page of a request), events, errors."""
    pages, events, errors = {}, [], []
    with Path(path).open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # the last line of a ledger still being written
                continue
            if r["kind"] == "page":
                pages.setdefault(r["request_id"], r)
            elif r["kind"] == "event":
                r["t"] = uuid7_ms(r["event_id"]) / 1000
                events.append(r)
            elif r["kind"] == "error":
                errors.append(r)
    return pages, events, errors


def summarize(pages: dict, events: list, errors: list, t0: float, t1: float, arm) -> dict:
    """Per arm (arm(page) -> name; None: skip), over pages served in [t0, t1): page views,
    users, failed page views (not attributable to an arm: reported for the window), degraded
    share, client latency P50 / P99, impressions, clicks and CTR."""
    by_arm = defaultdict(
        lambda: {
            "pages": 0,
            "users": set(),
            "degraded": 0,
            "reasons": Counter(),
            "ms": [],
            "imps": 0,
            "clicks": 0,
        }
    )
    rid_arm = {}
    for rid, p in pages.items():
        if t0 <= p["t"] < t1:
            a = arm(p)
            if a is None:
                continue
            rid_arm[rid] = a
            d = by_arm[a]
            d["pages"] += 1
            d["users"].add(p["user_id"])
            d["degraded"] += bool(p["degraded"])
            d["reasons"].update(p["degraded"])
            if "ms" in p:
                d["ms"].append(p["ms"])
    seen_click = set()
    for e in events:
        imp = e.get("impression_id")
        if not imp:
            continue
        a = rid_arm.get(imp.split(":")[0])
        if a is None:
            continue
        if e["type"] == "impression":
            by_arm[a]["imps"] += 1
        elif e["type"] == "click" and imp not in seen_click:  # a double click counts once
            seen_click.add(imp)
            by_arm[a]["clicks"] += 1
    out = {}
    for a, d in sorted(by_arm.items()):
        ms = np.asarray(d["ms"]) if d["ms"] else np.asarray([np.nan])
        out[a] = {
            "page_views": d["pages"],
            "users": len(d["users"]),
            "degraded_share": d["degraded"] / d["pages"] if d["pages"] else float("nan"),
            "degraded_reasons": dict(d["reasons"].most_common(3)),
            "p50_ms": float(np.nanpercentile(ms, 50)),
            "p99_ms": float(np.nanpercentile(ms, 99)),
            "impressions": d["imps"],
            "clicks": d["clicks"],
            "ctr": d["clicks"] / d["imps"] if d["imps"] else float("nan"),
        }
    failed = [e for e in errors if t0 <= e["t"] < t1]
    return {
        "arms": out,
        "failed_page_views": len(failed),
        "failed_by_status": _count(failed, "status"),
    }


def _count(rows: list, key: str) -> dict:
    out = defaultdict(int)
    for r in rows:
        out[str(r[key])] += 1
    return dict(out)


def timeline(pages: dict, errors: list, t0: float, t1: float, step: float = 10.0) -> list[dict]:
    """Per `step` seconds: page views by bundle version, failed page views, P99 latency."""
    n = max(1, int(np.ceil((t1 - t0) / step)))
    rows = [
        {"t": round(i * step, 1), "versions": defaultdict(int), "failed": 0, "ms": []}
        for i in range(n)
    ]
    for p in pages.values():
        i = int((p["t"] - t0) // step)
        if 0 <= i < n:
            rows[i]["versions"][p.get("model_version") or "?"] += 1
            if "ms" in p:
                rows[i]["ms"].append(p["ms"])
    for e in errors:
        i = int((e["t"] - t0) // step)
        if 0 <= i < n:
            rows[i]["failed"] += 1
    for r in rows:
        r["versions"] = dict(r["versions"])
        r["p99_ms"] = float(np.percentile(r["ms"], 99)) if r["ms"] else None
        del r["ms"]
    return rows


def ctr_by_user(
    pages: dict, events: list, t0: float, t1: float, arm
) -> dict[str, dict[int, list[int]]]:
    """arm -> user -> [impressions, clicks] (for a user-level bootstrap of the CTR difference)."""
    rid = {
        r: (arm(p), p["user_id"])
        for r, p in pages.items()
        if t0 <= p["t"] < t1 and arm(p) is not None
    }
    out = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    seen = set()
    for e in events:
        imp = e.get("impression_id")
        if not imp or imp.split(":")[0] not in rid:
            continue
        a, u = rid[imp.split(":")[0]]
        if e["type"] == "impression":
            out[a][u][0] += 1
        elif e["type"] == "click" and imp not in seen:
            seen.add(imp)
            out[a][u][1] += 1
    return {a: dict(us) for a, us in out.items()}


def degraded_reasons(pages: dict, t0: float, t1: float, top: int = 3) -> dict[str, int]:
    """The most frequent degradation reasons of the pages served in [t0, t1)."""
    c = Counter()
    for p in pages.values():
        if t0 <= p["t"] < t1:
            c.update(p["degraded"])
    return dict(c.most_common(top))


# Bins of a user's home-page count within a run: (first, last) page number, last None = open.
PAGE_COUNT_BINS = ((1, 3), (4, 15), (16, 63), (64, None))


def ctr_by_page_count(pages: dict, events: list, t0: float, t1: float, arm=None) -> list[dict]:
    """CTR of a user's n-th home page view within [t0, t1), by bins of n (PAGE_COUNT_BINS) and,
    when given, by arm. Shows how CTR moves as the same users keep coming back, which makes
    phases at different times differ in more than their version."""
    order = defaultdict(list)
    for r, p in pages.items():
        if p["surface"] == "home" and t0 <= p["t"] < t1:
            order[p["user_id"]].append((p["t"], r))
    rank = {}
    for lst in order.values():
        for n, (_, r) in enumerate(sorted(lst), 1):
            rank[r] = n
    counts = defaultdict(lambda: [0, 0])
    seen = set()
    for e in events:
        imp = e.get("impression_id")
        if not imp:
            continue
        r = imp.split(":")[0]
        n = rank.get(r)
        if n is None:
            continue
        b = next(b for b in PAGE_COUNT_BINS if n >= b[0] and (b[1] is None or n <= b[1]))
        key = (b, arm(pages[r]) if arm else None)
        if e["type"] == "impression":
            counts[key][0] += 1
        elif e["type"] == "click" and imp not in seen:
            seen.add(imp)
            counts[key][1] += 1
    return [
        {
            "pages": list(b),
            "arm": a,
            "impressions": i,
            "clicks": c,
            "ctr": c / i if i else float("nan"),
        }
        for (b, a), (i, c) in sorted(counts.items(), key=lambda kv: (kv[0][0][0], str(kv[0][1])))
    ]


def bootstrap_ctr_diff(
    a: dict, b: dict, reps: int = 2000, seed: int = 20261108
) -> tuple[float, float, float]:
    """CTR(b) - CTR(a) with a 95% interval from resampling users (ratio of sums per replicate)."""
    rng = np.random.default_rng(seed)
    xa = np.asarray(list(a.values()), dtype=np.float64)
    xb = np.asarray(list(b.values()), dtype=np.float64)
    if len(xa) == 0 or len(xb) == 0:
        return float("nan"), float("nan"), float("nan")
    point = xb[:, 1].sum() / xb[:, 0].sum() - xa[:, 1].sum() / xa[:, 0].sum()
    diffs = np.empty(reps)
    for i in range(reps):
        ia = rng.integers(0, len(xa), len(xa))
        ib = rng.integers(0, len(xb), len(xb))
        sa, sb = xa[ia].sum(axis=0), xb[ib].sum(axis=0)
        diffs[i] = sb[1] / sb[0] - sa[1] / sa[0]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(point), float(lo), float(hi)
