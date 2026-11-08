"""Closed-loop traffic simulator (docs/design.md §2.2).

Each session loops: request recommendations -> the user looks at the first N positions
(impressions) -> clicks by the click model -> some clicks become add-to-cart or purchase ->
report the events -> next request (the next page, the clicked item's page, or a refresh). Every
event is created once with a UUIDv7 event_id and written to the ledger; the fault switches
then change how it is delivered, never what it is:

    duplicate     the event is sent again in a later batch (same event_id): a client retry
    resend        the whole batch is POSTed twice (a retried request)
    reorder       the batch is shuffled and sent after the next one
    late          the event is held back and sent when the session ends
    double_click  a second click on the same impression (a new event: a business duplicate)
    forge         a click on a sponsored slot carries a tampered ad_token (billing must refuse it)

Sponsored slots (design §9.2) are treated like the web client treats them: their impression
and click events carry the slot's ad_token. A user may stay on a page for a while before acting
(--dwell-ms): its events are sent after that, which is the delay between serving an ad and its
click reaching billing.

A 5xx or a dropped connection is retried with the same body, like the web client. The ledger
(JSONL) and the "before" snapshot (hv per user, stock, orders) are what sim.verify checks the
pipeline's result against; sim.billing_verify checks the charge ledger against the clicks.

Usage: python -m sim.run --sessions 2000 --concurrency 16 [--faults duplicate=0.1,resend=0.05]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from sim import ads as ads_mod
from sim import world as world_mod
from sim.client import Api, uuid7

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "artifacts" / "sim"


@dataclass
class Faults:
    duplicate: float = 0.0
    resend: float = 0.0
    reorder: float = 0.0
    late: float = 0.0
    double_click: float = 0.0
    forge: float = 0.0

    @classmethod
    def parse(cls, spec: str) -> Faults:
        f = cls()
        names = {x.name for x in fields(cls)}
        for part in filter(None, (p.strip() for p in spec.split(","))):
            k, _, v = part.partition("=")
            if k not in names:
                raise SystemExit(f"unknown fault {k!r}: one of {sorted(names)}")
            setattr(f, k, float(v))
        return f


@dataclass
class Stats:
    sessions: int = 0
    page_views: Counter = field(default_factory=Counter)  # surface -> n
    status: Counter = field(default_factory=Counter)  # GET status -> n
    events: Counter = field(default_factory=Counter)  # type -> unique events created
    posts: int = 0
    post_retries: int = 0
    post_failures: int = 0
    delivered: int = 0  # event copies POSTed and accepted (duplicates included)
    faults: Counter = field(default_factory=Counter)
    ads: Counter = field(default_factory=Counter)  # sponsored slots served, impressions, clicks

    def merge(self, o: Stats) -> None:
        self.sessions += o.sessions
        self.page_views.update(o.page_views)
        self.status.update(o.status)
        self.events.update(o.events)
        self.posts += o.posts
        self.post_retries += o.post_retries
        self.post_failures += o.post_failures
        self.delivered += o.delivered
        self.faults.update(o.faults)
        self.ads.update(o.ads)


class Ledger:
    """Every unique event and every served page the simulator saw, as JSONL."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = path.open("w")
        self.lock = threading.Lock()

    def write(self, kind: str, **rec) -> None:
        line = json.dumps({"kind": kind, **rec}, separators=(",", ":"))
        with self.lock:
            self.f.write(line + "\n")

    def close(self) -> None:
        self.f.close()


class Session:
    def __init__(self, sim: Simulator, rng: random.Random, api: Api, stats: Stats):
        self.sim, self.rng, self.api, self.stats = sim, rng, api, stats
        self.w = sim.world
        self.faults = sim.faults
        self.carry: list[dict] = []  # duplicates and reordered batches for the next POST
        self.late: list[dict] = []

    # ------------------------------------------------------------- events

    def event(
        self,
        typ: str,
        user: int,
        item: int,
        impression: str | None = None,
        ad_token: str | None = None,
        forged: bool = False,
    ) -> dict:
        ev = {"event_id": uuid7(), "type": typ, "user_id": user, "item_id": item}
        if impression:
            ev["impression_id"] = impression
        if ad_token:
            ev["ad_token"] = ad_token
        self.stats.events[typ] += 1
        # The ledger also says which tokens the simulator forged: the truth billing must match.
        self.sim.ledger.write("event", **ev, **({"forged": True} if forged else {}))
        return ev

    def post(self, batch: list[dict]) -> None:
        if not batch:
            return
        body = {"events": batch}
        # Retried with the same event_ids for up to ~40 s (a broker restart takes a few
        # seconds), backing off to 2 s between attempts.
        for attempt in range(25):
            try:
                status, _ = self.api.post_json("/v1/events", body)
            except OSError:
                status = 0
            self.stats.posts += 1
            if status == 202:
                self.stats.delivered += len(batch)
                return
            if 400 <= status < 500:
                self.stats.post_failures += 1  # rejected: a bug in the simulator
                return
            self.stats.post_retries += 1
            time.sleep(min(2.0, 0.05 * 2**attempt))
        self.stats.post_failures += 1

    def deliver(self, batch: list[dict]) -> None:
        """Send one page view's events through the fault switches."""
        f, rng = self.faults, self.rng
        now = []
        for ev in batch:
            if rng.random() < f.late:
                self.late.append(ev)
                self.stats.faults["late"] += 1
            else:
                now.append(ev)
        dups = [ev for ev in now if rng.random() < f.duplicate]
        self.stats.faults["duplicate"] += len(dups)
        if now and rng.random() < f.reorder:
            rng.shuffle(now)
            self.stats.faults["reorder"] += 1
            self.carry.extend(now)  # goes out after the next page view's batch
            now = []
        out, self.carry = now + self.carry, dups
        self.post(out)
        if out and rng.random() < f.resend:
            self.stats.faults["resend"] += 1
            self.post(out)

    def flush(self) -> None:
        self.post(self.carry + self.late)
        self.carry, self.late = [], []

    # ------------------------------------------------------------- one session

    def run(self) -> None:
        s, rng = self.w.sessions, self.rng
        user, targets = self.w.users[rng.randrange(len(self.w.users))]
        targets = set(targets)
        surface, item, cursor = "home", None, None
        for _ in range(s.max_pages):
            if self.sim.deadline and time.time() > self.sim.deadline:
                break  # the run is over: the session ends here (its held-back events still go)
            if cursor:
                path = f"/v1/recommendations?user_id={user}&limit=20&cursor={cursor}"
                kind = "next_page"
            elif surface == "item":
                path = f"/v1/recommendations?user_id={user}&surface=item&item_id={item}&limit=20"
                kind = "item"
            else:
                path = f"/v1/recommendations?user_id={user}&limit=20"
                kind = "home"
            t_req = time.perf_counter()
            try:
                status, resp, _ = self.api.get_json(path)
            except OSError:
                status, resp = 0, None
            ms = (time.perf_counter() - t_req) * 1000
            self.stats.status[status] += 1
            self.stats.page_views[kind] += 1
            if status != 200 or not resp:
                # The release drill counts failed page views over time (M4).
                self.sim.ledger.write(
                    "error", user_id=user, surface=kind, t=round(time.time(), 3), status=status, ms=round(ms, 2)
                )
                surface, item, cursor = "home", None, None
                continue
            sponsored = [
                {"pos": r["position"], "item_id": r["item_id"], "campaign_id": r.get("campaign_id"),
                 "impression_id": r["impression_id"]}
                for r in resp["items"] if r.get("ad_token")
            ]  # fmt: skip
            self.stats.ads["slots_served"] += len(sponsored)
            self.sim.ledger.write(
                "page",
                request_id=resp["request_id"],
                user_id=user,
                surface=kind,
                t=round(time.time(), 3),
                offset=(resp["items"][0]["position"] - 1) if resp["items"] else 0,
                variants={k: v["variant"] for k, v in resp["experiments"].items()},
                buckets={k: v["bucket"] for k, v in resp["experiments"].items()},
                degraded=resp["degraded"],
                model_version=resp.get("model_version"),
                ms=round(ms, 2),
                **({"sponsored": sponsored} if sponsored else {}),
            )
            visible = rng.randint(s.visible_min, s.visible_max)
            batch, clicked = [], None
            for rec in resp["items"][:visible]:
                pos, it = rec["position"], rec["item_id"]
                tok = rec.get("ad_token")
                batch.append(self.event("impression", user, it, rec["impression_id"], tok))
                self.stats.ads["impressions"] += bool(tok)
                if clicked is None and rng.random() < self.w.p_click(user, it, pos):
                    clicked = rec
            bought = False
            if clicked:
                it = clicked["item_id"]
                batch.append(self.click(user, clicked))
                if rng.random() < self.faults.double_click:
                    self.stats.faults["double_click"] += 1
                    batch.append(self.click(user, clicked))
                p_cart = s.p_cart_target if it in targets else s.p_cart
                if rng.random() < p_cart:
                    batch.append(self.event("add_to_cart", user, it))
                    if rng.random() < s.p_buy:
                        batch.append(self.event("purchase", user, it))
                        bought = True
            if self.sim.dwell_ms[1] > 0:  # the user stays on the page before acting
                time.sleep(rng.uniform(*self.sim.dwell_ms) / 1000)
            self.deliver(batch)
            if self.sim.think_ms:
                time.sleep(rng.uniform(*self.sim.think_ms) / 1000)
            if rng.random() < s.p_end or (bought and rng.random() < s.p_end_after_buy):
                break
            cursor = None
            if clicked and rng.random() < s.p_item_page:
                surface, item = "item", clicked["item_id"]
            elif not clicked and resp.get("next_cursor") and rng.random() < s.p_next_page:
                cursor = resp["next_cursor"]
            else:
                surface, item = "home", None
        self.flush()
        self.stats.sessions += 1

    def click(self, user: int, rec: dict) -> dict:
        """A click on rec; on a sponsored slot it carries the slot's ad_token (or, with the forge
        fault, a tampered one)."""
        tok = rec.get("ad_token")
        forged = bool(tok) and self.rng.random() < self.faults.forge
        if forged:
            tok = ads_mod.tamper(tok, self.rng)
            self.stats.faults["forge"] += 1
        self.stats.ads["clicks"] += bool(tok)
        return self.event("click", user, rec["item_id"], rec["impression_id"], tok, forged)


class Simulator:
    def __init__(
        self,
        world,
        api_url,
        faults,
        ledger,
        think_ms=(0.0, 0.0),
        seed=world_mod.SEED,
        dwell_ms=(0.0, 0.0),
    ):
        self.world, self.api_url, self.faults, self.ledger = world, api_url, faults, ledger
        self.think_ms, self.seed, self.dwell_ms = think_ms, seed, dwell_ms
        self.deadline: float | None = None

    def run(self, sessions: int, concurrency: int, deadline: float | None = None) -> Stats:
        self.deadline = deadline
        lock = threading.Lock()
        started = [0]
        total = Stats()

        def worker(wid: int) -> None:
            rng = random.Random(self.seed * 1000 + wid)
            api = Api(self.api_url)
            st = Stats()
            try:
                while True:
                    with lock:
                        # self.deadline, not the argument: a drill may move it while running.
                        if started[0] >= sessions or (self.deadline and time.time() > self.deadline):
                            break
                        started[0] += 1
                    Session(self, rng, api, st).run()
            finally:
                api.close()
                with lock:
                    total.merge(st)

        with ThreadPoolExecutor(concurrency) as pool:
            list(pool.map(worker, range(concurrency)))
        return total


def snapshot(user_ids: list[int]) -> dict:
    """hv of every simulated user, stock of every item and the order count, before a run."""
    import psycopg
    import redis

    r = redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:16390/0"))
    pipe = r.pipeline(transaction=False)
    for u in user_ids:
        pipe.get(f"u:{u}:hv")
    hv = {str(u): int(v or 0) for u, v in zip(user_ids, pipe.execute(), strict=True)}
    url = os.environ.get(
        "DATABASE_URL", "postgres://recloop:recloop@localhost:15432/recloop?sslmode=disable"
    )
    with psycopg.connect(url) as conn:
        stock = dict(conn.execute("SELECT item_id, stock FROM items").fetchall())
        orders = conn.execute("SELECT count(*) FROM orders").fetchone()[0]
    return {"hv": hv, "stock": {str(k): v for k, v in stock.items()}, "orders": orders}


def run_main(args, on_start=None) -> dict:
    w = world_mod.load(args.world)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    faults = Faults.parse(args.faults)
    before = snapshot([u for u, _ in w.users])
    (out / "before.json").write_text(json.dumps(before))
    ledger = Ledger(out / "ledger.jsonl")
    think = tuple(float(x) for x in args.think_ms.split(","))
    dwell = tuple(float(x) for x in args.dwell_ms.split(","))
    sim = Simulator(w, args.api, faults, ledger, think_ms=think, seed=args.seed, dwell_ms=dwell)
    t0 = time.time()
    if on_start:
        on_start()
    deadline = t0 + args.max_seconds if args.max_seconds else None
    stats = sim.run(args.sessions, args.concurrency, deadline)
    ledger.close()
    took = time.time() - t0
    summary = {
        "api": args.api,
        "world": w.digest,
        "seed": args.seed,
        "faults": asdict(faults),
        "started": t0,
        "seconds": round(took, 1),
        "sessions": stats.sessions,
        "page_views": dict(stats.page_views),
        "get_status": {str(k): v for k, v in stats.status.items()},
        "events_created": dict(stats.events),
        "events_unique": sum(stats.events.values()),
        "event_copies_delivered": stats.delivered,
        "posts": stats.posts,
        "post_retries": stats.post_retries,
        "post_failures": stats.post_failures,
        "fault_counts": dict(stats.faults),
        "ads": dict(stats.ads),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    pv = sum(stats.page_views.values())
    print(
        f"sim      {stats.sessions:,} sessions, {pv:,} page views ({pv / took:.0f}/s), "
        f"{summary['events_unique']:,} events, {stats.delivered:,} copies delivered, "
        f"{stats.post_retries} retried POSTs, {took:.1f}s -> {out.relative_to(ROOT)}"
    )
    return summary


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--api", default=os.environ.get("RECLOOP_API_URL", "http://127.0.0.1:18080"))
    p.add_argument("--world", type=Path, default=world_mod.DEFAULT_WORLD)
    p.add_argument("--sessions", type=int, default=2000)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--think-ms", default="0,0", help="min,max pause between page views")
    p.add_argument(
        "--dwell-ms", default="0,0", help="min,max time on a page before its events are sent"
    )
    p.add_argument("--max-seconds", type=float, default=0)
    p.add_argument(
        "--faults", default="", help="e.g. duplicate=0.1,resend=0.05,late=0.05,forge=0.02"
    )
    p.add_argument("--seed", type=int, default=world_mod.SEED)
    p.add_argument("--out", default=str(DEFAULT_OUT / "run"))
    return p


def main(argv: list[str] | None = None) -> None:
    run_main(parser().parse_args(argv))


if __name__ == "__main__":
    main()
