"""The simulated world (docs/design.md §2.2): who visits, what they want, how they click.

Users are the 2,024 dataset users with test-period interactions; their test-period items are
what they are looking for ("targets"). Everything before is treated as having happened (the
Redis state `make seed` loads). The click model never looks at any model's scores:

    P(click on the item at position p) = examination(p) x relevance(u, i)
    examination(p) = p ** -position_exponent
    relevance(u, i) = rel_target                       if i is one of u's targets
                    = rel_similar * ((s - s0) / (1 - s0)) ** 2   if s = max cosine similarity
                                                        of i to u's targets (text embeddings,
                                                        bge-small) is above s0
                    = rel_floor                        otherwise (noise clicks)

The world file (artifacts/sim/world.json) holds, per user, the targets and the precomputed
relevance of the most similar items, so the Python simulator (sim.run) and the k6 load test
(loadtest/recs.js) apply exactly the same model by table lookup.

Usage: python -m sim.world [--out artifacts/sim/world.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
DEFAULT_WORLD = ROOT / "artifacts" / "sim" / "world.json"
FORMAT = 1
SEED = 20261106


@dataclass(frozen=True)
class ClickModel:
    position_exponent: float = 0.7
    rel_target: float = 0.8
    rel_similar: float = 0.6  # < rel_target: a similar item never beats the wanted one
    similarity_floor: float = 0.6  # s0: below it an item counts as unrelated
    rel_floor: float = 0.015
    neighbours: int = 50  # similar items kept per user

    def examination(self, position: int) -> float:
        return position**-self.position_exponent

    def relevance_from_similarity(self, s: float) -> float:
        if s <= self.similarity_floor:
            return self.rel_floor
        x = (s - self.similarity_floor) / (1 - self.similarity_floor)
        return max(self.rel_floor, min(self.rel_similar, self.rel_similar * x * x))


@dataclass(frozen=True)
class Sessions:
    """A session: home page first, then more page views until it ends."""

    p_end: float = 0.3  # after each page view; mean session ~3.3 page views
    max_pages: int = 12
    visible_min: int = 4  # positions the user looks at on a page (uniform)
    visible_max: int = 10
    p_next_page: float = 0.25  # no click: scroll to the next page (cursor) instead of a refresh
    p_item_page: float = 0.7  # after a click: the clicked item's page (surface=item)
    p_cart: float = 0.15  # click -> add to cart
    p_cart_target: float = 0.5  # ... when the item is one of the user's targets
    p_buy: float = 0.5  # add to cart -> purchase
    p_end_after_buy: float = 0.5


@dataclass(frozen=True)
class World:
    users: list[tuple[int, list[int]]]  # (user_id, targets)
    relevance: dict[int, dict[int, float]]  # user -> item -> relevance (targets + similar)
    click: ClickModel = field(default_factory=ClickModel)
    sessions: Sessions = field(default_factory=Sessions)
    seed: int = SEED
    digest: str = ""

    def rel(self, user: int, item: int) -> float:
        return self.relevance.get(user, {}).get(item, self.click.rel_floor)

    def p_click(self, user: int, item: int, position: int) -> float:
        return self.click.examination(position) * self.rel(user, item)


def build(
    samples: pd.DataFrame, emb: np.ndarray, click: ClickModel | None = None
) -> tuple[list, dict]:
    """Users with their targets, and per user the relevance of targets and similar items."""
    click = click or ClickModel()
    test = samples[samples.split == "test"].sort_values(["user_id", "seq_idx"])
    users, relevance = [], {}
    for uid, g in test.groupby("user_id", sort=True):
        targets = list(dict.fromkeys(int(i) for i in g.item_id))  # in order, unique
        sims = emb @ emb[targets].T  # (items, targets)
        best = sims.max(axis=1)
        best[targets] = -1.0
        order = np.argsort(-best, kind="stable")[: click.neighbours]
        rel = {t: click.rel_target for t in targets}
        for i in order:
            r = click.relevance_from_similarity(float(best[i]))
            if r > click.rel_floor:
                rel[int(i)] = round(r, 6)
        users.append((int(uid), targets))
        relevance[int(uid)] = rel
    return users, relevance


def to_json(w: World) -> dict:
    return {
        "format": FORMAT,
        "seed": w.seed,
        "click_model": asdict(w.click),
        "sessions": asdict(w.sessions),
        # Compact rows for k6: [user_id, [targets], [[item, relevance], ...]]
        "users": [
            [uid, targets, [[i, r] for i, r in w.relevance[uid].items() if i not in targets]]
            for uid, targets in w.users
        ],
    }


def from_json(d: dict) -> World:
    if d.get("format") != FORMAT:
        raise SystemExit(f"world file format {d.get('format')}, expected {FORMAT}: rebuild it")
    click = ClickModel(**d["click_model"])
    users, relevance = [], {}
    for uid, targets, similar in d["users"]:
        users.append((uid, targets))
        rel = {t: click.rel_target for t in targets}
        rel.update({i: r for i, r in similar})
        relevance[uid] = rel
    return World(
        users=users,
        relevance=relevance,
        click=click,
        sessions=Sessions(**d["sessions"]),
        seed=d["seed"],
        digest=d.get("digest", ""),
    )


def load(path: Path = DEFAULT_WORLD) -> World:
    if not path.exists():
        raise SystemExit(f"{path} is missing: run `make sim-world`")
    return from_json(json.loads(path.read_text()))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=DEFAULT_WORLD)
    args = p.parse_args(argv)
    t0 = time.monotonic()
    samples = pd.read_parquet(PROCESSED / "samples.parquet")
    emb = np.load(PROCESSED / "text_emb.npy")
    users, relevance = build(samples, emb)
    w = World(users=users, relevance=relevance)
    doc = to_json(w)
    body = json.dumps(doc, separators=(",", ":"), sort_keys=True)
    doc["digest"] = hashlib.sha256(body.encode()).hexdigest()[:16]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, separators=(",", ":"), sort_keys=True))
    n_rel = sum(len(r) for r in relevance.values())
    n_targets = sum(len(t) for _, t in users)
    print(
        f"world    {len(users):,} users, {n_targets:,} targets, {n_rel:,} relevance entries, "
        f"digest {doc['digest']}, {args.out.stat().st_size / 1e6:.1f} MB, "
        f"{time.monotonic() - t0:.1f}s -> {args.out.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
