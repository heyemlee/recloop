"""Standing campaigns for the M4 runs: sponsored slots throughout sample collection, the release
drill and the A/B test, so the pCTR model has sponsored impressions to learn from and the A/B
sees the ads stage as it runs in production.

16 advertisers with one campaign each: bids 8 to 68 cents (4 apart), a daily budget of $1,000
(the runs never spend it), a frequency cap of 30 a day, 8 in-stock items with a known price
each, chosen with a fixed seed. Reused when they exist (same names, still active).

Usage: python -m sim.standing   (prints the campaigns)
"""

from __future__ import annotations

import os

from sim import ads as ads_mod

NAME = "M4 standing"
CAMPAIGNS = 16
ITEMS = 8
SEED = 20261108


def ensure(ads: ads_mod.Ads) -> list[dict]:
    mine = [c for c in ads.active_campaigns() if c["name"].startswith(NAME)]
    if len(mine) >= CAMPAIGNS:
        return mine
    if mine:
        ads.end([c["campaign_id"] for c in mine])
    items = ads.pick_items(CAMPAIGNS * ITEMS, SEED)
    specs = [
        {
            "bid": 8 + 4 * i,
            "budget": 100_000,
            "freq_cap": 30,
            "items": items[i * ITEMS : (i + 1) * ITEMS],
        }
        for i in range(CAMPAIGNS)
    ]
    return ads.create(NAME, specs)


def main() -> None:
    ads = ads_mod.Ads(
        os.environ.get("RECLOOP_API_URL", "http://127.0.0.1:18080"),
        os.environ.get(
            "DATABASE_URL", "postgres://recloop:recloop@localhost:15432/recloop?sslmode=disable"
        ),
    )
    cs = ensure(ads)
    print(
        f"standing {len(cs)} campaigns: "
        + ", ".join(f"{c['campaign_id']} ({c['bid_cpc_cents']}c)" for c in cs)
    )
    ads.close()


if __name__ == "__main__":
    main()
