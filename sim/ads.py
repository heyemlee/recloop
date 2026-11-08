"""Ads for the simulator's billing runs (docs/design.md §9): campaigns created through the
advertiser api, the ads-layer experiment that sets the serving policy (safety margin, pacing),
and the ad_token helpers.

The simulator never verifies a token (it has no key): it reads the claims the way any client
could (a JWT's payload is only base64) to know what it was served, and tampers with a token to
send a forged click.
"""

from __future__ import annotations

import base64
import json
import random
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import psycopg

from sim import verify as verify_mod
from sim.client import Api

ROOT = Path(__file__).resolve().parent.parent


def claims(token: str) -> dict:
    """The payload of an ad_token: rid, pos, cid, item, price, uid, iat, exp."""
    payload = token.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))


def tamper(token: str, rng: random.Random) -> str:
    """Change one character of the token's payload (as a forger raising the price would)."""
    head, payload, sig = token.split(".")
    i = rng.randrange(len(payload))
    c = "A" if payload[i] != "A" else "B"
    return f"{head}.{payload[:i]}{c}{payload[i + 1 :]}.{sig}"


def utc_day(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).date().isoformat()


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat().replace("+00:00", "Z")


class Ads:
    """Campaigns and the serving policy of one run."""

    def __init__(self, api_url: str, database_url: str):
        self.api = Api(api_url, timeout=30)
        self.database_url = database_url

    def _post(self, path: str, doc: dict) -> dict:
        status, body = self.api.post_json(path, doc)
        if status not in (200, 201):
            raise SystemExit(f"POST {path}: {status} {body}")
        return body

    def active_campaigns(self) -> list[dict]:
        status, body, _ = self.api.get_json("/v1/campaigns")
        if status != 200:
            raise SystemExit(f"GET /v1/campaigns: {status} {body}")
        return [c for c in body if c["status"] == "active"]

    def require_no_active_campaigns(self) -> None:
        """Other active campaigns would compete in the auctions and take clicks: refuse."""
        other = self.active_campaigns()
        if other:
            ids = ", ".join(str(c["campaign_id"]) for c in other)
            raise SystemExit(
                f"{len(other)} active campaign(s) ({ids}) would compete with the run's campaigns: "
                "pause or end them first (advertiser console or DELETE /v1/campaigns/{id})"
            )

    def pick_items(self, n: int, seed: int) -> list[int]:
        """n distinct in-stock items with a known price, chosen with a fixed seed."""
        with psycopg.connect(self.database_url) as conn:
            ids = [
                r[0]
                for r in conn.execute(
                    "SELECT item_id FROM items WHERE stock > 20 AND price_cents IS NOT NULL ORDER BY item_id"
                )
            ]
        return sorted(random.Random(seed).sample(ids, n))

    def create(self, name: str, specs: list[dict]) -> list[dict]:
        """One advertiser per campaign (an advertiser takes one slot per auction). A spec has
        bid, budget, items and optionally freq_cap, start, end (unix seconds)."""
        out = []
        for i, s in enumerate(specs):
            adv = self._post("/v1/advertisers", {"name": f"{name} {i + 1}"})
            body = {
                "advertiser_id": adv["advertiser_id"],
                "name": f"{name} {i + 1}",
                "status": "active",
                "daily_budget_cents": s["budget"],
                "bid_cpc_cents": s["bid"],
                "freq_cap_per_day": s.get("freq_cap", 3),
                "item_ids": s["items"],
            }
            if s.get("start"):
                body["start_at"] = iso(s["start"])
            if s.get("end"):
                body["end_at"] = iso(s["end"])
            out.append(self._post("/v1/campaigns", body))
        return out

    def end(self, campaign_ids: list[int]) -> None:
        """End the campaigns (DELETE: terminal, the rows stay for the ledger)."""
        for cid in campaign_ids:
            status = self.api.delete(f"/v1/campaigns/{cid}")
            if status != 200:
                raise SystemExit(f"DELETE /v1/campaigns/{cid}: {status}")

    # ------------------------------------------------------------- serving policy

    def set_policy(self, exp_id: str, margin: int, pacing: bool) -> int:
        """Run the ads-layer experiment exp_id over every bucket with one variant carrying the
        policy (any other running ads-layer experiment of a run is stopped first). Returns its
        config_version."""
        params = {"safety_margin_clicks": margin, "pacing": pacing}
        variants = [
            {"name": f"m{margin}{'_paced' if pacing else ''}", "share": 100, "params": params}
        ]
        with psycopg.connect(self.database_url) as conn:
            conn.execute(
                "UPDATE experiments SET status = 'stopped', end_at = now() "
                "WHERE layer = 'ads' AND status = 'running' AND exp_id <> %s",
                (exp_id,),
            )
            version = conn.execute(
                "SELECT COALESCE(max(config_version), 0) + 1 FROM experiments"
            ).fetchone()[0]
            conn.execute(
                """INSERT INTO experiments (exp_id, layer, status, bucket_from, bucket_to, variants,
                                            config_version, start_at)
                   VALUES (%s, 'ads', 'running', 0, 1000, %s, %s, now())
                   ON CONFLICT (exp_id) DO UPDATE SET status = 'running', variants = EXCLUDED.variants,
                     config_version = EXCLUDED.config_version, start_at = now(), end_at = NULL""",
                (exp_id, json.dumps(variants), version),
            )
        return version

    def stop_policy(self, exp_id: str) -> None:
        with psycopg.connect(self.database_url) as conn:
            conn.execute(
                "UPDATE experiments SET status = 'stopped', end_at = now() WHERE exp_id = %s",
                (exp_id,),
            )

    def wait_policy(self, margin: int, pacing: bool, user: int, timeout: float = 30) -> None:
        """Wait until the api serves with this policy (it reloads experiments every 10 s)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            status, body, _ = self.api.get_json(
                f"/v1/recommendations?user_id={user}&limit=3&debug=1"
            )
            ads = (body or {}).get("debug", {}).get("ads") if status == 200 else None
            if ads and ads["policy"] == {
                "safety_margin_clicks": margin,
                "pacing": pacing,
                "min_pctr": ads["policy"]["min_pctr"],
            }:
                return
            time.sleep(1)
        raise SystemExit(f"the api did not pick up margin {margin} / pacing {pacing} in {timeout}s")

    def wait_campaigns(self, ids: list[int], user: int, timeout: float = 20) -> None:
        """Wait until the api's campaign index has the run's campaigns (refreshed every 5 s)."""
        deadline = time.time() + timeout
        want = len(ids)
        while time.time() < deadline:
            status, body, _ = self.api.get_json(
                f"/v1/recommendations?user_id={user}&limit=3&debug=1"
            )
            ads = (body or {}).get("debug", {}).get("ads") if status == 200 else None
            if ads and ads["stats"]["campaigns"] >= want:
                return
            time.sleep(0.5)
        raise SystemExit(f"the api did not load the run's {want} campaigns in {timeout}s")

    def candidate_cache(self) -> bool:
        """Whether the api runs with the candidate cache (GET /readyz)."""
        status, body, _ = self.api.get_json("/readyz")
        return bool(body and body.get("candidate_cache"))

    def close(self) -> None:
        self.api.close()


def reset_redis() -> None:
    """Start a run from the seeded user state (`make seed`: FLUSHDB of both Redis instances, then
    the dataset users' histories), as the load tests do: wait until the consumers have applied
    everything, so no
    event of an earlier run lands afterwards, then reseed. The spend counters of earlier
    campaigns go too; billing's reconciliation rebuilds them from the ledger."""
    verify_mod.wait_for_consumers(("billing", "log-sink", "aggregator"))
    subprocess.run(
        [sys.executable, "-m", "ml.data.seed_redis"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
