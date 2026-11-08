"""The store's Go api, seen from the assistant (design §10.2, §10.3 (5)).

  * `recommend`: GET /v1/recommendations?surface=chat, the personalisation signal: the feed's
    sources over the user's history followed by the assistant's seed items (no ads, not paged).
  * `select_ad`: POST /internal/ads/select, the sponsored item of a reply. The ads module runs
    its auction and gates among the items the assistant's code has already checked; at most one.

Both sit on the reply's critical path and both are optional: a timeout or an error leaves the
reply without personalisation or without an ad and is reported, like the feed's degradation
(design §5.1). The client is the standard library's http.client: one keep-alive connection per
thread, no proxies (the api is an internal address).
"""

from __future__ import annotations

import http.client
import json
import threading
import time
import urllib.parse
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

DEFAULT_API_URL = "http://127.0.0.1:18080"
# The api answers within its own budgets (100 ms for recommendations, 25 ms for ads); these
# only bound a stuck connection.
RECS_TIMEOUT_S = 0.3
ADS_TIMEOUT_S = 0.2
MAX_AD_CANDIDATES = 500  # the api's limit


@dataclass(frozen=True)
class RecHit:
    item_id: int
    rank: int  # 1-based position in the chat surface's list
    source: str  # the source that ranked it best (sasrec, itemcf, pop, backfill)


@dataclass
class Recommendations:
    hits: list[RecHit]
    request_id: str | None = None
    model_version: str | None = None
    degraded: list[str] = field(default_factory=list)  # as the api reported them
    error: str | None = None  # "timeout" or "error" when the call itself failed
    detail: str = ""
    ms: float = 0.0


@dataclass(frozen=True)
class SponsoredSlot:
    item_id: int
    campaign_id: int
    position: int
    impression_id: str  # "{request_id}:{position}" of the ads module's served event
    ad_token: str  # sent back with the slot's impression and click events
    request_id: str
    label: str


@dataclass
class AdSelection:
    slot: SponsoredSlot | None
    request_id: str | None = None
    outcome: str | None = None  # the ads stage's outcome: filled, empty, not_allowed, stale, ...
    degraded: list[str] = field(default_factory=list)
    error: str | None = None
    detail: str = ""
    ms: float = 0.0


class Recommender(Protocol):
    def recommend(self, user_id: int, seed_items: Sequence[int], limit: int) -> Recommendations: ...


class AdSelector(Protocol):
    def select_ad(
        self,
        user_id: int,
        candidate_item_ids: Sequence[int],
        exclude_item_ids: Sequence[int],
        max_price_cents: int | None,
    ) -> AdSelection: ...


class APIError(Exception):
    def __init__(self, kind: str, detail: str):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind  # "timeout" | "error"
        self.detail = detail


class _HTTP:
    """Keep-alive HTTP/1.1 to one host: a connection per thread."""

    def __init__(self, base_url: str):
        u = urllib.parse.urlsplit(base_url)
        if u.scheme != "http" or not u.hostname:
            raise ValueError(f"the api URL must be http://host[:port]: {base_url!r}")
        self.host, self.port = u.hostname, u.port or 80
        self.prefix = u.path.rstrip("/")
        self._local = threading.local()

    def _drop(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
        self._local.conn = None

    def request(self, method: str, path: str, body: Any, timeout_s: float) -> tuple[int, bytes]:
        payload = None if body is None else json.dumps(body).encode()
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        for attempt in (0, 1):
            conn: http.client.HTTPConnection | None = getattr(self._local, "conn", None)
            reused = conn is not None
            if conn is None:
                conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout_s)
                self._local.conn = conn
            elif conn.sock is not None:
                conn.sock.settimeout(timeout_s)
            try:
                conn.request(method, self.prefix + path, body=payload, headers=headers)
                resp = conn.getresponse()
                data = resp.read()
            except (http.client.RemoteDisconnected, ConnectionResetError, BrokenPipeError) as e:
                self._drop()
                if reused and attempt == 0:
                    continue  # the server had closed the idle connection: once more, fresh
                raise APIError("error", f"{type(e).__name__}: {e}") from e
            except TimeoutError as e:
                self._drop()
                raise APIError("timeout", f"no answer within {timeout_s * 1000:.0f} ms") from e
            except OSError as e:
                self._drop()
                raise APIError("error", f"{type(e).__name__}: {e}") from e
            if resp.will_close:
                self._drop()
            return resp.status, data
        raise AssertionError("unreachable")


def _json(status: int, data: bytes) -> dict[str, Any]:
    if status != 200:
        raise APIError("error", f"HTTP {status}: {data[:200].decode(errors='replace')}")
    try:
        out = json.loads(data)
    except ValueError as e:
        raise APIError("error", f"not JSON: {e}") from e
    if not isinstance(out, dict):
        raise APIError("error", "not a JSON object")
    return out


class StoreAPI:
    """Recommender and AdSelector over the Go api."""

    def __init__(
        self,
        base_url: str = DEFAULT_API_URL,
        recs_timeout_s: float = RECS_TIMEOUT_S,
        ads_timeout_s: float = ADS_TIMEOUT_S,
    ):
        self.base_url = base_url
        self.http = _HTTP(base_url)
        self.recs_timeout_s = recs_timeout_s
        self.ads_timeout_s = ads_timeout_s

    def recommend(self, user_id: int, seed_items: Sequence[int], limit: int) -> Recommendations:
        q = {"user_id": str(user_id), "surface": "chat", "limit": str(limit)}
        if seed_items:
            q["seed_items"] = ",".join(str(i) for i in seed_items)
        t0 = time.perf_counter()
        try:
            status, data = self.http.request(
                "GET", "/v1/recommendations?" + urllib.parse.urlencode(q), None, self.recs_timeout_s
            )
            body = _json(status, data)
            hits = [
                RecHit(int(it["item_id"]), int(it["position"]), str(it.get("source", "")))
                for it in body.get("items") or []
            ]
        except (APIError, KeyError, TypeError, ValueError) as e:
            kind, detail = (e.kind, e.detail) if isinstance(e, APIError) else ("error", repr(e))
            ms = (time.perf_counter() - t0) * 1000
            return Recommendations([], error=kind, detail=detail, ms=ms)
        return Recommendations(
            hits,
            request_id=body.get("request_id"),
            model_version=body.get("model_version"),
            degraded=list(body.get("degraded") or []),
            ms=(time.perf_counter() - t0) * 1000,
        )

    def select_ad(
        self,
        user_id: int,
        candidate_item_ids: Sequence[int],
        exclude_item_ids: Sequence[int],
        max_price_cents: int | None,
    ) -> AdSelection:
        body: dict[str, Any] = {
            "user_id": user_id,
            "candidate_item_ids": list(candidate_item_ids)[:MAX_AD_CANDIDATES],
            "exclude_item_ids": list(exclude_item_ids),
        }
        if max_price_cents is not None:
            body["max_price_cents"] = max_price_cents
        t0 = time.perf_counter()
        try:
            status, data = self.http.request(
                "POST", "/internal/ads/select", body, self.ads_timeout_s
            )
            out = _json(status, data)
            items = out.get("items") or []
            slot = None
            if items:
                it = items[0]
                slot = SponsoredSlot(
                    item_id=int(it["item_id"]),
                    campaign_id=int(it["campaign_id"]),
                    position=int(it["position"]),
                    impression_id=str(it["impression_id"]),
                    ad_token=str(it["ad_token"]),
                    request_id=str(out["request_id"]),
                    label=str(it.get("label") or "Sponsored"),
                )
        except (APIError, KeyError, TypeError, ValueError) as e:
            kind, detail = (e.kind, e.detail) if isinstance(e, APIError) else ("error", repr(e))
            ms = (time.perf_counter() - t0) * 1000
            return AdSelection(None, error=kind, detail=detail, ms=ms)
        ads = out.get("ads") or {}
        return AdSelection(
            slot,
            request_id=out.get("request_id"),
            outcome=ads.get("outcome"),
            degraded=list(out.get("degraded") or []),
            ms=(time.perf_counter() - t0) * 1000,
        )
