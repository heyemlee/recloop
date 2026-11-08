"""The api client against a local stand-in for the Go api (skipped where ports cannot be bound,
e.g. in a sandbox)."""

from __future__ import annotations

import json
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent.store_api import StoreAPI

RID = "01970000-0000-7000-8000-0000000000aa"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive

    def log_message(self, *args):
        pass

    def setup(self):
        super().setup()
        self.server.connections += 1

    def _send(self, status: int, body: object) -> None:
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        u = urllib.parse.urlsplit(self.path)
        q = dict(urllib.parse.parse_qsl(u.query))
        self.server.seen.append(("GET", u.path, q))
        if q.get("user_id") == "500":
            return self._send(500, {"error": "boom"})
        if q.get("user_id") == "999":
            time.sleep(0.5)
        items = [
            {"position": 1, "item_id": 10, "source": "sasrec"},
            {"position": 2, "item_id": 11, "source": "itemcf"},
        ]
        self._send(
            200,
            {
                "request_id": RID,
                "model_version": "v1",
                "degraded": ["pages_skipped"],
                "items": items,
            },
        )

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.seen.append(("POST", self.path, body))
        items = []
        if body["candidate_item_ids"]:
            items = [{"position": 1, "item_id": body["candidate_item_ids"][0], "slot": "sponsored",
                      "label": "Sponsored", "impression_id": f"{RID}:1", "campaign_id": 7,
                      "ad_token": "a.b.c"}]  # fmt: skip
        self._send(200, {"request_id": RID, "surface": "chat", "degraded": [], "items": items,
                         "ads": {"outcome": "filled" if items else "empty"}})  # fmt: skip


@pytest.fixture
def server():
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    except PermissionError as e:
        pytest.skip(f"cannot bind a local port here: {e}")
    srv.seen, srv.connections = [], 0
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def test_recommend(server):
    api = StoreAPI(f"http://127.0.0.1:{server.server_port}")
    recs = api.recommend(3, [5, 6], 50)
    assert recs.error is None and recs.request_id == RID and recs.degraded == ["pages_skipped"]
    assert [(h.item_id, h.rank, h.source) for h in recs.hits] == [
        (10, 1, "sasrec"),
        (11, 2, "itemcf"),
    ]
    assert server.seen[0] == (
        "GET",
        "/v1/recommendations",
        {"user_id": "3", "surface": "chat", "limit": "50", "seed_items": "5,6"},
    )
    api.recommend(3, [], 10)
    assert "seed_items" not in server.seen[1][2]
    assert server.connections == 1  # one kept-alive connection


def test_select_ad(server):
    api = StoreAPI(f"http://127.0.0.1:{server.server_port}")
    sel = api.select_ad(3, [8, 9], [1, 2], 2500)
    assert sel.slot is not None and sel.slot.item_id == 8 and sel.slot.ad_token == "a.b.c"
    assert (
        sel.slot.impression_id == f"{RID}:1"
        and sel.slot.request_id == RID
        and sel.outcome == "filled"
    )
    assert server.seen[0][2] == {"user_id": 3, "candidate_item_ids": [8, 9], "exclude_item_ids": [1, 2],
                                 "max_price_cents": 2500}  # fmt: skip
    sel = api.select_ad(3, [], [], None)
    assert (
        sel.slot is None and sel.outcome == "empty" and "max_price_cents" not in server.seen[1][2]
    )


def test_failures(server):
    api = StoreAPI(f"http://127.0.0.1:{server.server_port}", recs_timeout_s=0.1)
    slow = api.recommend(999, [], 10)
    assert slow.error == "timeout" and slow.hits == [] and slow.ms < 400
    bad = api.recommend(500, [], 10)
    assert bad.error == "error" and "HTTP 500" in bad.detail
    assert api.recommend(3, [], 10).error is None  # a fresh connection after the failures


def test_connection_refused():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler) if _can_bind() else None
    if srv is None:
        pytest.skip("cannot bind a local port here")
    port = srv.server_port
    srv.server_close()  # nothing listens there now
    r = StoreAPI(f"http://127.0.0.1:{port}").select_ad(1, [2], [], None)
    assert r.slot is None and r.error == "error"


def _can_bind() -> bool:
    try:
        ThreadingHTTPServer(("127.0.0.1", 0), Handler).server_close()
        return True
    except PermissionError:
        return False


def test_bad_url():
    with pytest.raises(ValueError):
        StoreAPI("https://example.com")
