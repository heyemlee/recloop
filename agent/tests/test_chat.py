"""The store's assistant: personalisation, the sponsored item, the reply payload and /v1/chat.

Fakes stand in for the Go api (agent/tests/test_store_api.py tests the real client)."""

from __future__ import annotations

import json
import uuid

import pytest
from fastapi.testclient import TestClient

from agent.agent import AgentConfig, ShoppingAgent
from agent.chat import ChatService, reply_payload, uuid7
from agent.parsing import FixedParser
from agent.schema import ShoppingRequest
from agent.server import create_app
from agent.store_api import AdSelection, RecHit, Recommendations, SponsoredSlot
from agent.tests.conftest import USER

ADS_REQUEST = "01970000-0000-7000-8000-000000000001"


def req(**kw) -> ShoppingRequest:
    kw.setdefault("search_text", "item")
    return ShoppingRequest(**kw)


GEL = req(category="gel_pens", search_text="gel pens")
MG3620 = req(
    category="ink_cartridges", compatible_with="Canon PIXMA MG3620", search_text="ink MG3620"
)


class FakeRecs:
    def __init__(self, items=(), error=None, degraded=()):
        self.items, self.error, self.degraded = list(items), error, list(degraded)
        self.calls: list[tuple[int, list[int], int]] = []

    def recommend(self, user_id, seed_items, limit):
        self.calls.append((user_id, list(seed_items), limit))
        if self.error:
            return Recommendations([], error=self.error)
        hits = [RecHit(i, r, "sasrec") for r, i in enumerate(self.items, start=1)]
        return Recommendations(hits, request_id="rec-1", degraded=list(self.degraded))


class FakeAds:
    def __init__(self, pick=None, error=None):
        self.pick, self.error = pick, error
        self.calls: list[tuple[int, list[int], list[int], int | None]] = []

    def select_ad(self, user_id, candidate_item_ids, exclude_item_ids, max_price_cents):
        self.calls.append(
            (user_id, list(candidate_item_ids), list(exclude_item_ids), max_price_cents)
        )
        if self.error:
            return AdSelection(None, error=self.error)
        if self.pick is None:
            return AdSelection(None, request_id=ADS_REQUEST, outcome="empty")
        slot = SponsoredSlot(
            self.pick, 42, 1, f"{ADS_REQUEST}:1", "tok.en.sig", ADS_REQUEST, "Sponsored"
        )
        return AdSelection(slot, request_id=ADS_REQUEST, outcome="filled")


def agent(catalog, request, recs=None, ads=None, **cfg):
    return ShoppingAgent(catalog, FixedParser(request), None, AgentConfig(**cfg), None, recs, ads)


def shown(r) -> list[int]:
    return [s.item.item_id for s in r.shown]


def checked_order(catalog, request) -> list[int]:
    """The order the checks leave, before personalisation (all kept items)."""
    r = agent(catalog, request, max_results=10).run("msg", USER)
    return shown(r)


# --- personalisation ------------------------------------------------------------------------


def test_personalisation_reorders_close_matches_only(catalog):
    base = checked_order(catalog, GEL)
    assert len(base) == 4
    last = base[-1]
    recs = FakeRecs([last])
    r = agent(catalog, GEL, recs).run("msg", USER)
    # The recommender's first pick, 4th in search order, passes the first three.
    assert shown(r) == [last, *base[:2]]
    assert recs.calls == [(USER, base, 50)]  # seeds: the first checked items
    assert r.personal is not None and r.personal.moved and r.personal.hits == 1
    assert r.shown[0].personal == RecHit(last, 1, "sasrec")
    assert [t.name for t in r.trace][-1] == "get_recommendations"
    # A narrower window: the 4th item stays where the search put it.
    r = agent(catalog, GEL, FakeRecs([last]), personal_window=2).run("msg", USER)
    assert shown(r) == base[:3] and not r.personal.moved


def test_personalisation_never_adds_items(catalog):
    base = checked_order(catalog, GEL)
    r = agent(catalog, GEL, FakeRecs([11, 12, 1])).run("msg", USER)  # none of them gel pens
    assert shown(r) == base[:3] and r.personal.hits == 0 and not r.personal.moved


def test_personalisation_keeps_confirmed_devices_first(catalog):
    base = checked_order(catalog, MG3620)
    r = agent(catalog, MG3620, max_results=10).run("msg", USER)
    unconfirmed = [s.item.item_id for s in r.shown if s.compat.status != "confirmed"]
    assert unconfirmed, "the fixture needs an unconfirmed item"
    r = agent(catalog, MG3620, FakeRecs([unconfirmed[-1]]), max_results=10).run("msg", USER)
    status = [s.compat.status for s in r.shown]
    assert status == sorted(status, key=lambda s: s != "confirmed")
    assert set(shown(r)) == set(base)


def test_recommendation_failures_degrade_quietly(catalog):
    base = checked_order(catalog, GEL)
    r = agent(catalog, GEL, FakeRecs(error="timeout")).run("msg", USER)
    assert shown(r) == base[:3] and r.degraded == ["recommendations_timeout"]
    r = agent(catalog, GEL, FakeRecs([base[-1]], degraded=["sasrec_timeout"])).run("msg", USER)
    assert r.degraded == ["recommendations:sasrec_timeout"]


def test_no_recommender_no_call(catalog):
    r = agent(catalog, GEL).run("msg", USER)
    assert r.personal is None and "get_recommendations" not in [t.name for t in r.trace]
    assert "personalize" not in r.timings_ms and "ads" not in r.timings_ms


# --- the sponsored item ---------------------------------------------------------------------


def test_sponsored_item_comes_from_the_checked_items_not_shown(catalog):
    base = checked_order(catalog, GEL)
    budget = req(category="gel_pens", search_text="gel pens", budget_max_cents=2000)
    ads = FakeAds(pick=base[3])
    r = agent(catalog, budget, ads=ads).run("msg", USER)
    assert ads.calls == [(USER, [base[3]], sorted(base[:3]), 2000)]
    assert r.sponsored is not None and r.sponsored.item.item_id == base[3]
    assert r.ads["called"] and r.ads["outcome"] == "filled" and r.ads["candidates"] == 1
    # Labelled by code, after the organic options.
    lines = r.reply.splitlines()
    i = next(n for n, line in enumerate(lines) if line.startswith("Sponsored: "))
    assert r.sponsored.item.title[:40] in lines[i] and "$" in lines[i]
    assert lines[i - 2].startswith("3. ") and lines[i + 1] == f"   {r.sponsored.reason}"


def test_ads_module_answer_outside_the_candidates_is_dropped(catalog):
    base = checked_order(catalog, GEL)
    for pick in (base[0], 11):  # a shown item; an item the checks never saw
        r = agent(catalog, GEL, ads=FakeAds(pick=pick)).run("msg", USER)
        assert r.sponsored is None and r.degraded == ["ads_not_a_candidate"]
        assert "Sponsored" not in r.reply


def test_no_ad_call_without_candidates(catalog):
    # Every checked MG3620 ink is shown or unconfirmed: nothing the ads module could advertise.
    ads = FakeAds(pick=5)
    r = agent(catalog, MG3620, ads=ads).run("msg", USER)
    assert ads.calls == [] and r.sponsored is None and r.ads == {"called": False, "candidates": 0}
    # Unconfirmed items are never candidates.
    r = agent(catalog, MG3620, ads=ads, max_results=1).run("msg", USER)
    assert all(c in (1, 2, 3) for c in ads.calls[0][1])


def test_ads_failure_leaves_the_reply(catalog):
    r = agent(catalog, GEL, ads=FakeAds(error="timeout")).run("msg", USER)
    assert r.status == "ok" and len(r.shown) == 3 and r.sponsored is None
    assert r.degraded == ["ads_timeout"]


# --- the reply payload and the endpoint ------------------------------------------------------


def test_uuid7():
    u = uuid7()
    assert u.version == 7 and u.variant == uuid.RFC_4122
    assert uuid7().int >> 80 >= u.int >> 80  # time-ordered


def test_reply_payload_cards(catalog):
    base = checked_order(catalog, GEL)
    r = agent(catalog, GEL, FakeRecs([base[1]]), FakeAds(pick=base[3])).run("msg", USER)
    rid = uuid7()
    p = reply_payload(r, rid)
    assert [c["impression_id"] for c in p["items"]] == [f"{rid}:{i}" for i in (1, 2, 3)]
    assert all(c["slot"] == "organic" and "ad_token" not in c for c in p["items"])
    sp = p["sponsored"]
    assert sp["label"] == "Sponsored" and sp["slot"] == "sponsored" and sp["position"] == 4
    assert sp["impression_id"] == f"{ADS_REQUEST}:1" and sp["ad_token"] == "tok.en.sig"
    assert sp["campaign_id"] == 42 and sp["item_id"] == base[3]
    assert p["intro"][0].startswith("Here are 3 options") and p["text"] == r.reply
    assert p["personal"]["request_id"] == "rec-1"
    json.dumps(p)


def parse_sse(body: str) -> list[tuple[str, dict]]:
    out = []
    for block in body.replace("\r\n", "\n").split("\n\n"):
        name, data = None, []
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].strip())
        if name:
            out.append((name, json.loads("\n".join(data))))
    return out


@pytest.fixture
def client(catalog):
    base = checked_order(catalog, GEL)
    service = ChatService(
        catalog, FixedParser(GEL), recommender=FakeRecs([base[1]]), ads=FakeAds(pick=base[3])
    )
    with TestClient(create_app(lambda: service)) as c:
        yield c


def test_chat_streams_the_turn(client):
    r = client.post("/v1/chat", json={"user_id": USER, "message": "  gel pens please "})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    names = [n for n, _ in events]
    assert names[0] == "meta" and names[-2:] == ["reply", "done"]
    stages = [d["stage"] for n, d in events if n == "status"]
    assert stages == ["parse", "search", "check", "personalize", "reasons", "ads"]
    meta, reply, done = events[0][1], events[-2][1], events[-1][1]
    assert reply["reply_id"] == meta["reply_id"] and meta["parser"] == "oracle"
    assert len(reply["items"]) == 3 and reply["sponsored"]["label"] == "Sponsored"
    assert done["llm"]["calls"] == 0 and done["llm"]["cost_usd"] == 0
    assert {"search", "check", "personalize", "ads", "render", "total"} <= set(done["timings_ms"])


def test_chat_rejects_bad_requests(client):
    for body in ({"user_id": USER, "message": "   "}, {"user_id": -1, "message": "pens"},
                 {"user_id": USER, "message": "pens", "extra": 1}, {"message": "pens"}):  # fmt: skip
        assert client.post("/v1/chat", json=body).status_code == 422, body
    assert client.get("/healthz").json() == {"status": "ok", "parser": "oracle"}


def test_chat_reports_a_failed_turn(catalog):
    class Broken:
        name = "broken"

        def parse(self, message):
            raise RuntimeError("boom")

    with TestClient(create_app(lambda: ChatService(catalog, Broken()))) as c:
        events = parse_sse(c.post("/v1/chat", json={"user_id": 1, "message": "pens"}).text)
    assert [n for n, _ in events] == ["meta", "status", "error"]
