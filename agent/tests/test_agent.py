"""The pipeline's guarantees, with scripted model replies (design §10.3)."""

import json
from datetime import UTC, datetime

from agent.agent import AgentConfig, ShoppingAgent
from agent.catalog import Purchase
from agent.parsing import FixedParser, LLMParser
from agent.schema import ShoppingRequest
from agent.tests.conftest import USER


def req(**kw) -> ShoppingRequest:
    kw.setdefault("search_text", "item")
    return ShoppingRequest(**kw)


def run(catalog, parser, llm=None, message="msg", **cfg):
    return ShoppingAgent(catalog, parser, llm, AgentConfig(**cfg)).run(message, USER)


def test_shown_items_always_meet_the_hard_constraints(catalog):
    r = run(
        catalog,
        FixedParser(req(category="ink_cartridges", budget_max_cents=2500, search_text="ink")),
    )
    assert r.status == "ok"
    for s in r.shown:
        assert s.item.stock > 0
        assert s.item.price_cents is not None and s.item.price_cents <= 2500
    assert 3 not in [s.item.item_id for s in r.shown]  # unknown price


def test_reply_prices_and_titles_come_from_the_rows(catalog, fake_llm):
    parse = json.dumps({"category": "gel_pens", "search_text": "gel pens"})
    explain = json.dumps(
        {"reasons": [{"item_id": 7, "reason": "Only $1.99 today and the best pen ever."}]}
    )
    llm = fake_llm(parse=[parse], explain=[explain])
    r = run(catalog, LLMParser(llm), llm, message="gel pens")
    assert "$1.99" not in r.reply
    shown = {s.item.item_id: s for s in r.shown}
    for s in r.shown:
        assert s.item.title[:40] in r.reply and f"${s.item.price_cents / 100:.2f}" in r.reply
    if 7 in shown:
        assert shown[7].reason_source == "code" and shown[7].reason_note == "mentions a price"


def test_model_reasons_are_kept_when_clean(catalog, fake_llm):
    parse = json.dumps({"category": "gel_pens", "brands": ["Pentel"], "search_text": "gel pens"})
    explain = json.dumps(
        {"reasons": [{"item_id": 9, "reason": "A smooth retractable gel pen from Pentel."}]}
    )
    llm = fake_llm(parse=[parse], explain=[explain])
    r = run(catalog, LLMParser(llm), llm)
    assert [s.item.item_id for s in r.shown] == [9]
    assert r.shown[0].reason_source == "model" and "Pentel" in r.shown[0].reason
    assert llm.purposes() == ["parse", "explain"]


def test_device_matches_by_code_first_and_the_model_only_for_the_rest(catalog, fake_llm):
    parse = json.dumps(
        {"category": "toner", "compatible_with": "Brother HL-2240", "search_text": "toner HL-2240"}
    )
    llm = fake_llm(parse=[parse], explain=[json.dumps({"reasons": []})])
    r = run(catalog, LLMParser(llm), llm)
    # Items 13, 14 and 16 all name the HL-2240: three code confirmations, no judge call.
    assert {s.item.item_id for s in r.shown} == {13, 14, 16}
    assert all(s.compat.status == "confirmed" and s.compat.source == "code" for s in r.shown)
    assert "compat" not in llm.purposes()


def test_judge_evidence_is_verified(catalog, fake_llm):
    parse = json.dumps(
        {"category": "ink_cartridges", "compatible_with": "HP Deskjet 2540", "search_text": "ink"}
    )
    judged = json.dumps(
        {
            "items": [
                # Invented quote: must not be trusted.
                {
                    "item_id": 1,
                    "verdict": "compatible",
                    "evidence": [{"field": "title", "quote": "works with Deskjet 2540"}],
                },
                {
                    "item_id": 6,
                    "verdict": "incompatible",
                    "evidence": [{"field": "features", "quote": "HP ENVY 4500, 5540"}],
                },
            ]
        }
    )
    reasons = json.dumps(
        {"reasons": [{"item_id": 1, "reason": "Works great with your Deskjet 2540."}]}
    )
    llm = fake_llm(parse=[parse], compat=[judged], explain=[reasons])
    r = run(catalog, LLMParser(llm), llm, max_results=3)
    by_id = {s.item.item_id: s for s in r.shown}
    assert by_id[5].compat.status == "confirmed"  # code: 'Deskjet 1000, 1010, 1050, 2540'
    assert 6 not in by_id  # verified 'incompatible'
    assert by_id[1].compat.status == "unconfirmed"
    # The reason claimed a compatibility the listing does not confirm: replaced.
    assert by_id[1].reason_source == "code" and "compatibility" in by_id[1].reason_note
    assert "not confirmed" in r.reply
    assert list(by_id)[0] == 5  # confirmed first


def test_parse_failure_calls_no_tools(catalog, fake_llm):
    llm = fake_llm(parse=["{}", "{}"])
    r = run(catalog, LLMParser(llm), llm)
    assert r.status == "parse_failed" and r.trace == [] and "couldn't" in r.reply


def test_history_only_when_the_request_refers_to_it(catalog):
    r = run(catalog, FixedParser(req(category="gel_pens")))
    assert [t.name for t in r.trace] == ["search_items", "get_item_details"]
    r = run(catalog, FixedParser(req(category="gel_pens", last_purchase="different_brand")))
    assert [t.name for t in r.trace] == ["get_purchase_history", "search_items", "get_item_details"]
    assert r.trace[0].args["user_id"] == USER
    assert all(s.item.brand not in ("Pilot", "Pilot Corporation") for s in r.shown)
    assert r.trace[1].args["filters"]["exclude_brands"] == ["Pilot"]
    assert "Your last purchase" in r.reply


def test_blocked_requests_show_nothing(catalog):
    # 'Cheaper than last time', but the earlier ink cartridge (item 3) has no known price.
    parser = FixedParser(req(category="ink_cartridges", last_purchase="cheaper"))
    bought = [Purchase(3, datetime(2026, 10, 1, tzinfo=UTC))]
    agent = ShoppingAgent(catalog, parser, purchases=lambda user, limit: bought)
    r = agent.run("cheaper ink than last time", USER)
    assert r.status == "blocked" and not r.shown and "unknown" in r.reply
    assert [t.name for t in r.trace] == ["get_purchase_history"]


def test_no_results_reply(catalog):
    r = run(catalog, FixedParser(req(category="gel_pens", budget_max_cents=100)))
    assert r.status == "no_results" and "couldn't find" in r.reply


def test_record_is_json_serialisable(catalog):
    r = run(catalog, FixedParser(req(category="toner", compatible_with="Brother HL-2240")))
    rec = json.loads(json.dumps(r.record()))
    assert rec["status"] == "ok" and rec["shown"][0]["compat"]["status"] == "confirmed"
    assert {t["name"] for t in rec["trace"]} == {"search_items", "get_item_details"}


def test_reply_says_when_no_listing_names_the_device(catalog):
    r = run(
        catalog, FixedParser(req(category="ink_cartridges", compatible_with="Canon PIXMA TS3320"))
    )
    assert r.shown and all(s.compat.status == "unconfirmed" for s in r.shown)
    assert r.reply.startswith("No listing I found says it works with the Canon PIXMA TS3320")
    assert "check compatibility before buying" in r.reply


def test_genuine_reason_names_the_device_maker(catalog):
    c = req(category="toner", compatible_with="Brother HL-2240", oem_only=True)
    r = run(catalog, FixedParser(c))
    reseller = [s for s in r.shown if s.item.brand == "Brother Reseller"]
    assert reseller and "A genuine Brother supply." in reseller[0].reason
    assert 14 not in [s.item.item_id for s in r.shown]  # LINKYO: third-party
