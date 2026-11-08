import json

import pytest

from agent import prompts
from agent.parsing import LLMParser, RuleParser

GOOD = json.dumps({"category": "gel_pens", "budget_max_cents": 1500, "search_text": "gel pens"})


def test_valid_reply_is_accepted_first_time(fake_llm):
    llm = fake_llm(parse=[GOOD])
    out = LLMParser(llm).parse("gel pens under $15")
    assert out.request.category == "gel_pens" and out.request.budget_max_cents == 1500
    assert out.attempts == 1 and not out.errors and len(out.llm_calls) == 1
    purpose, system, user, schema = llm.calls[0]
    assert purpose == "parse" and system == prompts.PARSE_SYSTEM
    assert json.loads(user) == {"message": "gel pens under $15"}
    assert "search_text" in schema["required"]


def test_invalid_reply_is_retried_once_with_the_error(fake_llm):
    bad = json.dumps({"category": "gel_pens", "compatible_with": "my printer", "search_text": "x"})
    llm = fake_llm(parse=[bad, GOOD])
    out = LLMParser(llm).parse("gel pens")
    assert out.request is not None and out.attempts == 2
    retry = json.loads(llm.calls[1][2])
    assert retry["previous_reply"] == bad
    assert "model number" in retry["validation_error"]
    assert len(out.llm_calls) == 2  # both calls are billed


def test_two_failures_give_up(fake_llm):
    llm = fake_llm(parse=["not json", json.dumps({"category": "spaceships", "search_text": "x"})])
    out = LLMParser(llm).parse("??")
    assert out.request is None and out.attempts == 2 and len(out.errors) == 2


def test_refusal_counts_as_an_attempt(fake_llm):
    llm = fake_llm(parse=[RuntimeError("refusal"), GOOD])
    out = LLMParser(llm).parse("gel pens")
    assert out.request is not None and out.attempts == 2 and "refusal" in out.errors[0]


@pytest.fixture
def rules():
    return RuleParser(["Pilot", "Paper Mate", "Canon", "Brother", "Sharpie"])


@pytest.mark.parametrize(
    "message, cents",
    [
        ("gel pens under $12", 1200),
        ("gel pens for less than $9.99", 999),
        ("no more than 20 dollars", 2000),
        ("under 10 bucks", 1000),
        ("forty dollars or less", None),  # number words: not handled by the rules
    ],
)
def test_rules_budget(rules, message, cents):
    assert rules.parse(message).request.budget_max_cents == cents


def test_rules_device_brand_and_category(rules):
    r = rules.parse("Toner cartridges for my Brother HL-2240 under $40").request
    assert r.category == "toner" and r.compatible_with == "Brother HL-2240"
    assert r.brands == []  # the device's brand is not a brand constraint
    assert r.budget_max_cents == 4000 and "40" not in r.search_text.split()
    r = rules.parse("Toner compatible with the HP Color LaserJet Pro M252dw").request
    assert r.compatible_with == "HP Color LaserJet Pro M252dw"
    r = rules.parse("Pilot gel pens").request
    assert r.brands == ["Pilot"] and r.category == "gel_pens"
    assert rules.parse("Genuine ink for my Canon MG3620").request.oem_only


def test_rules_do_not_understand_negation(rules):
    # The documented limitation the evaluation measures.
    assert rules.parse("Gel pens, anything but Pilot").request.brands == ["Pilot"]


def test_brand_dictionary_drops_noise():
    parser = RuleParser.from_catalog(
        ["Pilot", "Pilot", "Need", "Need", "Colored Pencils", "Colored Pencils", "Solo"]
    )
    assert parser.parse("I need gel pens").request.brands == []
    assert parser.parse("colored pencils").request.brands == []
    assert parser.parse("Solo stapler").request.brands == []  # one item only
    assert parser.parse("Pilot pens").request.brands == ["Pilot"]
    # The most common spelling of a brand, whichever listing comes first.
    parser = RuleParser.from_catalog([".Pilot", "Pilot", "Pilot", "PILOT"])
    assert parser.parse("Pilot pens").request.brands == ["Pilot"]
