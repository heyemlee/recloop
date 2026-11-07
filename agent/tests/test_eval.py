from collections import Counter

import pytest

from agent import constraints
from agent.eval import cases as case_io
from agent.eval import harness, judge, metrics, report
from agent.schema import Constraints
from agent.tests.conftest import ITEMS

ROWS = {it.item_id: it for it in ITEMS}


def test_case_file_is_valid_and_balanced():
    cases = case_io.load()
    assert len(cases) == 150
    assert len({c.id for c in cases}) == 150 and len({c.user_id for c in cases}) == 150
    counts = Counter(c.template for c in cases)
    assert counts["compat"] == 20 and counts["unsatisfiable"] == 8
    for c in cases:
        c.oracle_request()  # valid ShoppingRequest
        refers = c.gold.last_purchase != "none"
        assert bool(c.purchases) == refers, c.id
        assert c.gold.category is not None, c.id


def test_case_history_is_most_recent_first():
    case = case_io.Case(
        id="x", template="t", message="m", user_id=1, gold=Constraints(),
        purchases=[case_io.CasePurchase(item_id=5, days_ago=30), case_io.CasePurchase(item_id=6, days_ago=2)],
    )  # fmt: skip
    assert [p.item_id for p in case.history(1, 10)] == [6, 5]
    assert case.history(2, 10) == []


def test_parse_fields_compare_meaning_not_spelling():
    gold = Constraints(brands=["Paper Mate"], compatible_with="Brother HL-2240")
    parsed = Constraints(brands=["paper mate"], compatible_with="HL2240")
    assert metrics.field_matches("brands", parsed, gold)
    assert metrics.field_matches("compatible_with", parsed, gold)
    assert not metrics.field_matches(
        "compatible_with", Constraints(compatible_with="HL-2270DW"), gold
    )


def call(name, args, result_ids=()):
    return {"name": name, "args": args, "result_ids": list(result_ids), "ms": 1.0}


def filters(**kw):
    base = {
        "category": "gel_pens",
        "max_price_cents": None,
        "in_stock": True,
        "brands": [],
        "exclude_brands": [],
    }
    return {**base, **kw}


def test_tool_problems():
    eff = constraints.Effective(Constraints(category="gel_pens"))
    ok = [
        call("search_items", {"query": "q", "filters": filters()}, [7, 8]),
        call("get_item_details", {"ids": [7, 8]}, [7, 8]),
    ]
    assert metrics.tool_problems(ok, eff, 1) == []
    extra = [call("get_purchase_history", {"user_id": 1}), *ok]
    assert metrics.tool_problems(extra, eff, 1) == ["unneeded get_purchase_history"]
    wrong = [
        call("search_items", {"query": "q", "filters": filters(max_price_cents=500)}, [7]),
        ok[1],
    ]
    problems = metrics.tool_problems(wrong, eff, 1)
    assert any("max_price_cents" in p for p in problems)
    assert any("did not return" in p for p in problems)
    twice = [ok[0], ok[0], ok[1]]
    assert "search_items called 2 times" in metrics.tool_problems(twice, eff, 1)
    hist_eff = constraints.Effective(Constraints(category="gel_pens", last_purchase="same_brand"))
    assert "missing get_purchase_history" in metrics.tool_problems(ok, hist_eff, 1)


def record(shown, parsed, trace=(), status="ok"):
    return {
        "status": status,
        "shown": [{"item_id": i, "compat": None} for i in shown],
        "parse": {"request": parsed},
        "trace": list(trace),
        "llm_calls": [],
        "timings_ms": {"total": 5.0},
    }


def test_score_counts_violations_against_the_gold_constraints():
    gold = Constraints(category="gel_pens", exclude_brands=["Pilot"])
    eff = constraints.Effective(gold)
    sat = metrics.satisfiable(ITEMS, eff)
    assert sat == {8, 9}
    parsed = {**Constraints(category="gel_pens", brands=["Pilot"]).model_dump(), "search_text": "x"}
    s = metrics.score_case("c", "t", 1, gold, eff, sat, ROWS, record([7, 10], parsed))
    assert (s.hard_violations, s.all_violations, s.success, s.hit) == (0, 2, False, False)
    assert not s.parse_fields["brands"] and not s.parse_fields["exclude_brands"]
    good = metrics.score_case(
        "c", "t", 1, gold, eff, sat, ROWS, record([9, 7], {**gold.model_dump(), "search_text": "x"})
    )
    assert good.hit and not good.success  # item 7 is Pilot
    assert metrics.score_case(
        "c", "t", 1, gold, eff, sat, ROWS, record([9], {**gold.model_dump(), "search_text": "x"})
    ).success


def test_unsatisfiable_case_succeeds_only_with_nothing_violating():
    gold = Constraints(category="gel_pens", budget_max_cents=100)
    eff = constraints.Effective(gold)
    assert metrics.satisfiable(ITEMS, eff) == set()
    empty = metrics.score_case(
        "c", "t", 1, gold, eff, set(), ROWS, record([], None, status="parse_failed")
    )
    assert empty.success and not empty.parse_ok
    shown = metrics.score_case("c", "t", 1, gold, eff, set(), ROWS, record([7], None))
    assert shown.hard_violations == 1 and not shown.success


def test_snapshot_records_orders_and_the_report_says_if_stock_is_as_ingested(catalog):
    snap = harness.snapshot(catalog.all_items(), catalog.order_count())
    assert snap["orders"] == 2 and snap["items"] == len(ITEMS)
    assert "fresh database" in report.database_note({**snap, "orders": 0})
    assert report.database_note({**snap, "orders": 22246}).startswith("22,246 orders")
    assert "belong to this snapshot" in report.database_note({"items": 3})


def test_quantile_and_summary():
    assert report.quantile([5, 1, 3, 2, 4], 0.5) == 3
    assert report.quantile([1, 2, 3, 4], 0.95) == 4
    assert report.quantile([], 0.5) is None
    gold = Constraints(category="gel_pens")
    eff = constraints.Effective(gold)
    s = metrics.score_case(
        "c",
        "budget",
        1,
        gold,
        eff,
        {7},
        ROWS,
        record([7], {**gold.model_dump(), "search_text": "x"}),
    )
    summary = report.summarize([{"score": s.as_dict()}])
    assert summary["success"] == 1.0 and summary["hard"] == 1.0 and summary["shown"] == 1


def test_judge_agreement():
    j = {("a", 1): 2, ("a", 2): 0, ("b", 3): 1, ("b", 4): 2}
    h = {("a", 1): 2, ("a", 2): 0, ("b", 3): 2, ("b", 4): 2}
    out = judge.agreement(j, h)
    assert out["pairs"] == 4 and out["agreement"] == pytest.approx(0.75)
    assert 0 < out["kappa"] < 1
    assert judge.agreement({}, h)["pairs"] == 0
