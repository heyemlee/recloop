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


# --- paid runs: caps, the judge's grades, the report's chat and relevance sections ------------


def test_parse_caps():
    from agent.eval.run import parse_caps

    models = ["claude-opus-5-5", "claude-haiku-5-5"]
    assert parse_caps("12", models) == {"claude-opus-5-5": 12.0, "claude-haiku-5-5": 12.0}
    assert parse_caps("claude-opus-5-5=12,claude-haiku-5-5=1", models)["claude-haiku-5-5"] == 1.0
    with pytest.raises(ValueError):
        parse_caps("claude-opus-5-5=12", models)  # no cap for haiku
    with pytest.raises(ValueError):
        parse_caps(None, models)


def test_llm_run_stops_before_passing_its_cap(catalog, fake_llm):
    import json as js

    cases = [c for c in case_io.load() if c.template == "budget"][:4]
    parse = js.dumps({"category": "gel_pens", "budget_max_cents": 1200, "search_text": "gel pens"})
    explain = js.dumps({"reasons": []})
    llm = fake_llm(parse=[parse] * 4, explain=[explain] * 4)  # $0.001 per call, 2 calls a case
    spec = harness.llm_spec("fake-model", 1)
    run = harness.run_agent(spec, cases, catalog, llm, max_cost_usd=0.005)
    # Case 1 costs $0.002; case 2 could take the run to $0.006: not started.
    assert run["partial"] and len(run["cases"]) == 1 and run["cost_usd"] <= 0.005


class _Calls:
    def __init__(self, grades):
        self.grades, self.model, self.seen = grades, "fake-judge", []

    def complete_json(self, purpose, system, user, schema):
        import json as js

        from agent.llm import LLMCall, LLMReply

        ids = [p["item_id"] for p in js.loads(user)["products"]]
        self.seen.append(ids)
        out = {"grades": [{"item_id": i, "grade": self.grades.get(i, 1), "why": "x"} for i in ids]}
        return LLMReply(js.dumps(out), LLMCall(purpose, self.model, cost_usd=0.01))


def _run(agent, shown_by_case):
    return {
        "agent": agent,
        "spec": {},
        "cases": [
            {
                "case_id": cid,
                "message": f"message {cid}",
                "record": {"shown": [{"item_id": i} for i in ids]},
            }
            for cid, ids in shown_by_case.items()
        ],
    }


def test_judge_grades_each_shown_item_once(catalog, tmp_path):
    calls = _Calls({7: 2, 8: 0})
    runs = [_run("baseline", {"a": [7, 8], "b": [9]}), _run("oracle", {"a": [8, 10], "b": [9]})]
    path = tmp_path / "judge" / judge.GRADES_FILE
    out = judge.grade_runs(judge.ClaudeJudge(calls), runs, catalog, path, max_cost_usd=1.0)
    assert sorted(map(sorted, calls.seen)) == [[7, 8, 10], [9]]  # one call per case, shared items
    assert out["cases"]["a"] == {"7": 2, "8": 0, "10": 1} and out["calls"] == 2
    assert out["cost_usd"] == pytest.approx(0.02) and not out["partial"]
    # Nothing new to grade: no calls; a new run's new item: one call.
    judge.grade_runs(judge.ClaudeJudge(calls), runs, catalog, path, 1.0)
    runs.append(_run("llm", {"b": [9, 11]}))
    out = judge.grade_runs(judge.ClaudeJudge(calls), runs, catalog, path, 1.0)
    assert calls.seen[-1] == [11] and out["calls"] == 3
    # The report: grades shared by the agents that showed the item.
    text = "\n".join(report.relevance_section(runs, path))
    assert "| baseline | 3 | 100.0% | 1.00 | 33.3% | 33.3% |" in text
    assert "| oracle | 3 | 100.0% | 0.67 | 0.0% | 33.3% |" in text


def test_judge_stops_at_its_cap(catalog, tmp_path):
    calls = _Calls({})
    runs = [_run("baseline", {c: [7] for c in "abcdef"})]
    path = tmp_path / "grades.json"
    out = judge.grade_runs(judge.ClaudeJudge(calls), runs, catalog, path, max_cost_usd=0.035)
    # $0.01 a call; stops before a call that could pass $0.035 (twice the dearest so far).
    assert out["partial"] and out["calls"] == 2 and out["cost_usd"] <= 0.035


def test_chat_section(tmp_path):
    import json as js

    reply = {"status": "ok", "items": 3, "sponsored": True, "personal": {"hits": 1, "moved": True},
             "ads": {"called": True, "candidates": 4, "outcome": "filled"}, "degraded": [],
             "tools": ["search_items", "get_item_details"], "client_reply_ms": 15.0,
             "client_total_ms": 15.5, "server_ms": 14.0,
             "timings_ms": {"parse": 0.1, "history": 0.0, "search": 10.0, "personalize": 2.0,
                            "ads": 0.7, "total": 14.0},
             "llm": {"calls": 0, "cost_usd": 0}}  # fmt: skip
    meta = {"started": "2026-11-08T10:30:00+00:00", "git": "abc", "cases": 1, "repeats": 2,
            "warmup": 30, "users": "2 users", "parser": "rules", "cpus": 12,
            "load_start": [3.0], "load_end": [2.5], "quiet_start": False, "waited_s": 1200,
            "compose_project": "recloop-m6", "other_containers": ["x-1"], "campaigns": "1 active."}  # fmt: skip
    path = tmp_path / "latency.json"
    path.write_text(js.dumps({"meta": meta, "replies": [reply, reply]}))
    text = "\n".join(report.chat_section([], path))
    assert (
        "| search | search_items: BM25, bge query encoding, pgvector KNN, SQL filters | 10 ms | 10 ms |"
        in text
    )
    assert (
        "| history | get_purchase_history (when the message refers to it) (not called in this run) | – | – |"
        in text
    )
    assert "**Indicative only**" in text and "`recloop-m6`): 1." in text and "x-1" not in text
    assert "filled the slot in 2 (100.0% of the asks)" in text
    assert "| claude-opus-5-5 | not run |" in text
    assert "Not measured yet" in "\n".join(report.chat_section([], tmp_path / "missing.json"))
