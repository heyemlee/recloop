import csv
import json
from collections import Counter

import pytest

from agent.eval import judge, report, review
from agent.tests.conftest import ITEMS

ROWS = {it.item_id: it for it in ITEMS}


def _grades(n_cases: int) -> dict:
    """A judge's grade store: case c showed the fixture's items 1 .. c % 6 (none for some)."""
    cases = {
        f"case-{c:03d}": {str(i): (c + i) % 3 for i in range(1, c % 6 + 1)} for c in range(n_cases)
    }
    return {"model": "judge-model", "cases": cases, "calls": 1, "errors": 0, "cost_usd": 0.0}


def test_sample_is_fixed_bounded_and_judged():
    g = _grades(80)
    pairs = review.sample_pairs(g)
    assert pairs == review.sample_pairs(g)  # fixed seed
    per_case = Counter(c for c, _ in pairs)
    assert len(per_case) == review.REVIEW_CASES
    judged = review.judged_pairs(g)
    assert all(p in judged for p in pairs) and len(set(pairs)) == len(pairs)
    for c, n in per_case.items():
        assert n == min(review.ITEMS_PER_CASE, len(g["cases"][c])) and n > 0


def _write(path, rows, fields, encoding="utf-8"):
    with path.open("w", newline="", encoding=encoding) as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def test_sheet_shows_what_the_judge_saw_and_reads_back(tmp_path):
    g = _grades(8)
    pairs = review.sample_pairs(g, n_cases=5)
    rows = review.sheet_rows(pairs, {c: f"message {c}" for c in g["cases"]}, ROWS)
    path = tmp_path / "review.csv"
    review.write_sheet(rows, path)
    with path.open() as f:
        got = list(csv.DictReader(f))
    assert list(got[0]) == review.COLUMNS and "judge" not in " ".join(review.COLUMNS)
    assert [r["grade"] for r in got] == [""] * len(pairs)
    first = ROWS[pairs[0][1]]
    assert got[0]["message"] == f"message {pairs[0][0]}" and got[0]["title"] == first.title
    assert got[0]["category"] == " > ".join(first.categories[2:])
    # Filled in and saved by a spreadsheet app: 2 written as 2.0, a byte-order mark.
    judged = review.judged_pairs(g)
    for r in got:
        r["grade"] = f"{float(judged[(r['case_id'], int(r['item_id']))]):.1f}"
    _write(path, got, review.COLUMNS, encoding="utf-8-sig")
    assert review.read_sheet(path, judged) == {p: judged[p] for p in pairs}
    # A sheet with grades in it is not replaced by accident.
    with pytest.raises(review.SheetError, match="already has grades"):
        review.write_sheet(rows, path)
    review.write_sheet(rows, path, force=True)


def test_sheet_errors_name_the_rows(tmp_path):
    judged = {("a", 1): 2, ("a", 2): 0, ("b", 3): 1}
    path = tmp_path / "review.csv"
    _write(
        path,
        [
            {"case_id": "a", "item_id": "1", "grade": ""},
            {"case_id": "a", "item_id": "2", "grade": "3"},
            {"case_id": "b", "item_id": "9", "grade": "1"},
            {"case_id": "b", "item_id": "3", "grade": "1"},
            {"case_id": "b", "item_id": "3", "grade": "1"},
        ],
        ["case_id", "item_id", "grade"],
    )
    with pytest.raises(review.SheetError) as e:
        review.read_sheet(path, judged)
    msg = str(e.value)
    assert "4 rows to fix" in msg and "line 2 (a, item 1): no grade" in msg
    assert "line 3 (a, item 2): grade '3' is not 0, 1 or 2" in msg
    assert "line 4 (b, item 9): the judge did not grade this pair" in msg
    assert "line 6 (b, item 3): graded twice" in msg


def test_agreement_with_case_intervals_and_the_report(tmp_path):
    judged = {("a", 1): 2, ("a", 2): 0, ("b", 3): 1, ("b", 4): 2, ("c", 5): 2, ("c", 6): 1}
    human = {("a", 1): 2, ("a", 2): 0, ("b", 3): 2, ("b", 4): 2, ("c", 5): 2, ("c", 6): 1}
    s = review.summarize(judged, human, b=200)
    assert s["pairs"] == 6 and s["cases"] == 3
    assert s["agreement"] == pytest.approx(5 / 6)
    assert s["kappa"] == pytest.approx(judge.agreement(judged, human)["kappa"])
    assert s["table"][2] == [0, 1, 3]  # the person's 2s: the judge gave one of them a 1
    assert s["agreement_ci"][0] <= s["agreement"] <= s["agreement_ci"][1]
    assert s["mean_judge"] == pytest.approx(8 / 6) and s["mean_human"] == pytest.approx(9 / 6)
    assert s == review.summarize(judged, human, b=200)  # seeded
    # The report: the pointer until there is a human.json, then its numbers and table.
    grades = tmp_path / "judge" / "grades.json"
    grades.parent.mkdir()
    grades.write_text(json.dumps(_grades(0) | {"model": "m"}))
    assert "Human check: not done yet" in "\n".join(report.relevance_section([], grades))
    (grades.parent / "human.json").write_text(json.dumps(s))
    text = "\n".join(report.relevance_section([], grades))
    assert "a person graded 6 items of 3 cases" in text and "Exact agreement 83.3% [" in text
    assert "| person 2 | 0 | 1 | 3 |" in text


def test_kappa_matches_agreement_and_is_undefined_for_one_grade():
    assert judge.kappa([2, 2], [2, 2]) is None
    assert judge.kappa([0, 2], [0, 2]) == pytest.approx(1.0)
    assert judge.kappa([0, 2], [2, 0]) < 0
