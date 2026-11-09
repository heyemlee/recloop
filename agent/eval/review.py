"""The person's check of the relevance judge (design §10.4): a review sheet, then the agreement.

  python -m agent.eval.review sheet        (make agent-review)
  python -m agent.eval.review agreement    (make agent-agreement)

`sheet` draws 50 cases (fixed seed) among those whose shown items the judge graded and, in each,
up to 3 of the graded items: the items every agent showed, so the rules parser's misses are in
it too. It writes them to <runs>/judge/review.csv with what the judge saw: the message and the
product's title, brand, category and first five features (prices and stock were checked by
code; the judge grades relevance only). The grade column is empty, and the judge's grades are
not in the file. A person fills in 0, 1 or 2 by the judge's own rule (RULE); the note column is
free text and is not read.

`agreement` reads the filled sheet, checks every grade, compares them with the judge's
(<runs>/judge/grades.json) and writes <runs>/judge/human.json: the pairs, exact agreement and
Cohen's kappa with 95 % intervals from a case-level bootstrap, both mean grades and the 3 x 3
table. The report (`agent.eval.report`) shows it under the judge's grades.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from agent import paths
from agent.catalog import ITEM_COLUMNS, Item
from agent.eval import judge

REVIEW_CASES = 50
ITEMS_PER_CASE = 3
SHEET_FILE = "review.csv"
HUMAN_FILE = "human.json"
BOOTSTRAP = 1000
BOOTSTRAP_SEED = 20261106  # the evaluation protocol's seed
COLUMNS = [
    "case_id",
    "message",
    "item_id",
    "title",
    "brand",
    "category",
    "features",
    "grade",
    "note",
]
RULE = (
    "Grade each product for its message: 2 = what was asked for, 1 = acceptable but not quite "
    "what was asked, 0 = does not fit what was asked. Judge only from the message and the "
    "product data; prices and stock were already checked."
)


class SheetError(ValueError):
    """The filled sheet cannot be read: what is wrong and where."""


def sample_pairs(
    grades: dict,
    n_cases: int = REVIEW_CASES,
    per_case: int = ITEMS_PER_CASE,
    seed: int = judge.REVIEW_SEED,
) -> list[tuple[str, int]]:
    """(case_id, item_id) pairs for the person: n_cases cases with graded items, up to per_case
    of each case's graded items; the same pairs every time."""
    rng = random.Random(seed)
    cases = sorted(c for c, items in grades["cases"].items() if items)
    pairs: list[tuple[str, int]] = []
    for c in sorted(rng.sample(cases, min(n_cases, len(cases)))):
        items = sorted(int(i) for i in grades["cases"][c])
        pairs += [(c, i) for i in sorted(rng.sample(items, min(per_case, len(items))))]
    return pairs


def sheet_rows(
    pairs: Sequence[tuple[str, int]], messages: dict[str, str], items: dict[int, Item]
) -> list[dict[str, str]]:
    """One row per pair with what the judge saw (judge.ClaudeJudge's payload), grade empty."""
    rows = []
    for case_id, item_id in pairs:
        it = items[item_id]
        rows.append(
            {
                "case_id": case_id,
                "message": messages[case_id],
                "item_id": str(item_id),
                "title": it.title,
                "brand": it.brand or "",
                "category": " > ".join(it.categories[2:]),
                "features": " | ".join(it.features[:5]),
                "grade": "",
                "note": "",
            }
        )
    return rows


def write_sheet(rows: Sequence[dict[str, str]], out: Path, force: bool = False) -> None:
    if out.exists() and not force and any(r["grade"].strip() for r in _read_rows(out)):
        raise SheetError(f"{out} already has grades in it; pass --force to start a new sheet")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def _read_rows(path: Path) -> list[dict[str, str]]:
    # utf-8-sig: a spreadsheet app may save the file with a byte-order mark.
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = {"case_id", "item_id", "grade"} - set(reader.fieldnames or [])
        if missing:
            raise SheetError(f"{path}: no {', '.join(sorted(missing))} column")
        return [{k: (v or "") for k, v in row.items() if k is not None} for row in reader]


def _grade(text: str) -> int | None:
    try:
        x = float(text.strip())  # a spreadsheet may write 2 as 2.0
    except ValueError:
        return None
    return int(x) if x in (0, 1, 2) else None


def read_sheet(path: Path, judged: dict[tuple[str, int], int]) -> dict[tuple[str, int], int]:
    """The person's grades by (case_id, item_id). Every row needs a grade of 0, 1 or 2 for a pair
    the judge graded, once; otherwise SheetError lists the rows (sheet line numbers)."""
    human: dict[tuple[str, int], int] = {}
    problems: list[str] = []
    for line, row in enumerate(_read_rows(path), start=2):
        try:
            key = (row["case_id"].strip(), int(row["item_id"]))
        except ValueError:
            problems.append(f"line {line}: item_id {row['item_id']!r} is not a number")
            continue
        g = _grade(row["grade"])
        if not row["grade"].strip():
            problems.append(f"line {line} ({key[0]}, item {key[1]}): no grade")
        elif g is None:
            problems.append(
                f"line {line} ({key[0]}, item {key[1]}): grade {row['grade']!r} is not 0, 1 or 2"
            )
        elif key not in judged:
            problems.append(
                f"line {line} ({key[0]}, item {key[1]}): the judge did not grade this pair"
            )
        elif key in human:
            problems.append(f"line {line} ({key[0]}, item {key[1]}): graded twice")
        else:
            human[key] = g
    if problems:
        shown = "\n  ".join(problems[:10])
        more = f"\n  ... and {len(problems) - 10} more" if len(problems) > 10 else ""
        raise SheetError(f"{path}: {len(problems)} rows to fix:\n  {shown}{more}")
    if not human:
        raise SheetError(f"{path}: no graded rows")
    return human


def judged_pairs(grades: dict) -> dict[tuple[str, int], int]:
    return {(c, int(i)): g for c, items in grades["cases"].items() for i, g in items.items()}


def summarize(
    judged: dict[tuple[str, int], int],
    human: dict[tuple[str, int], int],
    b: int = BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Agreement on the pairs both graded, with intervals from resampling whole cases (a case's
    items share its message, so they are not independent)."""
    keys = sorted(set(judged) & set(human))
    out = judge.agreement(judged, human)
    table = [[0, 0, 0] for _ in range(3)]  # rows: the person's grade; columns: the judge's
    for k in keys:
        table[human[k]][judged[k]] += 1
    by_case: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for k in keys:
        by_case[k[0]].append(k)
    cases = sorted(by_case)
    rng = np.random.default_rng(seed)
    agree, kappas = [], []
    for _ in range(b):
        ks = [k for d in rng.integers(0, len(cases), len(cases)) for k in by_case[cases[d]]]
        a, h = [judged[k] for k in ks], [human[k] for k in ks]
        agree.append(sum(x == y for x, y in zip(a, h, strict=True)) / len(ks))
        kappa = judge.kappa(a, h)
        if kappa is not None:
            kappas.append(kappa)
    return {
        "pairs": len(keys),
        "cases": len(cases),
        "agreement": out["agreement"],
        "agreement_ci": _ci(agree),
        "kappa": out["kappa"],
        "kappa_ci": _ci(kappas),
        "mean_judge": sum(judged[k] for k in keys) / len(keys),
        "mean_human": sum(human[k] for k in keys) / len(keys),
        "table": table,
        "bootstrap": {"b": b, "seed": seed, "unit": "case"},
    }


def _ci(xs: list[float]) -> list[float] | None:
    return [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))] if xs else None


def fetch_items(item_ids: Sequence[int], url: str) -> dict[int, Item]:
    """The items' current rows, on a read-only session."""
    import psycopg

    with psycopg.connect(url, connect_timeout=5, autocommit=True) as conn:
        conn.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
        rows = conn.execute(
            f"SELECT {ITEM_COLUMNS} FROM items WHERE item_id = ANY(%s)", (list(item_ids),)
        ).fetchall()
    return {it.item_id: it for it in map(Item.from_row, rows)}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("command", choices=["sheet", "agreement"])
    ap.add_argument("--runs-dir", type=Path, default=paths.RUNS, help="where the run files are")
    ap.add_argument("--force", action="store_true", help="sheet: replace a sheet that has grades")
    args = ap.parse_args()
    judge_dir = args.runs_dir / "judge"
    grades_path = judge_dir / judge.GRADES_FILE
    if not grades_path.exists():
        sys.exit(f"no judge grades at {grades_path}: the judge has not run (make agent-eval-llm)")
    grades = json.loads(grades_path.read_text())
    sheet = judge_dir / SHEET_FILE
    try:
        if args.command == "sheet":
            from agent.eval import report

            pairs = sample_pairs(grades)
            messages = {
                c["case_id"]: c["message"]
                for r in report.load_runs(args.runs_dir)
                for c in r["cases"]
            }
            items = fetch_items(sorted({i for _, i in pairs}), paths.database_url())
            write_sheet(sheet_rows(pairs, messages, items), sheet, force=args.force)
            n_cases = len({c for c, _ in pairs})
            print(f"wrote {sheet}: {len(pairs)} items of {n_cases} cases to grade.\n{RULE}")
            print("Fill in the grade column (0, 1 or 2), save as CSV, then: make agent-agreement")
        else:
            if not sheet.exists():
                sys.exit(f"no sheet at {sheet}: make agent-review first")
            judged = judged_pairs(grades)
            res = summarize(judged, read_sheet(sheet, judged))
            res |= {"sheet": SHEET_FILE, "judge_model": grades["model"]}
            (judge_dir / HUMAN_FILE).write_text(json.dumps(res, indent=1) + "\n")
            kappa = "undefined" if res["kappa"] is None else f"{res['kappa']:.2f}"
            print(
                f"{res['pairs']} pairs of {res['cases']} cases: agreement "
                f"{res['agreement']:.1%}, kappa {kappa} -> {judge_dir / HUMAN_FILE}"
            )
    except SheetError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
