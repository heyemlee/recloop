"""docs/results/agent.md from the run files in artifacts/eval/agent/ (written by agent.eval.run).

Everything except the latency section is a function of the cases, the code and the items table:
re-running on the same database gives the same numbers. Latency is wall time on this machine.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from agent import paths
from agent.eval.cases import load
from agent.eval.metrics import FIELDS
from agent.llm import PRICES_PER_MTOK

ORDER = ["baseline", "oracle", "oracle-bm25", "oracle-vector", "oracle-noprefix"]
PLANNED_MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
TEMPLATES = [
    ("budget", "category + budget, ten phrasings (number words, ranges)"),
    ("brand", "category + brand"),
    ("exclude_brand", "category, not brand X"),
    ("compat", "ink / toner for a named printer"),
    ("compat_budget", "printer + budget"),
    ("compat_oem", "printer + genuine only"),
    ("last_different_brand", "complaint about the last purchase, another brand"),
    ("last_same_brand", "same brand as the last purchase"),
    ("last_cheaper", "cheaper than the last purchase"),
    ("paraphrase", "hand-written paraphrases"),
    ("unsatisfiable", "nothing in the catalog fits"),
    ("category_only", "just a kind of product"),
]


def load_runs(directory: Path = paths.RUNS) -> list[dict[str, Any]]:
    runs = [json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))]

    def key(r: dict) -> tuple:
        name = r["agent"]
        return (ORDER.index(name), "") if name in ORDER else (len(ORDER), name)

    return sorted(runs, key=key)


def pct(x: float | None, digits: int = 1) -> str:
    return (
        "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{digits}f}%"
    )


def ratio(num: float, den: float) -> float | None:
    return num / den if den else None


def quantile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))]


def summarize(cases: Iterable[dict]) -> dict[str, Any]:
    cs = [c["score"] for c in cases]
    shown = sum(len(c["shown"]) for c in cs)
    possible = [c for c in cs if c["possible"]]
    impossible = [c for c in cs if not c["possible"]]
    device = [c for c in cs if c["compat_shown"]]
    return {
        "n": len(cs),
        "shown": shown,
        "hard": ratio(shown - sum(c["hard_violations"] for c in cs), shown),
        "all": ratio(shown - sum(c["all_violations"] for c in cs), shown),
        "cases_violating": sum(c["all_violations"] > 0 for c in cs),
        "success": ratio(sum(c["success"] for c in cs), len(cs)),
        "hit": ratio(sum(c["hit"] for c in possible), len(possible)),
        "n_possible": len(possible),
        "empty_ok": ratio(sum(c["success"] for c in impossible), len(impossible)),
        "n_impossible": len(impossible),
        "parse": ratio(sum(c["parse_ok"] for c in cs), len(cs)),
        "fields": {f: ratio(sum(c["parse_fields"][f] for c in cs), len(cs)) for f in FIELDS},
        "tools": ratio(sum(c["tools_ok"] for c in cs), len(cs)),
        "confirmed": ratio(
            sum(c["compat_confirmed"] for c in device), sum(c["compat_shown"] for c in device)
        ),
        "confirmed_wrong": sum(c["compat_confirmed_wrong"] for c in cs),
        "p50": quantile([c["latency_ms"] for c in cs], 0.5),
        "p95": quantile([c["latency_ms"] for c in cs], 0.95),
        "llm_calls": ratio(sum(c["llm_calls"] for c in cs), len(cs)),
        "in_tokens": ratio(
            sum(c["input_tokens"] + c["cache_read_tokens"] + c["cache_write_tokens"] for c in cs),
            len(cs),
        ),
        "out_tokens": ratio(sum(c["output_tokens"] for c in cs), len(cs)),
        "cost": ratio(sum(c["cost_usd"] for c in cs), len(cs)),
        "cost_p95": quantile([c["cost_usd"] for c in cs], 0.95),
    }


def _ms(x: float | None) -> str:
    return "–" if x is None else (f"{x:.0f} ms" if x >= 10 else f"{x:.1f} ms")


def _usd(x: float | None) -> str:
    return "–" if x is None else f"${x:.4f}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def _stage_p50(run: dict, stage: str) -> float | None:
    return quantile([c["record"]["timings_ms"].get(stage, 0.0) for c in run["cases"]], 0.5)


def _failure(c: dict) -> str:
    s = c["score"]
    if s["violations"]:
        why = "shows " + "; ".join(s["violations"][:2])
    elif s["possible"] and not s["hit"]:
        why = f"nothing from the {len(c['satisfiable'])} satisfying items shown"
        if s["status"] != "ok":
            why += f" (status {s['status']})"
    else:
        why = "; ".join(s["tool_problems"][:2]) or "?"
    wrong = [f for f, ok in s["parse_fields"].items() if not ok]
    if wrong:
        why += f"; parse wrong: {', '.join(wrong)}"
    return why


def _device_case(c: dict) -> bool:
    return c["template"].startswith("compat") and bool(c["satisfiable"])


def _retrieval(run: dict) -> tuple[float | None, float | None]:
    """Mean share of satisfying items among the search results, median rank of the first."""
    shares, firsts = [], []
    for c in run["cases"]:
        if not _device_case(c):
            continue
        hits = next(
            (t["result_ids"] for t in c["record"]["trace"] if t["name"] == "search_items"), []
        )
        sat = set(c["satisfiable"])
        shares.append(len(sat.intersection(hits)) / min(len(sat), max(len(hits), 1)))
        ranks = [i for i, item_id in enumerate(hits, start=1) if item_id in sat]
        firsts.append(ranks[0] if ranks else float("inf"))
    return ratio(sum(shares), len(shares)), quantile(firsts, 0.5)


def render(runs: list[dict[str, Any]]) -> str:
    cases = load()
    by_name = {r["agent"]: r for r in runs}
    snap = runs[0]["snapshot"] if runs else {}
    lines = [
        "# Shopping assistant: offline evaluation",
        "",
        "Generated by `make agent-eval` (`agent/eval/run.py`, then `agent/eval/report.py`); do not",
        "edit by hand. Design: `docs/design.md` §10. Cases: `agent/eval/cases.jsonl`, generated by",
        "`make agent-cases` from templates and catalog attributes.",
        "",
        "## Setup",
        "",
        f"- {len(cases)} cases in {len({c.template for c in cases})} templates (below); each has gold constraints, and the "
        "ones that refer to an earlier purchase carry their own purchase history.",
        f"- Items table at run time: {snap.get('items', '–'):,} items, {snap.get('in_stock', 0):,} in stock, "
        f"{snap.get('priced', 0):,} with a known price (fingerprint `{snap.get('fingerprint', '–')}`). Stock is "
        "simulated and changes with purchases, so the numbers below belong to this snapshot.",
        "- Every agent shares the pipeline: tools (`search_items`: BM25 + pgvector, RRF k=60, price / "
        "stock / category filters in SQL; `get_item_details`; `get_purchase_history`), code checks of "
        "every constraint on the item rows, device matching in code, replies rendered from the rows. "
        "At most 3 items per reply.",
        "- Agents: **baseline** parses with rules (budget regex, category aliases, a brand "
        "dictionary, 'genuine / original / OEM', 'for / with <model>'), no language model. "
        "**oracle** is handed the gold constraints: the ceiling of the pipeline when the request is "
        "understood perfectly. The oracle variants change only the search.",
        "",
        "## Results",
        "",
        "Hard constraints are price, stock and category (design §10.3); 'all' adds brands, excluded "
        "brands, genuine-only and the item the shopper complained about. Both are over the items "
        "shown, checked against the gold constraints. Success: an item meeting every gold constraint "
        "shown when the catalog has one, and nothing violating shown when it has none.",
        "",
    ]
    header = ["agent", "items shown", "hard constraints met", "all constraints met", "success",
              "parse exact", "tool calls correct", "LLM calls / turn", "cost / turn"]  # fmt: skip
    rows = []
    for r in runs:
        s = summarize(r["cases"])
        part = " (partial)" if r.get("partial") else ""
        rows.append([f"{r['agent']}{part}", str(s["shown"]), pct(s["hard"]), pct(s["all"]), pct(s["success"]),
                     pct(s["parse"]), pct(s["tools"]), f"{s['llm_calls']:.2f}", _usd(s["cost"])])  # fmt: skip
    for m in PLANNED_MODELS:
        if not any(r["spec"].get("model") == m for r in runs):
            rows.append([f"{m} (×3)", "not run", "", "", "", "", "", "", ""])
    lines += _table(header, rows)
    lines += [
        "",
        "Language-model agents (Opus 5.5 default; Sonnet 5.5 and Haiku 5.5 to compare; 3 runs "
        "each) have not been run: they need an API key and a spending cap.",
        "",
    ]

    base, oracle = by_name.get("baseline"), by_name.get("oracle")
    if base and oracle:
        lines += ["## By template", "", "Success rate per template (n cases):", ""]
        trows = []
        for t, desc in TEMPLATES:
            bs = summarize(c for c in base["cases"] if c["template"] == t)
            os_ = summarize(c for c in oracle["cases"] if c["template"] == t)
            trows.append([t, desc, str(bs["n"]), pct(bs["success"], 0), pct(bs["parse"], 0), pct(bs["all"], 0),
                          pct(os_["success"], 0)])  # fmt: skip
        lines += _table(["template", "what", "n", "baseline success", "baseline parse exact",
                         "baseline all constraints met", "oracle success"], trows)  # fmt: skip
        lines += [""]

        lines += [
            "## Parse accuracy by field",
            "",
            f"Share of the {len(cases)} cases with the field right (an unset field must stay unset).",
            "",
        ]
        fs = summarize(base["cases"])["fields"]
        lines += _table(["field", "baseline (rules)"], [[f, pct(fs[f])] for f in FIELDS])
        lines += [""]

        bsum, osum = summarize(base["cases"]), summarize(oracle["cases"])
        lines += [
            "## Where the baseline fails",
            "",
            f"- {bsum['cases_violating']} of {bsum['n']} baseline replies show at least one item that breaks a "
            f"gold constraint; the oracle's: {osum['cases_violating']}. The checks run on what the parser "
            "understood, so a misread request passes its own checks and fails the gold ones.",
            f"- Requests the catalog can satisfy ({bsum['n_possible']}): baseline shows a satisfying item in "
            f"{pct(bsum['hit'])}, the oracle in {pct(osum['hit'])}. Requests it cannot ({bsum['n_impossible']}): "
            f"nothing violating shown in {pct(bsum['empty_ok'])} (oracle {pct(osum['empty_ok'])}).",
            f"- Device requests: {pct(bsum['confirmed'])} of the baseline's shown items there are "
            f"code-confirmed compatible (oracle {pct(osum['confirmed'])}); the rest are labelled "
            "'not confirmed' in the reply.",
            "",
            "First two baseline failures per template:",
            "",
        ]
        frows = []
        for t, _ in TEMPLATES:
            fails = [c for c in base["cases"] if c["template"] == t and not c["score"]["success"]]
            for c in fails[:2]:
                frows.append(
                    [c["case_id"], c["message"].replace("|", "/"), _failure(c).replace("|", "/")]
                )
        lines += _table(["case", "message", "what went wrong"], frows) if frows else ["(none)"]
        lines += [""]

    search = [
        by_name[n]
        for n in ("oracle", "oracle-bm25", "oracle-vector", "oracle-noprefix")
        if n in by_name
    ]
    if search:
        lines += [
            "## Search ablation (gold constraints)",
            "",
            "With the gold constraints every item that passes the SQL filters meets the hard "
            "constraints, so success above cannot separate search variants. Device requests can: a "
            "compatible item must be among the 30 results for the code to find it. Measured on the "
            f"{sum(1 for c in search[0]['cases'] if _device_case(c))} device cases the catalog can satisfy, "
            "before compatible items are moved up:",
            "",
        ]
        srows = []
        for r in search:
            rec, first = _retrieval(r)
            srows.append(
                [
                    r["agent"],
                    r["spec"]["description"],
                    pct(rec),
                    "–" if first is None else f"{first:g}",
                ]
            )
        lines += _table(
            [
                "agent",
                "search",
                "compatible items in the 30 results",
                "rank of the first one (median)",
            ],
            srows,
        )
        lines += [
            "",
            "Relevance beyond the constraints (is it the right kind of gel pen?) needs graded labels: "
            "that is the judge below.",
            "",
        ]

    lines += [
        "## Relevance (language-model judge)",
        "",
        "Not run yet. `agent/eval/judge.py` grades each shown item 0 / 1 / 2 with Claude Sonnet 5.5; "
        "50 cases (fixed seed) go to a person for the same grades, and the report gets exact "
        "agreement and Cohen's kappa.",
        "",
        "## Latency (this machine, varies between runs)",
        "",
        "Wall time of a turn in the evaluation process: parsing, tools against the local PostgreSQL "
        "(Docker), checks, rendering; the query encoder (bge-small, CPU) is part of `search_items`.",
        "",
    ]
    lrows = []
    for r in runs:
        s = summarize(r["cases"])
        lrows.append([r["agent"], _ms(s["p50"]), _ms(s["p95"]), _ms(_stage_p50(r, "parse")),
                      _ms(_stage_p50(r, "search")), _ms(_stage_p50(r, "details")), _ms(_stage_p50(r, "compat"))])  # fmt: skip
    lines += _table(
        ["agent", "P50", "P95", "parse P50", "search P50", "details P50", "compat P50"], lrows
    )
    lines += [
        "",
        "## Prices used for costs",
        "",
        "USD per million tokens (input / output / cache write / cache read), Claude API list prices "
        "as of 2026-10; cache-write prices assume 1.25x input. Cost is computed at the price of the "
        "model that served each reply (server-side fallbacks can change it).",
        "",
    ]
    lines += _table(["model", "input", "output", "cache write", "cache read"],
                    [[m, *(f"{p:.3g}" for p in ps)] for m, ps in PRICES_PER_MTOK.items()])  # fmt: skip
    return "\n".join(lines) + "\n"


def main() -> None:
    runs = load_runs()
    if not runs:
        raise SystemExit(f"no runs in {paths.RUNS}: run `make agent-eval` first")
    paths.REPORT.write_text(render(runs))
    print(f"wrote {paths.REPORT.relative_to(paths.ROOT)} from {len(runs)} runs")


if __name__ == "__main__":
    main()
