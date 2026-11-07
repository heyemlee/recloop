"""Scores of one agent turn against a case's gold constraints (design §10.4).

* hard-constraint satisfaction: every shown item meets the gold price, stock and category;
  'all constraints' adds brands, excluded brands, genuine-only and the complained-about item.
  Checked on the item rows of the same run, with the gold constraints (a parsing mistake that
  lets a wrong item through counts against the agent).
* success: something satisfying shown when the catalog has it (the 'satisfiable' set: all
  gold constraints, in stock, and for device requests a code-confirmed match), nothing
  violating shown otherwise.
* parse accuracy: field by field against the gold labels.
* tool calls: the calls the gold constraints imply, with their arguments, and nothing else.
* latency and cost: wall time of the turn; language-model calls, tokens and dollars.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from agent import compat, constraints
from agent.catalog import Item, Purchase, norm_brand
from agent.schema import Constraints

FIELDS = (
    "category",
    "budget_max_cents",
    "brands",
    "exclude_brands",
    "oem_only",
    "compatible_with",
    "last_purchase",
)


def _device_key(s: str | None) -> str | None:
    d = compat.parse_device(s) if s else None
    return None if d is None else (d.prefix or "") + d.model


def _brand_set(xs: list[str]) -> set[str]:
    return {norm_brand(x) for x in xs}


def field_matches(name: str, parsed: Constraints, gold: Constraints) -> bool:
    a, b = getattr(parsed, name), getattr(gold, name)
    if name in ("brands", "exclude_brands"):
        return _brand_set(a) == _brand_set(b)
    if name == "compatible_with":
        return _device_key(a) == _device_key(b)
    return a == b


def gold_effective(
    gold: Constraints, history: list[tuple[Purchase, Item]]
) -> constraints.Effective:
    return constraints.resolve(gold, history)


def satisfiable(items: list[Item], eff: constraints.Effective) -> set[int]:
    if eff.blocked:
        return set()
    out = set()
    for it in items:
        if not constraints.satisfies(it, eff):
            continue
        if eff.device is not None and compat.code_verdict(it, eff.device) is None:
            continue
        out.add(it.item_id)
    return out


def expected_filters(eff: constraints.Effective) -> dict[str, Any]:
    c = eff.constraints
    return {
        "category": c.category,
        "max_price_cents": c.budget_max_cents,
        "in_stock": True,
        "brands": sorted(_brand_set(c.brands)),
        "exclude_brands": sorted(_brand_set(c.exclude_brands)),
    }


def tool_problems(trace: list[dict], eff: constraints.Effective, user_id: int) -> list[str]:
    """What is wrong with a turn's tool calls; empty when they are exactly the expected ones."""
    problems: list[str] = []
    by_name: dict[str, list[dict]] = {}
    for call in trace:
        by_name.setdefault(call["name"], []).append(call)
    for name, calls in by_name.items():
        if len(calls) > 1:
            problems.append(f"{name} called {len(calls)} times")

    wants_history = eff.constraints.last_purchase != "none"
    hist = by_name.get("get_purchase_history", [])
    if wants_history and not hist:
        problems.append("missing get_purchase_history")
    if hist and not wants_history:
        problems.append("unneeded get_purchase_history")
    if hist and hist[0]["args"].get("user_id") != user_id:
        problems.append("get_purchase_history for the wrong user")

    search = by_name.get("search_items", [])
    if eff.blocked:
        if search:
            problems.append("search_items although nothing can satisfy the request")
    elif not search:
        problems.append("missing search_items")
    else:
        got = dict(search[0]["args"]["filters"])
        got["brands"] = sorted(_brand_set(got.get("brands", [])))
        got["exclude_brands"] = sorted(_brand_set(got.get("exclude_brands", [])))
        want = expected_filters(eff)
        for k in want:
            if got.get(k) != want[k]:
                problems.append(f"search_items filter {k}={got.get(k)!r}, expected {want[k]!r}")

    details = by_name.get("get_item_details", [])
    hits = search[0]["result_ids"] if search else []
    if hits and not details:
        problems.append("missing get_item_details")
    if details and set(details[0]["args"]["ids"]) - set(hits):
        problems.append("get_item_details for items search_items did not return")
    for name in by_name:
        if name not in ("get_purchase_history", "search_items", "get_item_details"):
            problems.append(f"unexpected tool {name}")
    return problems


@dataclass
class CaseScore:
    case_id: str
    template: str
    status: str
    shown: list[int]
    possible: bool
    hit: bool
    success: bool
    hard_violations: int  # shown items breaking price, stock or category
    all_violations: int  # shown items breaking any constraint
    violations: list[str]
    parse_ok: bool
    parse_fields: dict[str, bool]
    tools_ok: bool
    tool_problems: list[str]
    compat_shown: int = 0
    compat_confirmed: int = 0
    compat_confirmed_wrong: int = 0  # 'confirmed' but not a code-confirmed match of the gold device
    latency_ms: float = 0.0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_case(
    case_id: str,
    template: str,
    user_id: int,
    gold: Constraints,
    gold_eff: constraints.Effective,
    satisfiable_ids: set[int],
    rows: dict[int, Item],
    record: dict[str, Any],
) -> CaseScore:
    shown = [s["item_id"] for s in record["shown"]]
    violations: list[str] = []
    hard = every = 0
    for item_id in shown:
        fails = constraints.failures(rows[item_id], gold_eff)
        if gold_eff.blocked:
            fails = [constraints.Check("item", False, "nothing satisfies this request")]
        if fails:
            every += 1
            if any(f.name in constraints.HARD for f in fails):
                hard += 1
            violations += [f"{item_id}: {f.name} ({f.detail})" for f in fails]
    possible = bool(satisfiable_ids)
    hit = possible and any(i in satisfiable_ids for i in shown)
    success = every == 0 and (hit or not possible)

    parsed = record["parse"]["request"]
    if parsed is None:
        fields = dict.fromkeys(FIELDS, False)
    else:
        pc = Constraints.model_validate({k: parsed[k] for k in FIELDS})
        fields = {f: field_matches(f, pc, gold) for f in FIELDS}

    problems = tool_problems(record["trace"], gold_eff, user_id)

    compat_shown = confirmed = wrong = 0
    if gold_eff.device is not None:
        for s in record["shown"]:
            compat_shown += 1
            if s["compat"] and s["compat"]["status"] == "confirmed":
                confirmed += 1
                if compat.code_verdict(rows[s["item_id"]], gold_eff.device) is None:
                    wrong += 1

    calls = record["llm_calls"]
    return CaseScore(
        case_id=case_id,
        template=template,
        status=record["status"],
        shown=shown,
        possible=possible,
        hit=hit,
        success=success,
        hard_violations=hard,
        all_violations=every,
        violations=violations,
        parse_ok=all(fields.values()),
        parse_fields=fields,
        tools_ok=not problems,
        tool_problems=problems,
        compat_shown=compat_shown,
        compat_confirmed=confirmed,
        compat_confirmed_wrong=wrong,
        latency_ms=record["timings_ms"].get("total", 0.0),
        llm_calls=len(calls),
        input_tokens=sum(c["input_tokens"] for c in calls),
        output_tokens=sum(c["output_tokens"] for c in calls),
        cache_write_tokens=sum(c["cache_creation_input_tokens"] for c in calls),
        cache_read_tokens=sum(c["cache_read_input_tokens"] for c in calls),
        cost_usd=sum(c["cost_usd"] for c in calls),
    )
