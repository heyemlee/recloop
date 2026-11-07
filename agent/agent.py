"""One turn of the shopping assistant: parse -> tools -> code checks -> compatibility -> reply.

    parse        Parser (language model or rules) -> ShoppingRequest
    history      get_purchase_history, only if the request refers to an earlier purchase
    search       search_items(search_text, filters from the constraints)
    details      get_item_details(hit ids): current rows, the only source of titles and prices
    check        every hard and structured constraint on every row (`agent.constraints`)
    compat       code match first; the model judges the rest only when the code confirmed fewer
                 items than the reply shows, and its verdicts need verified quotes
    reasons      the model's one-liners, replaced by code-written ones when they mention a
                 price or claim an unconfirmed compatibility
    reply        rendered from the rows (`agent.render`)

The model never picks or reorders items: order is search rank, confirmed-compatible first.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import ValidationError

from agent import compat, constraints, prompts, render, taxonomy
from agent.catalog import Catalog, Item, SearchFilters, SearchHit
from agent.llm import LLM, LLMCall, LLMError
from agent.parsing import ParseOutcome, Parser
from agent.schema import CompatJudgements, ItemReasons
from agent.tools import PurchaseSource, ToolCall, Tools

Status = Literal["ok", "no_results", "parse_failed", "blocked"]
JUDGE_TEXT_CHARS = 1500
_PRICE_TALK = re.compile(
    r"\$|€|£|\b\d+(?:\.\d+)?\s*(?:dollars?|usd|bucks|cents?)\b|\bfree shipping\b", re.IGNORECASE
)


@dataclass
class AgentConfig:
    max_results: int = 3
    search_limit: int = 30
    judge_limit: int = 8
    search_mode: str = "hybrid"


@dataclass
class Shown:
    item: Item
    rank: int  # 1-based position in the search results
    compat: compat.Verdict | None
    reason: str
    reason_source: Literal["model", "code"]
    reason_note: str = ""  # why a model-written reason was replaced


@dataclass
class AgentResult:
    message: str
    user_id: int
    status: Status
    reply: str
    parse: ParseOutcome
    effective: constraints.Effective | None = None
    hits: list[SearchHit] = field(default_factory=list)
    shown: list[Shown] = field(default_factory=list)
    dropped: dict[str, int] = field(default_factory=dict)
    trace: list[ToolCall] = field(default_factory=list)
    llm_calls: list[LLMCall] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        """JSON-serialisable summary (the evaluation stores one per case)."""
        e = self.effective
        return {
            "status": self.status,
            "reply": self.reply,
            "parse": {
                "source": self.parse.source,
                "attempts": self.parse.attempts,
                "errors": self.parse.errors,
                "request": self.parse.request.model_dump() if self.parse.request else None,
            },
            "effective": e.constraints.model_dump() if e else None,
            "reference_item": e.reference.item.item_id
            if e and e.reference and e.reference.item
            else None,
            "blocked": e.blocked if e else None,
            "shown": [
                {
                    "item_id": s.item.item_id,
                    "rank": s.rank,
                    "compat": None
                    if s.compat is None
                    else {
                        "status": s.compat.status,
                        "source": s.compat.source,
                        "evidence": [
                            {"field": ev.field, "snippet": ev.snippet} for ev in s.compat.evidence
                        ],
                    },
                    "reason": s.reason,
                    "reason_source": s.reason_source,
                    "reason_note": s.reason_note,
                }
                for s in self.shown
            ],
            "dropped": self.dropped,
            "trace": [
                {"name": t.name, "args": t.args, "result_ids": t.result_ids, "ms": round(t.ms, 3)}
                for t in self.trace
            ],
            "llm_calls": [c.as_dict() for c in self.llm_calls],
            "timings_ms": {k: round(v, 3) for k, v in self.timings_ms.items()},
        }


class ShoppingAgent:
    def __init__(
        self,
        catalog: Catalog,
        parser: Parser,
        llm: LLM | None = None,
        config: AgentConfig | None = None,
        purchases: PurchaseSource | None = None,
    ):
        self.catalog = catalog
        self.parser = parser
        self.llm = llm
        self.config = config or AgentConfig()
        self.purchases = purchases

    def run(self, message: str, user_id: int) -> AgentResult:
        cfg = self.config
        tools = Tools(self.catalog, self.purchases, cfg.search_mode)
        timings: dict[str, float] = {}
        t_start = t = time.perf_counter()

        def lap(name: str) -> None:
            nonlocal t
            now = time.perf_counter()
            timings[name] = (now - t) * 1000
            t = now

        parse = self.parser.parse(message)
        lap("parse")
        result = AgentResult(message, user_id, "ok", "", parse, trace=tools.trace)
        result.llm_calls.extend(parse.llm_calls)
        result.timings_ms = timings
        if parse.request is None:
            result.status = "parse_failed"
            result.reply = render.parse_failed()
            timings["total"] = (time.perf_counter() - t_start) * 1000
            return result

        request = parse.request
        wanted = request.constraints()
        history = tools.get_purchase_history(user_id) if wanted.last_purchase != "none" else []
        lap("history")
        eff = constraints.resolve(wanted, history)
        result.effective = eff
        if eff.blocked:
            result.status = "blocked"
            result.reply = render.reply(result)
            timings["total"] = (time.perf_counter() - t_start) * 1000
            return result

        c = eff.constraints
        filters = SearchFilters(
            category=c.category,
            max_price_cents=c.budget_max_cents,
            in_stock=True,
            brands=c.brands,
            exclude_brands=c.exclude_brands,
        )
        result.hits = tools.search_items(request.search_text, filters, cfg.search_limit)
        lap("search")
        items = tools.get_item_details([h.item_id for h in result.hits]) if result.hits else []
        lap("details")

        rank = {h.item_id: r for r, h in enumerate(result.hits, start=1)}
        kept: list[Item] = []
        dropped: Counter[str] = Counter()
        for item in items:
            failed = constraints.failures(item, eff)
            if failed:
                dropped[failed[0].name] += 1
            else:
                kept.append(item)
        result.dropped = dict(dropped)
        lap("check")

        verdicts: dict[int, compat.Verdict] = {}
        if eff.device is not None:
            found = {it.item_id: compat.code_verdict(it, eff.device) for it in kept}
            confirmed = sum(v is not None for v in found.values())
            if confirmed < cfg.max_results and self.llm is not None:
                unresolved = [it for it in kept if found[it.item_id] is None][: cfg.judge_limit]
                if unresolved:
                    found.update(self._judge(eff.device, unresolved, result.llm_calls))
            verdicts = {i: v or compat.UNCONFIRMED for i, v in found.items()}
            kept = [it for it in kept if verdicts[it.item_id].status != "incompatible"]
            kept.sort(key=lambda it: (verdicts[it.item_id].status != "confirmed", rank[it.item_id]))
        lap("compat")

        selected = kept[: cfg.max_results]
        reasons = self._reasons(message, eff, selected, verdicts, result.llm_calls)
        result.shown = [
            Shown(it, rank[it.item_id], verdicts.get(it.item_id), *reasons[it.item_id])
            for it in selected
        ]
        lap("reasons")
        result.status = "ok" if result.shown else "no_results"
        result.reply = render.reply(result)
        lap("render")
        timings["total"] = (time.perf_counter() - t_start) * 1000
        return result

    # --- language-model steps ---------------------------------------------------------------

    def _judge(
        self, device: compat.Device, items: list[Item], calls: list[LLMCall]
    ) -> dict[int, compat.Verdict]:
        assert self.llm is not None
        payload = [
            {
                "item_id": it.item_id,
                "title": it.title,
                "features": it.field_text("features")[:JUDGE_TEXT_CHARS],
                "description": it.field_text("description")[:JUDGE_TEXT_CHARS],
            }
            for it in items
        ]
        try:
            reply = self.llm.complete_json(
                "compat",
                prompts.JUDGE_SYSTEM,
                prompts.judge_user(device.raw, payload),
                CompatJudgements.model_json_schema(),
            )
        except LLMError as e:
            calls.append(e.call)
            return {}
        calls.append(reply.call)
        try:
            judged = CompatJudgements.model_validate_json(reply.text)
        except ValidationError:
            return {}
        by_id = {it.item_id: it for it in items}
        return {
            j.item_id: compat.model_verdict(by_id[j.item_id], device, j)
            for j in judged.items
            if j.item_id in by_id
        }

    def _reasons(
        self,
        message: str,
        eff: constraints.Effective,
        items: list[Item],
        verdicts: dict[int, compat.Verdict],
        calls: list[LLMCall],
    ) -> dict[int, tuple[str, Literal["model", "code"], str]]:
        out = {
            it.item_id: (render.code_reason(it, eff, verdicts.get(it.item_id)), "code", "")
            for it in items
        }
        if self.llm is None or not items:
            return out  # type: ignore[return-value]
        payload = [_facts(it, eff, verdicts.get(it.item_id)) for it in items]
        try:
            reply = self.llm.complete_json(
                "explain",
                prompts.EXPLAIN_SYSTEM,
                prompts.explain_user(message, payload),
                ItemReasons.model_json_schema(),
            )
        except LLMError as e:
            calls.append(e.call)
            return out  # type: ignore[return-value]
        calls.append(reply.call)
        try:
            written = ItemReasons.model_validate_json(reply.text)
        except ValidationError:
            return out  # type: ignore[return-value]
        for r in written.reasons:
            if r.item_id not in out:
                continue
            reason = " ".join(r.reason.split())
            problem = _reason_problem(reason, eff, verdicts.get(r.item_id))
            if problem:
                code = out[r.item_id][0]
                out[r.item_id] = (code, "code", problem)
            elif reason:
                out[r.item_id] = (render.clip(reason, 200), "model", "")
        return out  # type: ignore[return-value]


def _facts(item: Item, eff: constraints.Effective, verdict: compat.Verdict | None) -> dict:
    c = eff.constraints
    facts: dict[str, Any] = {
        "item_id": item.item_id,
        "title": item.title,
        "brand": item.brand,
        "category": taxonomy.CATEGORIES[c.category].label if c.category else None,
        "features": [render.clip(f, 160) for f in item.features[:3]],
    }
    if eff.device is not None:
        if verdict is not None and verdict.status == "confirmed":
            facts["compatibility"] = f"confirmed: {verdict.evidence[0].snippet}"
        else:
            facts["compatibility"] = "not confirmed by the listing"
    met = []
    if c.brands:
        met.append(f"brand {item.brand} as requested")
    if c.exclude_brands:
        met.append("not from " + ", ".join(c.exclude_brands))
    if c.oem_only:
        met.append("genuine manufacturer supply")
    if c.budget_max_cents is not None:
        met.append("within budget")
    facts["meets"] = met
    return facts


def _reason_problem(reason: str, eff: constraints.Effective, verdict: compat.Verdict | None) -> str:
    if _PRICE_TALK.search(reason):
        return "mentions a price"
    d = eff.device
    if d is not None and (verdict is None or verdict.status != "confirmed"):
        words = set(re.findall(r"[0-9a-z]+", reason.lower()))
        if words.intersection(d.keys):
            return "claims a compatibility the listing does not confirm"
    return ""
