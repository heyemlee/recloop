"""The store's shopping assistant behind POST /v1/chat (design §10, §13.1).

`ChatService.turn(user_id, message, emit)` runs one turn of the agent with the store's tools: the
PostgreSQL catalog (read-only), the purchases the feed counts as bought, the chat surface's
recommendations and the ads module. It emits the events the endpoint streams:

  meta     {reply_id, parser}
  status   {stage} as each stage starts: parse, search, check, personalize, reasons, ads
  reply    the reply: its text, and the same reply in parts (intro, item cards, the sponsored
           card, footer), the constraints it was checked against, degradations
  done     {timings_ms, llm: {calls, cost_usd}}

Every card carries the impression_id its impression and click events must use: "{reply_id}:
{position}" for the organic items, the ads module's for the sponsored one, together with its
ad_token (billing and the frequency caps need both). The sponsored card is labelled here, by
code; the reply's text and prices come from the item rows (`agent.render`).
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

import numpy as np

from agent import paths, render
from agent.agent import AgentConfig, AgentResult, ShoppingAgent, Shown, Sponsored
from agent.catalog import Catalog, Item
from agent.compat import Verdict
from agent.llm import LLM
from agent.parsing import Parser
from agent.store_api import DEFAULT_API_URL, AdSelector, Recommender
from agent.tools import PurchaseSource

Emit = Callable[[str, dict[str, Any]], None]


def uuid7() -> uuid.UUID:
    """A UUIDv7 (time-ordered, like the api's request ids): 48-bit Unix ms, then randomness."""
    ms = time.time_ns() // 1_000_000
    rand = int.from_bytes(os.urandom(10), "big")
    value = (ms & ((1 << 48) - 1)) << 80 | 0x7 << 76 | (rand >> 68) << 64 | 0b10 << 62
    return uuid.UUID(int=value | (rand & ((1 << 62) - 1)))


class LockedEncoder:
    """One query at a time through the sentence encoder (its tokenizer is not thread-safe)."""

    def __init__(self, inner):
        self.inner = inner
        self.name = getattr(inner, "name", "encoder")
        self._lock = threading.Lock()

    def encode(self, query: str) -> np.ndarray:
        with self._lock:
            return self.inner.encode(query)


def _compat(v: Verdict | None, line: str | None) -> dict[str, Any] | None:
    if v is None:
        return None
    return {
        "status": v.status,
        "source": v.source,
        "line": line,
        "evidence": [{"field": e.field, "snippet": e.snippet} for e in v.evidence],
        "note": v.note,
    }


def _card(item: Item) -> dict[str, Any]:
    return {
        "item_id": item.item_id,
        "title": item.title,
        "brand": item.brand,
        "price_cents": item.price_cents,
        "image_url": item.image_url,
        "stock": item.stock,
    }


def organic_card(s: Shown, position: int, reply_id: uuid.UUID, line: str | None) -> dict:
    return {
        "position": position,
        "slot": "organic",
        "impression_id": f"{reply_id}:{position}",
        **_card(s.item),
        "reason": s.reason,
        "reason_source": s.reason_source,
        "search_rank": s.rank,
        "personal": None
        if s.personal is None
        else {"rank": s.personal.rank, "source": s.personal.source},
        "compat": _compat(s.compat, line),
    }


def sponsored_card(sp: Sponsored, position: int, line: str | None) -> dict:
    return {
        "position": position,
        "slot": "sponsored",
        "label": render.SPONSORED_LABEL,
        "impression_id": sp.slot.impression_id,
        "ad_token": sp.slot.ad_token,
        "campaign_id": sp.slot.campaign_id,
        "request_id": sp.slot.request_id,
        **_card(sp.item),
        "reason": sp.reason,
        "compat": _compat(sp.compat, line),
    }


def reply_payload(r: AgentResult, reply_id: uuid.UUID) -> dict[str, Any]:
    p = render.parts(r)
    e = r.effective
    items = [
        organic_card(s, pos, reply_id, line)
        for pos, (s, line) in enumerate(zip(r.shown, p.compat, strict=False), start=1)
    ]
    sponsored = None
    if r.sponsored is not None and r.shown:
        sponsored = sponsored_card(r.sponsored, len(items) + 1, p.sponsored_compat)
    ref = e.reference if e else None
    return {
        "reply_id": str(reply_id),
        "user_id": r.user_id,
        "status": r.status,
        "text": r.reply,
        "intro": p.intro,
        "items": items,
        "sponsored": sponsored,
        "footer": p.footer,
        "summary": render.summary(e) if e and not e.blocked else None,
        "constraints": e.constraints.model_dump() if e else None,
        "reference_item_id": ref.item.item_id if ref and ref.item else None,
        "parse": {"source": r.parse.source, "attempts": r.parse.attempts},
        "personal": None
        if r.personal is None
        else {
            "request_id": r.personal.request_id,
            "seeds": r.personal.seeds,
            "hits": r.personal.hits,
            "moved": r.personal.moved,
        },
        "ads": r.ads,
        "degraded": r.degraded,
        "tools": [t.name for t in r.trace],
    }


class ChatService:
    def __init__(
        self,
        catalog: Catalog,
        parser: Parser,
        llm: LLM | None = None,
        purchases: PurchaseSource | None = None,
        recommender: Recommender | None = None,
        ads: AdSelector | None = None,
        config: AgentConfig | None = None,
    ):
        self.catalog = catalog
        self.parser_name = getattr(parser, "name", "rules") if llm is None else llm.model
        self.agent = ShoppingAgent(catalog, parser, llm, config, purchases, recommender, ads)

    @classmethod
    def from_env(cls) -> ChatService:
        """The store's service: DATABASE_URL, RECLOOP_API_URL, and RECLOOP_AGENT_MODEL (empty:
        the rules parser and no language model; a model id: Claude, which needs
        ANTHROPIC_API_KEY and is billed per reply)."""
        from agent.catalog import PgCatalog
        from agent.encoder import BgeQueryEncoder
        from agent.history import DatasetPurchases, StorePurchases
        from agent.parsing import LLMParser, RuleParser
        from agent.store_api import StoreAPI

        encoder = LockedEncoder(BgeQueryEncoder())
        catalog = PgCatalog.pooled(
            paths.database_url(), encoder, size=int(os.environ.get("AGENT_DB_POOL", "4"))
        )
        model = os.environ.get("RECLOOP_AGENT_MODEL", "").strip()
        llm: LLM | None = None
        if model:
            from agent.llm import ClaudeLLM

            llm = ClaudeLLM(model)
            parser: Parser = LLMParser(llm)
        else:
            parser = RuleParser.from_catalog(it.brand for it in catalog.all_items())
        api = StoreAPI(os.environ.get("RECLOOP_API_URL", DEFAULT_API_URL))
        purchases = StorePurchases(catalog, DatasetPurchases.load())
        return cls(catalog, parser, llm, purchases, api, api)

    def close(self) -> None:
        close = getattr(self.catalog, "close", None)
        if close is not None:
            close()

    def turn(self, user_id: int, message: str, emit: Emit) -> AgentResult:
        t0 = time.perf_counter()
        reply_id = uuid7()
        emit("meta", {"reply_id": str(reply_id), "parser": self.parser_name})
        result = self.agent.run(message, user_id, on_stage=lambda s: emit("status", {"stage": s}))
        emit("reply", reply_payload(result, reply_id))
        calls = result.llm_calls
        emit(
            "done",
            {
                "timings_ms": {k: round(v, 3) for k, v in result.timings_ms.items()},
                "server_ms": round((time.perf_counter() - t0) * 1000, 3),
                "llm": {
                    "calls": len(calls),
                    "cost_usd": round(sum(c.cost_usd for c in calls), 6),
                    "input_tokens": sum(c.input_tokens for c in calls),
                    "output_tokens": sum(c.output_tokens for c in calls),
                },
            },
        )
        return result
