"""Evaluation cases: a shopper message, its gold constraints and, when the message refers to an
earlier purchase, the shopper's purchase history (a fixture, so cases never depend on the
`orders` table)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from agent import paths, taxonomy
from agent.catalog import Purchase
from agent.schema import Constraints, ShoppingRequest

# Purchase dates are relative to a fixed day so a case means the same thing on every run.
REFERENCE_DAY = datetime(2026, 11, 1, 12, tzinfo=UTC)


class CasePurchase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item_id: int
    days_ago: int


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    template: str
    message: str
    user_id: int
    gold: Constraints
    purchases: list[CasePurchase] = Field(default_factory=list)
    note: str = ""

    def history(self, user_id: int, limit: int) -> list[Purchase]:
        """get_purchase_history's source for this case (most recent first)."""
        if user_id != self.user_id:
            return []
        rows = sorted(self.purchases, key=lambda p: (p.days_ago, p.item_id))[:limit]
        return [Purchase(p.item_id, REFERENCE_DAY - timedelta(days=p.days_ago)) for p in rows]

    def oracle_request(self) -> ShoppingRequest:
        """The gold constraints with a keyword query built from them (the 'oracle' agent)."""
        g = self.gold
        words = [*g.brands]
        if g.category:
            words.append(taxonomy.CATEGORIES[g.category].label)
        if g.compatible_with:
            words.append(g.compatible_with)
        query = " ".join(words) or self.message
        return ShoppingRequest(**g.model_dump(), search_text=query)


def load(path: Path = paths.CASES) -> list[Case]:
    with path.open() as f:
        return [Case.model_validate_json(line) for line in f if line.strip()]


def dump(cases: list[Case], path: Path = paths.CASES) -> None:
    lines = [json.dumps(c.model_dump(mode="json"), ensure_ascii=False) for c in cases]
    path.write_text("\n".join(lines) + "\n")
