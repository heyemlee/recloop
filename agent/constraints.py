"""Constraint checks run by code on database rows (design §10.3 (2)).

Hard constraints (the headline metric): price, stock, category. An unknown price never meets a
budget, an item without a category path never meets a category. The structured ones are checked
the same way: brands, excluded brands, genuine-only, and the item the shopper complained about.

`resolve` turns a reference to the shopper's last purchase ('a different brand than last time',
'cheaper than last time') into plain constraints, from the purchase history tool's rows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from agent import compat, taxonomy
from agent.catalog import Item, Purchase, brand_allowed, brand_matches
from agent.schema import Constraints

CheckName = Literal["price", "stock", "category", "brands", "exclude_brands", "oem", "item"]
HARD: tuple[CheckName, ...] = ("price", "stock", "category")

# Third-party markers in a title or feature list. Descriptions are not searched: genuine HP
# listings say 'up to 2x more prints with Original HP ink vs refill cartridges'.
_THIRD_PARTY = re.compile(
    r"\b(remanufactured|refurbished|refilled|re-?filled|compatible (ink|toner|cartridge)s?|"
    r"replacement (for|compatible)|non-?oem|third-?party|generic)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Check:
    name: CheckName
    ok: bool
    detail: str = ""


@dataclass(frozen=True)
class Reference:
    """The earlier purchase a request referred to."""

    relation: str
    item: Item | None
    note: str = ""


@dataclass(frozen=True)
class Effective:
    """Constraints after resolving the last-purchase reference."""

    constraints: Constraints
    exclude_item_ids: frozenset[int] = frozenset()
    reference: Reference | None = None
    blocked: str | None = None  # set when no item can satisfy the request (e.g. unknown price)
    device: compat.Device | None = field(default=None, compare=False)


def dollars(cents: int) -> str:
    return f"${cents // 100:,}.{cents % 100:02d}"


def resolve(c: Constraints, history: list[tuple[Purchase, Item]]) -> Effective:
    """Apply `c.last_purchase` using `history` (most recent first)."""
    device = compat.parse_device(c.compatible_with) if c.compatible_with else None
    if c.last_purchase == "none":
        return Effective(c, device=device)
    same_kind = [
        it
        for _, it in history
        if c.category is None or taxonomy.in_category(it.categories, c.category)
    ]
    if not same_kind:
        ref = Reference(c.last_purchase, None, "no earlier purchase of this kind was found")
        return Effective(c, reference=ref, device=device)
    item = same_kind[0]
    ref = Reference(c.last_purchase, item)
    update: dict = {}
    exclude_ids: frozenset[int] = frozenset()
    blocked = None
    if c.last_purchase == "same_brand":
        if item.brand:
            update["brands"] = [item.brand]
        else:
            ref = Reference(c.last_purchase, item, "the earlier purchase has no brand on record")
    elif c.last_purchase == "different_brand":
        exclude_ids = frozenset({item.item_id})
        if item.brand and not any(brand_matches(item.brand, b) for b in c.exclude_brands):
            update["exclude_brands"] = [*c.exclude_brands, item.brand]
    elif c.last_purchase == "cheaper":
        exclude_ids = frozenset({item.item_id})
        if item.price_cents is None:
            blocked = (
                "the price of the earlier purchase is unknown, so nothing can be shown as cheaper"
            )
        else:
            cap = item.price_cents - 1
            update["budget_max_cents"] = min(cap, c.budget_max_cents or cap)
            if update["budget_max_cents"] < 1:
                blocked = "nothing can be cheaper than the earlier purchase"
                update.pop("budget_max_cents")
    resolved = c.model_copy(update=update)
    return Effective(resolved, exclude_ids, ref, blocked, device)


def _is_genuine(item: Item, device: compat.Device | None) -> tuple[bool, str]:
    marked = _THIRD_PARTY.search(item.title) or _THIRD_PARTY.search(" ".join(item.features))
    if marked:
        return False, f"listed as '{marked.group(0)}'"
    if device and device.brand:
        if brand_matches(item.brand, device.brand):
            return True, ""
        return False, f"made by {item.brand or 'an unknown brand'}, not {device.brand.upper()}"
    if item.brand and any(brand_matches(item.brand, b) for b in compat.DEVICE_BRANDS):
        return True, ""
    return False, "not from a device manufacturer"


def check(item: Item, e: Effective) -> list[Check]:
    c = e.constraints
    out: list[Check] = [Check("stock", item.stock > 0, "" if item.stock > 0 else "sold out")]
    if c.budget_max_cents is not None:
        if item.price_cents is None:
            out.append(Check("price", False, "price unknown"))
        else:
            ok = item.price_cents <= c.budget_max_cents
            detail = "" if ok else f"{dollars(item.price_cents)} > {dollars(c.budget_max_cents)}"
            out.append(Check("price", ok, detail))
    if c.category is not None:
        ok = taxonomy.in_category(item.categories, c.category)
        out.append(Check("category", ok, "" if ok else f"not in {c.category}"))
    if c.brands:
        ok = brand_allowed(item.brand, c.brands, ())
        out.append(Check("brands", ok, "" if ok else f"brand {item.brand!r}"))
    if c.exclude_brands:
        ok = brand_allowed(item.brand, (), c.exclude_brands)
        out.append(Check("exclude_brands", ok, "" if ok else f"brand {item.brand!r} excluded"))
    if c.oem_only:
        ok, detail = _is_genuine(item, e.device)
        out.append(Check("oem", ok, detail))
    if e.exclude_item_ids:
        ok = item.item_id not in e.exclude_item_ids
        out.append(Check("item", ok, "" if ok else "the item from the earlier purchase"))
    return out


def failures(item: Item, e: Effective) -> list[Check]:
    return [c for c in check(item, e) if not c.ok]


def satisfies(item: Item, e: Effective, hard_only: bool = False) -> bool:
    if e.blocked:
        return False
    return all(c.ok for c in check(item, e) if not hard_only or c.name in HARD)
