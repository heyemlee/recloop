"""The reply text, rendered from database rows and code verdicts (design §10.3 (4)).

Titles and prices come from `items` rows fetched in the same turn; the language model only
contributes the one-line reasons (already vetted by `agent.agent`). Compatibility is stated as
the code found it: confirmed with the quoted listing text, or explicitly not confirmed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent import taxonomy
from agent.catalog import Item
from agent.compat import Verdict
from agent.constraints import Effective, dollars

if TYPE_CHECKING:
    from agent.agent import AgentResult

TITLE_CHARS = 140
FIELD_NAMES = {"title": "title", "description": "description", "features": "feature list"}


def clip(s: str, n: int) -> str:
    s = " ".join(s.split())
    if len(s) <= n:
        return s
    cut = s[: n - 1].rsplit(" ", 1)[0]
    return cut + "…"


def price(item: Item) -> str:
    return dollars(item.price_cents) if item.price_cents is not None else "price not listed"


def summary(e: Effective) -> str:
    """'printer ink cartridges for Canon PIXMA MG3620, up to $30.00, not Pilot, in stock'."""
    c = e.constraints
    head = taxonomy.CATEGORIES[c.category].label if c.category else "items"
    if c.compatible_with:
        head += f" for {c.compatible_with}"
    if c.brands:
        head += " from " + " or ".join(c.brands)
    tail = []
    if c.oem_only:
        tail.append("genuine only")
    if c.budget_max_cents is not None:
        tail.append(f"up to {dollars(c.budget_max_cents)}")
    if c.exclude_brands:
        tail.append("not " + ", ".join(c.exclude_brands))
    tail.append("in stock")
    return head + ", " + ", ".join(tail)


def code_reason(item: Item, e: Effective, verdict: Verdict | None) -> str:
    c = e.constraints
    bits = []
    if verdict is not None and verdict.status == "confirmed":
        bits.append(f"Its {FIELD_NAMES[verdict.evidence[0].field]} names the {c.compatible_with}.")
    if c.oem_only:
        # Name the device maker ('Brother'), not the listing's brand field ('Brother Reseller').
        maker = c.compatible_with.split()[0] if e.device and e.device.brand else item.brand
        bits.append(f"A genuine {maker} supply.")
    elif c.brands:
        bits.append(f"{item.brand}, as you asked.")
    if c.exclude_brands:
        bits.append(f"Not from {', '.join(c.exclude_brands)}.")
    if not bits:
        label = taxonomy.CATEGORIES[c.category].label if c.category else "the catalog"
        bits.append(f"A top match among {label}.")
    return " ".join(bits)


def _compat_line(e: Effective, v: Verdict | None) -> str | None:
    if e.device is None or v is None:
        return None
    if v.status == "confirmed":
        ev = v.evidence[0]
        who = "" if v.source == "code" else " (checked by the assistant)"
        where = FIELD_NAMES[ev.field]
        return f"Works with {e.constraints.compatible_with}: confirmed{who}, {where}: “{clip(ev.snippet, 160)}”"
    return f"Works with {e.constraints.compatible_with}: not confirmed ({v.note or 'no evidence'})"


def parse_failed() -> str:
    return (
        "Sorry, I couldn't turn that into a search. Could you tell me the product you need, "
        "and any budget, brand or device model?"
    )


def reply(r: AgentResult) -> str:
    e = r.effective
    assert e is not None
    lines: list[str] = []
    ref = e.reference
    if ref is not None:
        if ref.item is not None:
            lines.append(
                f"Your last purchase of this kind: {clip(ref.item.title, 100)} ({ref.item.brand or 'no brand'})."
            )
        if ref.note:
            lines.append(f"Note: {ref.note}.")
    if r.status == "blocked":
        lines.append(f"I can't show anything for this request: {e.blocked}.")
        return "\n".join(lines)
    if not r.shown:
        lines.append(f"I couldn't find {summary(e)}.")
        lines.append("Try a higher budget, another brand, or fewer requirements.")
        return "\n".join(lines)
    n = len(r.shown)
    confirmed = sum(s.compat is not None and s.compat.status == "confirmed" for s in r.shown)
    if e.device is not None and confirmed == 0:
        lines.append(
            f"No listing I found says it works with the {e.constraints.compatible_with}. The "
            f"closest {'match' if n == 1 else f'{n} matches'} ({summary(e)}); check compatibility before buying:"
        )
    else:
        lines.append(f"{'Here is 1 option' if n == 1 else f'Here are {n} options'}: {summary(e)}.")
    for i, s in enumerate(r.shown, start=1):
        lines.append(f"{i}. {clip(s.item.title, TITLE_CHARS)} — {price(s.item)}")
        compat_line = _compat_line(e, s.compat)
        if compat_line:
            lines.append(f"   {compat_line}")
        lines.append(f"   {s.reason}")
    if r.dropped:
        left = ", ".join(f"{k}: {v}" for k, v in sorted(r.dropped.items()))
        lines.append(f"Left out after checking the listings ({left}).")
    return "\n".join(lines)
