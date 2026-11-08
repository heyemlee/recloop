"""The reply, rendered from database rows and code verdicts (design §10.3 (4), (5)).

Titles and prices come from `items` rows fetched in the same turn; the language model only
contributes the one-line reasons (already vetted by `agent.agent`). Compatibility is stated as
the code found it: confirmed with the quoted listing text, or explicitly not confirmed. The
sponsored item, if the ads module picked one, gets its label and its reason here, from code.

`parts` is the reply in pieces (the chat page shows the items as cards); `reply` is the same
reply as text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agent import taxonomy
from agent.catalog import Item
from agent.compat import Verdict
from agent.constraints import Effective, dollars

if TYPE_CHECKING:
    from agent.agent import AgentResult

TITLE_CHARS = 140
FIELD_NAMES = {"title": "title", "description": "description", "features": "feature list"}
# Added by this code to the ads module's item; the model never sees or writes it.
SPONSORED_LABEL = "Sponsored"
SPONSORED_REASON = "Placed by an advertiser; it passed the same checks as the options above."


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


@dataclass
class ReplyParts:
    """The lines around the items, and each item's compatibility line (None: no device asked)."""

    intro: list[str]
    compat: list[str | None] = field(default_factory=list)  # per shown item, in order
    sponsored_compat: str | None = None
    footer: list[str] = field(default_factory=list)


def parts(r: AgentResult) -> ReplyParts:
    if r.status == "parse_failed":
        return ReplyParts([parse_failed()])
    e = r.effective
    assert e is not None
    intro: list[str] = []
    ref = e.reference
    if ref is not None:
        if ref.item is not None:
            intro.append(
                f"Your last purchase of this kind: {clip(ref.item.title, 100)} ({ref.item.brand or 'no brand'})."
            )
        if ref.note:
            intro.append(f"Note: {ref.note}.")
    if r.status == "blocked":
        intro.append(f"I can't show anything for this request: {e.blocked}.")
        return ReplyParts(intro)
    if not r.shown:
        intro.append(f"I couldn't find {summary(e)}.")
        intro.append("Try a higher budget, another brand, or fewer requirements.")
        return ReplyParts(intro)
    n = len(r.shown)
    confirmed = sum(s.compat is not None and s.compat.status == "confirmed" for s in r.shown)
    if e.device is not None and confirmed == 0:
        intro.append(
            f"No listing I found says it works with the {e.constraints.compatible_with}. The "
            f"closest {'match' if n == 1 else f'{n} matches'} ({summary(e)}); check compatibility before buying:"
        )
    else:
        intro.append(f"{'Here is 1 option' if n == 1 else f'Here are {n} options'}: {summary(e)}.")
    out = ReplyParts(intro, [_compat_line(e, s.compat) for s in r.shown])
    if r.sponsored is not None:
        out.sponsored_compat = _compat_line(e, r.sponsored.compat)
    if r.dropped:
        left = ", ".join(f"{k}: {v}" for k, v in sorted(r.dropped.items()))
        out.footer.append(f"Left out after checking the listings ({left}).")
    return out


def reply(r: AgentResult) -> str:
    p = parts(r)
    lines = list(p.intro)
    for i, (s, compat_line) in enumerate(zip(r.shown, p.compat, strict=False), start=1):
        lines.append(f"{i}. {clip(s.item.title, TITLE_CHARS)} — {price(s.item)}")
        if compat_line:
            lines.append(f"   {compat_line}")
        lines.append(f"   {s.reason}")
    sp = r.sponsored
    if sp is not None and r.shown:
        lines.append(f"{SPONSORED_LABEL}: {clip(sp.item.title, TITLE_CHARS)} — {price(sp.item)}")
        if p.sponsored_compat:
            lines.append(f"   {p.sponsored_compat}")
        lines.append(f"   {sp.reason}")
    lines.extend(p.footer)
    return "\n".join(lines)
