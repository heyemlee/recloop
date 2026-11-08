"""Device compatibility: code first, then the language model with evidence (design §10.3 (3)).

A device string such as 'Brother HL-2240' or 'HP Deskjet 2540' is reduced to its model number
(`2240`, with the series prefix `hl`; `2540` with the family word `deskjet`). The code confirms
an item when one of its fields (title, description, features) contains

  * the model as written: 'HL-2240', 'HL2240', 'HL 2240', 'MG3620', 'Pro 8710'; or
  * the bare number in a list, when the nearest series word before it (within 25 words) is the
    prefix or a family word: 'HL-2130, 2132, 2220, 2240', 'Deskjet 1000, 1010, ..., 2540', but
    not 'ENVY 4500' for an OfficeJet 4500.

Every confirmation carries the field and the original text around the match. When the code
finds nothing, a model may judge the item, but its 'compatible' only counts if it quotes words
that really are in the field it names and that contain the model number; otherwise the item is
'unconfirmed' and the reply says so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from agent import text
from agent.catalog import Item, ItemField
from agent.schema import CompatJudgement

FIELDS: tuple[ItemField, ...] = ("title", "features", "description")
LIST_WINDOW = 25  # words between a series prefix / family word and a bare model number
SNIPPET_CONTEXT = 60  # characters of context on each side of a match

# Manufacturers whose own supplies are the 'genuine' ones (for oem_only).
DEVICE_BRANDS = frozenset(
    [
        "hp",
        "canon",
        "epson",
        "brother",
        "lexmark",
        "xerox",
        "samsung",
        "dell",
        "ricoh",
        "kyocera",
        "oki",
        "okidata",
        "panasonic",
        "sharp",
        "konica",
        "dymo",
    ]
)
_GENERIC_WORDS = frozenset(
    [
        "printer",
        "printers",
        "all",
        "in",
        "one",
        "series",
        "wireless",
        "color",
        "colour",
        "the",
        "my",
    ]
)
_WORD = re.compile(r"[0-9a-z]+")

Status = Literal["confirmed", "unconfirmed", "incompatible"]


@dataclass(frozen=True)
class Evidence:
    field: ItemField
    snippet: str


@dataclass(frozen=True)
class Verdict:
    status: Status
    source: Literal["code", "model", "none"]
    evidence: tuple[Evidence, ...] = ()
    note: str = ""


UNCONFIRMED = Verdict("unconfirmed", "none", note="the listing does not mention this device")


@dataclass(frozen=True)
class Device:
    raw: str
    brand: str | None  # a known device maker, from the first word
    model: str  # last word with a digit: '2240', 'mg3620', '7860dw'
    prefix: str | None  # short series prefix written before a digit-led model: 'hl', 'wf'
    family: tuple[str, ...]  # other descriptive words: ('deskjet',), ('officejet', 'pro')

    @property
    def keys(self) -> tuple[str, ...]:
        """Single-word forms of the model ('hl2240', '2240'); used to vet quoted evidence."""
        return tuple(k for k in ((self.prefix or "") + self.model, self.model) if k)


def parse_device(raw: str) -> Device | None:
    ws = text.words(raw)
    idx = [i for i, w in enumerate(ws) if any(c.isdigit() for c in w)]
    if not idx:
        return None
    i = idx[-1]
    model = ws[i]
    brand = ws[0] if ws and ws[0] in DEVICE_BRANDS and i > 0 else None
    before = ws[i - 1] if i > 0 else ""
    short_word = before.isalpha() and len(before) <= 4 and before != brand
    prefix = before if model[0].isdigit() and short_word else None
    skip = {brand, prefix} | _GENERIC_WORDS
    family = tuple(w for w in ws[:i] if w.isalpha() and len(w) > 1 and w not in skip)
    return Device(raw, brand, model, prefix, family)


def _spans(s: str) -> list[tuple[int, int]]:
    return [m.span() for m in _WORD.finditer(s.lower())]


def _snippet(s: str, start: int, end: int) -> str:
    a, b = max(0, start - SNIPPET_CONTEXT), min(len(s), end + SNIPPET_CONTEXT)
    while a > 0 and not s[a - 1].isspace():
        a -= 1
    while b < len(s) and not s[b].isspace():
        b += 1
    out = " ".join(s[a:b].split())
    return ("…" if a > 0 else "") + out + ("…" if b < len(s) else "")


def _match_in(s: str, d: Device) -> tuple[int, int] | None:
    """Character span of the first place `s` names device `d`, or None."""
    spans = _spans(s)
    ws = [s[a:b].lower() for a, b in spans]
    m = d.model
    letters = re.match(r"^([a-z]+)(\d.*)$", m)
    for j, w in enumerate(ws):
        # Written as one word: 'HL2240', 'MG3620', 'M252dw'.
        if w == (d.prefix or "") + m and (d.prefix or letters):
            return spans[j]
        # Prefix and number as two words: 'HL-2240', 'Pro 8710'.
        if d.prefix and w == m and j > 0 and ws[j - 1] == d.prefix:
            return spans[j - 1][0], spans[j][1]
        # Letters and number split: 'MG 3620'.
        if letters and w == letters.group(1) and j + 1 < len(ws) and ws[j + 1] == letters.group(2):
            return spans[j][0], spans[j + 1][1]
        # A bare number in a list opened by the prefix or a family word.
        if not letters and w == m:
            if not d.family and not d.prefix:  # the device string was only a number
                return spans[j]
            if _list_opened_by(ws, j, d):
                return spans[j]
    return None


_LIST_FILLER = frozenset(
    [
        "and",
        "or",
        "series",
        "printer",
        "printers",
        "all",
        "in",
        "one",
        "model",
        "models",
        "with",
        "for",
    ]
)


def _list_opened_by(ws: list[str], j: int, d: Device) -> bool:
    """Whether the nearest series word before ws[j] (skipping numbers, brands and filler) is
    the device's prefix or family word: 'Deskjet 1000, 1010, 2540' yes; 'ENVY 4500' is not an
    OfficeJet 4500."""
    openers = set(d.family) | ({d.prefix} if d.prefix else set())
    for x in reversed(ws[max(0, j - LIST_WINDOW) : j]):
        if any(c.isdigit() for c in x):
            if d.prefix and x.startswith(d.prefix) and x[len(d.prefix) :][:1].isdigit():
                return True  # 'HL2130, 2240'
            continue
        if x in DEVICE_BRANDS or x in _LIST_FILLER or len(x) == 1:
            continue
        return x in openers
    return False


def code_verdict(item: Item, device: Device) -> Verdict | None:
    """'confirmed' with evidence when the item's text names the device; None when it does not."""
    for f in FIELDS:
        s = item.field_text(f)
        hit = _match_in(s, device) if s else None
        if hit:
            return Verdict("confirmed", "code", (Evidence(f, _snippet(s, *hit)),))
    return None


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def model_verdict(item: Item, device: Device, judgement: CompatJudgement) -> Verdict:
    """Turn a model's judgement into a verdict, keeping only evidence the code can verify."""
    verified = []
    for ev in judgement.evidence:
        quote = ev.quote.strip()
        if quote and _norm(quote) in _norm(item.field_text(ev.field)):
            verified.append((ev, set(text.words(quote))))
    if judgement.verdict == "compatible":
        for ev, ws in verified:
            if ws.intersection(device.keys):
                return Verdict(
                    "confirmed", "model", (Evidence(ev.field, " ".join(ev.quote.split())),)
                )
        return Verdict(
            "unconfirmed", "model", note="no quote naming this device was found in the listing"
        )
    if judgement.verdict == "incompatible" and verified:
        ev = verified[0][0]
        return Verdict("incompatible", "model", (Evidence(ev.field, " ".join(ev.quote.split())),))
    return UNCONFIRMED
