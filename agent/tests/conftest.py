"""A small synthetic catalog and a scripted language model; no database, no API calls."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from agent.catalog import Item, MemoryCatalog, Purchase
from agent.encoder import HashEncoder
from agent.llm import LLMCall, LLMError, LLMReply

INK = ("Office Products", "Office & School Supplies", "Printer Ink & Toner", "Inkjet Printer Ink")
TONER = (
    "Office Products",
    "Office & School Supplies",
    "Printer Ink & Toner",
    "Laser Printer Drums & Toner",
)
GEL = (
    "Office Products",
    "Office & School Supplies",
    "Writing & Correction Supplies",
    "Pens & Refills",
    "Rollerball Pens",
    "Gel Ink Rollerball Pens",
)
NOTEBOOK = ("Office Products", "Office & School Supplies", "Paper", "Notebooks & Writing Pads")
STICKY = (*NOTEBOOK, "Self-Stick Notes")


def item(item_id, title, brand, cats, price, stock=10, desc=None, feats=()):
    return Item(
        item_id,
        f"A{item_id:09d}",
        title,
        brand,
        tuple(cats),
        price,
        desc,
        tuple(feats),
        None,
        stock,
    )


ITEMS = [
    item(1, "Canon PG-240 Black Ink Cartridge, Compatible to MG3620, MG3520", "Canon", INK, 1899),
    item(2, "LD Remanufactured Ink Cartridge Replacement for Canon PG-240", "LD Products", INK, 999,
         feats=["For use in PIXMA MG3620, MG3520, MG2220"]),
    item(3, "Canon CL-241 Color Ink Cartridge", "Canon", INK, None, desc="Compatible to: MG3620, MG3520"),
    item(4, "HP 61 Black Ink Cartridge", "HP", INK, 2499, stock=0,
         feats=["HP 61 ink cartridges work with: HP Deskjet 1000, 1010, 1050, 2540, 2541"]),
    item(5, "HP 61 Tri-color Ink Cartridge", "HP", INK, 2799,
         feats=["HP 61 ink cartridges work with: HP Deskjet 1000, 1010, 1050, 2540, 2541"]),
    item(6, "HP 62 Black Ink Cartridge", "HP", INK, 2299,
         feats=["HP 62 ink cartridges work with: HP ENVY 4500, 5540, HP Officejet 200"]),
    item(7, "Pilot G2 Retractable Gel Pens, Fine Point, 12-Pack", "Pilot", GEL, 899),
    item(8, "uni-ball Signo 207 Gel Pens, Blue, 4-Pack", "Uni-ball", GEL, 1099),
    item(9, "Pentel EnerGel RTX Gel Pen, Black", "Pentel", GEL, 1299),
    item(10, "Pilot G2 Limited Gel Pen, Gold Barrel", "Pilot Corporation", GEL, 1599),
    item(11, "Mead Spiral Notebook, College Ruled, 70 Sheets", "Mead", NOTEBOOK, 499),
    item(12, "Post-it Super Sticky Notes, 3 x 3 in", "Post-it", STICKY, 599),
    item(13, "Brother Genuine High Yield Toner Cartridge, TN450", "Brother", TONER, 5399,
         feats=["Compatible with HL-2130, 2132, 2220, 2230, 2240, 2270DW and MFC-7860DW"]),
    item(14, "LINKYO Compatible Toner Cartridge Replacement for Brother TN450", "LINKYO", TONER, 2495,
         feats=["Suitable for use in Brother HL-2240, HL-2270DW"]),
    item(15, "Mystery desk thing", None, (), 799),
    item(16, "Brother Reseller TN420 Toner", "Brother Reseller", TONER, 4599,
         feats=["Works with HL-2240 printers"]),
]  # fmt: skip

USER = 7


@pytest.fixture
def items() -> list[Item]:
    return list(ITEMS)


@pytest.fixture
def catalog(items) -> MemoryCatalog:
    enc = HashEncoder()
    emb = np.stack([enc.encode(f"{it.title}. {'; '.join(it.features)}") for it in items])
    orders = {
        USER: [
            Purchase(7, datetime(2026, 10, 1, tzinfo=UTC)),  # Pilot gel pens
            Purchase(11, datetime(2026, 10, 20, tzinfo=UTC)),  # a notebook, more recent
        ]
    }
    return MemoryCatalog(items, emb, enc, orders)


class FakeLLM:
    """Replies are scripted per purpose ('parse', 'compat', 'explain'); an Exception entry is
    raised as an LLMError (refusal, truncation)."""

    model = "fake-model"

    def __init__(self, **replies: list):
        self.replies = {k: list(v) for k, v in replies.items()}
        self.calls: list[tuple[str, str, str, dict]] = []

    def complete_json(self, purpose: str, system: str, user: str, schema: dict) -> LLMReply:
        self.calls.append((purpose, system, user, schema))
        call = LLMCall(purpose, self.model, input_tokens=100, output_tokens=10, cost_usd=0.001)
        queue = self.replies.get(purpose)
        if not queue:
            raise AssertionError(f"unexpected {purpose} call")
        r = queue.pop(0)
        if isinstance(r, Exception):
            raise LLMError(str(r), call)
        return LLMReply(r, call)

    def purposes(self) -> list[str]:
        return [c[0] for c in self.calls]


@pytest.fixture
def fake_llm():
    return FakeLLM
