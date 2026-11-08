"""Generate agent/eval/cases.jsonl: 150 shopper messages from templates and catalog attributes.

Design §10.4: templates filled with budgets, brands, compatible devices and complaints about an
earlier purchase, each with its gold constraints. Budgets come from the catalog's price
distribution, brands from the categories' brand counts, devices from a hand-checked list of
printers the ink and toner listings name, and purchase histories from items of the category.
Hand-written cases cover paraphrases and requests nothing can satisfy.

The output is deterministic (fixed seed, sorted inputs) and committed; re-running it on the same
catalog gives the same file. Stock is ignored here: what is in stock is decided at evaluation
time against the live table.

Usage: python -m agent.eval.build_cases [--review]   (reads `items`, read-only)
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from collections.abc import Callable

from agent import compat, constraints, paths, taxonomy
from agent.catalog import Item, PgCatalog, norm_brand
from agent.encoder import HashEncoder
from agent.eval.cases import Case, CasePurchase, dump
from agent.schema import Constraints

SEED = 20261106
FIRST_USER_ID = 900_000  # not a dataset user: histories come from the case, not from `orders`

NICE_DOLLARS = [5, 8, 10, 12, 15, 20, 25, 30, 35, 40, 50, 60, 75, 100, 120, 150, 200, 250, 300]
NUMBER_WORDS = {
    5: "five", 8: "eight", 10: "ten", 12: "twelve", 15: "fifteen", 20: "twenty",
    25: "twenty-five", 30: "thirty", 35: "thirty-five", 40: "forty", 50: "fifty", 60: "sixty",
    75: "seventy-five", 100: "a hundred", 120: "a hundred and twenty", 150: "a hundred and fifty",
    200: "two hundred", 250: "two hundred and fifty", 300: "three hundred",
}  # fmt: skip

# How shoppers name the products of a category (plural noun phrases).
PHRASES = {
    "ink_cartridges": "ink cartridges", "toner": "toner cartridges", "gel_pens": "gel pens",
    "ballpoint_pens": "ballpoint pens", "fountain_pens": "fountain pens",
    "mechanical_pencils": "mechanical pencils", "pencils": "pencils", "notebooks": "notebooks",
    "sticky_notes": "sticky notes", "highlighters": "highlighters",
    "permanent_markers": "permanent markers", "dry_erase_markers": "dry erase markers",
    "staplers": "staplers", "binders": "binders", "folders": "file folders", "labels": "labels",
    "envelopes": "envelopes", "calculators": "calculators", "colored_pencils": "colored pencils",
    "tape_dispensers": "tape dispensers", "desk_organizers": "desk organizers",
    "erasers": "erasers", "pens": "pens", "planners": "planners", "glue": "glue sticks",
    "mailers": "mailers", "pen_refills": "pen refills", "index_dividers": "index dividers",
}  # fmt: skip

PROBLEMS = {
    "pens": ["kept skipping", "smudged everywhere", "dried out in a week"],
    "markers": ["dried out in a week", "bled through the paper"],
    "mechanical_pencils": ["kept jamming", "kept breaking the lead"],
    "pencils": ["kept breaking", "were hard to sharpen"],
    "notebooks": ["fell apart", "had paper that was too thin"],
    "sticky_notes": ["didn't stick", "curled up and fell off"],
    "staplers": ["jammed constantly", "broke after a month"],
    "binders": ["had rings that didn't close", "fell apart"],
    "calculators": ["stopped working after a month", "had sticky keys"],
    "tape_dispensers": ["broke after a month", "kept tipping over"],
    "default": ["were disappointing", "did not hold up"],
}
_PROBLEM_GROUP = {
    "gel_pens": "pens", "ballpoint_pens": "pens", "fountain_pens": "pens", "pens": "pens",
    "highlighters": "markers", "permanent_markers": "markers", "dry_erase_markers": "markers",
}  # fmt: skip

# Printers named in the ink and toner listings (checked by hand with --review: every listed
# item names the printer). Category of the supply, then how the shopper writes the device.
DEVICES = [
    ("ink_cartridges", "Canon PIXMA MG3620"), ("ink_cartridges", "Canon PIXMA MG5520"),
    ("ink_cartridges", "Canon PIXMA MX922"), ("ink_cartridges", "Canon PIXMA iP8720"),
    ("ink_cartridges", "Canon PIXMA MG7720"), ("ink_cartridges", "Canon PIXMA TS9020"),
    ("ink_cartridges", "HP OfficeJet Pro 6230"), ("ink_cartridges", "HP OfficeJet 6812"),
    ("ink_cartridges", "HP OfficeJet 4500"), ("ink_cartridges", "HP ENVY 5540"),
    ("ink_cartridges", "HP OfficeJet Pro 6968"), ("ink_cartridges", "HP OfficeJet Pro X451"),
    ("ink_cartridges", "HP Deskjet 2540"), ("ink_cartridges", "Epson WorkForce WF-2650"),
    ("toner", "Brother HL-2240"), ("toner", "Brother HL-2270DW"), ("toner", "Brother MFC-7860DW"),
    ("toner", "Brother MFC-7340"), ("toner", "HP Color LaserJet Pro M252dw"),
    ("toner", "HP LaserJet Pro M452dn"),
]  # fmt: skip


def cap(s: str) -> str:
    return s[0].upper() + s[1:]


def nice(cents: int) -> int:
    """Smallest round dollar amount at or above `cents`, in cents."""
    for d in NICE_DOLLARS:
        if d * 100 >= cents:
            return d * 100
    return NICE_DOLLARS[-1] * 100


def a(device: str) -> str:
    """'an HP OfficeJet', 'an Epson', 'a Canon'."""
    return f"an {device}" if device.startswith(("HP", "Epson")) else f"a {device}"


def money(cents: int) -> str:
    return str(cents // 100) if cents % 100 == 0 else f"{cents / 100:.2f}"


class Builder:
    def __init__(self, items: list[Item]):
        self.items = items
        self.rng = random.Random(SEED)
        self.cases: list[Case] = []
        self._by_cat = {
            k: [i for i in items if taxonomy.in_category(i.categories, k)] for k in taxonomy.KEYS
        }

    # --- catalog helpers --------------------------------------------------------------------

    def in_cat(self, key: str) -> list[Item]:
        return self._by_cat[key]

    def brands(self, key: str, min_items: int = 1) -> list[tuple[str, int]]:
        """(display name, count) of a category's brands, most common first."""
        counts: Counter[str] = Counter()
        spelling: dict[str, Counter[str]] = {}
        for it in self.in_cat(key):
            if it.brand:
                n = norm_brand(it.brand)
                counts[n] += 1
                spelling.setdefault(n, Counter())[it.brand] += 1
        rows = [(spelling[n].most_common(1)[0][0], c) for n, c in counts.items() if c >= min_items]
        return sorted(rows, key=lambda r: (-r[1], r[0]))

    def budget(self, key: str, q: float = 0.5, pool: list[Item] | None = None) -> int:
        prices = sorted(i.price_cents for i in (pool or self.in_cat(key)) if i.price_cents)
        return nice(prices[min(len(prices) - 1, int(q * len(prices)))])

    def satisfiable(self, gold: Constraints, purchases: list[CasePurchase] = ()) -> list[Item]:
        """Items meeting the gold constraints, stock ignored (it is checked at run time)."""
        by_id = {i.item_id: i for i in self.items}
        history = [(None, by_id[p.item_id]) for p in sorted(purchases, key=lambda p: p.days_ago)]
        eff = constraints.resolve(gold, history)  # type: ignore[arg-type]
        if eff.blocked:
            return []
        out = []
        for it in self.in_cat(gold.category) if gold.category else self.items:
            fails = [c for c in constraints.failures(it, eff) if c.name != "stock"]
            if fails:
                continue
            if eff.device and compat.code_verdict(it, eff.device) is None:
                continue
            out.append(it)
        return out

    def add(
        self, template: str, message: str, gold: Constraints, purchases=(), note="", expect=True
    ):
        purchases = list(purchases)
        sat = self.satisfiable(gold, purchases)
        if expect and len(sat) < 1:
            raise SystemExit(f"{template}: nothing satisfies {message!r}")
        if not expect and sat:
            raise SystemExit(
                f"{template}: {message!r} should be unsatisfiable, {len(sat)} items fit"
            )
        n = sum(c.template == template for c in self.cases) + 1
        self.cases.append(
            Case(
                id=f"{template}-{n:03d}",
                template=template,
                message=message,
                user_id=FIRST_USER_ID + len(self.cases),
                gold=gold,
                purchases=purchases,
                note=note,
            )
        )

    def distractors(self, key: str, k: int) -> list[CasePurchase]:
        """More recent purchases from other categories (the reference must be the latest of
        *this* category, not the latest overall)."""
        others = [
            i for i in self.items if i.categories and not taxonomy.in_category(i.categories, key)
        ]
        picks = self.rng.sample(others, k)
        return [CasePurchase(item_id=i.item_id, days_ago=self.rng.randint(1, 5)) for i in picks]


# --- templates --------------------------------------------------------------------------------

BUDGET_CATS = [
    "gel_pens", "ballpoint_pens", "fountain_pens", "mechanical_pencils", "notebooks",
    "sticky_notes", "highlighters", "permanent_markers", "dry_erase_markers", "staplers",
    "binders", "folders", "labels", "envelopes", "calculators", "ink_cartridges", "toner",
    "colored_pencils", "tape_dispensers", "desk_organizers",
]  # fmt: skip
BUDGET_FORMS: list[Callable[[str, int, int], str]] = [
    lambda p, b, lo: f"I need {p} under ${money(b)}.",
    lambda p, b, lo: f"Looking for {p} for less than ${money(b)}.",
    lambda p, b, lo: f"Can you find {p}? No more than {money(b)} dollars.",
    lambda p, b, lo: f"{cap(p)} with a budget of ${money(b)}.",
    lambda p, b, lo: f"Show me {p}, ${money(b)} max.",
    lambda p, b, lo: f"{cap(p)} under {money(b)} bucks",
    lambda p, b, lo: f"{cap(p)} for {NUMBER_WORDS[b // 100]} dollars or less",
    lambda p, b, lo: f"Any {p} below {NUMBER_WORDS[b // 100]} bucks?",
    lambda p, b, lo: f"{cap(p)} between ${money(lo)} and ${money(b)}",
    lambda p, b, lo: f"My budget is ${money(b)}. I want {p}.",
]


def budget_cases(b: Builder) -> None:
    for i, key in enumerate(BUDGET_CATS):
        budget = b.budget(key)
        lo = max(100, (budget // 200) * 100)
        msg = BUDGET_FORMS[i % len(BUDGET_FORMS)](PHRASES[key], budget, lo)
        b.add("budget", msg, Constraints(category=key, budget_max_cents=budget))


BRAND_CATS = [
    "gel_pens", "ballpoint_pens", "fountain_pens", "mechanical_pencils", "notebooks",
    "sticky_notes", "highlighters", "permanent_markers", "dry_erase_markers", "staplers",
    "binders", "folders", "labels", "calculators", "ink_cartridges", "erasers",
]  # fmt: skip


def brand_cases(b: Builder) -> None:
    for i, key in enumerate(BRAND_CATS):
        candidates = [name for name, n in b.brands(key, min_items=3)]
        brand = b.rng.choice(candidates)
        p = PHRASES[key]
        form = i % 4
        budget = b.budget(
            key, 0.7, [it for it in b.in_cat(key) if norm_brand(it.brand) == norm_brand(brand)]
        )
        msg = [
            f"{brand} {p}, please.",
            f"Do you have {brand} {p}?",
            f"Only {brand} {p}, under ${money(budget)}.",
            f"I'd like {p} from {brand}.",
        ][form]
        gold = Constraints(
            category=key, brands=[brand], budget_max_cents=budget if form == 2 else None
        )
        b.add("brand", msg, gold)


EXCLUDE_CATS = [
    "gel_pens", "ballpoint_pens", "fountain_pens", "mechanical_pencils", "notebooks",
    "sticky_notes", "highlighters", "permanent_markers", "dry_erase_markers", "staplers",
    "binders", "folders", "labels", "erasers",
]  # fmt: skip


def exclude_cases(b: Builder) -> None:
    for i, key in enumerate(EXCLUDE_CATS):
        brand = b.brands(key)[0][0]  # the most common brand: excluding it matters
        p = PHRASES[key]
        budget = b.budget(key, 0.6)
        form = i % 7
        msg = [
            f"{cap(p)}, anything but {brand}.",
            f"{cap(p)} under ${money(budget)}, but not {brand}.",
            f"No {brand} please. I need {p}.",
            f"{cap(p)} that aren't made by {brand}.",
            f"I'm done with {brand}. Show me other {p}.",
            f"Any {p} besides {brand}?",
            f"{cap(p)}, avoid {brand}, under {money(budget)} dollars.",
        ][form]
        has_budget = form in (1, 6)
        gold = Constraints(
            category=key, exclude_brands=[brand], budget_max_cents=budget if has_budget else None
        )
        b.add("exclude_brand", msg, gold)


def _supply(key: str) -> str:
    return "ink cartridges" if key == "ink_cartridges" else "toner cartridges"


def compat_cases(b: Builder) -> None:
    for i, (key, device) in enumerate(DEVICES):
        s = _supply(key)
        msg = [
            f"{cap(s)} for my {device}",
            f"Which {s} work with {a(device)}?",
            f"I have {a(device)} and need {s}.",
            f"{cap(s)} compatible with the {device}",
            f"Need {s} for our office {device}.",
        ][i % 5]
        b.add("compat", msg, Constraints(category=key, compatible_with=device))


def compat_budget_cases(b: Builder) -> None:
    for i, (key, device) in enumerate(DEVICES[:12]):
        s = _supply(key)
        fits = b.satisfiable(Constraints(category=key, compatible_with=device))
        prices = sorted(it.price_cents for it in fits if it.price_cents)
        budget = nice(prices[min(1, len(prices) - 1)])
        form = i % 4
        msg = [
            f"{cap(s)} for my {device} under ${money(budget)}",
            f"{cap(s)} that fit {a(device)}, no more than {money(budget)} dollars",
            f"I need {s} for {a(device)}. Budget ${money(budget)}.",
            f"{cap(s)} for the {device} below {NUMBER_WORDS[budget // 100]} bucks",
        ][form]
        b.add(
            "compat_budget",
            msg,
            Constraints(category=key, compatible_with=device, budget_max_cents=budget),
        )


def compat_oem_cases(b: Builder) -> None:
    picks = [d for d in DEVICES if d[0] == "toner"] + [
        d for d in DEVICES if d[0] == "ink_cartridges"
    ][:4]
    for i, (key, device) in enumerate(picks[:10]):
        s = _supply(key)
        msg = [
            f"Genuine {s} for my {device}",
            f"Original {s} only, for {a(device)}",
            f"OEM {s} for the {device}, no remanufactured ones",
            f"{cap(s)} for my {device}, but no third-party or refilled cartridges",
            f"Brand-name {s} for {a(device)}, not knock-offs",
        ][i % 5]
        b.add("compat_oem", msg, Constraints(category=key, compatible_with=device, oem_only=True))


HISTORY_CATS = [
    "gel_pens", "ballpoint_pens", "fountain_pens", "mechanical_pencils", "notebooks",
    "sticky_notes", "highlighters", "permanent_markers", "dry_erase_markers", "staplers",
    "binders", "calculators", "pencils", "tape_dispensers",
]  # fmt: skip


def _problem(b: Builder, key: str) -> str:
    group = key if key in PROBLEMS else _PROBLEM_GROUP.get(key, "default")
    return b.rng.choice(PROBLEMS[group])


def different_brand_cases(b: Builder) -> None:
    for i, key in enumerate(HISTORY_CATS):
        top = b.brands(key, min_items=2)[0][0]
        bought = b.rng.choice(
            [it for it in b.in_cat(key) if norm_brand(it.brand) == norm_brand(top)]
        )
        p = PHRASES[key]
        budget = b.budget(key, 0.6)
        form = i % 4
        msg = [
            f"The {p} I bought last time {_problem(b, key)}. Can you find a different brand?",
            f"The {bought.brand} {p} I got last time {_problem(b, key)}. Something from another brand under ${money(budget)}?",
            f"My last order of {p} {_problem(b, key)}. Show me other brands.",
            f"I'm not buying {bought.brand} {p} again after last time. Alternatives?",
        ][form]
        named = form in (1, 3)
        gold = Constraints(
            category=key,
            last_purchase="different_brand",
            exclude_brands=[bought.brand] if named else [],
            budget_max_cents=budget if form == 1 else None,
        )
        purchases = [CasePurchase(item_id=bought.item_id, days_ago=b.rng.randint(10, 60))]
        purchases += b.distractors(key, 1 + i % 2)
        b.add("last_different_brand", msg, gold, purchases)


def same_brand_cases(b: Builder) -> None:
    for i, key in enumerate(HISTORY_CATS[:8]):
        brand = b.brands(key, min_items=3)[min(1, len(b.brands(key, 3)) - 1)][0]
        bought = b.rng.choice(
            [it for it in b.in_cat(key) if norm_brand(it.brand) == norm_brand(brand)]
        )
        p = PHRASES[key]
        budget = b.budget(
            key, 0.8, [it for it in b.in_cat(key) if norm_brand(it.brand) == norm_brand(brand)]
        )
        form = i % 4
        msg = [
            f"I loved the {p} I bought last time. More from the same brand, please.",
            f"Same brand as my last {p} order, under ${money(budget)}.",
            f"Those {p} from my last order were great. Anything else by that brand?",
            f"I want {p} again, same brand as before.",
        ][form]
        gold = Constraints(
            category=key, last_purchase="same_brand", budget_max_cents=budget if form == 1 else None
        )
        purchases = [CasePurchase(item_id=bought.item_id, days_ago=b.rng.randint(10, 60))]
        purchases += b.distractors(key, 1)
        b.add("last_same_brand", msg, gold, purchases)


def cheaper_cases(b: Builder) -> None:
    for i, key in enumerate(HISTORY_CATS[6:]):
        priced = sorted(
            (it for it in b.in_cat(key) if it.price_cents),
            key=lambda it: (it.price_cents, it.item_id),
        )
        bought = b.rng.choice(priced[len(priced) // 2 : (3 * len(priced)) // 4 + 1])
        p = PHRASES[key]
        msg = [
            f"I need {p} again, but cheaper than what I paid last time.",
            f"Find {p} that cost less than the ones I bought last time.",
            f"My last {p} were too expensive. Something cheaper?",
            f"Show me {p} for less than my previous order.",
        ][i % 4]
        gold = Constraints(category=key, last_purchase="cheaper")
        purchases = [CasePurchase(item_id=bought.item_id, days_ago=b.rng.randint(10, 60))]
        purchases += b.distractors(key, 1)
        b.add("last_cheaper", msg, gold, purchases)


# Paraphrases the rules were not written for; gold labels by hand.
PARAPHRASES = [
    ("post-its for my desk", Constraints(category="sticky_notes")),
    ("something to sharpen my pencils", Constraints(category="pencil_sharpeners")),
    ("Sharpies, under 10 bucks", Constraints(category="permanent_markers", brands=["Sharpie"], budget_max_cents=1000)),
    ("a pen that writes nicely for journaling, nothing over forty dollars", Constraints(category="pens", budget_max_cents=4000)),
    ("refills for my Parker pen", Constraints(category="pen_refills", brands=["Parker"])),
    ("a whiteboard for the kitchen under $20", Constraints(category="dry_erase_boards", budget_max_cents=2000)),
    ("ink for my fountain pen", Constraints(category="bottled_ink")),
    ("a cheap calculator for my kid's math class, ten bucks at most", Constraints(category="calculators", budget_max_cents=1000)),
    ("highlighters, but not the Sharpie ones", Constraints(category="highlighters", exclude_brands=["Sharpie"])),
    ("tape for wrapping boxes I'm shipping", Constraints(category="packing_tape")),
]  # fmt: skip

# Nothing in the catalog satisfies these; the right answer shows nothing.
UNSATISFIABLE = [
    ("A fountain pen under $1", Constraints(category="fountain_pens", budget_max_cents=100)),
    ("Toner cartridges for my Brother HL-2240 under $5", Constraints(category="toner", compatible_with="Brother HL-2240", budget_max_cents=500)),
    ("Expo fountain pens", Constraints(category="fountain_pens", brands=["Expo"])),
    ("Lamy staplers", Constraints(category="staplers", brands=["Lamy"])),
    ("Ink cartridges for my Epson EcoTank ET-2850", Constraints(category="ink_cartridges", compatible_with="Epson EcoTank ET-2850")),
    ("Sharpie gel pens under $2", Constraints(category="gel_pens", brands=["Sharpie"], budget_max_cents=200)),
    ("Genuine Canon toner for my Brother HL-2240", Constraints(category="toner", brands=["Canon"], oem_only=True, compatible_with="Brother HL-2240")),
    ("A calculator for under $2", Constraints(category="calculators", budget_max_cents=200)),
]  # fmt: skip

CATEGORY_ONLY = [
    ("I need sticky notes.", "sticky_notes"), ("Show me some fountain pens.", "fountain_pens"),
    ("Do you sell staplers?", "staplers"), ("Recommend a calculator.", "calculators"),
    ("Notebooks for school", "notebooks"), ("A desk organizer for pens and clips", "desk_organizers"),
    ("Mailing labels", "labels"), ("Binder clips", "clips"), ("Envelopes", "envelopes"),
    ("A planner for next year", "planners"),
]  # fmt: skip


def build(items: list[Item]) -> list[Case]:
    b = Builder(items)
    budget_cases(b)
    brand_cases(b)
    exclude_cases(b)
    compat_cases(b)
    compat_budget_cases(b)
    compat_oem_cases(b)
    different_brand_cases(b)
    same_brand_cases(b)
    cheaper_cases(b)
    for msg, gold in PARAPHRASES:
        b.add("paraphrase", msg, gold)
    for msg, gold in UNSATISFIABLE:
        b.add("unsatisfiable", msg, gold, expect=False)
    for msg, key in CATEGORY_ONLY:
        b.add("category_only", msg, Constraints(category=key))
    return b.cases


def review(items: list[Item]) -> None:
    """Print every item the code matcher confirms for each device, for a manual check."""
    for key, device in DEVICES:
        d = compat.parse_device(device)
        assert d is not None
        hits = [
            (it, compat.code_verdict(it, d))
            for it in items
            if taxonomy.in_category(it.categories, key)
        ]
        hits = [(it, v) for it, v in hits if v]
        print(f"\n{device} ({key}): {len(hits)} confirmed")
        for it, v in hits:
            print(
                f"  {it.item_id:5d} {it.brand or '-':12.12s} {it.title[:70]:70s} | {v.evidence[0].field}: {v.evidence[0].snippet[:90]}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--review", action="store_true", help="print the device matches and exit")
    args = ap.parse_args()
    catalog = PgCatalog.connect(paths.database_url(), HashEncoder())
    items = catalog.all_items()
    if args.review:
        review(items)
        return
    cases = build(items)
    dump(cases)
    counts = Counter(c.template for c in cases)
    print(
        f"{len(cases)} cases -> {paths.CASES.relative_to(paths.ROOT)}: "
        + ", ".join(f"{k} {v}" for k, v in counts.items())
    )


if __name__ == "__main__":
    main()
