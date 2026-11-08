from datetime import UTC, datetime

from agent import constraints
from agent.catalog import Purchase
from agent.schema import Constraints
from agent.tests.conftest import GEL, INK, ITEMS, TONER, item

BY_ID = {it.item_id: it for it in ITEMS}


def names(it, c, **kw):
    return [f.name for f in constraints.failures(it, constraints.Effective(c, **kw))]


def test_unknown_price_never_meets_a_budget():
    unknown = item(1, "x", "A", INK, None)
    assert names(unknown, Constraints(budget_max_cents=10_000)) == ["price"]
    assert names(unknown, Constraints()) == []  # no budget, nothing to check


def test_hard_checks():
    it = item(1, "Gel pen", "Pilot", GEL, 1500, stock=0)
    assert names(it, Constraints(category="gel_pens", budget_max_cents=1500)) == ["stock"]
    assert names(it, Constraints(category="notebooks", budget_max_cents=1499)) == [
        "stock",
        "price",
        "category",
    ]
    assert names(item(2, "x", None, (), 100), Constraints(category="pens")) == ["category"]


def test_brand_checks():
    it = item(1, "Gel pen", "Pilot Corporation", GEL, 900)
    assert names(it, Constraints(brands=["Pilot"])) == []
    assert names(it, Constraints(brands=["Pentel"])) == ["brands"]
    assert names(it, Constraints(exclude_brands=["pilot"])) == ["exclude_brands"]


def test_genuine_only():
    d = constraints.compat.parse_device("Brother HL-2240")
    genuine = BY_ID[13]
    third_party = BY_ID[14]
    reseller = BY_ID[16]
    c = Constraints(oem_only=True, compatible_with="Brother HL-2240")
    assert names(genuine, c, device=d) == []
    assert names(reseller, c, device=d) == []  # 'Brother Reseller' is Brother
    assert names(third_party, c, device=d) == ["oem"]
    assert names(BY_ID[2], Constraints(oem_only=True)) == ["oem"]  # 'Remanufactured' in the title
    assert names(item(3, "Toner", "Canon", TONER, 1), c, device=d) == [
        "oem"
    ]  # not the device maker


def history(*ids_and_days):
    return [
        (Purchase(i, datetime(2026, 10, 31 - d, tzinfo=UTC)), BY_ID[i]) for i, d in ids_and_days
    ]


def test_different_brand_excludes_the_brand_and_the_item():
    # Most recent purchase overall is a notebook; the gel-pen purchase is the reference.
    e = constraints.resolve(
        Constraints(category="gel_pens", last_purchase="different_brand"), history((11, 1), (7, 20))
    )
    assert e.reference.item.item_id == 7
    assert e.constraints.exclude_brands == ["Pilot"]
    assert e.exclude_item_ids == {7}
    assert not constraints.satisfies(BY_ID[10], e)  # 'Pilot Corporation'
    assert constraints.satisfies(BY_ID[8], e)


def test_different_brand_does_not_repeat_a_named_brand():
    c = Constraints(category="gel_pens", last_purchase="different_brand", exclude_brands=["pilot"])
    e = constraints.resolve(c, history((7, 3)))
    assert e.constraints.exclude_brands == ["pilot"]


def test_same_brand_and_missing_reference():
    e = constraints.resolve(
        Constraints(category="gel_pens", last_purchase="same_brand"), history((7, 3))
    )
    assert e.constraints.brands == ["Pilot"]
    none = constraints.resolve(
        Constraints(category="toner", last_purchase="same_brand"), history((7, 3))
    )
    assert none.reference.item is None and "no earlier purchase" in none.reference.note
    assert none.constraints.brands == []


def test_cheaper_caps_the_budget_below_the_earlier_price():
    e = constraints.resolve(
        Constraints(category="gel_pens", last_purchase="cheaper"), history((8, 3))
    )
    assert e.constraints.budget_max_cents == 1098  # item 8 cost $10.99
    assert e.exclude_item_ids == {8}
    tighter = constraints.resolve(
        Constraints(category="gel_pens", last_purchase="cheaper", budget_max_cents=500),
        history((8, 3)),
    )
    assert tighter.constraints.budget_max_cents == 500


def test_cheaper_than_an_unknown_price_blocks():
    e = constraints.resolve(
        Constraints(category="ink_cartridges", last_purchase="cheaper"), history((3, 2))
    )
    assert e.blocked and "unknown" in e.blocked
    assert not constraints.satisfies(BY_ID[1], e)


def test_hard_only_ignores_structured_constraints():
    e = constraints.Effective(Constraints(category="notebooks", brands=["Five Star"]))
    assert constraints.satisfies(BY_ID[11], e, hard_only=True)
    assert not constraints.satisfies(BY_ID[11], e)
