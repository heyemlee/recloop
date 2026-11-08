import pytest

from agent import compat
from agent.schema import CompatEvidence, CompatJudgement
from agent.tests.conftest import INK, TONER, item


def dev(s):
    d = compat.parse_device(s)
    assert d is not None
    return d


def test_parse_device():
    d = dev("Brother HL-2240")
    assert (d.brand, d.prefix, d.model, d.family) == ("brother", "hl", "2240", ())
    d = dev("HP OfficeJet Pro 6230")
    assert (d.brand, d.prefix, d.model, d.family) == ("hp", "pro", "6230", ("officejet",))
    d = dev("Canon PIXMA MG3620")
    assert (d.prefix, d.model, d.family) == (None, "mg3620", ("pixma",))
    d = dev("HP Deskjet 2540")
    assert (d.prefix, d.model, d.family) == (None, "2540", ("deskjet",))
    assert dev("HP Color LaserJet Pro M252dw").model == "m252dw"
    assert compat.parse_device("my printer") is None


@pytest.mark.parametrize(
    "device, text",
    [
        ("Brother HL-2240", "Works with HL-2240 and HL-2270DW"),
        ("Brother HL-2240", "for HL2240 printers"),
        ("Brother HL-2240", "Compatible with HL-2130, 2132, 2220, 2230, 2240"),
        ("Canon PIXMA MG3620", "Compatible to MG3620,MG3520"),
        ("Canon PIXMA MG3620", "fits the MG 3620"),
        ("HP Deskjet 2540", "work with: HP Deskjet 1000, 1010, 1050, 2050, 2510, 2540"),
        ("HP OfficeJet Pro 6230", "HP Officejet Pro 6230, 6830"),
        ("HP OfficeJet Pro 6830", "HP Officejet Pro 6230, 6830"),
        ("HP Color LaserJet Pro M252dw", "for HP Color Laserjet Pro M252dw M277"),
        ("Brother MFC-7340", "DCP-7030/7045N, MFC-7320/7440/7340, HL-2140"),
    ],
)
def test_code_confirms_with_evidence(device, text):
    v = compat.code_verdict(item(1, "Cartridge", "X", INK, 100, feats=[text]), dev(device))
    assert v is not None and v.status == "confirmed" and v.source == "code"
    assert v.evidence[0].field == "features"
    assert v.evidence[0].snippet.replace("…", "") in text


@pytest.mark.parametrize(
    "device, text",
    [
        ("HP OfficeJet 4500", "HP ENVY 4500, 4502, 5530"),  # another series' 4500
        ("Brother HL-2240", "HL-2240D only"),  # a different model
        ("Brother HL-2240", "Page yield 2240 pages"),  # a number, not a model
        ("Canon PIXMA MG3620", "Compatible to MG3520"),
        ("HP Deskjet 2540", "Up to 2540 pages"),
    ],
)
def test_code_does_not_confirm_lookalikes(device, text):
    assert (
        compat.code_verdict(item(1, "Cartridge", "X", INK, 100, feats=[text]), dev(device)) is None
    )


def test_title_is_checked_first():
    it = item(1, "Toner for HL-2240", "X", TONER, 100, desc="also HL-2240", feats=["HL-2240"])
    assert compat.code_verdict(it, dev("Brother HL-2240")).evidence[0].field == "title"


def judgement(verdict, *evidence):
    return CompatJudgement(
        item_id=1, verdict=verdict, evidence=[CompatEvidence(field=f, quote=q) for f, q in evidence]
    )


def test_model_confirmation_needs_a_real_quote_naming_the_model():
    it = item(
        1, "Toner", "X", TONER, 100, desc="Use in the Brother 2200 family: HL2240 and HL2270DW."
    )
    d = dev("Brother HL-2240")
    ok = compat.model_verdict(
        it, d, judgement("compatible", ("description", "HL2240 and HL2270DW"))
    )
    assert (ok.status, ok.source) == ("confirmed", "model")
    # A quote that is not in the field: invented evidence.
    fake = compat.model_verdict(
        it, d, judgement("compatible", ("description", "works with HL-2240"))
    )
    assert fake.status == "unconfirmed"
    # A real quote that does not name the model.
    vague = compat.model_verdict(
        it, d, judgement("compatible", ("description", "Brother 2200 family"))
    )
    assert vague.status == "unconfirmed"
    # Right words, wrong field.
    wrong_field = compat.model_verdict(
        it, d, judgement("compatible", ("title", "HL2240 and HL2270DW"))
    )
    assert wrong_field.status == "unconfirmed"
    assert compat.model_verdict(it, d, judgement("compatible")).status == "unconfirmed"


def test_incompatible_needs_verified_evidence():
    it = item(1, "Toner", "X", TONER, 100, feats=["Works with HL-2130 and HL-2132 only"])
    d = dev("Brother HL-2240")
    no = compat.model_verdict(
        it, d, judgement("incompatible", ("features", "HL-2130 and HL-2132 only"))
    )
    assert no.status == "incompatible"
    unproven = compat.model_verdict(it, d, judgement("incompatible", ("features", "HL-9999")))
    assert unproven.status == "unconfirmed"
    assert compat.model_verdict(it, d, judgement("unknown")).status == "unconfirmed"
