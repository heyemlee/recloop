import anthropic
import pytest
from pydantic import ValidationError

from agent import taxonomy
from agent.schema import CompatJudgements, Constraints, ItemReasons, ShoppingRequest


def test_defaults_mean_no_constraint():
    c = Constraints()
    assert c.category is None and c.budget_max_cents is None and not c.brands
    assert c.last_purchase == "none" and not c.oem_only


@pytest.mark.parametrize("bad", [0, -5, 10_000_001])
def test_budget_bounds(bad):
    with pytest.raises(ValidationError, match="budget_max_cents"):
        Constraints(budget_max_cents=bad)


def test_device_needs_a_model_number():
    with pytest.raises(ValidationError, match="model number"):
        Constraints(compatible_with="my HP printer")
    assert Constraints(compatible_with="  Brother   HL-2240 ").compatible_with == "Brother HL-2240"
    assert Constraints(compatible_with="  ").compatible_with is None


def test_brands_are_deduplicated_and_cannot_overlap():
    assert Constraints(brands=["Pilot", "pilot", " Pilot "]).brands == ["Pilot"]
    with pytest.raises(ValidationError, match="overlap"):
        Constraints(brands=["Post-it"], exclude_brands=["post it"])


def test_unknown_category_and_extra_fields_are_rejected():
    with pytest.raises(ValidationError):
        Constraints(category="spaceships")
    with pytest.raises(ValidationError):
        Constraints(colour="blue")


def test_search_text_is_required_and_trimmed():
    with pytest.raises(ValidationError):
        ShoppingRequest.model_validate({"search_text": "   "})
    r = ShoppingRequest(search_text="  gel   pens ")
    assert r.search_text == "gel pens"
    assert r.constraints() == Constraints()


def test_json_schemas_work_with_structured_outputs():
    schema = ShoppingRequest.model_json_schema()
    assert set(schema["properties"]["category"]["anyOf"][0]["enum"]) == set(taxonomy.KEYS)
    for model in (ShoppingRequest, CompatJudgements, ItemReasons):
        transformed = anthropic.transform_schema(model.model_json_schema())
        assert transformed["type"] == "object"
        assert transformed.get("additionalProperties") is False
