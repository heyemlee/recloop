"""What the shopper asked for, as constraints the code can check (design §10.1, §10.3).

`Constraints` is shared by the parsers (language model, rules, gold) and the evaluation's
gold labels; `ShoppingRequest` adds the search query. The JSON schema the language model must
follow is generated from `ShoppingRequest`; validators below go beyond what a JSON schema can
say, and a failure is sent back to the model once (`parsing.LLMParser`).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent import taxonomy, text

CategoryKey = Literal[taxonomy.KEYS]  # type: ignore[valid-type]
LastPurchase = Literal["none", "same_brand", "different_brand", "cheaper"]

MAX_BUDGET_CENTS = 10_000_000  # $100,000: anything above is a parsing error, not a budget


def _clean_brands(values: list[str]) -> list[str]:
    out: dict[str, str] = {}
    for v in values:
        v = " ".join(v.split())
        if v and text.norm_phrase(v) not in out:
            out[text.norm_phrase(v)] = v
    return list(out.values())


class Constraints(BaseModel):
    """The checkable part of a request. Unset fields mean 'no constraint'."""

    model_config = ConfigDict(extra="forbid")

    category: CategoryKey | None = Field(
        default=None,
        description="The kind of product, as one of the store's category keys. Null if the "
        "shopper did not name a kind of product or none of the keys fits.",
    )
    budget_max_cents: int | None = Field(
        default=None,
        description="Highest price the shopper accepts, inclusive, in US cents ('under $40' -> "
        "4000, 'no more than 12 dollars' -> 1200). Null if the shopper gave no budget.",
    )
    brands: list[str] = Field(
        default_factory=list,
        description="Brands the shopper asked for; only these are acceptable. Empty if any "
        "brand is fine.",
    )
    exclude_brands: list[str] = Field(
        default_factory=list,
        description="Brands the shopper does not want.",
    )
    oem_only: bool = Field(
        default=False,
        description="True if the shopper wants genuine supplies from the device's manufacturer "
        "(no remanufactured, refilled or third-party 'compatible' products).",
    )
    compatible_with: str | None = Field(
        default=None,
        description="The device the product must work with, as the shopper named it, with its "
        "model number (e.g. 'Canon PIXMA MG3620'). Null if no device was named.",
    )
    last_purchase: LastPurchase = Field(
        default="none",
        description="How the wanted product relates to the shopper's most recent purchase of "
        "this kind of product: 'same_brand' (stay with that brand), 'different_brand' (not "
        "that brand again), 'cheaper' (must cost less than it), or 'none' if the shopper did "
        "not refer to an earlier purchase.",
    )

    @field_validator("budget_max_cents")
    @classmethod
    def _budget(cls, v: int | None) -> int | None:
        if v is not None and not 1 <= v <= MAX_BUDGET_CENTS:
            raise ValueError(f"budget_max_cents must be between 1 and {MAX_BUDGET_CENTS} cents")
        return v

    @field_validator("brands", "exclude_brands")
    @classmethod
    def _brands(cls, v: list[str]) -> list[str]:
        return _clean_brands(v)

    @field_validator("compatible_with")
    @classmethod
    def _device(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = " ".join(v.split())
        if not v:
            return None
        if not any(ch.isdigit() for ch in v):
            raise ValueError(
                "compatible_with must name a device with its model number (e.g. 'Brother "
                "HL-2240'); use null when the shopper named no model"
            )
        return v

    @model_validator(mode="after")
    def _no_overlap(self) -> Constraints:
        both = {text.norm_phrase(b) for b in self.brands} & {
            text.norm_phrase(b) for b in self.exclude_brands
        }
        if both:
            raise ValueError(f"brands and exclude_brands overlap: {sorted(both)}")
        return self


class ShoppingRequest(Constraints):
    """A parsed request: the constraints plus the query for the catalog search."""

    search_text: str = Field(
        description="A short English keyword query for the catalog search: product type, brand, "
        "model numbers and key attributes (color, size, pack). No prices and no filler words.",
    )

    @field_validator("search_text")
    @classmethod
    def _search_text(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("search_text must not be empty")
        return v[:200]

    def constraints(self) -> Constraints:
        return Constraints.model_validate(self.model_dump(exclude={"search_text"}))


# --- Structured outputs of the two other language-model steps --------------------------------


class CompatEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: Literal["title", "description", "features"] = Field(
        description="The item field the quote is taken from."
    )
    quote: str = Field(description="Words copied exactly from that field.")


class CompatJudgement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item_id: int
    verdict: Literal["compatible", "incompatible", "unknown"] = Field(
        description="'compatible' only if the item's own text says it works with the device; "
        "'incompatible' if the text names the devices it works with and this one is not among "
        "them; otherwise 'unknown'."
    )
    evidence: list[CompatEvidence] = Field(
        default_factory=list,
        description="Exact quotes supporting the verdict. Required for 'compatible'.",
    )


class CompatJudgements(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[CompatJudgement]


class ItemReason(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item_id: int
    reason: str = Field(description="One sentence, at most 25 words.")


class ItemReasons(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reasons: list[ItemReason]
