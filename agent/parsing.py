"""Message -> ShoppingRequest: by a language model (validated, one retry) or by rules.

`LLMParser` follows design §10.3 (1): the model's JSON is validated with Pydantic (the schema
plus the validators in `agent.schema`); a reply that fails is sent back once with the error,
and a second failure ends the turn with a 'could not understand' reply rather than a guess.

`RuleParser` is the no-LLM baseline's understanding: a budget regex, the category aliases of
`agent.taxonomy`, a dictionary of catalog brands, 'genuine / original / OEM', and 'for / works
with <words> <model number>'. It does not handle negation, paraphrased numbers or references to
earlier purchases; the evaluation shows where that matters.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import ValidationError

from agent import prompts, taxonomy, text
from agent.llm import LLM, LLMCall, LLMError
from agent.schema import ShoppingRequest


@dataclass
class ParseOutcome:
    request: ShoppingRequest | None
    source: str
    attempts: int = 1
    errors: list[str] = field(default_factory=list)
    llm_calls: list[LLMCall] = field(default_factory=list)


class Parser(Protocol):
    name: str

    def parse(self, message: str) -> ParseOutcome: ...


def _short_error(e: ValidationError) -> str:
    parts = []
    for err in e.errors()[:5]:
        loc = ".".join(str(x) for x in err.get("loc", ())) or "(root)"
        parts.append(f"{loc}: {err.get('msg')}")
    return "; ".join(parts)


REQUEST_SCHEMA = ShoppingRequest.model_json_schema()


class LLMParser:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.name = f"llm:{llm.model}"

    def parse(self, message: str) -> ParseOutcome:
        out = ParseOutcome(None, self.name, attempts=0)
        rejected: str | None = None
        error: str | None = None
        for _ in range(2):
            out.attempts += 1
            user = prompts.parse_user(message, rejected, error)
            try:
                reply = self.llm.complete_json("parse", prompts.PARSE_SYSTEM, user, REQUEST_SCHEMA)
            except LLMError as e:
                out.llm_calls.append(e.call)
                out.errors.append(str(e))
                rejected, error = "", str(e)
                continue
            out.llm_calls.append(reply.call)
            try:
                out.request = ShoppingRequest.model_validate_json(reply.text)
                return out
            except ValidationError as e:
                rejected, error = reply.text, _short_error(e)
                out.errors.append(error)
        return out


# --- Rules (baseline) ------------------------------------------------------------------------

_AMOUNT = r"(\d{1,6}(?:\.\d{1,2})?)"
_BUDGET_PATTERNS = [
    re.compile(r"\$\s?" + _AMOUNT, re.IGNORECASE),
    re.compile(_AMOUNT + r"\s?(?:dollars|bucks|usd)\b", re.IGNORECASE),
]
_DEVICE = re.compile(
    r"\b(?:for|with|fits?|fit in)\s+(?:my|our|a|an|the)?\s*"
    r"((?:[A-Za-z][\w&.\-]*\s+){0,4}[A-Za-z]*-?\d[\w\-]*)",
)
_OEM = re.compile(r"\b(genuine|original|oem|brand[- ]name)\b", re.IGNORECASE)


def _budget(message: str) -> tuple[int | None, str]:
    """Cents of the first price in the message, and the message without price phrases."""
    found = None
    rest = message
    for pat in _BUDGET_PATTERNS:
        m = pat.search(rest)
        if m and found is None:
            found = round(float(m.group(1)) * 100)
        rest = pat.sub(" ", rest)
    return (found if found and found > 0 else None), rest


class RuleParser:
    name = "rules"

    def __init__(self, brands: Iterable[str]):
        # Longest first so 'Paper Mate' wins over a shorter brand inside it.
        names = {text.norm_phrase(b): b for b in brands if b and text.norm_phrase(b)}
        self._brands = sorted(names.items(), key=lambda kv: (-len(kv[0]), kv[0]))

    @classmethod
    def from_catalog(cls, brands: Iterable[str | None], min_items: int = 2) -> RuleParser:
        """A brand dictionary from the catalog's brand column, without the noise a rule writer
        would remove: brands on fewer than `min_items` items, and 'brands' that are ordinary
        words or product names in this catalog ('Need', 'Colored Pencils')."""
        counts: dict[str, int] = {}
        spelling: dict[str, str] = {}
        for b in brands:
            n = text.norm_phrase(b or "")
            if n:
                counts[n] = counts.get(n, 0) + 1
                spelling.setdefault(n, b or "")
        keep = []
        for n, c in counts.items():
            if c < min_items or len(n) < 2 or n in text.QUERY_STOPWORDS:
                continue
            if taxonomy.match_alias(n) is not None:
                continue
            keep.append(spelling[n])
        return cls(keep)

    def _find_brand(self, message: str) -> str | None:
        hay = f" {text.norm_phrase(message)} "
        for norm, raw in self._brands:
            if f" {norm} " in hay:
                return raw
        return None

    def parse(self, message: str) -> ParseOutcome:
        budget, rest = _budget(message)
        device = None
        m = _DEVICE.search(rest)
        if m and any(ch.isalpha() for ch in m.group(1)):
            device = m.group(1).strip(" .,;")
            rest = rest.replace(m.group(1), " ")
        cat = taxonomy.match_alias(rest)
        brand = self._find_brand(rest)
        request = ShoppingRequest(
            category=cat[0] if cat else None,
            budget_max_cents=budget,
            brands=[brand] if brand else [],
            oem_only=bool(_OEM.search(message)),
            compatible_with=device,
            search_text=" ".join(text.words(rest)) or message,
        )
        if device:
            request = request.model_copy(update={"search_text": f"{request.search_text} {device}"})
        return ParseOutcome(request, self.name)


class FixedParser:
    """Returns a given request for every message (the evaluation's 'oracle' agent)."""

    def __init__(self, request: ShoppingRequest, name: str = "oracle"):
        self.request = request
        self.name = name

    def parse(self, message: str) -> ParseOutcome:
        return ParseOutcome(self.request, self.name)
