"""The language-model seam: one JSON-in-a-schema call, with usage and cost accounting.

The agent asks a model for three things (design §10.3): parse the shopper's message, judge
device compatibility from item text, and write one-line reasons. Each is `complete_json(purpose,
system, user, schema)`; tests and the evaluation's offline agents use fakes, so nothing here runs
without an explicit API key.

`ClaudeLLM` calls the Messages API (anthropic SDK 2.x):

  * structured outputs (`output_config.format`, the schema from `anthropic.transform_schema`),
    validated again with Pydantic by the caller, which keeps the usage of rejected replies;
  * `output_config.effort` (default "low": extraction, not open-ended reasoning). Thinking stays
    adaptive (Claude Opus 5.5 cannot turn it off; its tokens are billed as output);
  * the system prompt is the cached prefix (`cache_control` on it); the shopper's text and item
    data come after it;
  * server-side refusal fallbacks (`fallbacks="default"`) on Opus 5.5 and Sonnet 5.5; Haiku 5.5
    has none. Cost is computed at the price of the model that actually served the reply.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol

# USD per million tokens: input, output, 5-minute cache write, cache read. From the Claude API
# price list as of 2026-10; cache-write prices assume the usual 1.25x of input, and Haiku 5.5's
# cache read the usual 0.1x. The report prints this table next to the costs.
PRICES_PER_MTOK: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-5-5": (4.00, 20.00, 5.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 2.50, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.125, 0.01),
    # Where fallbacks="default" may route a refused request.
    "claude-opus-5": (5.00, 25.00, 6.25, 0.50),
    "claude-opus-4-8": (5.00, 25.00, 6.25, 0.50),
}
DEFAULT_MODEL = "claude-opus-5-5"
SERVER_FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(Exception):
    """A reply that cannot be used (refusal, truncated, no text). Carries the call for costs."""

    def __init__(self, message: str, call: LLMCall):
        super().__init__(message)
        self.call = call


@dataclass
class LLMCall:
    purpose: str
    model: str  # the model that served the reply
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    latency_ms: float = 0.0
    stop_reason: str | None = None
    cost_usd: float = 0.0
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LLMReply:
    text: str
    call: LLMCall


class LLM(Protocol):
    model: str

    def complete_json(self, purpose: str, system: str, user: str, schema: dict) -> LLMReply:
        """One request whose text reply must follow `schema`. Raises LLMError."""
        ...


def cost_usd(model: str, inp: int, out: int, cache_write: int, cache_read: int) -> float:
    p = PRICES_PER_MTOK.get(model)
    if p is None:
        return float("nan")
    return (inp * p[0] + out * p[1] + cache_write * p[2] + cache_read * p[3]) / 1e6


class ClaudeLLM:
    """Claude through the Messages API. Constructing one without `client` needs credentials."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        client: Any = None,
        effort: str = "low",
        max_tokens: int = 8000,
        timeout_s: float = 60.0,
    ):
        if client is None:
            import anthropic

            client = anthropic.Anthropic(timeout=timeout_s, max_retries=2)
        self.client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    def request(self, system: str, user: str, schema: dict) -> dict[str, Any]:
        """The keyword arguments of the Messages call (separate so tests can check them)."""
        import anthropic

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user}],
            "output_config": {
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": anthropic.transform_schema(schema)},
            },
        }
        if self.model in SERVER_FALLBACK_MODELS:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return kwargs

    def complete_json(self, purpose: str, system: str, user: str, schema: dict) -> LLMReply:
        t0 = time.perf_counter()
        response = self.client.beta.messages.create(**self.request(system, user, schema))
        call = _call_from_response(purpose, self.model, response, time.perf_counter() - t0)
        if response.stop_reason == "refusal":
            call.error = "refusal"
            raise LLMError(f"{purpose}: the model declined the request", call)
        if response.stop_reason == "max_tokens":
            call.error = "max_tokens"
            raise LLMError(f"{purpose}: the reply hit max_tokens", call)
        texts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
        if not texts:
            call.error = "no_text"
            raise LLMError(f"{purpose}: the reply has no text block", call)
        return LLMReply(texts[0], call)


def _call_from_response(purpose: str, requested: str, response: Any, seconds: float) -> LLMCall:
    usage = response.usage
    served = getattr(response, "model", None) or requested
    inp = int(getattr(usage, "input_tokens", 0) or 0)
    out = int(getattr(usage, "output_tokens", 0) or 0)
    cw = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    cr = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    return LLMCall(
        purpose=purpose,
        model=served,
        input_tokens=inp,
        output_tokens=out,
        cache_creation_input_tokens=cw,
        cache_read_input_tokens=cr,
        latency_ms=seconds * 1000,
        stop_reason=getattr(response, "stop_reason", None),
        cost_usd=cost_usd(served, inp, out, cw, cr),
    )
