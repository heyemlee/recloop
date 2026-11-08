"""ClaudeLLM against a stand-in client: request shape, usage and cost, stop reasons. No network."""

from types import SimpleNamespace

import pytest

from agent.llm import ClaudeLLM, LLMError, cost_usd
from agent.schema import ShoppingRequest


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def response(stop="end_turn", model=None, blocks=None):
    usage = SimpleNamespace(
        input_tokens=1000,
        output_tokens=200,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=3000,
    )
    blocks = blocks if blocks is not None else [
        SimpleNamespace(type="thinking", thinking=""),
        SimpleNamespace(type="text", text='{"search_text": "x"}'),
    ]  # fmt: skip
    return SimpleNamespace(content=blocks, usage=usage, stop_reason=stop, model=model)


def client(resp):
    msgs = FakeMessages(resp)
    return SimpleNamespace(beta=SimpleNamespace(messages=msgs)), msgs


SCHEMA = ShoppingRequest.model_json_schema()


@pytest.mark.parametrize(
    "model, fallbacks",
    [("claude-opus-5-5", True), ("claude-sonnet-5-5", True), ("claude-haiku-5-5", False)],
)
def test_request_shape(model, fallbacks):
    c, msgs = client(response())
    llm = ClaudeLLM(model, client=c)
    llm.complete_json("parse", "SYSTEM", "USER", SCHEMA)
    k = msgs.kwargs
    assert k["model"] == model
    assert k["system"] == [
        {"type": "text", "text": "SYSTEM", "cache_control": {"type": "ephemeral"}}
    ]
    assert k["messages"] == [{"role": "user", "content": "USER"}]
    assert k["output_config"]["effort"] == "low"
    assert k["output_config"]["format"]["type"] == "json_schema"
    assert k["output_config"]["format"]["schema"]["additionalProperties"] is False
    assert "thinking" not in k and "temperature" not in k and "tool_choice" not in k
    if fallbacks:
        assert k["fallbacks"] == "default" and k["betas"] == ["server-side-fallback-2026-07-01"]
    else:
        assert "fallbacks" not in k and "betas" not in k


def test_text_block_usage_and_cost():
    c, _ = client(response())
    reply = ClaudeLLM("claude-opus-5-5", client=c).complete_json("parse", "S", "U", SCHEMA)
    assert reply.text == '{"search_text": "x"}'
    assert reply.call.input_tokens == 1000 and reply.call.cache_read_input_tokens == 3000
    assert reply.call.cost_usd == pytest.approx((1000 * 4 + 200 * 20 + 3000 * 0.2) / 1e6)


def test_cost_uses_the_model_that_served_the_reply():
    c, _ = client(response(model="claude-opus-4-8"))
    call = ClaudeLLM("claude-opus-5-5", client=c).complete_json("parse", "S", "U", SCHEMA).call
    assert call.model == "claude-opus-4-8"
    assert call.cost_usd == pytest.approx(cost_usd("claude-opus-4-8", 1000, 200, 0, 3000))


@pytest.mark.parametrize("stop, blocks", [("refusal", []), ("max_tokens", None), ("end_turn", [])])
def test_unusable_replies_raise_with_the_call(stop, blocks):
    c, _ = client(response(stop=stop, blocks=blocks))
    with pytest.raises(LLMError) as e:
        ClaudeLLM("claude-opus-5-5", client=c).complete_json("parse", "S", "U", SCHEMA)
    assert e.value.call.error in {"refusal", "max_tokens", "no_text"}
    assert e.value.call.input_tokens == 1000  # still accounted for


def test_unknown_model_price_is_nan():
    import math

    assert math.isnan(cost_usd("some-other-model", 1, 1, 0, 0))
