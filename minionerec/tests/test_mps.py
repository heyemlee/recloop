"""Why MiniOneRec runs with eager attention on MPS (upstream.ATTN_IMPLEMENTATION).

Only where PyTorch sees the Apple GPU; inside the Claude Code sandbox, and off Apple Silicon,
these tests skip.
"""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")

# Query row 0 attends to no key, as the pad positions of a left-padded prompt do.
MASK = torch.tensor([[[[False, False, False], [True, True, False], [True, True, True]]]])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_mps_sdpa_returns_nan_for_a_query_row_that_attends_to_nothing(dtype):
    """The bug eager attention works around. If this fails, the pinned torch returns finite
    values here (as the CPU does) and ATTN_IMPLEMENTATION could go back to sdpa."""
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(1, 2, 3, 8, generator=g).to("mps", dtype) for _ in range(3))
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=MASK.to("mps"))
    assert torch.isnan(out.float()).any(dim=-1)[0, 0].tolist() == [True, False, False]
    on_cpu = F.scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu(), attn_mask=MASK)
    assert torch.isfinite(on_cpu.float()).all()


@pytest.fixture(scope="module")
def tiny_qwen2(tmp_path_factory):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    out = tmp_path_factory.mktemp("tiny_qwen2_mps")
    cfg = Qwen2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    Qwen2ForCausalLM(cfg).save_pretrained(out)
    return out


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_left_padded_batch_on_mps_needs_eager_attention(tiny_qwen2, dtype):
    from transformers import AutoModelForCausalLM

    ids = torch.randint(0, 128, (2, 6), generator=torch.Generator().manual_seed(1))
    attention_mask = torch.ones_like(ids)
    attention_mask[1, :3] = 0  # row 1 is left-padded by three positions

    def last_logits(impl: str) -> torch.Tensor:
        model = AutoModelForCausalLM.from_pretrained(
            tiny_qwen2, torch_dtype=getattr(torch, dtype), attn_implementation=impl
        ).to("mps")
        model.eval()
        with torch.no_grad():
            out = model(input_ids=ids.to("mps"), attention_mask=attention_mask.to("mps"))
        return out.logits[:, -1].float().cpu()

    sdpa, eager = last_logits("sdpa"), last_logits("eager")
    assert torch.isnan(sdpa[1]).any() and torch.isfinite(sdpa[0]).all()
    assert torch.isfinite(eager).all()
    tol = 1e-4 if dtype == "float32" else 5e-2
    torch.testing.assert_close(eager[0], sdpa[0], rtol=tol, atol=tol)  # the unpadded row
