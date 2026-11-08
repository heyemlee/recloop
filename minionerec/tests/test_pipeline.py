"""End to end on the CPU with a tiny random-weight Qwen2 and the checkpoint's real tokenizer.

The patched evaluate.py runs under the pinned transformers, the constrained beam search must
return only catalog SIDs (calc.py's CC = 0, the check MiniOneRec's README asks for), and the
outputs have the shapes the scorer expects. Runs in seconds; MPS is not needed (inside the
Claude Code sandbox torch sees no MPS device and evaluate.py falls back to the CPU).
"""

import json
import shutil

import numpy as np
import pytest

from minionerec import generate, upstream
from ml import paths

TOKENIZER_FILES = (
    "added_tokens.json",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)

pytestmark = pytest.mark.skipif(
    not (paths.MINIONEREC / "evaluate.py").exists()
    or not all((paths.HF_CKPT / f).exists() for f in TOKENIZER_FILES),
    reason="needs third_party/MiniOneRec and the checkpoint's tokenizer (make data-weights)",
)


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    out = tmp_path_factory.mktemp("tiny_qwen2")
    real = json.loads((paths.HF_CKPT / "config.json").read_text())
    cfg = Qwen2Config(
        vocab_size=real["vocab_size"],
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=512,
        tie_word_embeddings=True,
        bos_token_id=real["bos_token_id"],
        eos_token_id=real["eos_token_id"],
    )
    torch.manual_seed(0)
    Qwen2ForCausalLM(cfg).save_pretrained(out)
    for name in TOKENIZER_FILES:
        shutil.copyfile(paths.HF_CKPT / name, out / name)
    return out


def test_patched_evaluate_runs_and_constrains_to_catalog_sids(tmp_path, tiny_model):
    script = upstream.stage(tmp_path / "stage")
    test_csv = paths.public_split_csv("test")
    rows = np.array([0, 1, 2])
    input_csv = generate.write_subset(test_csv, rows, tmp_path / "input.csv")
    cmd = upstream.evaluate_command(
        script,
        input_csv,
        tmp_path / "result.json",
        tmp_path / "timing.json",
        num_beams=10,
        batch_size=2,
        max_new_tokens=256,
        length_penalty=0.0,
        dtype="float32",
        base_model=tiny_model,
    )
    res = upstream.run_logged(cmd, script.parent, tmp_path / "evaluate.log")
    assert res["returncode"] == 0, (tmp_path / "evaluate.log").read_text()[-3000:]
    assert res["memory"]["lifetime_max_footprint"] > 0

    result = json.loads((tmp_path / "result.json").read_text())
    assert len(result) == len(rows)
    assert all(len(r["predict"]) == 10 and len(set(r["predict"])) == 10 for r in result)
    assert generate.predictions_valid(tmp_path / "result.json") == (True, 0)
    timing = json.loads((tmp_path / "timing.json").read_text())
    assert [b["size"] for b in timing["batches"]] == [2, 1]
    assert timing["dtype"] == "float32" and timing["setup_seconds"] > 0

    calc = upstream.run_calc(tmp_path / "result.json", tmp_path / "calc.txt")
    assert calc["invalid_predictions"] == 0
    assert len(calc["HR"]) == len(calc["NDCG"]) == 4  # K = 1, 3, 5, 10
