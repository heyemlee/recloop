"""minionerec.engine against the patched evaluate.py, on the CPU with a tiny random Qwen2.

The engine must generate exactly what evaluate.py generates for the same batches (its own
code path, prompt encoded per beam), sharing the prompt pass across the beams must not change a
single SID or beam score on the CPU, and prompts built from SID lists (the nearline worker) must
be the token ids EvalSidDataset builds from the CSV rows. Whether the GPU also computes the
shared pass alike is what `make minionerec-replay` checks on the real model.
"""

import ast
import functools
import json

import numpy as np
import pandas as pd
import pytest

from minionerec import engine, generate, replay, upstream
from minionerec.tests.test_pipeline import TOKENIZER_FILES, tiny_model  # noqa: F401 (fixture)
from ml import paths

pytestmark = pytest.mark.skipif(
    not (paths.MINIONEREC / "evaluate.py").exists()
    or not all((paths.HF_CKPT / f).exists() for f in TOKENIZER_FILES),
    reason="needs third_party/MiniOneRec and the checkpoint's tokenizer (make data-weights)",
)

ROWS = np.arange(6)
BEAMS = 10


@pytest.fixture(scope="module")
def subset_csv(tmp_path_factory):
    out = tmp_path_factory.mktemp("subset") / "input.csv"
    return generate.write_subset(paths.public_split_csv("test"), ROWS, out)


def _engine(model, prefill_once):
    return engine.Generator(model, num_beams=BEAMS, dtype="float32", prefill_once=prefill_once, device="cpu")


def test_engine_reproduces_evaluate_py(tmp_path, tiny_model, subset_csv):  # noqa: F811
    script = upstream.stage(tmp_path / "stage")
    cmd = upstream.evaluate_command(
        script,
        subset_csv,
        tmp_path / "result.json",
        tmp_path / "timing.json",
        num_beams=BEAMS,
        batch_size=2,
        max_new_tokens=256,
        length_penalty=0.0,
        dtype="float32",
        base_model=tiny_model,
    )
    res = upstream.run_logged(cmd, script.parent, tmp_path / "evaluate.log", sample_memory=False)
    assert res["returncode"] == 0, (tmp_path / "evaluate.log").read_text()[-3000:]
    theirs = json.loads((tmp_path / "result.json").read_text())
    for prefill_once in (False, True):
        rows, timing = replay.evaluate_csv(_engine(tiny_model, prefill_once), subset_csv, 2)
        assert [r["predict"] for r in rows] == [r["predict"] for r in theirs], prefill_once
        assert [r["output"] for r in rows] == [r["output"] for r in theirs]
        assert [b["size"] for b in timing] == [2, 2, 2]


def test_prompt_once_changes_no_sid_and_no_score(tiny_model, subset_csv):  # noqa: F811
    once, per_beam = _engine(tiny_model, True), _engine(tiny_model, False)
    ds = per_beam.dataset(subset_csv)
    enc = [ds[i] for i in range(len(ds))]
    a = once.generate_encoded(enc[:4])
    b = per_beam.generate_encoded(enc[:4])
    assert a.sids == b.sids
    assert a.scores == b.scores
    assert all(len(s) == BEAMS for s in a.sids)


def test_share_prompt_pass_runs_the_first_pass_on_one_copy(tiny_model, subset_csv):  # noqa: F811
    gen = _engine(tiny_model, False)
    original, seen = gen.model.forward, []

    @functools.wraps(original)  # keeps the signature generate() inspects
    def spy(*args, **kwargs):
        seen.append(kwargs["input_ids"].shape[0])
        return original(*args, **kwargs)

    gen.model.forward = spy
    engine.share_prompt_pass(gen.model, BEAMS)
    ds = gen.dataset(subset_csv)
    gen.generate_encoded([ds[0], ds[1]])
    assert seen[0] == 2  # the first pass: one row per prompt
    assert len(seen) > 1 and all(n == 2 * BEAMS for n in seen[1:])  # decoding: one per beam


def test_prompts_from_sid_lists_match_the_csv_path(tiny_model, subset_csv):  # noqa: F811
    gen = _engine(tiny_model, True)
    ds = gen.dataset(subset_csv)
    from_csv = [ds[i]["input_ids"] for i in range(len(ds))]
    histories = [
        ast.literal_eval(h) for h in pd.read_csv(subset_csv).history_item_sid.tolist()
    ]
    from_lists = [e["input_ids"] for e in gen.encode_histories(histories)]
    assert from_lists == from_csv
