"""The patch applies to the pinned upstream files; subsets and workloads are deterministic."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from minionerec import cost, generate, upstream
from ml import paths

needs_clone = pytest.mark.skipif(
    not (paths.MINIONEREC / "evaluate.py").exists(),
    reason="third_party/MiniOneRec missing: run `make data` first",
)


@needs_clone
def test_patch_applies_to_the_pinned_evaluate_py(tmp_path):
    script = upstream.stage(tmp_path)
    assert upstream.sha256(script) == upstream.PATCHED_EVALUATE_SHA256
    text = script.read_text()
    assert "bitsandbytes" not in text
    assert 'device = "mps"' in text
    assert "torch_dtype=getattr(torch, dtype)" in text and 'device_map="auto"' not in text
    assert "attn_implementation=attn_implementation or None" in text
    assert "torch.mps.empty_cache()" in text
    # the generation itself is untouched
    original = (paths.MINIONEREC / "evaluate.py").read_text()
    start = original.index("    def evaluate(")
    end = original.index("    model = model.to(device)")
    assert original[start:end] in text
    for name in ("data.py", "LogitProcessor.py", "npu_support.py"):
        assert (tmp_path / name).read_bytes() == (paths.MINIONEREC / name).read_bytes()


@needs_clone
def test_stage_refuses_a_changed_upstream_file(tmp_path, monkeypatch):
    monkeypatch.setitem(upstream.UPSTREAM_SHA256, "data.py", "0" * 64)
    with pytest.raises(SystemExit, match="data.py"):
        upstream.stage(tmp_path)


def test_evaluate_command_uses_evaluate_sh_settings(tmp_path):
    cmd = upstream.evaluate_command(
        tmp_path / "evaluate.py",
        tmp_path / "in.csv",
        tmp_path / "out.json",
        tmp_path / "t.json",
        dtype="bfloat16",
        **upstream.EVALUATE_SH,
    )
    args = dict(zip(cmd[3::2], cmd[4::2], strict=True))
    assert args["--num_beams"] == "50" and args["--batch_size"] == "8"
    assert args["--max_new_tokens"] == "256" and args["--length_penalty"] == "0.0"
    assert args["--category"] == "Office_Products" and args["--dtype"] == "bfloat16"
    assert args["--attn_implementation"] == "eager"  # see upstream.ATTN_IMPLEMENTATION


def _toy_csv(path):
    df = pd.DataFrame(
        {
            "user_id": ["A1", "A1", "A2", "A3", "A3", "A3", "A4"],
            "history_item_title": ['["a, b"]', "['x']", "['y \"q\"']", "[]", "[]", "[]", "[]"],
            "item_sid": ["<a_1><b_1><c_1>"] * 7,
        }
    )
    df.to_csv(path, index=False)
    return df


def test_select_rows_takes_every_sample_of_the_chosen_users(tmp_path):
    csv = tmp_path / "test.csv"
    df = _toy_csv(csv)
    np.testing.assert_array_equal(generate.select_rows(csv, None), np.arange(len(df)))
    rows = generate.select_rows(csv, 2, seed=1)
    chosen = set(df.user_id.iloc[rows])
    assert len(chosen) == 2
    np.testing.assert_array_equal(rows, np.flatnonzero(df.user_id.isin(chosen)))
    np.testing.assert_array_equal(rows, generate.select_rows(csv, 2, seed=1))
    # nested: the users of a smaller subset are among those of a larger one (same seed)
    assert set(generate.select_rows(csv, 1, seed=1)) <= set(rows)
    assert set(rows) <= set(generate.select_rows(csv, 3, seed=1))


def test_write_subset_keeps_values_and_reuses_the_full_file(tmp_path):
    csv = tmp_path / "test.csv"
    df = _toy_csv(csv)
    assert generate.write_subset(csv, np.arange(len(df)), tmp_path / "x.csv") == csv
    out = generate.write_subset(csv, np.array([1, 2, 5]), tmp_path / "sub.csv")
    back = pd.read_csv(out, dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(
        back, df.astype(str).iloc[[1, 2, 5]].reset_index(drop=True), check_dtype=False
    )


def test_cost_workload_is_one_sample_per_user(tmp_path):
    csv = tmp_path / "test.csv"
    df = _toy_csv(csv)
    rows = cost.one_sample_per_user(csv, 3, seed=5)
    assert len(rows) == 3 and len(set(df.user_id.iloc[rows])) == 3
    assert list(rows) == sorted(rows)
    np.testing.assert_array_equal(rows, cost.one_sample_per_user(csv, 3, seed=5))
    names = [c["name"] for c in cost.configs()]
    assert len(names) == len(set(names)) == 9
    once = [c for c in cost.configs() if c.get("prefill") == "once"]
    assert [c["name"] for c in once] == ["beam50_batch1_once", "beam50_batch8_once"]
    cmd = cost.engine_command(
        once[0], tmp_path / "in.csv", tmp_path, SimpleNamespace(dtype="bfloat16", base_model="m")
    )
    assert cmd[1:5] == ["-u", "-m", "minionerec.replay", "csv"]  # the engine, not evaluate.py
    assert json.dumps(cost.configs())  # serializable into the manifest


@needs_clone
def test_predictions_valid_counts_unknown_sids(tmp_path):
    sid = paths.PUBLIC_INFO.read_text().splitlines()[0].split("\t")[0]
    result = tmp_path / "r.json"
    result.write_text(json.dumps([{"predict": [sid, '"' + sid + '"\n', "<a_1>"]}]))
    assert generate.predictions_valid(result) == (False, 1)


def test_shard_bounds_are_split_py_slices():
    bounds = upstream.shard_bounds(4866, 8)
    assert [hi - lo for lo, hi in bounds] == [608, 608, 608, 609, 608, 608, 608, 609]
    assert bounds[0][0] == 0 and bounds[-1][1] == 4866
    assert all(a[1] == b[0] for a, b in zip(bounds, bounds[1:], strict=False))
    assert upstream.shard_bounds(5, 1) == [(0, 5)]
