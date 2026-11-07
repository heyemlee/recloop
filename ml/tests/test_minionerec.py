"""MiniOneRec scoring: SIDs to ranks, segments, calc.py agreement, the report (synthetic data)."""

import json
import math

import numpy as np
import pytest

from ml import paths
from ml.eval import bootstrap, calc_check, metrics, minionerec, minionerec_report
from ml.eval.splits import Split

# Twelve items; items 1 and 3 share a SID (like the 15 duplicate listings in the catalog).
SIDS = [f"<a_{i}><b_{i}><c_{i}>" for i in range(12)]
SIDS[3] = SIDS[1]
# popularity order: item 3 first, then 0, 1, 4, 2, 5, 6, ...
TIE_ORDER = np.array([3, 0, 1, 4, 2, 5, 6, 7, 8, 9, 10, 11])
BEAMS = 10


def test_sid_items_lists_shared_sids_in_tie_order():
    by_sid = minionerec.sid_items(SIDS, TIE_ORDER)
    assert by_sid["<a_1><b_1><c_1>"] == [3, 1]
    assert by_sid["<a_0><b_0><c_0>"] == [0]


def test_generated_ranks_under_both_rules():
    by_sid = minionerec.sid_items(SIDS, TIE_ORDER)
    s = SIDS
    predict = [
        [s[2], s[1], s[0]],  # target 1: SID at beam 2; items 2, 3, 1 -> item rank 3
        [s[2], s[1], s[0]],  # target 3: shares the SID; item rank 2
        ['"' + s[4] + '"\n', s[0]],  # target 4: calc.py's normalization; rank 1 / 1
        ["<a_99><b_99><c_99>", s[0], s[0]],  # target 0: unknown SID keeps its beam; repeat
        [s[2]],  # target 4: missing
    ]
    target = np.array([1, 3, 4, 0, 4])
    r = minionerec.generated_ranks(predict, target, SIDS, by_sid)
    np.testing.assert_array_equal(r["sid"], [2, 2, 1, 2, minionerec.MISS])
    np.testing.assert_array_equal(r["item"], [3, 2, 1, 1, minionerec.MISS])


def _calc_py_formula(predict, target_sids, ks):
    """calc.py's loop, transcribed (minID = first beam equal to the target)."""
    hr, ndcg = np.zeros(len(ks)), np.zeros(len(ks))
    for beams, t in zip(predict, target_sids, strict=True):
        beams = [b.strip('"\n').strip() for b in beams]
        min_id = next((i for i, b in enumerate(beams) if b == t), 1_000_000)
        for j, k in enumerate(ks):
            if min_id < k:
                hr[j] += 1
                ndcg[j] += 1 / math.log(min_id + 2)
    return hr / len(predict), ndcg / len(predict) / (1 / math.log(2))


def test_sid_rule_equals_calc_py_formula():
    rng = np.random.default_rng(3)
    by_sid = minionerec.sid_items(SIDS, TIE_ORDER)
    uniq = sorted(set(SIDS))
    predict = [list(rng.permutation(uniq)[: rng.integers(1, 8)]) for _ in range(200)]
    target = rng.integers(0, len(SIDS), 200)
    ranks = minionerec.generated_ranks(predict, target, SIDS, by_sid)["sid"]
    ks = (1, 3)
    ours = metrics.summarize(metrics.per_sample(ranks, ks))
    hr, ndcg = _calc_py_formula(predict, [SIDS[t] for t in target], ks)
    for j, k in enumerate(ks):
        assert ours[f"HR@{k}"] == pytest.approx(hr[j], abs=1e-12)
        assert ours[f"NDCG@{k}"] == pytest.approx(ndcg[j], abs=1e-12)


def test_head_items_includes_ties_at_the_threshold():
    counts = np.array([9.0, 5, 5, 5, 1, 0, 7, 2, 2, 3])
    order = np.lexsort((np.arange(10), -counts))
    head, threshold = minionerec.head_items(counts, order, share=0.2)  # 2 items -> count >= 7
    assert threshold == 7 and head.tolist() == [c >= 7 for c in counts]
    head, threshold = minionerec.head_items(counts, order, share=0.3)  # 3rd item has 5: ties in
    assert threshold == 5 and head.sum() == 5


def _toy_split():
    history = [
        np.array([0]),
        np.array([0, 1, 2, 3]),
        np.array([4, 4]),
        np.array([1, 2, 3, 0, 1]),
        np.array([2]),
        np.array([3, 1]),
    ]
    return Split("test", np.array([10, 10, 11, 12, 12, 13]), np.array([1, 2, 4, 0, 3, 3]), history)


def test_segment_masks():
    test = _toy_split()
    counts = np.array([5.0, 9, 1, 3, 0])
    order = np.lexsort((np.arange(5), -counts))
    cats = np.array(["A", "B", "A", "unknown", "C"], dtype=object)
    same_day = np.array([True, False, True, False, True, False])
    masks, info = minionerec.segment_masks(test, counts, order, cats, same_day)
    assert masks["history"]["1-3 items"].tolist() == [True, False, True, False, True, True]
    # head = top ceil(0.2 * 5) = 1 item: item 1 (9 interactions)
    assert masks["popularity"]["head"].tolist() == [True, False, False, False, False, False]
    # target category not among the history's categories
    assert masks["category"]["cross-category"].tolist() == [True, False, False, False, True, False]
    assert masks["day"]["later day"].tolist() == (~same_day).tolist()
    assert info["targets_never_seen_in_training"] == 1


def _beams(*items):
    """A full list of BEAMS distinct SIDs that starts with the given items' SIDs."""
    head = [SIDS[i] for i in items]
    rest = [s for s in dict.fromkeys(SIDS) if s not in head]
    return (head + rest)[:BEAMS]


def _write_run(run_dir, rows, predict, test, beams=BEAMS):
    run_dir.mkdir(parents=True)
    (run_dir / "rows.json").write_text(json.dumps(rows))
    result = [
        {"input": "...", "output": SIDS[test.target[r]] + "\n", "predict": p}
        for r, p in zip(rows, predict, strict=True)
    ]
    (run_dir / "result.json").write_text(json.dumps(result))
    batches = [{"size": 2, "seconds": 1.5, "mps_driver_allocated": 2**30}] * 3
    (run_dir / "timing.json").write_text(json.dumps({"setup_seconds": 9.0, "batches": batches}))
    meta = {
        "checkpoint": minionerec.CHECKPOINT + " (pinned)",
        "selection": "all",
        "settings": {
            "num_beams": beams,
            "batch_size": 2,
            "length_penalty": 0.0,
            "max_new_tokens": 256,
            "dtype": "bfloat16",
        },
        "versions": {"torch": "2.6.0", "transformers": "4.57.1"},
        "machine": {
            "chip": "Apple M4 Pro",
            "gpu_cores": 16,
            "memory_gib": 24.0,
            "os": "macOS",
            "performance_cores": "8",
            "efficiency_cores": "4",
        },
        "started": "2026-11-07T01:00:00-0800",
        "wall_seconds": 30.0,
        "setup_seconds": 9.0,
        "generate_seconds": 4.5,
        "loadavg_start": [1.0, 1.0, 1.0],
        "memory": {"lifetime_max_footprint": 5 * 2**30},
    }
    (run_dir / "run.json").write_text(json.dumps(meta))


def _toy_context(test):
    n = len(test)
    rng = np.random.default_rng(0)
    baselines = {}
    for name in minionerec.BASELINES:
        ranks = rng.integers(1, 6, n)
        baselines[name] = {rule: metrics.per_sample(ranks) for rule in ("item", "sid")}
    counts = np.array([5.0, 9, 1, 3, 0, 1, 1, 1, 1, 1, 1, 1])
    return minionerec.Context(
        test=test,
        tie_order=TIE_ORDER,
        counts=counts,
        sids=SIDS,
        by_sid=minionerec.sid_items(SIDS, TIE_ORDER),
        baselines=baselines,
        descriptions={name: name for name in minionerec.BASELINES},
    )


def _calc_from_formula(run_dir):
    result = json.loads((run_dir / "result.json").read_text())
    predict = [r["predict"] for r in result]
    target = [r["output"].strip() for r in result]
    ks = [k for k in metrics.CALC_KS if k <= len(predict[0])]
    hr, ndcg = _calc_py_formula(predict, target, ks)
    return {"HR": hr.tolist(), "NDCG": ndcg.tolist(), "invalid_predictions": 0}


def test_build_and_render_end_to_end(tmp_path):
    test = _toy_split()
    predict = [_beams(1, 0), _beams(0, 2), _beams(4), _beams(2, 1), _beams(1, 0), _beams(2, 0)]
    _write_run(tmp_path / "runs" / "test", list(range(len(test))), predict, test)
    ctx = _toy_context(test)
    cats = np.array(["A", "B", "A", "unknown", "C"] + ["D"] * 7, dtype=object)
    same_day = np.array([True, False, True, False, True, False])
    out = minionerec.build(
        minionerec.finished_runs(tmp_path / "runs"),
        "test",
        ctx,
        None,
        cats,
        same_day,
        calc_fn=_calc_from_formula,
    )
    main = out["runs"]["test"]
    assert out["main_run"] == "test" and out["status"] == "final"
    assert main["calc_py"]["ok"]
    item = main["comparison"]["models"]["MiniOneRec"]["item"]
    expected = minionerec.generated_ranks(predict, test.target, SIDS, ctx.by_sid)["item"]
    assert item["HR@3"]["value"] == pytest.approx(np.mean(expected <= 3))
    assert set(main["segments"]) == {"history", "popularity", "category", "day"}
    # the contrast is MiniOneRec - SASRec in the first group minus the same in the second
    day = main["segments"]["day"]
    assert day["contrast"]["groups"] == ["same day", "later day"]
    diff = ctx.baselines["SASRec"]["item"]["HR@10"] * -1.0 + (expected <= 10)
    want = diff[same_day].mean() - diff[~same_day].mean()
    got = day["contrast"]["pairs"]["MiniOneRec - SASRec"]["HR@10"]
    assert got["value"] == pytest.approx(want)
    assert got["ci"][0] <= got["value"] <= got["ci"][1]
    text = minionerec_report.render(out)
    assert text == minionerec_report.render(json.loads(json.dumps(out)))  # deterministic
    for heading in ("## Reproduction", "## Compared with", "## Which users", "## Cost", "## Runs"):
        assert heading in text
    assert "Preliminary" not in text
    assert "Findings (HR@10" in text and "Between the two groups" in text


def test_group_contrast():
    users = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    res = bootstrap.Resampler.build(users, b=200, seed=1)
    a = np.array([True, False] * 4)
    # the same difference in both groups for every user: the contrast is exactly 0
    same = np.array([0.5, 0.5, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0])
    c = minionerec.group_contrast(same, a, ~a, res)
    assert c["value"] == 0 and c["ci"] == (0.0, 0.0) and c["replicates"] == 200
    # +1 in group a, 0 in group b for everyone: contrast 1 with no spread
    c = minionerec.group_contrast(a.astype(float), a, ~a, res)
    assert c["value"] == 1 and c["ci"] == (1.0, 1.0)
    # a group that some replicates never draw: those replicates are dropped
    only_user_0 = users == 0
    c = minionerec.group_contrast(np.ones(8), only_user_0, ~only_user_0, res)
    assert 0 < c["replicates"] < 200 and c["value"] == 0


def test_subset_run_is_preliminary(tmp_path):
    test = _toy_split()
    rows = [0, 1, 2]
    _write_run(tmp_path / "runs" / "trial", rows, [_beams(1, 0)] * 3, test)
    meta = json.loads((tmp_path / "runs" / "trial" / "run.json").read_text())
    meta["selection"] = {"users": 2, "seed": 1}
    (tmp_path / "runs" / "trial" / "run.json").write_text(json.dumps(meta))
    out = minionerec.build(
        minionerec.finished_runs(tmp_path / "runs"),
        "test",
        _toy_context(test),
        None,
        np.array(["A"] * 12, dtype=object),
        np.ones(len(test), dtype=bool),
        calc_fn=_calc_from_formula,
    )
    assert out["main_run"] == "trial" and out["status"] == "preliminary"
    text = minionerec_report.render(out)
    assert "**Preliminary.**" in text
    # 6 samples at 1.5 s each, plus 8 processes of (30 - 4.5) s start-up each
    assert "the full test split (6 samples as evaluate.sh's 8 shards, 26 s of start-up" in text
    assert "takes about 0.1 h." in text


def test_cost_summary_leaves_out_warmup_calls(tmp_path):
    cost = tmp_path / "cost"
    (cost / "beam10_batch1").mkdir(parents=True)
    batches = [{"size": 1, "seconds": s, "mps_driver_allocated": 3} for s in (9.0, 5.0, 1.0, 2.0)]
    (cost / "beam10_batch1" / "timing.json").write_text(json.dumps({"batches": batches}))
    manifest = {
        "checkpoint": minionerec.CHECKPOINT,
        "users": 4,
        "dtype": "bfloat16",
        "versions": {},
        "machine": {},
        "configs": {
            "beam10_batch1": {
                "num_beams": 10,
                "batch_size": 1,
                "max_new_tokens": 256,
                "samples": 4,
                "setup_seconds": 7.0,
                "memory": {"lifetime_max_footprint": 100},
                "loadavg_start": [0.5, 0, 0],
                "loadavg_end": [0.7, 0, 0],
                "invalid_predictions": 0,
                "quiet": {
                    "attempts": [
                        {
                            "waited_seconds": 0.0,
                            "start": {"load1": 0.5, "containers": []},
                            "containers_during": ["aihd-test-pg"],
                        },
                        {
                            "waited_seconds": 30.0,
                            "start": {"load1": 0.5, "containers": []},
                            "containers_during": ["aihd-test-kafka"],
                        },
                    ],
                    "quiet_start": True,
                    "disturbed": True,
                },
            }
        },
    }
    (cost / "manifest.json").write_text(json.dumps(manifest))
    out = minionerec.cost_summary(cost, tmp_path / "none.json")
    c = out["configs"]["beam10_batch1"]
    assert c["warmup_calls"] == 2 and c["per_call"]["calls"] == 2
    assert c["per_call"]["p50"] == pytest.approx(1.5)
    assert c["samples_per_second"] == pytest.approx(2 / 3)
    assert c["loadavg_end"] == 0.7 and c["quiet"]["disturbed"]
    assert minionerec_report._disturbed(c) and not minionerec_report._busy(c)
    notes = "\n".join(minionerec_report._quiet_notes(out))
    assert "`beam10_batch1` was run a second time" in notes
    assert "aihd-test-pg" in notes and "aihd-test-kafka" in notes and "(†)" in notes


@pytest.mark.skipif(
    not calc_check.CALC_PY.exists() or not paths.PUBLIC_INFO.exists(),
    reason="third_party/MiniOneRec missing: run `make data` first",
)
def test_real_calc_py_agrees_with_our_sid_rule(tmp_path):
    from ml.eval.splits import item_sids

    sids = item_sids()
    rng = np.random.default_rng(11)
    n, beams = 60, 50
    targets = rng.integers(0, len(sids), n)
    predict = []
    for t in targets:
        beam_items = rng.permutation(len(sids))[:beams]
        if rng.random() < 0.5:
            beam_items[rng.integers(0, beams)] = t
        predict.append([sids[i] for i in beam_items])
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            [
                {"input": "", "output": sids[t] + "\n", "predict": p}
                for t, p in zip(targets, predict, strict=True)
            ]  # fmt: skip
        )
    )
    calc = calc_check.run_calc_py(result)
    order = np.arange(len(sids))
    ranks = minionerec.generated_ranks(predict, targets, sids, minionerec.sid_items(sids, order))
    ours = metrics.summarize(metrics.per_sample(ranks["sid"], metrics.CALC_KS))
    check = minionerec.calc_cross_check(
        {"calc_ks": list(metrics.CALC_KS), "sid_calc_ks": ours}, calc
    )
    assert check["ok"], check["rows"]


def test_agreement_pairs_runs_on_shared_samples(tmp_path):
    test = _toy_split()
    full = [_beams(1, 0), _beams(0, 2), _beams(4), _beams(2, 1), _beams(1, 0), _beams(2, 0)]
    _write_run(tmp_path / "runs" / "test", list(range(len(test))), full, test)
    # rows 1 and 4, in another order; row 4 gets the same list, row 1 one SID that the main
    # run's top 10 does not have, in front of the main run's first nine
    other_row1 = [SIDS[11]] + full[1][:9]
    _write_run(tmp_path / "runs" / "beam10", [4, 1], [full[4], other_row1], test)
    out = minionerec.build(
        minionerec.finished_runs(tmp_path / "runs"),
        "test",
        _toy_context(test),
        None,
        np.array(["A"] * 12, dtype=object),
        np.ones(len(test), dtype=bool),
        calc_fn=_calc_from_formula,
    )
    (a,) = out["agreement"]
    assert a["run"] == "beam10" and a["reference"] == "test" and a["samples"] == 2
    assert a["identical_top10"] == pytest.approx(0.5)
    assert a["top10_overlap"] == pytest.approx((1.0 + 0.9) / 2)
    # row 1's target (item 2) moves from rank 2 to 3: a hit at 10 in both runs
    assert a["difference"]["HR@10"]["value"] == pytest.approx(0.0)
    assert a["difference"]["NDCG@10"]["value"] < 0
    assert "## Other settings on the same samples" in minionerec_report.render(out)


def test_runs_differing_in_one_setting_are_compared_with_each_other(tmp_path):
    test = _toy_split()
    full = [_beams(1, 0), _beams(0, 2), _beams(4), _beams(2, 1), _beams(1, 0), _beams(2, 0)]
    _write_run(tmp_path / "runs" / "test", list(range(len(test))), full, test)
    for name, dtype in (("bf16-b4", "bfloat16"), ("fp32-b4", "float32")):
        _write_run(tmp_path / "runs" / name, [0, 1], full[:2], test)
        meta_path = tmp_path / "runs" / name / "run.json"
        meta = json.loads(meta_path.read_text())
        meta["settings"].update(batch_size=4, dtype=dtype)
        meta_path.write_text(json.dumps(meta))
    gens = {
        p.name: minionerec.load_generation(p) for p in minionerec.finished_runs(tmp_path / "runs")
    }
    pairs = minionerec.comparisons(gens, "test")
    # both against the main run, and float32 against bfloat16 (the main run's dtype)
    assert sorted(pairs) == [("bf16-b4", "fp32-b4"), ("test", "bf16-b4"), ("test", "fp32-b4")]
    out = minionerec.build(
        minionerec.finished_runs(tmp_path / "runs"),
        "test",
        _toy_context(test),
        None,
        np.array(["A"] * 12, dtype=object),
        np.ones(len(test), dtype=bool),
        calc_fn=_calc_from_formula,
    )
    dtype_pair = next(a for a in out["agreement"] if a["reference"] == "bf16-b4")
    assert dtype_pair["run"] == "fp32-b4" and dtype_pair["identical_top10"] == 1.0
    assert "| fp32-b4 | 10 / 4 / float32 | bf16-b4 |" in minionerec_report.render(out)


def test_smoke_runs_are_left_out(tmp_path):
    test = _toy_split()
    _write_run(tmp_path / "runs" / "real", [0, 1], [_beams(1)] * 2, test)
    _write_run(tmp_path / "runs" / "smoke", [0, 1], [_beams(1)] * 2, test)
    meta = json.loads((tmp_path / "runs" / "smoke" / "run.json").read_text())
    meta["checkpoint"] = "/tmp/tiny_qwen2"
    (tmp_path / "runs" / "smoke" / "run.json").write_text(json.dumps(meta))
    assert [p.name for p in minionerec.finished_runs(tmp_path / "runs")] == ["real"]
    names = [p.name for p in minionerec.finished_runs(tmp_path / "runs", include_test_models=True)]
    assert names == ["real", "smoke"]


def _write_cost(cost, sasrec_json, disturbed="beam50_batch8"):
    """A cost directory like minionerec.cost writes, and a sasrec_cost.json."""
    configs = {}
    for beams in (10, 20, 50):
        for batch in (1, 8):
            configs[f"beam{beams}_batch{batch}"] = (beams, batch, 256, 4)
    configs["beam50_batch1_prefill"] = (50, 1, 1, 2)
    manifest = {
        "checkpoint": minionerec.CHECKPOINT,
        "users": 4,
        "dtype": "bfloat16",
        "versions": {},
        "machine": {
            "chip": "Apple M4 Pro",
            "gpu_cores": 16,
            "performance_cores": "8",
            "efficiency_cores": "4",
        },
        "configs": {},
    }
    for name, (beams, batch, new_tokens, samples) in configs.items():
        (cost / name).mkdir(parents=True)
        per_call = beams / 10 * (0.2 if new_tokens == 1 else 1.0) * batch
        calls = max(1, samples // batch) + 1
        timing = {"batches": [{"size": batch, "seconds": per_call} for _ in range(calls)]}
        (cost / name / "timing.json").write_text(json.dumps(timing))
        attempts = [{"waited_seconds": 0.0, "start": {"load1": 1.2, "containers": []}}]
        attempts[0]["containers_during"] = ["aihd-test-pg"] if name == disturbed else []
        if name == disturbed:
            attempts.append(
                {
                    "waited_seconds": 40.0,
                    "start": {"load1": 1.5, "containers": []},
                    "containers_during": [],
                }
            )
        manifest["configs"][name] = {
            "num_beams": beams,
            "batch_size": batch,
            "max_new_tokens": new_tokens,
            "samples": samples,
            "setup_seconds": 5.0,
            "device": "mps",
            "memory": {"lifetime_max_footprint": 6 * 2**30},
            "loadavg_start": [1.2, 0, 0],
            "loadavg_end": [1.8, 0, 0],
            "invalid_predictions": 0 if new_tokens > 1 else None,
            "swapped_out_bytes": 5 * 2**20 if name == disturbed else 0,
            "quiet": {"attempts": attempts, "quiet_start": True, "disturbed": False},
        }
    (cost / "manifest.json").write_text(json.dumps(manifest))
    runs = [
        {"batch_size": 1, "threads": 1, "seconds": [2e-4] * 10},
        {"batch_size": 1, "threads": 12, "seconds": [3e-4] * 10},
        {"batch_size": 256, "threads": 12, "seconds": [0.016] * 3},
    ]
    sasrec_json.write_text(
        json.dumps(
            {
                "footprint": {"current": 2**30, "peak": 2**30},
                "loadavg_start": [1.1, 0, 0],
                "loadavg_end": [1.3, 0, 0],
                "quiet": {
                    "attempts": [
                        {
                            "waited_seconds": 0.0,
                            "start": {"load1": 1.1, "containers": []},
                            "containers_during": [],
                        }
                    ],
                    "quiet_start": True,
                    "disturbed": False,
                },
                "runs": runs,
            }
        )
    )


def test_cost_section_renders_load_and_reruns(tmp_path):
    test = _toy_split()
    _write_run(tmp_path / "runs" / "test", list(range(len(test))), [_beams(1, 0)] * 6, test)
    meta_path = tmp_path / "runs" / "test" / "run.json"
    meta_path.write_text(json.dumps(dict(json.loads(meta_path.read_text()), swapped_out_bytes=0)))
    _write_cost(tmp_path / "cost", tmp_path / "sasrec_cost.json")
    cost = minionerec.cost_summary(tmp_path / "cost", tmp_path / "sasrec_cost.json")
    out = minionerec.build(
        minionerec.finished_runs(tmp_path / "runs"),
        "test",
        _toy_context(test),
        cost,
        np.array(["A"] * 12, dtype=object),
        np.ones(len(test), dtype=bool),
        calc_fn=_calc_from_formula,
    )
    text = minionerec_report.render(out)
    assert "| Load, start / end |" in text and "| 1.2 / 1.8 |" in text and "| 1.1 / 1.3 |" in text
    assert "`beam50_batch8` was run a second time" in text and "(aihd-test-pg)." in text
    assert "†" not in text  # the second attempt was quiet
    assert "**Where the time goes.**" in text and "**Against SASRec.**" in text
    assert "Preliminary timings" not in text
    assert "Nothing was swapped out while it ran" in text  # the main run
    assert "`beam50_batch8`: 5 MiB swapped out while it ran" in text
