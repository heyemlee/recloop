"""loadtest/report.py renders the same bytes from the same results and applies the SLO rule;
loadtest/run.py plans every repeat with its own configuration and resumes a results file."""

import copy
import json

import pytest

from loadtest import report, run


def fake_run(
    experiment,
    rate,
    mode="design",
    cache="on",
    fault="none",
    rep=1,
    p99=20.0,
    degraded=0.0,
    errors=0.0,
    breaker="time:10s:20:10",
    proxied=False,
):
    cfg = {
        "resilience": "naive" if mode == "naive" else "full",
        "cache": cache,
        "breaker": breaker,
        "proxied": proxied,
    }
    path = "toxiproxy" if proxied else "direct"
    return {
        "key": f"{experiment}/{rate}/{mode}/cache-{cache}/{breaker}/{path}/{fault}",
        "experiment": experiment,
        "rate": rate,
        "mode": mode,
        "config": cfg,
        "fault": fault,
        "rep": rep,
        "started": 1_794_000_000.0 + rep,
        "seconds": 60.0,
        "consumers_caught_up_s": 0.5,
        "load_avg_before": [1.0, 1.0, 1.0],
        "load_avg_after": [1.2, 1.0, 1.0],
        "k6": {
            "recs": {
                "med": p99 / 4,
                "p(95)": p99 / 2,
                "p(99)": p99 + rep,
                "p(99.9)": p99 * 2,
                "max": p99 * 3,
                "count": rate * 60,
            },
            "home": {"p(99)": p99},
            "item": {},
            "next_page": {},
            "events": {"p(99)": 3.0},
            "achieved_rps": float(rate),
            "errors": errors,
            "event_errors": 0.0,
            "degraded": degraded,
            "cache_hit_home": 0.6 if cache == "on" else None,
            "cache_hit_item": 0.4 if cache == "on" else None,
            "dropped_iterations": 0,
            "clicks": 10,
        },
        "server": {
            "p50_ms": p99 / 5,
            "p99_ms": p99 - 1,
            "stage_p99_ms": {"recall": 4.0, "user_context": 1.0},
            "degraded_reasons": {"sasrec_timeout": 12.0} if degraded else {},
            "breaker_transitions": {},
            "model_server_calls": {"OK": 60.0 * rate},
            "model_server_aborts": {},
            "model_server_injected": 0.0,
            "consumer_lag_max": {"aggregator": 3.0, "log-sink": 100.0},
        },
        "containers_mid_run": {"api": {"cpu_pct": 50.0, "mem": "80MiB"}},
    }


def results():
    runs = []
    for rep in (1, 2, 3):
        runs += [
            fake_run("ladder", 100, rep=rep, p99=10),
            fake_run("ladder", 200, rep=rep, p99=30),
            fake_run("ladder", 400, rep=rep, p99=60, degraded=0.02),  # misses the SLO
            fake_run("cache", 200, cache="on", rep=rep, p99=15),
            fake_run("cache", 200, cache="off", rep=rep, p99=25),
            fake_run("cache", 200, cache="on", rep=rep, p99=18, proxied=True),
            fake_run("faults", 100, mode="naive", cache="off", fault="ms_slow", rep=rep, p99=230),
            fake_run("faults", 100, cache="off", fault="ms_slow", rep=rep, p99=45, degraded=0.3),
            fake_run(
                "faults",
                100,
                mode="naive",
                cache="off",
                fault="redis_down",
                rep=rep,
                p99=120,
                errors=1.0,
            ),
            fake_run("faults", 100, cache="off", fault="redis_down", rep=rep, p99=12, degraded=1.0),
        ]
    for rep in (1, 2, 3):
        runs.append(
            {
                "key": "onset/100/design/cache-off/time:10s:20:10/toxiproxy/ms_down",
                "experiment": "onset",
                "rate": 100,
                "mode": "design",
                "config": {
                    "resilience": "full",
                    "cache": "off",
                    "breaker": "time:10s:20:10",
                    "proxied": True,
                },
                "fault": "ms_down",
                "rep": rep,
                "started": 1_794_000_100.0 + rep,
                "open_after_ms": 40.0 + rep,
                "failed_before_open": 10.0,
                "closed_after_recovery_s": 8.5,
                "k6": {},
            }
        )
    meta = {
        "plan": "full",
        "onset": [5.0, 15.0, 15.0], "repeats": 3, "duration": 60, "warmup": 10, "started": 1_794_000_000,
        "load_avg_end": [1.1, 1.0, 1.0],
        "environment": {
            "host": {"cpu": "Apple M4 Pro", "cores": 12, "memory_gb": 24, "os": "macOS 26"},
            "docker": {"server": "29.7.2", "vm_cpus": 8, "vm_memory_gb": 8.0, "other_containers_running": 0},
            "limits": {"api": {"cpus": 2.0, "memory_mb": 512}},
            "images": {},
            "k6": "k6 v2.3.0",
            "world": "abc",
            "load_avg_start": [1.0, 1.0, 1.0],
        },
    }  # fmt: skip
    return {"meta": meta, "runs": runs}


def test_report_is_deterministic_and_applies_the_slo():
    data = results()
    a = report.render(copy.deepcopy(data))
    b = report.render(copy.deepcopy(data))
    assert a == b
    assert "**200 requests/s** is the largest offered load" in a
    assert "| 400 ✗ |" in a  # 2 % degraded misses the SLO although P99 is fine
    assert "Preliminary" not in a  # 8 CPUs, idle, nothing else running
    assert "| model server: 30 % of calls +200 ms | naive | " in a
    assert "| Redis disconnected | this design |" in a
    assert "| on, through Toxiproxy |" in a
    assert "xychart-beta" in a
    assert "| model server disconnected | 42.0 [41.0–43.0] | 10 | 8.5 |" in a
    # the median of the three runs (p99 + rep: 31, 32, 33 -> 32) and the range
    assert "| 32.0 [31.0–33.0] |" in a


def test_banner_names_the_deviations():
    data = results()
    data["meta"]["environment"]["docker"]["vm_cpus"] = 12
    data["meta"]["environment"]["docker"]["other_containers_running"] = 5
    text = report.render(data)
    assert "**Preliminary.**" in text and "12 CPUs" in text and "5 containers" in text


def test_banner_names_active_campaigns():
    data = results()
    assert "campaigns were active" not in report.render(data)  # recorded empty, or not at all
    data["meta"]["active_campaigns"] = [81, 82]
    text = report.render(data)
    assert "**Preliminary.**" in text and "2 campaigns were active" in text


def test_every_repeat_of_an_experiment_uses_its_own_configuration():
    for plan in run.PLANS.values():
        runs = plan.runs()
        assert len({(r.key, r.rep) for r in runs}) == len(runs)
        for r in runs:
            if r.experiment == "ladder":  # the design's defaults in every repeat
                assert r.config == run.ApiConfig(), (r.key, r.rep)
            if r.experiment in ("faults", "onset"):
                assert r.config.proxied and r.config.cache == "off", (r.key, r.rep)


def test_resume_keeps_the_planned_runs_and_discards_the_rest(tmp_path, capsys):
    plan = run.PLANS["smoke"]
    runs = plan.runs()
    data = results()
    first = next(r for r in runs if r.experiment == "ladder")  # the smoke plan's first ladder run
    good = fake_run("ladder", first.rate)  # as the runner keys it
    assert (good["key"], good["rep"]) == (first.key, first.rep)
    stray = fake_run("ladder", first.rate, cache="off", proxied=True, rep=2)
    data["runs"] = [good, stray]
    out = tmp_path / "results.json"
    out.write_text(json.dumps(data))
    fresh = copy.deepcopy(data["meta"])
    fresh["environment"]["load_avg_start"] = [1.5, 1.2, 1.1]
    meta, kept, done = run.resume(out, fresh, runs, "runner fix", 30)
    assert kept == [good] and done == {(good["key"], 1)}
    assert meta["started"] == data["meta"]["started"]  # the first start's metadata stays
    assert meta["resumed"][0]["discarded"] == [f"{stray['key']}#2"]
    (gone,) = json.loads((tmp_path / "results-discarded.json").read_text())["runs"]
    assert gone["key"] == stray["key"] and gone["discarded_reason"] == "runner fix"
    text = report.render({"meta": meta, "runs": results()["runs"]})
    assert "- Resumed " in text and "runner fix. 1 runs kept, 1 discarded" in text
    assert "Preliminary" not in text
    meta["resumed"][0]["load_avg_start"] = [3.5, 2.0, 1.0]
    assert "load average was 3.5 before the runs" in report.render(
        {"meta": meta, "runs": results()["runs"]}
    )
    # --redo: a finished, planned run is discarded too and runs again; earlier discards stay
    out.write_text(json.dumps({"meta": meta, "runs": kept}))
    meta2, kept2, done2 = run.resume(out, fresh, runs, "redo", 5, [f"{good['key']}#1"])
    assert kept2 == [] and done2 == set() and len(meta2["resumed"]) == 2
    reasons = [
        r["discarded_reason"]
        for r in json.loads((tmp_path / "results-discarded.json").read_text())["runs"]
    ]
    assert reasons == ["runner fix", "redo"]
    with pytest.raises(SystemExit):
        run.resume(out, fresh, runs, "redo", 5, ["no/such/run#1"])
    capsys.readouterr()


def test_notes_and_the_appendix_cover_every_run():
    data = results()
    data["meta"]["notes"] = ["An earlier start was stopped; none of its runs are used."]
    text = report.render(data)
    assert "- An earlier start was stopped; none of its runs are used." in text
    appendix = text.split("## Appendix: every run", 1)[1]
    rows = [line for line in appendix.splitlines() if line.startswith("| ") and "---" not in line]
    assert len(rows) == 1 + len(data["runs"])  # the header and one row per run
    assert "| 1.00 → 1.20 |" in appendix  # load at the start and end of a run
    assert "ran before the runner recorded it" in appendix  # the fake onset runs have none


def test_a_rate_counts_its_worst_run_not_the_median():
    data = results()
    bad = next(
        r
        for r in data["runs"]
        if r["experiment"] == "ladder" and r["rate"] == 200 and r["rep"] == 2
    )
    bad["k6"]["degraded"] = 0.05  # one of three runs misses: the median would still pass
    bad["server"]["degraded_reasons"] = {"sasrec_circuit_open": 900.0}
    bad["server"]["breaker_transitions"] = {"open": 1.0}
    text = report.render(data)
    assert "**100 requests/s** is the largest offered load at which every run met the SLO" in text
    assert "At 200/s 2 of 3 runs met it; repeat 2 had P99 32.0 ms and 5.0 % degraded" in text
    assert "a burst of timeouts opened the breaker in front of" in text
    assert "| 200 ✗ | 2/3 |" in text and "| 100 | 3/3 |" in text


def test_the_cache_summary_leads_with_the_stable_effects_and_names_the_run_order():
    data = results()
    text = report.render(copy.deepcopy(data))
    line = next(x for x in text.splitlines() if x.startswith("- Candidate cache"))
    assert line.startswith("- Candidate cache at 200/s: P50 ")
    assert "slower one" not in line  # on is faster in every repeat of the fake data
    assert "right after the ladder's 400/s run" in line
    on3 = next(r for r in data["runs"] if r["experiment"] == "cache" and r["config"]["cache"] == "on"
               and not r["config"]["proxied"] and r["rep"] == 3)  # fmt: skip
    on3["k6"]["recs"]["p(99)"] = 40.0
    line = next(x for x in report.render(data).splitlines() if x.startswith("- Candidate cache"))
    assert "in repeat 3 the cache-on run was the slower one (40.0 vs 28.0 ms)" in line
    assert "the likely reason for the slower repeat" in line


def test_status_counts_name_the_runs_they_come_from():
    data = results()
    runs200 = [r for r in data["runs"] if r["experiment"] == "ladder" and r["rate"] == 200]
    runs200[0]["server"]["recs_status"] = {"410": 7.2}  # only one run kept a status record
    text = report.render(copy.deepcopy(data))
    assert "Errors by status: 200/s 410 ×7 (recorded in 1 of its 3 runs)" in text
    assert "ran before the runner kept one" in text
    for r in runs200[1:]:
        r["server"]["recs_status"] = {"410": 3.0}
    text = report.render(data)
    assert "200/s 410 ×3 (median per run)" in text and "ran before the runner kept one" not in text


def test_a_fault_summary_names_the_run_where_the_breaker_opened():
    data = results()
    slow = next(r for r in data["runs"] if r["experiment"] == "faults" and r["fault"] == "ms_slow"
                and r["mode"] == "design" and r["rep"] == 1)  # fmt: skip
    slow["k6"]["degraded"] = 0.7
    slow["server"]["breaker_transitions"] = {"open": 3.0}
    line = next(x for x in report.render(data).splitlines() if x.startswith("- Model server: 30 %"))
    assert line.endswith("30.0 % degraded (70.0 % in the run where the breaker opened).")


def test_plan_shuffles_the_cache_and_breaker_blocks():
    plan = run.PLANS["full"]
    runs = plan.runs()
    assert [(r.key, r.rep) for r in runs] == [(r.key, r.rep) for r in plan.runs()]  # deterministic
    default, cand = "time:10s:20:10", run.CANDIDATE_BREAKER
    for rep in (1, 2, 3):
        mine = [r for r in runs if r.rep == rep]
        assert [r.experiment for r in mine[:3]] == ["cache"] * 3  # the cache block opens the repeat
        last = max(i for i, r in enumerate(mine) if r.experiment == "ladder")
        brk = mine[last + 1 : last + 3]
        assert [r.experiment for r in brk] == ["breaker"] * 2
        assert {r.config.breaker for r in brk} == {default, cand}
        assert all(r.rate == plan.ladder[-1] and r.fault == "none" for r in brk)
        slow = [
            r.config.breaker for r in mine if r.fault == "ms_slow" and r.config.mode == "design"
        ]
        assert sorted(slow) == sorted([default, "calls:20", cand])
    orders = {
        tuple(r.key for r in runs if r.experiment == "cache" and r.rep == rep) for rep in (1, 2, 3)
    }
    assert len(orders) > 1  # the repeats do not all run the cache block in one order


def test_report_compares_the_breaker_rules_and_names_the_setup():
    data = results()
    cand = run.CANDIDATE_BREAKER
    for rep in (1, 2, 3):
        data["runs"] += [
            fake_run("breaker", 400, rep=rep, p99=30, degraded=0.15 if rep == 1 else 0.005),
            fake_run("breaker", 400, rep=rep, p99=28, degraded=0.004, breaker=cand),
            fake_run("faults", 100, cache="off", fault="ms_slow", rep=rep, p99=40, degraded=0.25,
                     breaker=cand, proxied=True),
        ]  # fmt: skip
    data["meta"] |= {
        "candidate_breaker": cand,
        "active_campaigns": [],
        "serving": {
            "bundle": "2026-11-08-sasrec64-ce60bf5d", "ranker": "lightgbm", "pctr": "logreg",
            "rank_experiments": [],
            "redis": {"state": {"policy": "noeviction", "maxmemory_mb": 1024},
                      "cache": {"policy": "allkeys-lru", "maxmemory_mb": 384}},
        },
    }  # fmt: skip
    text = report.render(data)
    assert text == report.render(data)
    assert "## 5. Breaker rule: in force vs 25 in a row, 2 probes" in text
    assert "| 400/s, no fault | 10 in a row, 5 probes |" in text
    assert "| 400/s, no fault | 25 in a row, 2 probes |" in text
    assert "| 100/s, 30 % of the calls +200 ms | 25 in a row, 2 probes |" in text
    assert "| this design, 25 in a row, 2 probes |" in text  # the fault table's comparison row
    assert "- Breaker rule, 10 in a row, 5 probes (in force) vs 25 in a row, 2 probes:" in text
    assert ("bundle `2026-11-08-sasrec64-ce60bf5d` with a LightGBM ranker and a logistic-regression "
            "pCTR model: a home request that misses the candidate cache calls the model server "
            "twice") in text  # fmt: skip
    assert (
        "The list keeps RRF's order (the api's default ranker; no rank-layer experiment ran)"
        in text
    )
    assert "state (`noeviction`, 1024 MB" in text and "no campaign was active" in text
    data["meta"]["serving"]["rank_experiments"] = [
        {"exp_id": "rank-ab", "buckets": [0, 1000], "rankers": {"rrf": None, "lgbm": "lightgbm"}}
    ]
    assert "(`rank-ab`, buckets [0, 1000): rrf RRF, lgbm LightGBM)" in report.render(data)
    # Without the new records (results before 2026-11-08) none of it appears.
    old = report.render(results())
    assert "## 5. Breaker rule" not in old and "- Served:" not in old


def test_plan_runs_the_warm_rates_after_a_lighter_start():
    plan = run.PLANS["full"]
    for rep in (1, 2, 3):
        mine = [r for r in plan.runs() if r.rep == rep]
        last = max(i for i, r in enumerate(mine) if r.experiment == "breaker")
        warm = mine[last + 1 : last + 1 + len(plan.warm)]
        assert [r.experiment for r in warm] == ["warm"] * len(plan.warm)
        assert [r.rate for r in warm] == plan.warm
        assert all(r.prewarm == plan.prewarm and r.config == run.ApiConfig() for r in warm)
    assert all(r.prewarm is None for r in plan.runs() if r.experiment != "warm")


def cold(r, hit, degraded, opened):
    """Make a fake run look like a cold start that never filled the cache."""
    r["k6"]["cache_hit_home"], r["k6"]["degraded"] = hit, degraded
    r["server"]["degraded_reasons"] = {"rank_circuit_open": 40000.0, "sasrec_circuit_open": 38000.0}
    r["server"]["breaker_transitions"] = {"open": opened}
    r["server"]["model_server_calls"] = {"OK": 370.0, "DEADLINE_EXCEEDED": 340.0}
    r["server"]["model_server_aborts"] = {"arrival": 280.0}
    return r


def test_report_explains_a_cold_start_and_shows_the_warm_runs():
    data = results()
    for rep in (1, 2, 3):
        data["runs"] += [
            cold(fake_run("ladder", 800, rep=rep, p99=25), 0.04, 0.92, 11.0),
            fake_run("warm", 800, rep=rep, p99=9) | {"prewarm": [400, 30]},
        ]
        for r in data["runs"]:
            if r["experiment"] == "ladder" and r["rate"] == 200 and r["rep"] == rep:
                r["k6"]["cache_hit_home"] = 0.9  # the ladder's best
    text = report.render(data)
    assert text == report.render(data)
    assert "- From 800/s on, the ladder's runs never filled the candidate cache (home hit rate 4.0 % " \
        "there, 90.0 % at 200/s)" in text  # fmt: skip
    assert "With the cache filled first (30 s at 400/s): 800/s 3/3 runs within the SLO" in text
    assert "At 800/s the model server received 710 calls in the measured 60 s, 340 of which ran " \
        "past their deadline (280 were already past it on arrival" in text  # fmt: skip
    assert "opened 11 times per run in all" in text
    assert (
        "| 800 ✗ | empty cache (ladder) | 0/3 |" in text and "| 800 | cache filled | 3/3 |" in text
    )
    assert "in every repeat right after the breaker block" in text
    for r in data["runs"]:
        if r["experiment"] == "warm":
            r["started"] += 10_000  # added by a restart: after every other run
    assert (
        "added to the plan by a restart (Setup), they ran after every other run"
        in report.render(data)
    )


def test_a_breaker_open_from_the_warm_up_is_named_as_such():
    data = results()
    for r in data["runs"]:
        if r["experiment"] == "ladder" and r["rate"] == 400:
            r["server"]["degraded_reasons"] = {
                "rank_circuit_open": 3800.0
            }  # no transition measured
    text = report.render(data)
    assert ("the breaker in front of the model server's Rank had opened in the 10-s warm-up, which "
            "starts with the candidate cache empty") in text  # fmt: skip
    assert "was open in 3 of 3 runs at 400/s (the worst run 2.0 % degraded; in 3 of them it had " \
        "opened in the warm-up)" in text  # fmt: skip
    assert "never filled" not in text  # the hit rate is the same at every rate
