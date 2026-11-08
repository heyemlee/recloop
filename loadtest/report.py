"""Render docs/results/latency.md from artifacts/latency/results.json (loadtest.run).

Deterministic: the same results file gives the same bytes. Every table shows the median of the
repeated runs of a configuration, with the range of P99 over the runs.

Usage: python -m loadtest.report [--results artifacts/latency/results.json] [--out ...]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "artifacts" / "latency" / "results.json"
REPORT = ROOT / "docs" / "results" / "latency.md"

SLO_P99_MS = 100.0
DEFAULT_BREAKER = "time:10s:20:10"
SLO_DEGRADED = 0.01
SLO_ERRORS = 0.001
FAULT_LABEL = {
    "none": "no fault",
    "ms_slow": "model server: 30 % of calls +200 ms",
    "ms_latency": "model server: every call +50 ms",
    "ms_down": "model server disconnected",
    "redis_down": "Redis disconnected",
}
FAULT_ORDER = ["none", "ms_slow", "ms_latency", "ms_down", "redis_down"]


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def group(runs: list[dict]) -> dict[str, list[dict]]:
    g = defaultdict(list)
    for r in runs:
        g[r["key"]].append(r)
    return g


def summarize(rs: list[dict]) -> dict:
    k = [r["k6"] for r in rs]
    s = [r["server"] for r in rs]
    p99s = [x["recs"].get("p(99)") for x in k]
    reasons = defaultdict(list)
    for x in s:
        for reason, n in x["degraded_reasons"].items():
            reasons[reason].append(n)
    ms_calls = [sum(x["model_server_calls"].values()) for x in s]
    secs = [r["seconds"] for r in rs]
    return {
        "n": len(rs),
        "ok_runs": sum(run_meets_slo(r) for r in rs),
        "per_run": [
            {
                "rep": r["rep"],
                "ok": run_meets_slo(r),
                "p99": r["k6"]["recs"].get("p(99)"),
                "degraded": r["k6"]["degraded"],
                "reasons": r["server"]["degraded_reasons"],
                "opened": r["server"]["breaker_transitions"].get("open", 0) >= 0.5,
            }
            for r in rs
        ],
        "rate": rs[0]["rate"],
        "achieved": med([x["achieved_rps"] for x in k]),
        "p50": med([x["recs"].get("med") for x in k]),
        "p95": med([x["recs"].get("p(95)") for x in k]),
        "p99": med(p99s),
        "p99_min": min(p for p in p99s if p is not None)
        if any(p is not None for p in p99s)
        else None,
        "p99_max": max(p for p in p99s if p is not None)
        if any(p is not None for p in p99s)
        else None,
        "p999": med([x["recs"].get("p(99.9)") for x in k]),
        "home_p99": med([x["home"].get("p(99)") for x in k]),
        "server_p99": med([x["p99_ms"] for x in s]),
        "errors": med([x["errors"] for x in k]),
        "degraded": med([x["degraded"] for x in k]),
        "degraded_min": min((x["degraded"] for x in k if x["degraded"] is not None), default=None),
        "degraded_max": max((x["degraded"] for x in k if x["degraded"] is not None), default=None),
        "dropped": med([x["dropped_iterations"] for x in k]),
        "hit_home": med([x["cache_hit_home"] for x in k]),
        "hit_item": med([x["cache_hit_item"] for x in k]),
        "events_p99": med([x["events"].get("p(99)") for x in k]),
        "ms_rps": med([c / t for c, t in zip(ms_calls, secs, strict=True)]),
        "stages": {
            st: med([x["stage_p99_ms"].get(st) for x in s])
            for st in sorted({st for x in s for st in x["stage_p99_ms"]})
        },
        "reasons": {r: med(v + [0] * (len(rs) - len(v))) for r, v in reasons.items()},
        # non-200 answers by status, over the runs that recorded them (older runs did not)
        "status_runs": sum("recs_status" in x for x in s),
        "status": {
            c: med([x["recs_status"].get(c, 0) for x in s if "recs_status" in x])
            for c in sorted({c for x in s for c in x.get("recs_status", {})})
        },
        "breaker_opened": med([x["breaker_transitions"].get("open", 0) for x in s]),
        # Prometheus' increase() extrapolates: one transition can read 0.9 or 1.1
        "breaker_runs": sum(x["breaker_transitions"].get("open", 0) >= 0.5 for x in s),
        "lag_max": {
            g: med([x["consumer_lag_max"].get(g, 0) for x in s]) for g in ("aggregator", "log-sink")
        },
        "caught_up": med([r["consumers_caught_up_s"] for r in rs]),
        "cpu": {
            c: med([r["containers_mid_run"].get(c, {}).get("cpu_pct") for r in rs])
            for c in (
                "api",
                "model-server",
                "redis",
                "redis-cache",
                "postgres",
                "redpanda",
                "log-sink",
            )
        },  # fmt: skip
    }


def ms(x, digits=1) -> str:
    return "–" if x is None else f"{x:.{digits}f}"


def pct(x, digits=1) -> str:
    return "–" if x is None else f"{100 * x:.{digits}f} %"


def pct_range(x, lo, hi) -> str:
    """The median and, when the runs differ, their range: 46.6 % [24.1–69.2]."""
    if x is None:
        return "–"
    if lo is None or hi is None or round(100 * lo, 1) == round(100 * hi, 1):
        return pct(x)
    return f"{pct(x)} [{100 * lo:.1f}–{100 * hi:.1f}]"


def run_meets_slo(r: dict) -> bool:
    """One run: P99, degraded share, errors and dropped requests all within the SLO."""
    k = r["k6"]
    p99 = k["recs"].get("p(99)")
    return (
        p99 is not None and p99 <= SLO_P99_MS and (k["degraded"] or 0) <= SLO_DEGRADED
        and (k["errors"] or 0) <= SLO_ERRORS and k["dropped_iterations"] == 0
    )  # fmt: skip


def meets_slo(s: dict) -> bool:
    """A configuration meets the SLO when every one of its runs does: the worst run counts."""
    return s["ok_runs"] == s["n"]


def top_reasons(s: dict, n: int = 3) -> str:
    rs = sorted(
        ((r, v) for r, v in s["reasons"].items() if v >= 0.5), key=lambda kv: (-kv[1], kv[0])
    )[:n]
    return ", ".join(f"{r} {v:.0f}" for r, v in rs) if rs else "–"


def load_line(runs: list[dict]) -> str:
    """The 1-minute load average when each measured run started and ended."""
    vals = [x[0] for r in runs for x in (r.get("load_avg_before"), r.get("load_avg_after")) if x]
    if not vals:
        return "no per-run load average"
    return f"over the runs median {statistics.median(vals):.2f}, max {max(vals):.2f} (start and end of each run)"


def starts(meta: dict) -> list[dict]:
    """The first start and every restart (`run.py --resume`), each with what it saw."""
    env = meta["environment"]
    first = {
        "load_avg_start": env["load_avg_start"],
        "vm_cpus": env["docker"]["vm_cpus"],
        "other_containers_running": env["docker"]["other_containers_running"],
    }
    return [first, *meta.get("resumed", [])]


def banner(meta: dict) -> list[str]:
    issues = []
    for s in starts(meta):
        if s["vm_cpus"] != 8:
            issues.append(f"the Docker VM had {s['vm_cpus']} CPUs, not the 8 of design §20")
        idle = s["load_avg_start"][0]  # before the runs; during them the test adds load
        if idle > 2:
            issues.append(
                f"the 1-minute load average was {idle:.1f} before the runs (above 2: not an idle machine)"
            )
        if s["other_containers_running"]:
            issues.append(
                f"{s['other_containers_running']} containers of other projects were running in the same VM"
            )
    issues = list(dict.fromkeys(issues))
    if meta["plan"] != "full":
        issues.append(f"plan `{meta['plan']}` (fewer, shorter runs)")
    if not issues:
        return []
    return ["> **Preliminary.** " + "; ".join(issues) + ". Numbers from such runs are noisier and "
            "lower than on a quiet machine; rerun `make load` to replace them.", ""]  # fmt: skip


def mermaid(rates: list[int], series: dict[str, list]) -> list[str]:
    top = max([v for vs in series.values() for v in vs if v is not None] + [SLO_P99_MS])
    ymax = int(top * 1.15 // 10 * 10 + 10)
    lines = ["```mermaid", "xychart-beta", '  title "Offered load vs latency (median of runs)"',
             "  x-axis \"requests/s\" [" + ", ".join(str(r) for r in rates) + "]",
             f'  y-axis "ms" 0 --> {ymax}']  # fmt: skip
    for name, vs in series.items():
        lines.append(f"  line [{', '.join('0' if v is None else f'{v:.1f}' for v in vs)}]")
        del name
    lines += ["```", ""]
    return lines


def render(data: dict) -> str:
    meta, runs = data["meta"], data["runs"]
    env = meta["environment"]
    onset = [r for r in runs if r["experiment"] == "onset"]
    g = group([r for r in runs if r["experiment"] != "onset"])
    S = {k: summarize(v) for k, v in g.items()}
    by_exp = defaultdict(list)
    for k, rs in g.items():
        by_exp[rs[0]["experiment"]].append((rs[0], S[k]))

    out = ["# Latency budget and degradation", ""]
    out += ["> Generated by `make load` (`loadtest/run.py`, rendered by `loadtest/report.py` from "
            "`artifacts/latency/results.json`). Do not edit by hand.", ""]  # fmt: skip
    out += banner(meta)

    # ---- summary
    ladder = sorted(by_exp["ladder"], key=lambda x: x[0]["rate"])
    first_fail = next((r["rate"] for r, s in ladder if not meets_slo(s)), None)
    # the largest rate below the first one where any run missed the SLO
    best = max(
        (r["rate"] for r, s in ladder if first_fail is None or r["rate"] < first_fail), default=None
    )
    out += ["## Summary", ""]
    if ladder:
        line = (f"- **{best} requests/s** is the largest offered load at which every run met the SLO "
                if best else "- At no rate of the ladder did every run meet the SLO ")  # fmt: skip
        line += (f"(P99 ≤ {SLO_P99_MS:.0f} ms, ≤ {SLO_DEGRADED:.0%} degraded responses, ≤ {SLO_ERRORS:.1%} "
                 "errors, k6 kept the rate), with this design and the candidate cache on.")  # fmt: skip
        if best and first_fail is None:
            line += f" Every run of the ladder met it: the limit lies above {best}/s."
        if first_fail:
            fs = next(s for r, s in ladder if r["rate"] == first_fail)
            bad = [x for x in fs["per_run"] if not x["ok"]]
            worst = max(bad, key=lambda x: (x["degraded"] or 0, x["p99"] or 0))
            line += (f" At {first_fail}/s {fs['ok_runs']} of {fs['n']} runs met it; repeat {worst['rep']} "
                     f"had P99 {ms(worst['p99'])} ms and {pct(worst['degraded'])} degraded")  # fmt: skip
            top = max(worst["reasons"].items(), key=lambda kv: kv[1], default=("", 0))[0]
            if worst["opened"] and top.endswith("_circuit_open"):
                dep = {"sasrec": "the model server", "text": "PostgreSQL text recall"}.get(
                    top.removesuffix("_circuit_open"), top.removesuffix("_circuit_open")
                )
                line += (": no fault was injected; a burst of timeouts reached the 10-failures-in-a-row "
                         f"rule of the breaker in front of {dep}, which then stayed open for at least "
                         "its 10 s")  # fmt: skip
            line += "."
        out.append(line)
    cache = {
        r["config"]["cache"] + ("+proxy" if r["config"].get("proxied") else ""): s
        for r, s in by_exp["cache"]
    }
    if "on" in cache and "off" in cache:
        on, off = cache["on"], cache["off"]
        line = (f"- Candidate cache at {on['rate']}/s: P50 {ms(off['p50'])} → {ms(on['p50'])} ms, model "
                f"server calls {off['ms_rps']:.0f} → {on['ms_rps']:.0f} per second, hit rate "
                f"{pct(on['hit_home'])} (home). P99 {ms(off['p99'])} ms off, {ms(on['p99'])} ms on "
                "(median)")  # fmt: skip

        def cache_p99(state: str) -> dict:
            return {r["rep"]: r["k6"]["recs"].get("p(99)") for r in runs if r["experiment"] == "cache"
                    and r["config"]["cache"] == state and not r["config"].get("proxied")}  # fmt: skip

        p_on, p_off = cache_p99("on"), cache_p99("off")
        slower = [rep for rep in sorted(p_on) if None not in (p_on[rep], p_off.get(rep))
                  and p_on[rep] > p_off[rep]]  # fmt: skip
        if slower:
            line += "; in " + ", ".join(
                f"repeat {rep} the cache-on run was the slower one ({ms(p_on[rep])} vs {ms(p_off[rep])} ms)"
                for rep in slower
            )
        line += "."
        if ladder:
            line += (f" The cache-on run always runs right after the ladder's {ladder[-1][0]['rate']}/s run, "
                     "before the cache-off run, so the P99 comparison carries that order"
                     + (", the likely reason for the slower repeat" + "s" * (len(slower) > 1) if slower else "")
                     + ".")  # fmt: skip
        out.append(line)
    faults = defaultdict(dict)
    for r, s in by_exp["faults"]:
        name = (
            r["mode"]
            if r["config"]["breaker"] == DEFAULT_BREAKER
            else f"design ({r['config']['breaker']})"
        )
        faults[r["fault"]][name] = s
    for f in FAULT_ORDER[1:]:
        if f in faults and "naive" in faults[f] and "design" in faults[f]:
            n, d = faults[f]["naive"], faults[f]["design"]
            line = (f"- {FAULT_LABEL[f].capitalize()}: naive P99 {ms(n['p99'])} ms with {pct(n['errors'])} "
                    f"errors; this design P99 {ms(d['p99'])} ms, {pct(d['errors'])} errors, "
                    f"{pct(d['degraded'])} degraded")  # fmt: skip
            k = d["breaker_runs"]
            if 0 < k < d["n"]:  # the breaker opened in some runs only: say what that cost
                line += (f" ({pct(d['degraded_max'])} in the run where the breaker opened)" if k == 1 else
                         f" (up to {pct(d['degraded_max'])} in the {k} runs where the breaker opened)")  # fmt: skip
            out.append(line + ".")
    out.append("")

    # ---- setup
    lim = env["limits"]

    def limit(c):
        x = lim.get(c)
        return f"{c} {x['cpus']:g} CPU / {x['memory_mb']} MB" if x else c

    started = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(meta["started"]))
    out += ["## Setup", ""]
    out += [
        f"- Machine: {env['host']['cpu']}, {env['host']['cores']} cores, {env['host']['memory_gb']} GB, "
        f"{env['host']['os']}, on mains power. Docker {env['docker']['server']}, VM with "
        f"{env['docker']['vm_cpus']} CPUs and {env['docker']['vm_memory_gb']} GB.",
        "- Containers (compose `deploy/docker-compose.yml`, bench profile): "
        + ", ".join(
            limit(c)
            for c in (
                "api",
                "model-server",
                "redis",
                # Two Redis instances since 2026-11-08 (state, cache); earlier runs had one.
                *(["redis-cache"] if "redis-cache" in lim else []),
                "postgres",
                "redpanda",
                "aggregator",
                "log-sink",
                "toxiproxy",
            )
        )
        + ". The api connects to the model server and Redis directly, except in the fault "
        "experiment, where every row (the no-fault one included) goes through Toxiproxy.",
        f"- Load: {env['k6']} on the host, open loop (`constant-arrival-rate`: requests start on "
        "schedule however slow the answers are, so a slow server cannot hide its tail). One "
        "iteration is one page view of a simulated session (`sim/world.py`: the 2,024 test-period "
        "users, click model on text similarity to each user's next items), followed by the page's "
        "impression / click / cart / purchase events on `POST /v1/events`; rates below count page "
        "views, i.e. recommendation requests.",
        f"- Each configuration ran {meta['repeats']} times: {meta['warmup']} s of warm-up, then "
        f"{meta['duration']} s measured. Before every run the consumers caught up, Redis was "
        "reseeded and every fault was off. Tables show the median over the runs; P99 in "
        "brackets is the range over the runs. Latency is measured by k6 (client side, all "
        'recommendation requests: home, item page and next page); "server P99" is the api\'s own '
        "histogram.",
        f"- Started {started}"
        + (
            f" (after waiting {meta['idle_wait_s']} s for the 1-minute load average to drop below 2)"
            if "idle_wait_s" in meta
            else ""
        )
        + f"; 1-minute load average {env['load_avg_start'][0]:.2f} at the start and "
        f"{(meta.get('load_avg_end') or [0])[0]:.2f} at the end; {load_line(runs)}; "
        f"{env['docker']['other_containers_running']} containers of other projects running.",
        *[f"- {note}" for note in meta.get("notes", [])],
        *[
            f"- Resumed {time.strftime('%H:%M UTC', time.gmtime(r['at']))} (after waiting "
            f"{r['idle_wait_s']} s; 1-minute load average {r['load_avg_start'][0]:.2f}; "
            f"{r['other_containers_running']} containers of other projects running): {r['reason']}. "
            f"{r['kept']} runs kept, {len(r['discarded'])} discarded"
            + (
                " (listed in `artifacts/latency/results-discarded.json`, not used here)."
                if r["discarded"]
                else "."
            )
            for r in meta.get("resumed", [])
        ],
        f"- SLO (design §5.1): P99 ≤ {SLO_P99_MS:.0f} ms and at most {SLO_DEGRADED:.0%} of responses "
        "degraded (`degraded` not empty), under normal load. A rate of the ladder also needs at "
        f"most {SLO_ERRORS:.1%} errors (answers other than 200, next pages included) and no "
        "requests k6 had to drop, in every one of its runs.",
        "",
    ]

    # ---- 1. ladder
    if ladder:
        out += ["## 1. Offered load vs P99", "",
                "This design, candidate cache on, no fault.", "",
                "| offered /s | runs within the SLO | achieved /s | P50 ms | P95 ms | P99 ms | server P99 ms | "
                "degraded | errors | dropped | cache hits (home) | top degradation reasons (count per run) |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]  # fmt: skip
        for r, s in ladder:
            mark = "" if meets_slo(s) else " ✗"
            out.append(
                f"| {r['rate']}{mark} | {s['ok_runs']}/{s['n']} | {s['achieved']:.0f} | {ms(s['p50'])} | {ms(s['p95'])} | "
                f"{ms(s['p99'])} [{ms(s['p99_min'])}–{ms(s['p99_max'])}] | {ms(s['server_p99'])} | "
                f"{pct_range(s['degraded'], s['degraded_min'], s['degraded_max'])} | "
                f"{pct(s['errors'], 2)} | {s['dropped']:.0f} | {pct(s['hit_home'])} | {top_reasons(s)} |"
            )
        out += ["", "✗: at least one run missed the SLO (a rate counts its worst run, not the median). "
                "Degraded: median of the runs [range].", ""]  # fmt: skip
        opened = [(r["rate"], s) for r, s in ladder if s["breaker_runs"]]
        if opened:
            out += ["Without any fault, the model server's breaker opened in "
                    + "; ".join(f"{s['breaker_runs']} of {s['n']} runs at {rate}/s (the worst run "
                                f"{pct(s['degraded_max'])} degraded)" for rate, s in opened)
                    + ": a burst of timeouts reached the 10-failures-in-a-row rule, and the breaker "
                    "then stayed open for at least its 10 s.", ""]  # fmt: skip
        status = [(r["rate"], s) for r, s in ladder if s["status_runs"] and s["status"]]
        if status:
            parts = []
            for rate, st in status:
                counts = ", ".join(f"{c} ×{n:.0f}" for c, n in sorted(st["status"].items()))
                where = ("median per run" if st["status_runs"] == st["n"]
                         else f"recorded in {st['status_runs']} of its {st['n']} runs")  # fmt: skip
                parts.append(f"{rate}/s {counts} ({where})")
            out += ["Errors by status: " + "; ".join(parts)
                    + ". 410: a next page whose result list had expired or been evicted (Redis "
                    "`maxmemory` with `volatile-ttl` drops the entries closest to expiry first); 503: "
                    "the page read missed its budget."
                    + (" Runs without a status record ran before the runner kept one."
                       if any(st["status_runs"] < st["n"] for _, st in status) else ""), ""]  # fmt: skip
        rates = [r["rate"] for r, _ in ladder]
        out += mermaid(
            rates, {"P99": [s["p99"] for _, s in ladder], "P50": [s["p50"] for _, s in ladder]}
        )
        out += ["Lines: P99 (upper) and P50 (lower), median of the runs.", ""]
        # Runs since the Redis split (2026-11-08) have a cache instance: its own CPU column.
        two = any(s["cpu"].get("redis-cache") is not None for _, s in ladder)
        redis_cols = "state Redis CPU | cache Redis CPU" if two else "Redis CPU"
        out += ["Per stage (server P99, ms) and where the CPU went (container CPU % mid-run, 100 % = one core):", "",
                "| offered /s | user context | cache get | recall | rank | filter | page store | render | "
                f"api CPU | model server CPU | {redis_cols} | PostgreSQL CPU | Redpanda CPU | log-sink lag (max) |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|" + ("---:|" if two else "")
                + "---:|---:|---:|"]  # fmt: skip
        for r, s in ladder:
            st = s["stages"]
            cpu = s["cpu"]
            redis_cpu = ms(cpu["redis"], 0) + (f" | {ms(cpu.get('redis-cache'), 0)}" if two else "")
            out.append(
                f"| {r['rate']} | {ms(st.get('user_context'))} | {ms(st.get('cache_get'))} | "
                f"{ms(st.get('recall'))} | {ms(st.get('rank'))} | {ms(st.get('filter'))} | "
                f"{ms(st.get('page_store'))} | {ms(st.get('render'))} | {ms(cpu['api'], 0)} | "
                f"{ms(cpu['model-server'], 0)} | {redis_cpu} | {ms(cpu['postgres'], 0)} | "
                f"{ms(cpu['redpanda'], 0)} | {s['lag_max']['log-sink']:.0f} |"
            )
        out.append("")

    # ---- 2. cache
    if "on" in cache and "off" in cache:
        out += [f"## 2. Candidate cache on / off ({cache['on']['rate']} requests/s)", "",
                "| candidate cache | P50 ms | P95 ms | P99 ms | home P99 ms | hit rate home | hit rate item | "
                "model server calls/s | degraded | errors |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]  # fmt: skip
        for name in ("off", "on", "on+proxy"):
            s = cache.get(name)
            if s is None:
                continue
            label = {"off": "off", "on": "on", "on+proxy": "on, through Toxiproxy"}[name]
            out.append(
                f"| {label} | {ms(s['p50'])} | {ms(s['p95'])} | {ms(s['p99'])} [{ms(s['p99_min'])}–{ms(s['p99_max'])}] | "
                f"{ms(s['home_p99'])} | {pct(s['hit_home'])} | {pct(s['hit_item'])} | {s['ms_rps']:.0f} | "
                f"{pct(s['degraded'], 2)} | {pct(s['errors'], 2)} |"
            )
        out += ["", "The cache keeps recall + ranking output per user and history version (`cand:{uid}:home:"
                "{variant}:{model_ver}:{hv}`) and per item for the item page; filters run on every request. "
                "A click, add-to-cart or purchase bumps hv, so the next request recomputes. The last row "
                "is the same configuration with Toxiproxy between the api and the model server / Redis, "
                "as in every row of the fault table below: the proxy's own cost.", ""]  # fmt: skip
        if ladder:
            out += [f"Within a repeat the order is fixed: the ladder up to {ladder[-1][0]['rate']}/s, then "
                    "the cache rows top to bottom, so the \"on\" run always starts right after the "
                    "ladder's highest rate.", ""]  # fmt: skip

    # ---- 3. faults
    if faults:
        rate = by_exp["faults"][0][0]["rate"]
        out += [f"## 3. Faults: naive vs this design ({rate} requests/s, candidate cache off, through Toxiproxy)", "",
                "| fault | implementation | P50 ms | P99 ms | errors | degraded | top degradation reasons (count per run) | breaker opened: times per run (runs) |",
                "|---|---|---:|---:|---:|---:|---|---:|"]  # fmt: skip
        for f in FAULT_ORDER:
            for name in (
                "naive",
                "design",
                *sorted(n for n in faults.get(f, {}) if n not in ("naive", "design")),
            ):
                s = faults.get(f, {}).get(name)
                if s is None:
                    continue
                label = {
                    "naive": "naive",
                    "design": "this design",
                    "design (calls:20)": "this design, count window",
                }.get(name, name.replace("design", "this design"))
                out.append(
                    f"| {FAULT_LABEL[f]} | {label} | {ms(s['p50'])} | {ms(s['p99'])} [{ms(s['p99_min'])}–{ms(s['p99_max'])}] | "
                    f"{pct(s['errors'], 2)} | {pct_range(s['degraded'], s['degraded_min'], s['degraded_max'])} | "
                    f"{top_reasons(s)} | {s['breaker_opened']:.0f} ({s['breaker_runs']}/{s['n']}) |"
                )
        slow = faults.get("ms_slow", {}).get("design")
        slow_note = []
        if slow and slow["breaker_runs"]:
            slow_note = [
                f"- 30 % of the calls slow: the 50 % rule of the 10-s window does not trip, yet the "
                f"breaker opened in {slow['breaker_runs']} of {slow['n']} runs. The 10-failures-in-a-row "
                "rule did it: independent 30 % failures make such a streak unlikely, so these were "
                "bursts of timeouts. A half-open breaker needs 5 successes in a row (at 70 % success, "
                "17 % per attempt), so it then stayed open for most of the run. Hence the spread of "
                "the degraded share between the runs."
            ]
        out += ["",
                "- Naive: no request deadline, no stage budgets, no circuit breaker, and any failed "
                "dependency fails the request (503). Same binary, `RECLOOP_RESILIENCE=naive`; Redis "
                "client defaults (3 s timeouts, 3 retries).",
                "- This design (§5.1): 100 ms per request, every stage takes its budget from what is "
                "left (user context 5 ms, recall 35 ms, 25 ms kept for ranking); the model server is "
                "behind a circuit breaker that opens for 10 s when at least half of the calls of the "
                "last 10 s failed (at least 20 calls) or after 10 failures in a row; every missing "
                "piece is written into `degraded` and the request goes on without it, down to the "
                "in-process popular list.",
                *slow_note,
                "- \"count window\": the same design with §5.1's first breaker rule, at least half of "
                "the last 20 calls. At hundreds of calls per second those 20 calls span a few tens of "
                "milliseconds; with 30 % of the calls slow, some of these short windows reach 10 "
                "failures, the breaker opens, and its half-open probes often fail again.",
                "- The 30 % fault is the model server's own switch (per call, which a proxy on one "
                "long-lived gRPC connection cannot do); the deadline the api sends ends the delay "
                "early. +50 ms on every call is a Toxiproxy latency toxic; disconnects are "
                "Toxiproxy proxies switched off.",
                ""]  # fmt: skip
    # ---- 4. onset
    if onset:
        rate = onset[0]["rate"]
        out += [f"## 4. Breaker onset and recovery ({rate} requests/s, this design)", "",
                f"The fault starts in the middle of steady traffic and ends {meta['onset'][1]:.0f} s later. Open after: from "
                "the fault to the breaker opening (the api's breaker state, read every 10 ms); failed "
                "calls: SASRec calls that failed or timed out (each waiting up to its budget) before "
                "it opened; closed after: from the model server's recovery to the breaker closing "
                "again (it stays open 10 s, then 5 probes must succeed).", "",
                "| fault | open after ms | failed calls before | closed after recovery s |",
                "|---|---:|---:|---:|"]  # fmt: skip
        for f in ("ms_down", "ms_latency"):
            rs = [r for r in onset if r["fault"] == f]
            if not rs:
                continue
            opened = [r["open_after_ms"] for r in rs if r["open_after_ms"] is not None]
            out.append(
                f"| {FAULT_LABEL[f]} | {ms(med(opened))} [{ms(min(opened, default=None))}–{ms(max(opened, default=None))}] | "
                f"{ms(med([r['failed_before_open'] for r in rs]), 0)} | "
                f"{ms(med([r['closed_after_recovery_s'] for r in rs]), 1)} |"
            )
        out.append("")

    # ---- appendix: every run, in the order it ran
    unstamped = any(r.get("load_avg_before") is None for r in runs)
    out += ["## Appendix: every run", "",
            "In the order they ran (UTC), with the host's 1-minute load average when each run's "
            "measurement started and ended; the test's own load is part of it."
            + (" Runs without one (onset) ran before the runner recorded it for them; the rows "
               "around them bracket them." if unstamped else "")
            + " P99: k6, all recommendation requests (onset: the whole run, fault included).", "",
            "| started | experiment | rate /s | api | fault | repeat | load start → end | P99 ms |",
            "|---|---|---:|---|---|---:|---|---:|"]  # fmt: skip
    for r in sorted(runs, key=lambda r: r["started"]):
        c = r["config"]
        api = [r["mode"], f"cache {c['cache']}", "via Toxiproxy" if c.get("proxied") else "direct"]
        if c["breaker"] != DEFAULT_BREAKER:
            api.append(c["breaker"])
        la = [x[0] for x in (r.get("load_avg_before"), r.get("load_avg_after")) if x]
        load = " → ".join(f"{x:.2f}" for x in la) if la else "–"
        p99 = (r.get("k6") or {}).get("recs", {}).get("p(99)")
        out.append(
            f"| {time.strftime('%H:%M:%S', time.gmtime(r['started']))} | {r['experiment']} | "
            f"{r['rate']} | {', '.join(api)} | {FAULT_LABEL.get(r['fault'], r['fault'])} | "
            f"{r['rep']} | {load} | {ms(p99)} |"
        )
    out.append("")
    return "\n".join(out).rstrip() + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", type=Path, default=RESULTS)
    p.add_argument("--out", type=Path, default=REPORT)
    args = p.parse_args(argv)
    args.results, args.out = args.results.resolve(), args.out.resolve()
    if not args.results.exists():
        raise SystemExit(f"{args.results} is missing: run `make load`")
    text = render(json.loads(args.results.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"report   {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out}")


if __name__ == "__main__":
    main()
