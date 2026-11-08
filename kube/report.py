"""Render docs/results/deploy.md (design §23.2, the M5 acceptance of §16) from the runs on the
kind cluster:

    artifacts/deploy/results.json        kube.run: rolling update, HPA, KEDA
    artifacts/deploy/drill/verify.json   kube.drill: the billing drill with pods deleted

Either may be missing (its sections then say how to produce them, and the report is marked
preliminary). The output depends only on those files: re-rendering gives the same bytes.

Usage: python -m kube.report [--results ...] [--drill ...] [--out ...]
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path

from analysis.billing_report import dollars, table

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts" / "deploy"
DEFAULT_OUT = ROOT / "docs" / "results" / "deploy.md"
BUSY_LOAD = 2.0  # as ml/eval/machine.py: a 1-minute load above this is not a quiet machine
SETTLE_S = 30  # after the last new api pod is ready: its connections spread (CONN_CLOSE)
DRAINED = 1000  # a backlog below this many records counts as drained


# ---------------------------------------------------------------- small helpers


def when(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d %H:%M UTC")


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def ms(x, digits: int = 1) -> str:
    return "–" if x is None else f"{x:,.{digits}f} ms"


def secs(x, digits: int = 0) -> str:
    return "–" if x is None else f"{x:.{digits}f} s"


def pct(num: float, den: float, digits: int = 2) -> str:
    return "–" if not den else f"{100 * num / den:.{digits}f} %"


def rng(xs, fmt) -> str:
    """median [min–max] of a run's values over its repeats."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return "–"
    if len(xs) == 1:
        return fmt(xs[0])
    return f"{fmt(med(xs))} [{fmt(min(xs))}–{fmt(max(xs))}]"


def action_time(run: dict, prefix: str) -> float | None:
    for a in run.get("actions", []):
        if a["action"].startswith(prefix):
            return a["t"]
    return None


def windows_in(run: dict, t0: float | None, t1: float | None) -> list[dict]:
    """The k6 windows that start in [t0, t1)."""
    if t0 is None or t1 is None:
        return []
    return [w for w in run["k6"]["windows"] if t0 <= w["t"] < t1]


def phase(ws: list[dict]) -> dict:
    """A phase of a run from its 5 s windows: requests, errors, degraded, and the P99 of the
    windows (median and worst; k6 keeps exact percentiles per window, not per phase)."""
    n = sum(w["n"] for w in ws)
    return {
        "windows": len(ws),
        "requests": n,
        "errors": sum(w["errors"] for w in ws),
        "degraded": sum(w["degraded"] for w in ws),
        "p99_med": med([w["p99"] for w in ws]),
        "p99_max": max((w["p99"] for w in ws), default=None),
        "rps": n / sum(w.get("s", 5) for w in ws) if ws else None,
    }


def series_mean(
    series: list[dict], t0: float, t1: float, pod_prefix: str | None = None
) -> float | None:
    """Mean over [t0, t1) of a Prometheus range result, summed over its series (e.g. pods)."""
    by_t: dict[float, float] = {}
    for s in series:
        if pod_prefix and not s["metric"].get("pod", "").startswith(pod_prefix):
            continue
        for t, v in s["values"]:
            if t0 <= t < t1:
                by_t[t] = by_t.get(t, 0.0) + v
    return statistics.fmean(by_t.values()) if by_t else None


def other_containers(run: dict) -> set[str]:
    return {c for m in run.get("machine", []) for c in m.get("other_containers") or []}


def load_range(runs: list[dict]) -> str:
    vals = [m["load1"] for r in runs for m in r.get("machine", []) if m.get("load1") is not None]
    if not vals:
        return "not recorded"
    return f"median {statistics.median(vals):.2f}, max {max(vals):.2f} (start and end of each run)"


# ---------------------------------------------------------------- rollout


def rollout_stats(r: dict) -> dict:
    """One rolling-update run: the rollout window and what the load saw in it."""
    trig = r["actions"][0]["t"] if r["actions"] else None
    done = action_time(r, "rollback complete") or action_time(r, "rollout complete")
    start = r["k6"]["windows"][0]["t"] if r["k6"]["windows"] else r["started"]
    before = phase(windows_in(r, start + 10, trig))  # the first 10 s warm the connections
    during = phase(windows_in(r, (trig or 0) - 5, (done or 0) + 5))
    whole = phase(r["k6"]["windows"])
    pods = r["watch"]["pods"]
    new = [
        p
        for p in pods
        if p["component"] == "model-server" and trig and (p["created"] or 0) >= trig - 1
    ]
    old = [p for p in pods if p["component"] == "model-server" and p not in new]
    ready_delay = [p["ready"] - p["started"] for p in new if p.get("ready") and p.get("started")]
    loads = [v for v in r.get("bundle_load_seconds", {}).values() if v is not None]
    ms_ready = [s["ready"].get("model-server", 0) for s in r["watch"]["samples"]
                if trig and done and trig <= s["t"] <= done]  # fmt: skip
    return {
        "trigger": trig,
        "done": done,
        "rollout_seconds": (done - trig) if trig and done else None,
        "before": before,
        "during": during,
        "whole": whole,
        "new_pods": len(new),
        "old_pods": len(old),
        "ready_after_start": ready_delay,
        "bundle_load": loads,
        "min_ready": min(ms_ready) if ms_ready else None,
        "new_never_ready": sum(1 for p in new if not p.get("ready")),
        "restarts": max((p.get("restarts", 0) for p in new), default=0),
        "waiting": sorted({w for p in new for w in p.get("waiting", [])}),
        "log_tail": next((p["log_tail"] for p in new if p.get("log_tail")), ""),
        "stalled_after": stall_seconds(r),
    }


def stall_seconds(r: dict) -> float | None:
    for a in r.get("actions", []):
        if a["action"].startswith("rollout reported failed"):
            return float(a["action"].split(" after ")[1].split(" s")[0])
    return None


def rollout_summary(runs: list[dict]) -> list[str]:
    gated = [rollout_stats(r) for r in runs if r["variant"] == "gated"]
    noprobe = [rollout_stats(r) for r in runs if r["variant"] == "no-probe"]
    broken = [rollout_stats(r) for r in runs if r["variant"] == "broken"]
    if not gated:
        return []
    err = sum(s["whole"]["errors"] for s in gated)
    n = sum(s["whole"]["requests"] for s in gated)
    deg = sum(s["during"]["degraded"] for s in gated)
    dn = sum(s["during"]["requests"] for s in gated)
    rate = runs[0]["rate"]
    head = (
        "The model server's rolling update does not interrupt the service."
        if err == 0
        else f"The model server's rolling update failed {err:,} requests."
    )
    out = [
        f"- **{head}** In "
        f"{len(gated)} rolling updates of both model-server pods under {rate} page views/s "
        f"(open loop), {err} of {n:,} requests failed and {deg} of the {dn:,} answered during "
        f"the rollouts were degraded ({pct(deg, dn)}). A new pod got traffic only once its "
        f"bundle had loaded and passed its checks: Ready {rng([med(s['ready_after_start']) for s in gated], secs1)} "
        f"after its container started; never fewer than "
        f"{min((s['min_ready'] for s in gated if s['min_ready'] is not None), default='–')} ready pods."
    ]  # fmt: skip
    if broken:
        b = broken[0]
        out.append(
            f"  A bundle that fails its checks (item-id mapping shuffled: checksums fine, golden "
            f"queries wrong) never became ready; the rollout was reported failed after "
            f"{secs(b['stalled_after'])} and undone, with {b['whole']['errors']} failed and "
            f"{b['during']['degraded']} degraded requests meanwhile."
        )
    if noprobe:
        nd = sum(s["during"]["degraded"] for s in noprobe)
        nn = sum(s["during"]["requests"] for s in noprobe)
        ne = sum(s["whole"]["errors"] for s in noprobe)
        gm = sum(ms_degraded(r) for r in runs if r["variant"] == "gated")
        nm = sum(ms_degraded(r) for r in runs if r["variant"] == "no-probe")
        out.append(
            f"  The same rollout without the readiness probe (pods get traffic as soon as they "
            f"start): {pct(nd, nn)} of the requests during the rollouts degraded, {ne} failed; "
            f"answers degraded for the model server (`sasrec_*`): {nm:,.0f} without the probe, "
            f"{gm:,.0f} with it."
        )
    return out


MS_REASONS = ("sasrec_", "model_version_mismatch")


def ms_degraded(r: dict) -> float:
    """Answers degraded because of the model server (SASRec recall), over the whole run."""
    return sum(v for k, v in r["server"]["degraded_reasons"].items() if k.startswith(MS_REASONS))


def secs1(x) -> str:
    return secs(x, 1)


def rollout_section(runs: list[dict]) -> list[str]:
    rs = [r for r in runs if r["experiment"] == "rollout"]
    out = ["## 1. Rolling update of the model server", ""]
    if not rs:
        return out + ["Not run yet: `make deploy-rollout`.", ""]
    r0 = rs[0]
    out += [
        f"k6 at {r0['rate']} page views/s (open loop, `loadtest/recs.js`) for "
        f"{r0['duration']} s; 30 s in, a rolling update of the model server's two pods "
        "(`maxSurge 1`, `maxUnavailable 0`, `minReadySeconds 5`). The api calls the model "
        "server over one gRPC connection per api pod; an old pod first leaves the Service "
        "(preStop sleep 5 s), then stops with `grpc.server.stop(grace=5)`, which tells the "
        "clients to reconnect (GOAWAY) and lets running calls finish. *During* = from the "
        "trigger to the end of the rollout (± one 5 s window).",
        "",
        "- **gated**: the chart. Readiness = the gRPC health service says SERVING, which the "
        "server does only after `model_server/bundle.py` accepted the bundle (checksums, item "
        "count, dimension, golden queries).",
        "- **no-probe**: the same rollout after removing the readiness and startup probes: a "
        "pod counts as ready once its container runs, before the bundle is loaded.",
        "- **broken**: the new pods get a copy of the bundle with `item_ids.json` shuffled and "
        "its checksum updated: everything matches except the golden queries.",
        "",
    ]
    rows = []
    for variant in ("gated", "no-probe", "broken"):
        ss = [rollout_stats(r) for r in rs if r["variant"] == variant]
        if not ss:
            continue
        rows.append([
            variant, len(ss), f"{sum(s['whole']['requests'] for s in ss):,}",
            sum(s["whole"]["errors"] for s in ss),
            pct(sum(s["during"]["degraded"] for s in ss), sum(s["during"]["requests"] for s in ss)),
            rng([s["before"]["p99_med"] for s in ss], ms), rng([s["during"]["p99_max"] for s in ss], ms),
            rng([s["rollout_seconds"] for s in ss], secs),
            min((s["min_ready"] for s in ss if s["min_ready"] is not None), default="–"),
            rng([med(s["ready_after_start"]) for s in ss], secs1),
        ])  # fmt: skip
    out += table(
        ["variant", "runs", "requests", "failed", "degraded during", "P99 before (median window)",
         "P99 during (worst window)", "rollout", "fewest ready pods", "Ready after start"],
        rows,
    )  # fmt: skip
    out += [
        "",
        "Degradation reasons over each variant's runs (the api's `recloop_degraded_total`):",
        "",
    ]
    reasons = {}
    for r in rs:
        for k, v in r["server"]["degraded_reasons"].items():
            reasons.setdefault(r["variant"], {}).setdefault(k, 0)
            reasons[r["variant"]][k] += v
    for variant, rs_ in reasons.items():
        top = ", ".join(
            f"`{k}` {round(v):,}" for k, v in sorted(rs_.items(), key=lambda kv: -kv[1])[:6]
        )
        out.append(f"- {variant}: {top or 'none'}")
    if not reasons:
        out.append("- none in any run")
    gated = [r for r in rs if r["variant"] == "gated"]
    if gated:
        r = gated[0]
        s = rollout_stats(r)
        out += ["", f"Pods of the first gated run (s from the trigger, {when(r['started'])}):", ""]
        rows = []
        for p in r["watch"]["pods"]:
            if p["component"] != "model-server":
                continue
            rel = lambda t: "–" if t is None or s["trigger"] is None else f"{t - s['trigger']:+.0f}"  # noqa: E731
            rows.append([p["name"], rel(p.get("created")), rel(p.get("started")), rel(p.get("ready")),
                         rel(p.get("deleting")), rel(p.get("gone"))])  # fmt: skip
        out += table(
            ["pod", "created", "started", "ready", "deletion asked", "gone"], rows, "lrrrrr"
        )
        loads = [v for rr in gated for v in rr.get("bundle_load_seconds", {}).values() if v]
        if loads:
            out += ["", f"Bundle load and checks in the new pods (their log): "
                    f"{rng(loads, secs1)} over {len(loads)} pods."]  # fmt: skip
    broken = [r for r in rs if r["variant"] == "broken"]
    if broken:
        s = rollout_stats(broken[0])
        out += [
            "",
            f"The broken bundle: the new pod restarted {s['restarts']} times "
            f"({', '.join(s['waiting']) or 'no waiting state seen'}), never ready; the Deployment "
            f"reported `ProgressDeadlineExceeded` {secs(s['stalled_after'])} after the change "
            "(`progressDeadlineSeconds: 120`), then `kubectl rollout undo`. Its last log line:",
            "",
            "```",
            s["log_tail"].strip().splitlines()[-1][:300] if s["log_tail"].strip() else "(no log)",
            "```",
        ]
    out.append("")
    return out


# ---------------------------------------------------------------- hpa


def hpa_stats(r: dict) -> dict:
    step_at, step_end = r["step_at"], r["step_end"]
    samples = r["watch"]["samples"]
    decide = next(
        (s["t"] for s in samples if s["t"] >= step_at - 5 and (s["desired"].get("api") or 1) > 1),
        None,
    )
    api_pods = [p for p in r["watch"]["pods"] if p["component"] == "api"]
    new = sorted((p for p in api_pods if (p["created"] or 0) >= step_at - 5 and p.get("ready")),
                 key=lambda p: p["ready"])  # fmt: skip
    first_ready = new[0]["ready"] if new else None
    last_ready = max((p["ready"] for p in new if p["ready"] <= step_end), default=None)
    hold = [s["ready"].get("api", 0) for s in samples if step_at <= s["t"] <= step_end]
    one_pod = phase(windows_in(r, step_at, first_ready or step_end))
    scaled = phase(windows_in(r, (last_ready or step_end) + SETTLE_S, step_end))
    base = phase(windows_in(r, r["k6"]["t0"] + 10, step_at - 1))
    cpu = r["server"]["cpu"]["api"]
    thr = r["server"]["throttled"]["api"]
    return {
        "decide_after": (decide - step_at) if decide else None,
        "first_ready_after": (first_ready - step_at) if first_ready else None,
        "last_ready_after": (last_ready - step_at) if last_ready else None,
        "max_replicas": max(hold, default=None),
        "end_replicas": hold[-1] if hold else None,
        "base": base,
        "one_pod": one_pod,
        "scaled": scaled,
        "cpu_one_pod": series_mean(cpu, step_at, first_ready or step_end),
        "cpu_scaled": series_mean(cpu, (last_ready or step_end) + SETTLE_S, step_end),
        "throttled_one_pod": series_mean(thr, step_at, first_ready or step_end),
        # What the HPA saw (its currentMetrics: % of the CPU request) before the first new pod.
        "hpa_cpu_one_pod": med(
            [
                x.get("metric", {}).get("api")
                for x in samples
                if step_at <= x["t"] < (first_ready or step_end)
            ]
        ),
        "dropped": r["k6"]["dropped_iterations"],
        "errors": r["k6"]["errors"],
    }


def api_cpu(meta: dict | None) -> tuple[str | None, str | None]:
    """The api container's CPU request and limit as the cluster had them (e.g. "500m")."""
    for c in ((meta or {}).get("environment") or {}).get("containers", []):
        if c.get("workload") == "api" and c.get("container") == "api" and not c.get("init"):
            return c["requests"].get("cpu"), c["limits"].get("cpu")
    return None, None


def cores(q: str | None) -> str:
    """A Kubernetes CPU quantity in cores: "500m" -> "0.5"."""
    if not q:
        return "?"
    x = float(q[:-1]) / 1000 if q.endswith("m") else float(q)
    return f"{x:g}"


def hpa_summary(runs: list[dict], meta: dict | None = None) -> list[str]:
    rs = [r for r in runs if r["experiment"] == "hpa"]
    if not rs:
        return []
    ss = [hpa_stats(r) for r in rs]
    r0 = rs[0]
    req, _ = api_cpu(meta)
    scaled = max((s["max_replicas"] or 0) for s in ss) > 1
    head = "The api scales out under load." if scaled else "The api did not scale out."
    request = f"{cores(req)}-core request" if req else "request"
    return [
        f"- **{head}** At a step from {r0['base']} to {r0['step']} page "
        f"views/s ({len(rs)} runs), the HorizontalPodAutoscaler (CPU, target "
        f"{r0['hpa']['metrics'][0]['resource']['target']['averageUtilization']} % of the "
        f"{request}) asked for more pods {rng([s['decide_after'] for s in ss], secs)} after the step "
        f"and the first new pod was ready {rng([s['first_ready_after'] for s in ss], secs)} after "
        f"it; the api reached {rng([s['max_replicas'] for s in ss], str)} pods. P99 per 5 s window "
        f"(median of the windows): {rng([s['one_pod']['p99_med'] for s in ss], ms)} on one pod, "
        f"{rng([s['scaled']['p99_med'] for s in ss], ms)} once scaled out; "
        f"{sum(s['errors'] for s in ss)} failed requests."
    ]


def hpa_section(runs: list[dict], meta: dict | None = None) -> list[str]:
    rs = [r for r in runs if r["experiment"] == "hpa"]
    out = ["## 2. The api's HorizontalPodAutoscaler", ""]
    if not rs:
        return out + ["Not run yet: `make deploy-hpa`.", ""]
    r0 = rs[0]
    h = r0["hpa"]
    req, lim = api_cpu(meta)
    share = (
        f"The api pod has exactly {cores(req)} core (request = limit = {req}, so utilisation "
        "means the share of it)"
        if req == lim
        else f"The api pod's CPU request is {req}, its limit {lim} (utilisation is relative to "
        "the request)"
    )
    out += [
        f"k6 open loop: {r0['base']} page views/s for 60 s, {r0['step']}/s for {r0['hold']} s, "
        f"then {r0['base']}/s for 120 s. {share}; HPA between "
        f"{h['minReplicas']} and {h['maxReplicas']} pods at "
        f"{h['metrics'][0]['resource']['target']['averageUtilization']} % CPU, metrics-server "
        "every 15 s, scale-up without stabilisation. kube-proxy places connections, not "
        f"requests: k6 closes {100 * r0['conn_close']:.0f} % of its requests' connections "
        "(`Connection: close`), so new connections, and with them requests, reach the new pods. "
        f"*One pod* = from the step until the first new pod is ready; *scaled out* = from "
        f"{SETTLE_S} s after the last new pod is ready to the end of the step.",
        "",
    ]
    rows = []
    for r in rs:
        s = hpa_stats(r)
        rows.append([
            r["rep"], secs(s["decide_after"]), secs(s["first_ready_after"]), s["max_replicas"],
            ms(s["base"]["p99_med"]), ms(s["one_pod"]["p99_med"]), ms(s["one_pod"]["p99_max"]),
            ms(s["scaled"]["p99_med"]), ms(s["scaled"]["p99_max"]),
            pct(s["one_pod"]["degraded"], s["one_pod"]["requests"]),
            pct(s["scaled"]["degraded"], s["scaled"]["requests"]),
            "–" if s["cpu_one_pod"] is None else f"{s['cpu_one_pod']:.2f}",
            "–" if s["throttled_one_pod"] is None else f"{100 * s['throttled_one_pod']:.0f} %",
            "–" if s["hpa_cpu_one_pod"] is None else f"{s['hpa_cpu_one_pod']:.0f} %",
            s["errors"], s["dropped"],
        ])  # fmt: skip
    out += table(
        ["run", "scale-up asked", "first new pod ready", "pods", "P99 base", "P99 one pod",
         "worst window", "P99 scaled out", "worst window", "degraded one pod",
         "degraded scaled out", "api CPU one pod (cores)", "throttled periods", "HPA's CPU one pod", "failed",
         "dropped iterations"],
        rows,
    )  # fmt: skip
    out += ["", "P99 per 5 s window of the first run (ms) and its ready api pods:", ""]
    out += hpa_chart(r0)
    return out


def hpa_chart(r: dict) -> list[str]:
    ws = r["k6"]["windows"]
    if not ws:
        return []
    ready = []
    for w in ws:
        s = [x for x in r["watch"]["samples"] if w["t"] <= x["t"] < w["t"] + 5]
        ready.append(max((x["ready"].get("api", 0) for x in s), default=0))
    ymax = max(w["p99"] for w in ws)
    top = int(ymax * 1.1 // 10 * 10 + 10)
    xs = ", ".join(f"{int(w['t'] - ws[0]['t'])}" for w in ws)
    return [
        "```mermaid",
        "xychart-beta",
        '  title "P99 per 5 s window"',
        f'  x-axis "s" [{xs}]',
        f'  y-axis "ms" 0 --> {top}',
        f"  line [{', '.join(f'{w["p99"]:.1f}' for w in ws)}]",
        "```",
        "",
        "```mermaid",
        "xychart-beta",
        '  title "Ready api pods"',
        f'  x-axis "s" [{xs}]',
        f'  y-axis "pods" 0 --> {max(ready) + 1}',
        f"  bar [{', '.join(str(x) for x in ready)}]",
        "```",
        "",
    ]


# ---------------------------------------------------------------- keda


def lag_series(r: dict, group: str) -> list[tuple[float, int]]:
    return [(x["t"], x[group]) for x in r["watch"]["lag"] if x.get(group) is not None]


def replicas_series(r: dict, comp: str) -> list[tuple[float, int]]:
    return [(s["t"], s["ready"].get(comp, 0)) for s in r["watch"]["samples"]]


def keda_stats(r: dict) -> dict:
    comp = "billing" if r["variant"] == "billing" else "aggregator"
    lag = lag_series(r, comp)
    reps = replicas_series(r, comp)
    if comp == "billing":
        t_on = action_time(r, "postgres +")
        t_off = action_time(r, "postgres back")
    else:
        t_on = action_time(r, "aggregator paused")
        t_off = action_time(r, "aggregator resumed")
    peak = max((v for _, v in lag), default=None)
    at_off = next((v for t, v in lag if t >= (t_off or 0)), None)
    drained = next((t for t, v in lag if t > (t_off or 0) and v < DRAINED), None)
    up = next((t for t, n in reps if t >= (t_on or 0) and n > 1), None)
    back = (
        next((t for t, n in reps if t > (drained or t_off or 0) and n == 1), None) if up else None
    )
    return {
        "component": comp,
        "peak_lag": peak,
        "lag_at_end_of_fault": at_off,
        "drained_after": (drained - t_off) if drained and t_off else None,
        "scaled_up_after": (up - t_on) if up and t_on else None,
        "max_replicas": max((n for _, n in reps), default=None),
        "back_to_one_after_drain": (back - drained) if back and drained else None,
        "errors": r["k6"]["errors"],
        "requests": r["k6"]["requests"],
    }


def keda_summary(runs: list[dict]) -> list[str]:
    rs = [r for r in runs if r["experiment"] == "keda"]
    if not rs:
        return []
    out = []
    b = [keda_stats(r) for r in rs if r["variant"] == "billing"]
    pg = next((r.get("pg_latency_ms") for r in rs if r["variant"] == "billing"), None)
    if b:
        head = (
            "KEDA scales the consumers on their lag."
            if max((s["max_replicas"] or 0) for s in b) > 1
            else "KEDA did not scale billing."
        )
        out.append(
            f"- **{head}** Billing behind a database {pg or '?'} ms "
            f"slower per reply: lag up to {rng([s['peak_lag'] for s in b], lambda x: f'{x:,.0f}')} "
            f"records, scaled to {rng([s['max_replicas'] for s in b], str)} pods "
            f"{rng([s['scaled_up_after'] for s in b], secs)} after the slowdown began, lag below "
            f"{DRAINED:,} {rng([s['drained_after'] for s in b], secs)} after it ended."
        )
    a = [keda_stats(r) for r in rs if r["variant"] == "aggregator"]
    c = [keda_stats(r) for r in rs if r["variant"] == "aggregator-capped"]
    if a:
        pause = next((r.get("pause_seconds") for r in rs if r.get("pause_seconds")), "?")
        scaled = max((s["max_replicas"] or 1) for s in a) > 1
        line = (
            f"  The aggregator after a {pause} s outage "
            f"(backlog {rng([s['lag_at_end_of_fault'] for s in a], lambda x: f'{x:,.0f}')} records): "
            f"drained in {rng([s['drained_after'] for s in a], secs)} "
            + (
                f"with KEDA ({rng([s['max_replicas'] for s in a], str)} pods)"
                if scaled
                else "by its one pod, before KEDA added any"
            )
        )
        if c:
            line += f", {rng([s['drained_after'] for s in c], secs)} capped at one pod"
        out.append(line + ".")
    return out


def keda_section(runs: list[dict]) -> list[str]:
    rs = [r for r in runs if r["experiment"] == "keda"]
    out = ["## 3. KEDA: aggregator and billing scaled on their consumer lag", ""]
    if not rs:
        return out + ["Not run yet: `make deploy-keda`.", ""]
    r0 = rs[0]
    so = r0.get("scaledobject") or {}
    trig = (so.get("triggers") or [{}])[0].get("metadata", {})
    lag = int(trig.get("lagThreshold", 0)) or None
    slow = next((r.get("slow_seconds") for r in rs if r.get("slow_seconds")), None)
    pause = next((r.get("pause_seconds") for r in rs if r.get("pause_seconds")), None)
    out += [
        "Both consumers are StatefulSets (pods `billing-0`, `billing-1`, ...: unique and stable "
        "static group member ids) with a KEDA ScaledObject: Kafka scaler on the group's lag over "
        f"`events`' 6 partitions, a pod per {f'{lag:,}' if lag else '?'} records of lag, between "
        f"{so.get('minReplicaCount', 1)} and {so.get('maxReplicaCount', '?')} pods, polled every "
        f"{so.get('pollingInterval', 5)} s; down after 60 s below target, one pod per 30 s. The lag "
        "below is the broker's (`rpk group describe`) every 5 s, k6 at "
        f"{r0['rate']} page views/s throughout.",
        "",
        "- **billing**: six live campaigns (large budgets) put sponsored slots on every home page, "
        "so billing has clicks to charge; 60 s in, every reply PostgreSQL sends billing is "
        f"{next((r.get('pg_latency_ms') for r in rs if r.get('pg_latency_ms')), 200)} ms late for "
        f"{slow or '?'} s (pg-proxy). A batch of 1,000 records that holds a click then takes over "
        "a second, and the topic receives more than that per second.",
        f"- **aggregator**: 30 s in the aggregator is paused at 0 pods (KEDA's paused-replicas) for "
        f"{pause or '?'} s, then resumed; **aggregator-capped**: the same with the ScaledObject's "
        "maxReplicaCount at 1, the drain of one pod.",
        "",
        "Scaling in has a cost here: a removed pod is a static member, which does not leave the "
        "group (by design: a restarted pod takes its partitions back at once), so its partitions "
        "wait for its 15 s session to time out before the others get them. The lag threshold sits "
        "above what that stall builds up at this rate, so it does not trigger a scale-up.",
        "",
    ]
    rows = []
    for r in rs:
        s = keda_stats(r)
        rows.append([
            r["variant"], r["rep"], "–" if s["peak_lag"] is None else f"{s['peak_lag']:,}",
            "–" if s["lag_at_end_of_fault"] is None else f"{s['lag_at_end_of_fault']:,}",
            secs(s["scaled_up_after"]), s["max_replicas"], secs(s["drained_after"]),
            secs(s["back_to_one_after_drain"]), s["errors"],
        ])  # fmt: skip
    out += table(
        ["run", "#", "peak lag", "lag when the fault ended", "more pods after", "pods",
         f"lag < {DRAINED:,} after the fault ended", "back to 1 pod after the drain", "failed"],
        rows, "lrrrrrrrr",
    )  # fmt: skip
    out.append("")
    return out


# ---------------------------------------------------------------- drill


def drill_summary(d: dict) -> list[str]:
    failed = [c["check"] for c in d["checks"] if not c["ok"]]
    pfailed = [c["check"] for c in d["pipeline"]["checks"] if not c["ok"]]
    deletes = [c for c in d["chaos"] if c["action"].startswith("deleted pod billing")]
    slow = sum(c["action"].startswith("postgres +") for c in d["chaos"])
    head = (
        "Deleting billing's pods mid-run leaves the ledger right."
        if not failed and not pfailed
        else "The billing drill on the cluster failed checks."
    )
    verdict = (
        "**the ledger equals the simulator's truth charge by charge**"
        if not failed
        else f"**{len(failed)} ledger check(s) failed**: " + "; ".join(failed)
    )
    return [
        f"- **{head}** Billing drill on the "
        f"cluster ({d['minutes']:.0f} min, {len(deletes)} billing pods deleted, Redpanda's pod "
        f"deleted {sum(c['action'] == 'deleted pod redpanda-0' for c in d['chaos'])} times, "
        f"PostgreSQL {d['pg_latency_ms']} ms slower for billing {slow} times): {verdict} "
        f"({d['charges']:,} charges, {d['charges_expected']:,} expected; "
        f"{len(d['checks']) - len(failed)}/{len(d['checks'])} ledger checks and "
        f"{len(d['pipeline']['checks']) - len(pfailed)}/{len(d['pipeline']['checks'])} "
        f"event-pipeline checks pass)."
    ]


def drill_section(d: dict | None) -> list[str]:
    out = ["## 4. The billing drill on the cluster", ""]
    if d is None:
        return out + ["Not run yet: `make deploy-drill`.", ""]
    sim = d["sim"]
    kube = d.get("kube", {})
    billing_pods = [p for p in kube.get("watch", {}).get("pods", []) if p["component"] == "billing"]
    out += [
        f"Run {when(sim['started'])}: `make deploy-drill` (`kube/drill.py`: `sim/billing_drill.py` "
        "with its chaos on pods), seed "
        f"{d['seed']}, the same traffic and checks as `make billing-drill` (billing.md). The "
        f"simulator ({d['concurrency']} sessions at a time) sent {sim['events_unique']:,} events "
        f"({sim['event_copies_delivered']:,} deliveries with its faults `{d['faults']}`) over "
        f"{sum(sim['page_views'].values()):,} page views; {d['slots_served']:,} sponsored slots "
        f"were served, {d['sponsored_click_events']:,} sponsored click events sent "
        f"({d['forged_click_events']} forged). Redis evicted {d.get('evicted_keys', '?')} keys.",
        "",
        "Chaos (s from the start of the traffic): a billing pod deleted, alternately graceful "
        "(SIGTERM) and forced (`--grace-period=0 --force`); its StatefulSet recreates it at once "
        "under the same name, the same static member of the `billing` group. Redpanda's pod "
        "deleted (graceful) and recreated on its volume. PostgreSQL slowed for billing through "
        "pg-proxy.",
        "",
    ]
    out += table(["t (s)", "action"], [[f"{c['t']:.1f}", c["action"]] for c in d["chaos"]], "rl")
    if billing_pods:
        samples = kube.get("watch", {}).get("samples", [])
        most = max((x["ready"].get("billing", 0) for x in samples), default=1)
        names = ", ".join(sorted({p["name"] for p in billing_pods}))
        out += ["", f"Billing pods during the run: {len(billing_pods)} incarnations ({names}); at "
                f"most {most} ready at once" + (" (KEDA did not add pods at this rate)." if most <= 1 else
                " (KEDA added pods when the lag grew).")]  # fmt: skip
    out += ["", "The ledger against the truth (`sim/billing_verify.py`):", ""]
    out += table(
        ["check", "expected", "actual", ""],
        [[c["check"], c["expected"], c["actual"], "ok" if c["ok"] else "**FAIL**"] for c in d["checks"]],
        "lrrl",
    )  # fmt: skip
    rec = d["reconcile"]
    out += [
        "",
        f"{d['budgets_exhausted']} of {len(d['days'])} campaign-days spent their whole budget; "
        f"{dollars(d['billed_cents'])} billed, {dollars(d['unbilled_cents'])} not billed. "
        f"Reconciliation after the drill: {rec['match']} campaign-days in agreement, "
        f"{rec['redis_behind']} Redis counters behind (repaired +{rec['repaired_cents']} ¢), "
        f"{rec['redis_ahead']} ahead, {rec['row_mismatch']} budget rows different from the ledger.",
        "",
        "The same run, checked as `make drill` checks the event pipeline:",
        "",
    ]
    out += table(
        ["check", "expected", "actual", ""],
        [[c["check"], c["expected"], c["actual"], "ok" if c["ok"] else "**FAIL**"] for c in d["pipeline"]["checks"]],
        "lrrl",
    )  # fmt: skip
    out.append("")
    return out


# ---------------------------------------------------------------- setup and the whole report


def setup_section(meta: dict | None, runs: list[dict]) -> list[str]:
    out = ["## Setup", ""]
    if not meta:
        return out + ["No run on the cluster yet.", ""]
    env = meta["environment"]
    mach, node = env["machine"], env["node"]
    charts = [p for p in env["pins"] if p[0] == "chart"]
    nodeimg = next((p[1] for p in env["pins"] if p[0] == "node"), "?")
    out += [
        f"- Machine: {mach.get('chip', '?')}, {mach.get('memory_gib', '?')} GiB, "
        f"{mach.get('os', '?')}. One kind node ({env['tools'].get('kind', '?')}, "
        f"`{nodeimg.split('@')[0]}`, Kubernetes {node['kubelet']}) in Docker Desktop's VM: "
        f"{node['allocatable']['cpu']} CPUs, {node['allocatable']['memory']} allocatable.",
        "- Infrastructure from official charts: "
        + ", ".join(f"{name} {ver}" for _, name, ver in charts)
        + ". Prometheus scrapes RecLoop's services every 5 s.",
        bundle_line(env.get("bundle")),
        f"- RecLoop's images (`deploy/kind/cluster.sh images`, loaded with `kind load`, not pushed "
        f"anywhere): {', '.join(f'{k} {v}' for k, v in env['images'].items()) or '?'}.",
        f"- Load: {env.get('k6') or 'k6'} on the host, open loop, to the api's NodePort through "
        "kind's port mapping.",
        f"- Machine load during the runs: {load_range(runs)}.",
    ]
    others = sorted({c for r in runs for c in other_containers(r)})
    if others:
        out.append(
            f"- {len(others)} containers of other projects ran in the same VM during the runs."
        )
    out += [
        "",
        "Requests and limits of every container of RecLoop's two charts (CPU in cores):",
        "",
    ]
    rows, seen = [], set()
    for c in env["containers"]:
        if c["namespace"] != "recloop" or c["init"]:
            continue
        comp = c.get("workload") or c["owner"]
        key = (comp, c["container"])
        if key in seen:
            continue
        seen.add(key)
        rq, lm = c["requests"], c["limits"]
        rows.append([comp, c["container"], rq.get("cpu", "–"), rq.get("memory", "–"),
                     lm.get("cpu", "–"), lm.get("memory", "–")])  # fmt: skip
    out += table(["workload", "container", "CPU request", "memory request", "CPU limit", "memory limit"],
                 sorted(rows), "llrrrr")  # fmt: skip
    out.append("")
    return out


def bundle_line(b: dict | None) -> str:
    if not b:
        return "- Bundle: not recorded."
    if not b.get("ranker") or b.get("rank") == "off":
        rank = "no Rank call (RRF order)"
    else:
        order = (
            f"the order of the rank experiments {', '.join(b['rank_experiments'])}"
            if b.get("rank_experiments")
            else f"`{b.get('default_ranker', 'rrf')}` order (no rank-layer experiment)"
        )
        rank = f"the model server's Rank (features, LightGBM scores) on every home request, {order}"
    pctr = "the model's pCTR for ads" if b.get("pctr") else "a fixed pCTR"
    return f"- Bundle `{b['version']}` (the api's `current`): {rank}; {pctr}."


def preliminary(data: dict | None, drill: dict | None) -> list[str]:
    issues = []
    runs = (data or {}).get("runs", [])
    have = {r["experiment"] for r in runs}
    missing = [e for e in ("rollout", "hpa", "keda") if e not in have]
    if missing:
        issues.append("not run yet: " + ", ".join(missing))
    if drill is None:
        issues.append("no billing drill on the cluster yet")
    busy = [
        m["load1"] for r in runs for m in r.get("machine", [])[:1] if m.get("load1", 0) > BUSY_LOAD
    ]
    if busy:
        issues.append(f"{len(busy)} runs started with a 1-minute load above {BUSY_LOAD:.0f}")
    if any(other_containers(r) for r in runs):
        issues.append("other projects' containers ran in the same VM")
    for x in (data or {}).get("meta", {}).get("resumed", []):
        issues.append(f"resumed ({x['reason']})")
    if not issues:
        return []
    return ["> **Preliminary.** " + "; ".join(issues) + ".", ""]


def render(data: dict | None, drill: dict | None) -> str:
    runs = (data or {}).get("runs", [])
    out = [
        "# Local Kubernetes deployment",
        "",
        "Generated by `make deploy-report` (`kube/report.py` over `artifacts/deploy/`); do not edit "
        "by hand. Design §23.2 and the M5 acceptance of §16: the model server's rolling update "
        "does not interrupt the service, the api scales out under load, and with billing's pods "
        "deleted mid-run the ledger still equals the truth charge by charge.",
        "",
    ]
    out += preliminary(data, drill)
    out += ["## Summary", ""]
    meta = (data or {}).get("meta")
    summary = rollout_summary(runs) + hpa_summary(runs, meta) + keda_summary(runs)
    if drill:
        summary += drill_summary(drill)
    out += summary or [
        "Nothing has run on the cluster yet (`make kind-up deploy`, then the `deploy-*` targets)."
    ]
    out.append("")
    out += setup_section(meta, runs)
    out += rollout_section(runs)
    out += hpa_section(runs, meta)
    out += keda_section(runs)
    out += drill_section(drill)
    out += [
        "## How it is deployed",
        "",
        "- `deploy/kind/cluster.yaml`: one node; the host reaches NodePort Services through "
        "kind's port mappings on 127.0.0.1 (no Ingress). The bundles directory "
        "(`artifacts/bundles`) is mounted read-only into the node and from there into the api "
        "and model-server pods, so both see every version and the `current` link as in compose; "
        "images stay code-only and a new bundle needs no image build.",
        "- `deploy/helm/recloop-infra`: PostgreSQL as a CloudNativePG `Cluster` (its standard "
        "image has pgvector), Redis as StatefulSets (state: noeviction; cache: allkeys-lru), the "
        "`events` topic. `deploy/helm/recloop`: the services, an HPA for the api, KEDA "
        "ScaledObjects for the consumers, ServiceMonitors, compose's Grafana dashboard.",
        "- Readiness: the api's `/healthz` (the listener starts after the bundle, the catalog and "
        "the experiments loaded), not `/readyz`, whose dependency report would take every api "
        "pod out of the Service when one dependency fails, instead of degrading (§5.1). The "
        "model server's readiness is its gRPC health service (SERVING after the bundle checks).",
        "",
    ]
    return "\n".join(out)


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", type=Path, default=ART / "results.json")
    p.add_argument("--drill", type=Path, default=ART / "drill" / "verify.json")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    text = render(load(args.results), load(args.drill))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    rel = args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out
    print(f"deploy report -> {rel}")


if __name__ == "__main__":
    main()
