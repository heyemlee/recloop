"""Rolling update, HPA and KEDA runs on the kind cluster (M5, docs/design.md §23.2)
-> artifacts/deploy/results.json; `python -m kube.report` renders docs/results/deploy.md.

    rollout   the model server's rolling update under open-loop load (k6 on the host):
                gated     the chart: a new pod gets traffic once its readiness probe passes,
                          i.e. once the bundle has loaded and passed its checks
                broken    a bundle whose item-id mapping is shuffled (checksums, counts and
                          dimensions fine, golden queries fail): the new pod never becomes ready,
                          the rollout stalls, then `kubectl rollout undo`
                no-probe  the same rollout as `gated` without the readiness / startup probes
                          (the counterexample: a pod gets traffic as soon as it starts)
    hpa       the api under a step in load (base -> step -> base): replicas, CPU, P99 and
              degraded share per 5 s window before, during and after the scale-out
    keda      billing     PostgreSQL 200 ms slower for billing (pg-proxy) for 2 minutes, with
                          sponsored campaigns live: billing's lag, replicas
              aggregator  the aggregator paused (KEDA paused-replicas 0) for 90 s, then resumed:
                          the backlog's drain (--capped: also with KEDA capped at one replica)

Every run starts from the same state: the consumers have caught up, both Redis instances are
reseeded (ml.data.seed_redis), the autoscaled workloads are back at one replica, the model
server has its two ready pods, no toxic is set. Each configuration runs `--repeats` times
(interleaved); the report takes medians.

Usage: python -m kube.run [--only rollout,hpa,keda] [--repeats 3] [--resume REASON]
       python -m kube.run --smoke   (one short run of each kind -> artifacts/deploy/smoke.json;
                                     for trying the cluster out, not for the report)
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

from kube import cluster as k8s
from kube.cluster import kubectl

ROOT = k8s.ROOT
OUT = ROOT / "artifacts" / "deploy"
K6_BIN = ROOT / ".cache" / "bin" / "k6"
AGGREGATOR = ROOT / ".cache" / "bin" / "aggregator"
WORLD = ROOT / "artifacts" / "sim" / "world.json"
RECS_JS = ROOT / "loadtest" / "recs.js"
BUNDLES = ROOT / "artifacts" / "bundles"
BROKEN = "m5-broken-item-ids"  # the bad bundle's directory under artifacts/bundles
WINDOW_S = 5


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------- load


class K6:
    """k6 (loadtest/recs.js) in the background, open loop, with per-window sub-metrics."""

    def __init__(self, profile: str, tag: str, out: Path, conn_close: float = 0.0):
        self.summary = out / f"{tag}.k6.json"
        self.cmd = [
            str(K6_BIN), "run", "--quiet", "--no-color",
            "-e", f"BASE_URL={k8s.API_URL}", "-e", f"PROFILE={profile}",
            "-e", f"WINDOW_S={WINDOW_S}", "-e", f"CONN_CLOSE={conn_close}",
            "-e", f"WORLD={WORLD}", "-e", f"SUMMARY={self.summary}",
            "--tag", f"run={tag}", str(RECS_JS),
        ]  # fmt: skip
        self.proc: subprocess.Popen | None = None
        self.started = 0.0

    def start(self) -> K6:
        self.summary.unlink(missing_ok=True)
        self.started = time.time()
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return self

    def at(self, seconds: float) -> None:
        """Sleep until `seconds` after the start of the load."""
        time.sleep(max(0.0, self.started + seconds - time.time()))

    def wait(self) -> dict:
        _, err = self.proc.communicate()
        if self.proc.returncode != 0 or not self.summary.exists():
            raise RuntimeError(f"k6 failed ({self.proc.returncode}): {err.decode()[-800:]}")
        return k6_result(json.loads(self.summary.read_text()))


def k6_result(s: dict) -> dict:
    m = s["metrics"]

    def trend(name: str) -> dict:
        v = m.get(name, {}).get("values", {})
        return {k: round(v[k], 3) for k in ("med", "p(95)", "p(99)", "max", "count") if k in v}

    def passes(name: str) -> tuple[int, int]:
        v = m.get(name, {}).get("values", {})
        return int(v.get("passes", 0)), int(v.get("passes", 0) + v.get("fails", 0))

    t0 = s["setup_data"]["t0"] / 1000
    size = s["setup_data"].get("window_s") or WINDOW_S
    windows = []
    for i in range(10_000):
        w = f"w{i:03d}"
        if f"recs_ms{{w:{w}}}" not in m:
            break
        tr = trend(f"recs_ms{{w:{w}}}")
        if not tr.get("count"):
            continue
        err, n = passes(f"recs_errors{{w:{w}}}")
        deg, ok = passes(f"recs_degraded{{w:{w}}}")
        windows.append({
            "t": round(t0 + i * size, 3), "s": size, "n": n, "p50": tr["med"], "p95": tr["p(95)"],
            "p99": tr["p(99)"], "max": tr["max"], "errors": err, "degraded": deg, "ok": ok,
        })  # fmt: skip
    err, n = passes("recs_errors")
    deg, ok = passes("recs_degraded")
    ev_err, ev_n = passes("events_errors")
    return {
        "t0": t0,
        "recs": trend("recs_ms"),
        "requests": n,
        "errors": err,
        "degraded": deg,
        "answered": ok,
        "event_posts": ev_n,
        "event_errors": ev_err,
        "dropped_iterations": int(
            m.get("dropped_iterations", {}).get("values", {}).get("count", 0)
        ),
        "windows": windows,
    }


# ---------------------------------------------------------------- state between runs


def ml_python(module: str, *args: str) -> None:
    """A module of the ml environment (this interpreter) against the cluster."""
    subprocess.run(
        [sys.executable, "-m", module, *args], cwd=ROOT, check=True,
        env={**os.environ, **k8s.host_env()}, stdout=subprocess.DEVNULL,
    )  # fmt: skip


def wait_consumers(groups=("aggregator", "billing", "log-sink"), timeout: str = "5m") -> float:
    t = time.time()
    for g in groups:
        subprocess.run(
            [str(AGGREGATOR), "wait-lag", g, timeout], check=True, stdout=subprocess.DEVNULL,
            env={**os.environ, **k8s.host_env()},
        )  # fmt: skip
    return round(time.time() - t, 1)


def pg_toxic(on: bool, ms: int = 200) -> None:
    url = f"{k8s.PG_PROXY_URL}/proxies/postgres/toxics"
    if on:
        k8s.http_json("POST", url, {"name": "pg_latency", "type": "latency",
                                    "stream": "downstream", "attributes": {"latency": ms}})  # fmt: skip
    else:
        with contextlib.suppress(OSError):  # 404: none set
            k8s.http_json("DELETE", url + "/pg_latency")


def keda_pause(name: str, replicas: int | None) -> None:
    """Pause a ScaledObject at `replicas` (KEDA's paused-replicas annotation), or resume it."""
    key = "autoscaling.keda.sh/paused-replicas"
    arg = f"{key}={replicas}" if replicas is not None else f"{key}-"
    kubectl("annotate", "scaledobject", name, arg, "--overwrite")


def keda_max(name: str, n: int) -> None:
    kubectl("patch", "scaledobject", name, "--type=merge", "-p",
            json.dumps({"spec": {"maxReplicaCount": n}}))  # fmt: skip


def reset(log=log) -> dict:
    """The common starting state of every run (see the module docstring)."""
    t = time.time()
    pg_toxic(False)
    for so in ("aggregator", "billing"):
        with contextlib.suppress(RuntimeError):
            keda_pause(so, None)
    caught_up = wait_consumers()
    ml_python("ml.data.seed_redis")
    k8s.wait_replicas("deployment/model-server", 2, log=log)
    for w in ("deployment/api", "statefulset/aggregator", "statefulset/billing"):
        k8s.wait_replicas(w, 1, timeout=900, log=log)
    return {"caught_up_seconds": caught_up, "reset_seconds": round(time.time() - t, 1)}


def machine() -> dict:
    from ml.eval import machine as m

    s = m.state()
    node = f"{k8s.CLUSTER}-control-plane"
    s["other_containers"] = [c for c in (s.pop("containers") or []) if c != node]
    s.pop("vm", None)
    return s


# ---------------------------------------------------------------- Prometheus after a run


def server_side(t0: float, t1: float) -> dict:
    """What the cluster's Prometheus saw during [t0, t1]: the api's degradation reasons and
    breaker transitions, the model server's calls, CPU and throttling per pod (5 s steps)."""
    w = f"{max(1, round(t1 - t0))}s"
    api, ms = 'job="api"', 'job="model-server"'

    def cpu(container: str) -> list[dict]:
        sel = f'namespace="{k8s.NS}",container="{container}"'
        return k8s.prom_range(
            f"sum by (pod) (rate(container_cpu_usage_seconds_total{{{sel}}}[20s]))", t0, t1
        )

    def throttled(container: str) -> list[dict]:
        sel = f'namespace="{k8s.NS}",container="{container}"'
        return k8s.prom_range(
            f"sum by (pod) (rate(container_cpu_cfs_throttled_periods_total{{{sel}}}[20s])) / "
            f"sum by (pod) (rate(container_cpu_cfs_periods_total{{{sel}}}[20s]))", t0, t1,
        )  # fmt: skip

    return {
        "degraded_reasons": k8s.prom_by(
            f"sum by (reason) (increase(recloop_degraded_total{{{api}}}[{w}]))", t1, "reason"
        ),
        "breaker_transitions": k8s.prom_by(
            f"sum by (to) (increase(recloop_breaker_transitions_total{{{api}}}[{w}]))", t1, "to"
        ),
        "model_server_calls": k8s.prom_by(
            f"sum by (code) (increase(recloop_ms_requests_total{{{ms}}}[{w}]))", t1, "code"
        ),
        "recs_status": k8s.prom_by(
            f'sum by (code) (increase(recloop_http_requests_total{{{api},route="GET /v1/recommendations"}}[{w}]))',
            t1,
            "code",
        ),  # fmt: skip
        "api_p99_ms": k8s.prom_range(
            "1000 * histogram_quantile(0.99, sum by (le) (rate("
            f'recloop_stage_duration_seconds_bucket{{{api},stage="total"}}[20s])))',
            t0,
            t1,
        ),  # fmt: skip
        "cpu": {c: cpu(c) for c in ("api", "model-server", "aggregator", "billing")},
        "throttled": {c: throttled(c) for c in ("api", "aggregator")},
    }


def bundle_load_seconds(pod: str) -> float | None:
    """The model server logs `bundle <version> loaded from <dir> in <s>s` once it has passed."""
    out = kubectl("logs", pod, check=False, timeout=30)
    for line in out.splitlines():
        if " loaded from " in line and " in " in line:
            with contextlib.suppress(ValueError, IndexError):
                return float(line.rsplit(" in ", 1)[1].split("s:")[0])
    return None


# ---------------------------------------------------------------- rollout


def broken_bundle(seed: int = 20261108) -> str:
    """A copy of the current bundle whose item_ids.json (FAISS row -> item id) is shuffled, with
    the manifest's checksum updated: it passes the checksums, the item count and dimension
    checks, and fails the golden queries (the wrong items for every user, silently)."""
    src = (BUNDLES / "current").resolve()
    dst = BUNDLES / BROKEN
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
    ids = json.loads((dst / "item_ids.json").read_text())
    random.Random(seed).shuffle(ids)
    data = json.dumps(ids, separators=(",", ":")).encode()
    (dst / "item_ids.json").write_bytes(data)
    manifest = json.loads((dst / "manifest.json").read_text())
    manifest["files"]["item_ids.json"] = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }
    manifest["version"] = f"{manifest['version']}-broken"
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return f"/bundles/{BROKEN}"


def new_pod_log(since: float) -> str:
    """The last lines the newest model-server pod's crashed container wrote (its previous run;
    the current one may still be starting)."""
    pods = k8s.kubectl_json("get", "pods", "-l", "app.kubernetes.io/name=model-server")["items"]
    new = [p for p in pods if (k8s.ts(p["metadata"]["creationTimestamp"]) or 0) >= since]
    if not new:
        return ""
    name = max(new, key=lambda p: p["metadata"]["creationTimestamp"])["metadata"]["name"]
    out = kubectl("logs", name, "--previous", "--tail=4", check=False, timeout=30)
    return (out or kubectl("logs", name, "--tail=4", check=False, timeout=30))[-800:]


def wait_condition(deploy: str, cond: str, reason: str | None, timeout: float) -> float | None:
    """Seconds until the Deployment's condition `cond` is False with `reason` (None: True)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = k8s.kubectl_json("get", deploy).get("status", {})
        for c in st.get("conditions", []):
            if c["type"] != cond:
                continue
            if reason is None and c["status"] == "True":
                return time.time() - t0
            if reason is not None and c.get("reason") == reason:
                return time.time() - t0
        time.sleep(2)
    return None


def run_rollout(variant: str, rep: int, args, out: Path) -> dict:
    deploy = "deployment/model-server"
    state = reset()
    broken_log = ""
    duration = {"gated": args.rollout_seconds, "no-probe": args.rollout_seconds,
                "broken": args.rollout_seconds + 120}[variant]  # fmt: skip
    tag = f"rollout-{variant}-{rep}"
    actions = []

    def act(what: str) -> None:
        actions.append({"t": round(time.time(), 2), "action": what})
        log(f"  {what}")

    m0 = machine()
    with k8s.Watch(["model-server", "api"]) as watch:
        load = K6(f"{args.rollout_rate}:{duration}s", tag, out).start()
        load.at(30)
        if variant == "gated":
            kubectl("rollout", "restart", deploy)
            act("kubectl rollout restart deployment/model-server")
            k8s.rollout_status(deploy, timeout="5m")
            act("rollout complete")
        elif variant == "no-probe":
            patch = [
                {"op": "remove", "path": "/spec/template/spec/containers/0/readinessProbe"},
                {"op": "remove", "path": "/spec/template/spec/containers/0/startupProbe"},
            ]
            kubectl("patch", deploy, "--type=json", "-p", json.dumps(patch))
            act("removed the readiness and startup probes (a rollout of new pods)")
            k8s.rollout_status(deploy, timeout="5m")
            act("rollout complete")
        else:  # broken
            path = broken_bundle()
            patch = [
                {"op": "replace", "path": "/spec/template/spec/containers/0/args/1", "value": path}
            ]
            kubectl("patch", deploy, "--type=json", "-p", json.dumps(patch))
            act(f"new pod template: --bundle {path} (item ids shuffled)")
            stalled = wait_condition(deploy, "Progressing", "ProgressDeadlineExceeded", 300)
            act(f"rollout reported failed (ProgressDeadlineExceeded) after {stalled or 0:.0f} s")
            broken_log = new_pod_log(load.started)  # before the undo removes the pod
            kubectl("rollout", "undo", deploy)
            act("kubectl rollout undo deployment/model-server")
            k8s.rollout_status(deploy, timeout="5m")
            act("rollback complete")
        k6res = load.wait()
    t1 = time.time()
    m1 = machine()
    w = watch.result()
    loads = {}
    for p in w["pods"]:
        if p["component"] == "model-server" and p["created"] and p["created"] >= load.started:
            loads[p["name"]] = bundle_load_seconds(p["name"])
            if variant == "broken":
                p["log_tail"] = broken_log
    if variant == "no-probe":  # back to the chart's probes (one more, unmeasured, rollout)
        kubectl("rollout", "undo", deploy)
        k8s.rollout_status(deploy, timeout="5m")
    if variant == "broken":
        shutil.rmtree(BUNDLES / BROKEN, ignore_errors=True)
    return {
        "experiment": "rollout", "variant": variant, "rep": rep, "rate": args.rollout_rate,
        "duration": duration, "started": load.started, "ended": t1, "actions": actions,
        "k6": k6res, "watch": w, "bundle_load_seconds": loads,
        "server": server_side(load.started, t1), "state": state, "machine": [m0, m1],
    }  # fmt: skip


# ---------------------------------------------------------------- hpa


def run_hpa(rep: int, args, out: Path) -> dict:
    state = reset()
    tag = f"hpa-{rep}"
    profile = f"{args.hpa_base}:60s,{args.hpa_step}:{args.hpa_hold}s,{args.hpa_base}:120s"
    hpa = k8s.kubectl_json("get", "hpa", "api")["spec"]
    m0 = machine()
    with k8s.Watch(["api"]) as watch:
        load = K6(profile, tag, out, conn_close=args.conn_close).start()
        k6res = load.wait()
    t1 = time.time()
    m1 = machine()
    return {
        "experiment": "hpa", "variant": "step", "rep": rep, "profile": profile,
        "base": args.hpa_base, "step": args.hpa_step, "hold": args.hpa_hold,
        "conn_close": args.conn_close, "hpa": hpa, "started": load.started, "ended": t1,
        "step_at": load.started + 61, "step_end": load.started + 61 + args.hpa_hold,
        "k6": k6res, "watch": watch.result(), "server": server_side(load.started, t1),
        "state": state, "machine": [m0, m1],
    }  # fmt: skip


# ---------------------------------------------------------------- keda


def run_keda_billing(rep: int, args, out: Path) -> dict:
    from sim import ads as ads_mod

    os.environ.update(k8s.host_env())
    ads = ads_mod.Ads(k8s.API_URL, k8s.DATABASE_URL)
    ads.require_no_active_campaigns()
    items = ads.pick_items(60, 20261108 + rep)
    # Budgets far above what the run can spend: the slots stay up for the whole run.
    specs = [{"bid": bid, "budget": 100_000, "freq_cap": 100, "items": items[i * 10 : (i + 1) * 10]}
             for i, bid in enumerate((60, 50, 40, 30, 20, 10))]  # fmt: skip
    camps = ads.create(f"KEDA run {rep}", specs)
    ids = [c["campaign_id"] for c in camps]
    try:
        ads.wait_campaigns(ids, user=1)
        state = reset()
        tag = f"keda-billing-{rep}"
        actions = []
        m0 = machine()
        with k8s.Watch(["billing"], groups=["billing"]) as watch:
            load = K6(f"{args.keda_rate}:{args.keda_seconds}s", tag, out).start()
            load.at(60)
            pg_toxic(True, args.pg_latency_ms)
            actions.append(
                {
                    "t": round(time.time(), 2),
                    "action": f"postgres +{args.pg_latency_ms} ms for billing",
                }
            )
            load.at(60 + args.slow_seconds)
            pg_toxic(False)
            actions.append({"t": round(time.time(), 2), "action": "postgres back to normal"})
            k6res = load.wait()
            settle(watch, "billing", args.settle_seconds)
        t1 = time.time()
    finally:
        pg_toxic(False)
        ads.end(ids)
        ads.close()
    return {
        "experiment": "keda", "variant": "billing", "rep": rep, "rate": args.keda_rate,
        "slow_seconds": args.slow_seconds, "pg_latency_ms": args.pg_latency_ms,
        "scaledobject": k8s.kubectl_json("get", "scaledobject", "billing")["spec"],
        "started": load.started, "load_ended": load.started + args.keda_seconds, "ended": t1,
        "actions": actions, "k6": k6res, "watch": watch.result(),
        "campaigns": ids, "server": server_side(load.started, t1), "keda_metric": keda_metric(
            "billing", load.started, t1), "clicks": k8s.prom_range(
            "sum(rate(recloop_billing_clicks_total[20s]))", load.started, t1),
        "state": state, "machine": [m0, machine()],
    }  # fmt: skip


def run_keda_aggregator(variant: str, rep: int, args, out: Path) -> dict:
    cap = variant == "aggregator-capped"
    state = reset()
    tag = f"keda-{variant}-{rep}"
    actions = []
    spec = k8s.kubectl_json("get", "scaledobject", "aggregator")["spec"]
    m0 = machine()
    try:
        if cap:
            keda_max("aggregator", 1)
        with k8s.Watch(["aggregator"], groups=["aggregator"]) as watch:
            load = K6(f"{args.keda_rate}:{args.keda_seconds}s", tag, out).start()
            load.at(30)
            keda_pause("aggregator", 0)
            actions.append({"t": round(time.time(), 2), "action": "aggregator paused (0 replicas)"})
            load.at(30 + args.pause_seconds)
            keda_pause("aggregator", None)
            actions.append({"t": round(time.time(), 2), "action": "aggregator resumed"})
            k6res = load.wait()
            settle(watch, "aggregator", args.settle_seconds)
        t1 = time.time()
    finally:
        keda_pause("aggregator", None)
        if cap:
            keda_max("aggregator", spec["maxReplicaCount"])
    return {
        "experiment": "keda", "variant": variant, "rep": rep, "rate": args.keda_rate,
        "pause_seconds": args.pause_seconds, "scaledobject": spec,
        "max_replicas": 1 if cap else spec["maxReplicaCount"], "started": load.started,
        "load_ended": load.started + args.keda_seconds, "ended": t1, "actions": actions,
        "k6": k6res, "watch": watch.result(), "keda_metric": keda_metric("aggregator", load.started, t1),
        "server": server_side(load.started, t1), "state": state, "machine": [m0, machine()],
    }  # fmt: skip


def settle(watch: k8s.Watch, workload: str, seconds: float) -> None:
    """After the load: keep watching until the group has no lag and the workload is back at
    one replica (scale-in), for at most `seconds`."""
    t0 = time.time()
    while time.time() - t0 < seconds:
        spec, ready = k8s.replicas(f"statefulset/{workload}")
        lag = k8s.group_lag(workload)
        if spec == 1 and ready == 1 and lag is not None and lag < 100:
            return
        time.sleep(5)


def keda_metric(name: str, t0: float, t1: float) -> list[dict]:
    """The value KEDA's scaler computed (total lag) for the ScaledObject, as KEDA exported it."""
    return k8s.prom_range(f'max(keda_scaler_metrics_value{{scaledObject="{name}"}})', t0, t1)


# ---------------------------------------------------------------- driver


def environment() -> dict:
    from ml.eval import machine as m

    images = {}
    env_file = ROOT / ".cache" / "kind" / "images.env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            k, _, v = line.partition("=")
            images[k] = v
    lock = (ROOT / "deploy" / "kind" / "versions.lock").read_text()
    pins = [ln.split()[:3] for ln in lock.splitlines() if ln.startswith(("chart ", "node "))]
    tools = {ln.split()[1]: ln.split()[2] for ln in lock.splitlines() if ln.startswith("tool ")}
    k6v = subprocess.run([str(K6_BIN), "version"], capture_output=True, text=True).stdout.strip()
    current = (BUNDLES / "current").resolve()
    files = set(json.loads((current / "manifest.json").read_text())["files"])
    return {
        "machine": m.info(),
        "node": k8s.node_info(),
        "pins": pins,
        "tools": tools,
        "k6": k6v.splitlines()[0] if k6v else None,
        # The bundle the cluster serves (artifacts/bundles/current): Rank runs when it has a
        # ranker (M4), ads get the model's pCTR when it has one.
        "bundle": {
            "version": current.name,
            "ranker": "ranker.onnx" in files,
            "pctr": "pctr.json" in files,
        },
        "images": images,
        "containers": k8s.containers_resources(),
        "hpa": k8s.kubectl_json("get", "hpa", "api")["spec"],
        "scaledobjects": {
            n: k8s.kubectl_json("get", "scaledobject", n)["spec"] for n in ("aggregator", "billing")
        },
    }


def plan(args) -> list[tuple]:
    reps = range(1, args.repeats + 1)
    out = []
    if "rollout" in args.only:
        for r in reps:  # interleaved: drift spreads over both variants
            out += [("rollout", "gated", r), ("rollout", "no-probe", r)]
        out.append(("rollout", "broken", 1))
    if "hpa" in args.only:
        out += [("hpa", "step", r) for r in reps]
    if "keda" in args.only:
        for r in reps:
            out += [("keda", "billing", r), ("keda", "aggregator", r)]
            # The aggregator drained its backlog before KEDA's first poll in the smoke runs
            # (2026-11-08): the capped comparison only on request.
            if args.capped:
                out.append(("keda", "aggregator-capped", r))
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--only", default="rollout,hpa,keda")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", default=str(OUT / "results.json"))
    p.add_argument("--resume", metavar="REASON", help="keep the finished runs of --out")
    p.add_argument("--rollout-rate", type=int, default=200)
    p.add_argument("--rollout-seconds", type=int, default=150)
    p.add_argument("--hpa-base", type=int, default=100)
    p.add_argument("--hpa-step", type=int, default=800)
    p.add_argument("--hpa-hold", type=int, default=240)
    p.add_argument("--conn-close", type=float, default=0.02)
    p.add_argument("--keda-rate", type=int, default=300)
    p.add_argument("--keda-seconds", type=int, default=300)
    p.add_argument("--pg-latency-ms", type=int, default=200)
    p.add_argument("--slow-seconds", type=int, default=120)
    p.add_argument("--pause-seconds", type=int, default=90)
    p.add_argument("--settle-seconds", type=int, default=240)
    p.add_argument("--smoke", action="store_true", help="one short run of each kind")
    p.add_argument("--capped", action="store_true", help="also the aggregator capped at 1 pod")
    args = p.parse_args(argv)
    if args.smoke:
        short = {"repeats": 1, "rollout_seconds": 90, "hpa_hold": 120, "keda_seconds": 180,
                 "slow_seconds": 60, "pause_seconds": 45, "settle_seconds": 120}  # fmt: skip
        for k, v in short.items():
            setattr(args, k, v)
        if args.out == str(OUT / "results.json"):
            args.out = str(OUT / "smoke.json")
    args.only = set(args.only.split(","))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {"meta": {}, "runs": []}
    if args.resume and out_path.exists():
        data = json.loads(out_path.read_text())
        data["meta"].setdefault("resumed", []).append({"at": time.time(), "reason": args.resume})
    done = {(r["experiment"], r["variant"], r["rep"]) for r in data["runs"]}
    data["meta"].setdefault("environment", environment())
    data["meta"]["args"] = {
        k: (sorted(v) if isinstance(v, set) else v) for k, v in vars(args).items()
    }
    for key in plan(args):
        if key in done:
            continue
        exp, variant, rep = key
        log(f"{exp} {variant} #{rep}")
        if exp == "rollout":
            r = run_rollout(variant, rep, args, out_path.parent)
        elif exp == "hpa":
            r = run_hpa(rep, args, out_path.parent)
        elif variant == "billing":
            r = run_keda_billing(rep, args, out_path.parent)
        else:
            r = run_keda_aggregator(variant, rep, args, out_path.parent)
        data["runs"].append(r)
        out_path.write_text(json.dumps(data, indent=1))
        k = r["k6"]
        log(f"  {k['requests']} requests, {k['errors']} errors, {k['degraded']} degraded, "
            f"P99 {k['recs'].get('p(99)')} ms")  # fmt: skip
    log(f"done: {out_path}")


if __name__ == "__main__":
    main()
