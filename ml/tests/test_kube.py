"""The Kubernetes deployment's logic without a cluster: the charts give every container requests
and limits (rendered with the pinned helm, when it is there), the k6 window parsing, the broken
bundle of the rollout run, the drill's chaos platforms, and deploy.md (kube/report.py)."""

import hashlib
import json
import subprocess
import threading
from pathlib import Path

import pytest

from kube import drill as kube_drill
from kube import report, run
from ml.tests.test_billing import drill_doc
from sim import billing_drill

ROOT = Path(__file__).resolve().parents[2]
HELM = ROOT / ".cache" / "bin" / "helm"


# ---------------------------------------------------------------- charts


def render(chart: str, *args: str) -> list[str]:
    env = {"HELM_CACHE_HOME": str(ROOT / ".cache/helm/cache"),
           "HELM_CONFIG_HOME": str(ROOT / ".cache/helm/config"),
           "HELM_DATA_HOME": str(ROOT / ".cache/helm/data"), "PATH": "/usr/bin:/bin"}  # fmt: skip
    out = subprocess.run(
        [str(HELM), "template", "x", str(ROOT / "deploy" / "helm" / chart), "--kube-version", "1.36.4", *args],
        capture_output=True, text=True, check=True, env=env,
    ).stdout  # fmt: skip
    return out.split("\n---\n")


@pytest.mark.skipif(not HELM.exists(), reason="no pinned helm (make kind-tools)")
@pytest.mark.parametrize("chart", ["recloop", "recloop-infra"])
def test_every_container_has_requests_and_limits(chart):
    yaml = pytest.importorskip("yaml")
    files = ["--set-file", "logSink.config=deploy/connect/log-sink.yaml",
             "--set-file", "grafana.dashboard=deploy/grafana/dashboards/recloop.json"]  # fmt: skip
    docs = [yaml.safe_load(d) for d in render(chart, *(files if chart == "recloop" else []))]
    pods = 0
    for d in docs:
        if not d or d["kind"] not in ("Deployment", "StatefulSet", "Job"):
            continue
        spec = d["spec"]["template"]["spec"]
        for c in spec.get("containers", []) + spec.get("initContainers", []):
            pods += 1
            for part in ("requests", "limits"):
                got = c.get("resources", {}).get(part, {})
                assert {"cpu", "memory"} <= set(got), (d["metadata"]["name"], c["name"], part)
    # api, model-server, aggregator, billing, log-sink, web, pg-proxy; two Redis + exporter, the topic Job
    assert pods == (7 if chart == "recloop" else 5)


THIRD_PARTY = {"kube-prometheus-stack": "monitoring", "metrics-server": "kube-system", "keda": "keda",
               "cloudnative-pg": "cnpg-system", "redpanda": "redpanda"}  # fmt: skip


def chart_file(name: str) -> Path | None:
    for line in (ROOT / "deploy" / "kind" / "versions.lock").read_text().splitlines():
        f = line.split()
        if f[:2] == ["chart", name]:
            p = ROOT / ".cache" / "charts" / f"{name}-{f[2]}.tgz"
            return p if p.exists() else None
    return None


@pytest.mark.skipif(not HELM.exists(), reason="no pinned helm (make kind-tools)")
@pytest.mark.parametrize("name", sorted(THIRD_PARTY))
def test_third_party_containers_have_requests_and_limits(name):
    yaml = pytest.importorskip("yaml")
    chart = chart_file(name)
    if chart is None:
        pytest.skip("chart not downloaded (deploy/kind/cluster.sh charts)")
    env = {"HELM_CACHE_HOME": str(ROOT / ".cache/helm/cache"),
           "HELM_CONFIG_HOME": str(ROOT / ".cache/helm/config"),
           "HELM_DATA_HOME": str(ROOT / ".cache/helm/data"), "PATH": "/usr/bin:/bin"}  # fmt: skip
    out = subprocess.run(
        [str(HELM), "template", name, str(chart), "-n", THIRD_PARTY[name], "--kube-version", "1.36.4",
         "-f", str(ROOT / "deploy" / "kind" / "values" / f"{name}.yaml")],
        capture_output=True, text=True, check=True, env=env,
    ).stdout  # fmt: skip
    for d in yaml.safe_load_all(out):
        if not d or d["kind"] not in ("Deployment", "StatefulSet", "DaemonSet", "Job"):
            continue
        spec = d["spec"]["template"]["spec"]
        for c in spec.get("containers", []) + spec.get("initContainers", []):
            for part in ("requests", "limits"):
                got = (c.get("resources") or {}).get(part, {})
                assert {"cpu", "memory"} <= set(got), (d["metadata"]["name"], c["name"], part)


@pytest.mark.skipif(not HELM.exists(), reason="no pinned helm (make kind-tools)")
def test_chart_wiring():
    yaml = pytest.importorskip("yaml")
    files = ["--set-file", "logSink.config=deploy/connect/log-sink.yaml"]
    docs = {
        (d["kind"], d["metadata"]["name"]): d
        for d in map(yaml.safe_load, render("recloop", *files))
        if d
    }
    ms = docs[("Deployment", "model-server")]["spec"]
    c = ms["template"]["spec"]["containers"][0]
    # Readiness is the gRPC health service (SERVING only after the bundle checks), and a
    # rolling update never goes below the replica count.
    assert c["readinessProbe"]["grpc"]["port"] == 18081
    assert ms["strategy"]["rollingUpdate"] == {"maxSurge": 1, "maxUnavailable": 0}
    # The consumers' static member id is the StatefulSet pod name.
    billing = docs[("StatefulSet", "billing")]["spec"]["template"]["spec"]["containers"][0]
    env = {e["name"]: e.get("value") for e in billing["env"]}
    assert env["BILLING_INSTANCE_ID"] == "$(POD_NAME)"
    assert "@pg-proxy:15433/" in env["DATABASE_URL"]
    assert (
        docs[("ScaledObject", "billing")]["spec"]["triggers"][0]["metadata"]["consumerGroup"]
        == "billing"
    )
    # The HPA owns the api's replica count.
    assert "replicas" not in docs[("Deployment", "api")]["spec"]
    api = docs[("Deployment", "api")]["spec"]["template"]["spec"]["containers"][0]
    assert {e["name"] for e in api["env"]} >= {"REDIS_URL", "CACHE_REDIS_URL", "MODEL_SERVER_ADDR"}


# ---------------------------------------------------------------- k6


def k6_summary() -> dict:
    def trend(p99, n):
        return {
            "values": {"med": p99 / 2, "p(95)": p99 * 0.9, "p(99)": p99, "max": p99 * 2, "count": n}
        }

    def rate(passes, fails):
        return {
            "values": {"rate": passes / max(1, passes + fails), "passes": passes, "fails": fails}
        }

    return {
        "setup_data": {"t0": 1_800_000_000_000},
        "metrics": {
            "recs_ms": trend(9.0, 30),
            "recs_errors": rate(1, 29),
            "recs_degraded": rate(2, 27),
            "events_errors": rate(0, 12),
            "dropped_iterations": {"values": {"count": 3}},
            "recs_ms{w:w000}": trend(4.0, 10),
            "recs_errors{w:w000}": rate(0, 10),
            "recs_degraded{w:w000}": rate(0, 10),
            "recs_ms{w:w001}": {"values": {"count": 0}},
            "recs_ms{w:w002}": trend(9.0, 20),
            "recs_errors{w:w002}": rate(1, 19),
            "recs_degraded{w:w002}": rate(2, 17),
        },
    }


def test_k6_windows():
    r = run.k6_result(k6_summary())
    assert (r["requests"], r["errors"], r["degraded"], r["answered"]) == (30, 1, 2, 29)
    assert r["dropped_iterations"] == 3
    # The empty window is skipped; the others keep their start time.
    assert [(w["t"], w["n"], w["p99"], w["errors"], w["degraded"]) for w in r["windows"]] == [
        (1_800_000_000.0, 10, 4.0, 0, 0),
        (1_800_000_010.0, 20, 9.0, 1, 2),
    ]


# ---------------------------------------------------------------- the broken bundle


def test_broken_bundle_passes_checksums_and_shuffles_ids(tmp_path, monkeypatch):
    src = tmp_path / "v1"
    src.mkdir()
    ids = json.dumps(list(range(50))).encode()
    (src / "item_ids.json").write_bytes(ids)
    (src / "manifest.json").write_text(json.dumps({"version": "v1", "files": {
        "item_ids.json": {"sha256": hashlib.sha256(ids).hexdigest(), "bytes": len(ids)}}}))  # fmt: skip
    (tmp_path / "current").symlink_to("v1")
    monkeypatch.setattr(run, "BUNDLES", tmp_path)
    path = run.broken_bundle()
    dst = tmp_path / run.BROKEN
    assert path == f"/bundles/{run.BROKEN}"
    data = (dst / "item_ids.json").read_bytes()
    got = json.loads(data)
    assert sorted(got) == list(range(50)) and got != list(range(50))
    m = json.loads((dst / "manifest.json").read_text())
    assert m["files"]["item_ids.json"] == {
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }
    assert m["version"] == "v1-broken"
    assert json.loads((src / "item_ids.json").read_text()) == list(
        range(50)
    )  # the original is intact


# ---------------------------------------------------------------- the drill's platforms


class FakePlatform:
    name = "fake"

    def __init__(self):
        self.calls = []

    def kill_billing(self, down, note):
        self.calls.append(("kill", down))
        note("killed")

    def restart_redpanda(self, note):
        self.calls.append(("redpanda",))
        note("restarted")

    def ensure_billing(self):
        self.calls.append(("ensure",))

    def reconciler_during(self, since):
        return {}


def test_chaos_uses_the_platform(monkeypatch):
    monkeypatch.setattr(billing_drill, "pg_slow", lambda on, ms: None)
    steps = [{"t": 0.0, "action": "kill_billing", "down": 3.5}, {"t": 0.0, "action": "restart_redpanda"},
             {"t": 0.0, "action": "pg_slow", "seconds": 0.01}]  # fmt: skip
    p, log = FakePlatform(), []
    billing_drill.chaos(steps, 200, log, threading.Event(), p)
    assert p.calls == [("kill", 3.5), ("redpanda",)]
    assert [x["action"] for x in log] == ["killed", "restarted", "postgres +200 ms for billing",
                                          "postgres back to normal (0 s)"]  # fmt: skip


def test_kube_deletes_billing_pods_alternately_graceful_and_forced(monkeypatch):
    calls = []
    pods = {"items": [{"metadata": {"name": "billing-0"}}, {"metadata": {"name": "billing-1"}},
                      {"metadata": {"name": "billing-2", "deletionTimestamp": "x"}}]}  # fmt: skip
    monkeypatch.setattr(kube_drill.k8s, "kubectl_json", lambda *a, **k: pods)
    monkeypatch.setattr(kube_drill, "kubectl", lambda *a, **k: calls.append(a))
    notes = []
    k = kube_drill.Kube(seed=1)
    for _ in range(4):
        k.kill_billing(5, notes.append)
    assert [("--force" in c) for c in calls] == [False, True, False, True]
    assert all(c[2] in ("billing-0", "billing-1") for c in calls)  # never one already terminating
    assert notes[0].endswith("(graceful, SIGTERM)") and notes[1].endswith("(forced, grace 0)")


# ---------------------------------------------------------------- deploy.md


def pod(name, comp, created, started=None, ready=None, deleting=None, gone=None, **kw):
    p = {"name": name, "component": comp, "created": created, "first_seen": created, "restarts": 0}
    for k, v in (("started", started), ("ready", ready), ("deleting", deleting), ("gone", gone)):
        if v is not None:
            p[k] = v
    return {**p, **kw}


def windows(t0, n, p99=5.0, errors=0, degraded=0, per=100):
    return [{"t": t0 + 5 * i, "n": per, "p50": 1.0, "p95": 3.0, "p99": p99, "max": 2 * p99,
             "errors": errors, "degraded": degraded, "ok": per - errors} for i in range(n)]  # fmt: skip


def k6(t0, ws):
    return {"t0": t0, "recs": {"p(99)": max(w["p99"] for w in ws)}, "requests": sum(w["n"] for w in ws),
            "errors": sum(w["errors"] for w in ws), "degraded": sum(w["degraded"] for w in ws),
            "answered": sum(w["ok"] for w in ws), "event_posts": 0, "event_errors": 0,
            "dropped_iterations": 0, "windows": ws}  # fmt: skip


def server():
    return {"degraded_reasons": {"sasrec_error": 3.0}, "breaker_transitions": {}, "model_server_calls": {},
            "recs_status": {"200": 1000.0}, "api_p99_ms": [], "cpu": {c: [] for c in ("api", "model-server",
            "aggregator", "billing")}, "throttled": {"api": [], "aggregator": []}}  # fmt: skip


def rollout_run(variant, t0=1_800_000_000.0):
    trig = t0 + 30
    actions = [{"t": trig, "action": "kubectl rollout restart deployment/model-server"},
               {"t": trig + 40, "action": "rollout complete"}]  # fmt: skip
    pods = [pod("model-server-a", "model-server", t0 - 600, t0 - 599, t0 - 596, trig + 15, trig + 26),
            pod("model-server-b", "model-server", trig + 1, trig + 2, trig + 6),
            pod("model-server-c", "model-server", trig + 21, trig + 22, trig + 26)]  # fmt: skip
    if variant == "broken":
        actions = [{"t": trig, "action": "new pod template: --bundle /bundles/x"},
                   {"t": trig + 121, "action": "rollout reported failed (ProgressDeadlineExceeded) after 121 s"},
                   {"t": trig + 122, "action": "kubectl rollout undo deployment/model-server"},
                   {"t": trig + 125, "action": "rollback complete"}]  # fmt: skip
        pods = [pod("model-server-a", "model-server", t0 - 600, t0 - 599, t0 - 596),
                pod("model-server-x", "model-server", trig + 1, trig + 2, None, trig + 123, trig + 124,
                    restarts=4, waiting=["CrashLoopBackOff"],
                    log_tail="... BundleError: /bundles/x: golden queries failed: {'recall': 0.01}\n")]  # fmt: skip
    ws = windows(t0, 30, degraded=1 if variant == "no-probe" else 0)
    samples = [
        {"t": t0 + i, "ready": {"model-server": 2, "api": 1}, "pods": {}, "desired": {}}
        for i in range(150)
    ]
    return {"experiment": "rollout", "variant": variant, "rep": 1, "rate": 200, "duration": 150,
            "started": t0, "ended": t0 + 150, "actions": actions, "k6": k6(t0, ws),
            "watch": {"pods": pods, "samples": samples, "lag": [], "poll_errors": 0},
            "bundle_load_seconds": {"model-server-b": 1.9, "model-server-c": 2.1},
            "server": server(), "machine": [{"load1": 1.5, "other_containers": []}] * 2}  # fmt: skip


def hpa_run(t0=1_800_001_000.0):
    step_at, step_end = t0 + 61, t0 + 61 + 240
    ws = windows(t0, 12, p99=4.0) + [  # base
        {**w, "p99": 80.0} for w in windows(t0 + 60, 8)] + windows(t0 + 100, 52, p99=6.0)  # fmt: skip
    samples = []
    for i in range(0, 430):
        t = t0 + i
        n = 1 if t < step_at + 32 else 3
        samples.append({"t": t, "ready": {"api": n}, "pods": {"api": n},
                        "desired": {"api": 1 if t < step_at + 18 else 3}})  # fmt: skip
    pods = [pod("api-a", "api", t0 - 900, t0 - 899, t0 - 897),
            pod("api-b", "api", step_at + 19, step_at + 20, step_at + 22),
            pod("api-c", "api", step_at + 19, step_at + 21, step_at + 32)]  # fmt: skip
    hpa = {
        "minReplicas": 1,
        "maxReplicas": 4,
        "metrics": [{"resource": {"target": {"averageUtilization": 60}}}],
    }
    return {"experiment": "hpa", "variant": "step", "rep": 1, "base": 100, "step": 800, "hold": 240,
            "conn_close": 0.02, "hpa": hpa, "started": t0, "ended": t0 + 430, "step_at": step_at,
            "step_end": step_end, "k6": k6(t0, ws), "watch": {"pods": pods, "samples": samples,
            "lag": [], "poll_errors": 0}, "server": server(), "machine": [{"load1": 1.2, "other_containers": []}]}  # fmt: skip


def keda_run(variant, t0=1_800_002_000.0):
    comp = "billing" if variant == "billing" else "aggregator"
    if comp == "billing":
        actions = [{"t": t0 + 60, "action": "postgres +200 ms for billing"},
                   {"t": t0 + 180, "action": "postgres back to normal"}]  # fmt: skip
    else:
        actions = [{"t": t0 + 30, "action": "aggregator paused (0 replicas)"},
                   {"t": t0 + 120, "action": "aggregator resumed"}]  # fmt: skip
    t_on, t_off = actions[0]["t"], actions[1]["t"]
    lag = [{"t": t, comp: (0 if t < t_on else 40_000 if t < t_off else 500 if t > t_off + 40 else 20_000)}
           for t in range(int(t0), int(t0) + 400, 5)]  # fmt: skip
    cap = variant == "aggregator-capped"
    samples = [{"t": t, "ready": {comp: (1 if cap or t < t_on + 20 or t > t_off + 150 else 3)}, "pods": {},
                "desired": {}} for t in range(int(t0), int(t0) + 400)]  # fmt: skip
    return {"experiment": "keda", "variant": variant, "rep": 1, "rate": 300, "started": t0,
            "ended": t0 + 400, "actions": actions, "k6": k6(t0, windows(t0, 60)),
            "watch": {"pods": [], "samples": samples, "lag": lag, "poll_errors": 0},
            "server": server(), "machine": [{"load1": 1.0, "other_containers": []}]}  # fmt: skip


def results_doc():
    env = {"machine": {"chip": "Apple M4 Pro", "memory_gb": 24, "os": "macOS"},
           "node": {"kubelet": "v1.36.4", "allocatable": {"cpu": "8", "memory": "12000000Ki"}},
           "pins": [["chart", "keda", "2.21.0"], ["node", "kindest/node:v1.36.4@sha256:abc", ""]],
           "tools": {"kind": "v0.33.0"}, "k6": "k6 v2.3.0", "images": {"GO_TAG": "abc"},
           "containers": [{"namespace": "recloop", "pod": "api-5d9c-x2", "owner": "api-5d9c", "workload": "api", "container": "api",
                           "init": False, "image": "i", "requests": {"cpu": "500m", "memory": "256Mi"},
                           "limits": {"cpu": "500m", "memory": "512Mi"}}]}  # fmt: skip
    runs = [rollout_run("gated"), rollout_run("no-probe"), rollout_run("broken"), hpa_run(),
            keda_run("billing"), keda_run("aggregator"), keda_run("aggregator-capped")]  # fmt: skip
    return {"meta": {"environment": env}, "runs": runs}


def kube_drill_doc():
    d = drill_doc()
    d["chaos"] = [{"t": 60.0, "action": "deleted pod billing-0 (graceful, SIGTERM)"},
                  {"t": 90.0, "action": "deleted pod redpanda-0"}]  # fmt: skip
    return d


def test_deploy_report_renders_deterministically():
    text = report.render(results_doc(), kube_drill_doc())
    assert text == report.render(results_doc(), kube_drill_doc())
    assert "**Preliminary.**" not in text
    # rollout: 0 failed of 3,000, 0 degraded in the rollout window, Ready 4 s after start.
    assert (
        "1 rolling updates of both model-server pods under 200 page views/s (open loop), 0 of 3,000"
        in text
    )
    assert "Ready 4.0 s after its container started; never fewer than 2 ready pods" in text
    assert "the rollout was reported failed after 121 s and undone" in text
    assert "golden queries failed" in text
    # hpa: scale-up asked 18 s after the step, first new pod ready 22 s after it, 3 pods.
    assert (
        "asked for more pods 18 s after the step and the first new pod was ready 22 s after it"
        in text
    )
    assert "the api reached 3 pods" in text
    assert "80.0 ms on one pod, 6.0 ms once scaled out" in text
    # keda
    assert "lag up to 40,000 records, scaled to 3 pods 20 s after the slowdown began" in text
    assert "drained in 45 s with KEDA (3 pods), 45 s capped at one pod" in text
    # drill
    assert "**the ledger equals the simulator's truth charge by charge**" in text
    assert "| 60.0 | deleted pod billing-0 (graceful, SIGTERM) |" in text
    assert "| api | api | 500m | 256Mi | 500m | 512Mi |" in text


def test_deploy_report_without_runs_is_preliminary():
    text = report.render(None, None)
    assert (
        "**Preliminary.** not run yet: rollout, hpa, keda; no billing drill on the cluster yet."
        in text
    )
    assert "Not run yet: `make deploy-rollout`." in text
    assert "Not run yet: `make deploy-drill`." in text
