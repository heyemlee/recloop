"""The kind cluster from the host (deploy/kind/cluster.sh made it): kubectl with the repo's
kubeconfig, the NodePort addresses, Prometheus queries, consumer lag, and a watcher that records
pods and replica counts once a second while a run goes on.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / ".cache" / "bin"
KUBECONFIG = ROOT / ".cache" / "kind" / "kubeconfig"
CLUSTER = os.environ.get("KIND_CLUSTER", "recloop")
NS = os.environ.get("KIND_NAMESPACE", "recloop")

# kind's port mappings (deploy/kind/cluster.yaml).
API_URL = os.environ.get("KIND_API_URL", "http://127.0.0.1:30080")
PROMETHEUS_URL = os.environ.get("KIND_PROMETHEUS_URL", "http://127.0.0.1:30090")
DATABASE_URL = os.environ.get(
    "KIND_DATABASE_URL", "postgres://recloop:recloop@localhost:30432/recloop?sslmode=disable"
)
REDIS_URL = os.environ.get("KIND_REDIS_URL", "redis://localhost:30379/0")
CACHE_REDIS_URL = os.environ.get("KIND_CACHE_REDIS_URL", "redis://localhost:30380/0")
KAFKA_BROKERS = os.environ.get("KIND_KAFKA_BROKERS", "localhost:31092")
PG_PROXY_URL = os.environ.get("KIND_PG_PROXY_URL", "http://127.0.0.1:30474")


def host_env() -> dict[str, str]:
    """The environment that points the host's tools (the simulator, ml.data.seed_redis,
    `aggregator wait-lag`, `billing reconcile`) at the cluster instead of compose."""
    return {
        "RECLOOP_API_URL": API_URL,
        "DATABASE_URL": DATABASE_URL,
        "REDIS_URL": REDIS_URL,
        "CACHE_REDIS_URL": CACHE_REDIS_URL,
        "KAFKA_BROKERS": KAFKA_BROKERS,
        "PG_PROXY_URL": PG_PROXY_URL,
        "PROMETHEUS_URL": PROMETHEUS_URL,
    }


def kubectl(*args: str, ns: str | None = NS, check: bool = True, timeout: float = 300) -> str:
    cmd = [str(BIN / "kubectl"), "--kubeconfig", str(KUBECONFIG), "--context", f"kind-{CLUSTER}"]
    if ns:
        cmd += ["-n", ns]
    r = subprocess.run([*cmd, *args], capture_output=True, text=True, timeout=timeout, check=False)
    if check and r.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args)}: {r.stderr.strip()[-500:]}")
    return r.stdout


def kubectl_json(*args: str, ns: str | None = NS) -> dict:
    return json.loads(kubectl(*args, "-o", "json", ns=ns))


def ts(s: str | None) -> float | None:
    """A Kubernetes timestamp (RFC 3339, UTC) as unix seconds."""
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def http_json(method: str, url: str, body: dict | None = None, timeout: float = 10):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw else None


# ---------------------------------------------------------------- Prometheus


def prom_range(expr: str, start: float, end: float, step: float = 5) -> list[dict]:
    """A range query: [{metric: {...}, values: [[t, value], ...]}, ...] (values as floats)."""
    q = urlencode({"query": expr, "start": f"{start:.3f}", "end": f"{end:.3f}", "step": step})
    r = http_json("GET", f"{PROMETHEUS_URL}/api/v1/query_range?{q}", timeout=30)
    out = []
    for s in r.get("data", {}).get("result", []):
        vals = [[float(t), float(v)] for t, v in s["values"] if v not in ("NaN", "+Inf", "-Inf")]
        out.append({"metric": s["metric"], "values": vals})
    return out


def prom_value(expr: str, at: float) -> float | None:
    q = urlencode({"query": expr, "time": f"{at:.3f}"})
    r = http_json("GET", f"{PROMETHEUS_URL}/api/v1/query?{q}", timeout=30)
    res = r.get("data", {}).get("result", [])
    if not res:
        return None
    v = float(res[0]["value"][1])
    return None if v != v else v


def prom_by(expr: str, at: float, label: str) -> dict[str, float]:
    q = urlencode({"query": expr, "time": f"{at:.3f}"})
    r = http_json("GET", f"{PROMETHEUS_URL}/api/v1/query?{q}", timeout=30)
    out = {}
    for s in r.get("data", {}).get("result", []):
        v = float(s["value"][1])
        if v == v and v > 0:
            out[s["metric"].get(label, "")] = round(v, 4)
    return out


# ---------------------------------------------------------------- Redpanda

_LAG = re.compile(r"^TOTAL-LAG\s+(\d+)", re.M)


def group_lag(group: str) -> int | None:
    """Records the consumer group has not committed yet, summed over the partitions (rpk in the
    broker's pod: works whether or not the group's members, or the aggregator's lag exporter,
    are running)."""
    out = kubectl(
        "exec", "redpanda-0", "-c", "redpanda", "--", "rpk", "group", "describe", group,
        "--print-summary", ns="redpanda", check=False, timeout=30,
    )  # fmt: skip
    m = _LAG.search(out)
    return int(m[1]) if m else None


# ---------------------------------------------------------------- watching a run


class Watch:
    """Once a second, in a thread: the pods of `components` (app.kubernetes.io/name values) in
    the namespace, their workloads' replica counts and the autoscalers' desired counts, and the
    lag of `groups` (every `lag_every` seconds). After the run, `pods` has one entry per pod
    incarnation (created, started, ready, deletion requested, gone; from the API's own
    timestamps where it has them, else when the watch saw it) and `samples` the time series."""

    def __init__(
        self,
        components: list[str],
        groups: list[str] | None = None,
        interval: float = 1.0,
        lag_every: float = 5.0,
    ):
        self.components = components
        self.groups = groups or []
        self.interval, self.lag_every = interval, lag_every
        self.pods: dict[str, dict] = {}
        self.samples: list[dict] = []
        self.lag: list[dict] = []
        self.errors = 0
        self._stop = threading.Event()
        self._threads = [threading.Thread(target=self._loop, daemon=True)]
        if self.groups:
            self._threads.append(threading.Thread(target=self._lag_loop, daemon=True))

    def __enter__(self) -> Watch:
        for t in self._threads:
            t.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        for t in self._threads:
            t.join()
        try:
            self._poll()
        except (RuntimeError, ValueError, subprocess.TimeoutExpired):
            self.errors += 1

    def _loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self._poll()
            except (RuntimeError, ValueError, subprocess.TimeoutExpired):
                self.errors += 1
            self._stop.wait(max(0.0, self.interval - (time.monotonic() - t0)))

    def _lag_loop(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            self.lag.append({"t": round(now, 2), **{g: group_lag(g) for g in self.groups}})
            self._stop.wait(self.lag_every)

    def _poll(self) -> None:
        now = time.time()
        sel = f"app.kubernetes.io/name in ({','.join(self.components)})"
        pods = kubectl_json("get", "pods", "-l", sel)["items"]
        seen = set()
        for p in pods:
            uid = p["metadata"]["uid"]
            seen.add(uid)
            rec = self.pods.setdefault(
                uid,
                {
                    "name": p["metadata"]["name"],
                    "component": p["metadata"]["labels"].get("app.kubernetes.io/name"),
                    "created": ts(p["metadata"]["creationTimestamp"]),
                    "first_seen": round(now, 2),
                },
            )
            st = p.get("status", {})
            cs = (st.get("containerStatuses") or [{}])[0]
            started = cs.get("state", {}).get("running", {}).get("startedAt")
            if started and "started" not in rec:
                rec["started"] = ts(started)
            rec["restarts"] = cs.get("restartCount", 0)
            waiting = cs.get("state", {}).get("waiting", {}).get("reason")
            if waiting and waiting not in rec.setdefault("waiting", []):
                rec["waiting"].append(waiting)  # e.g. CrashLoopBackOff
            for c in st.get("conditions", []):
                if c["type"] == "Ready" and c["status"] == "True" and "ready" not in rec:
                    rec["ready"] = ts(c.get("lastTransitionTime")) or round(now, 2)
            dt = p["metadata"].get("deletionTimestamp")
            if dt and "deleting" not in rec:
                grace = p["metadata"].get("deletionGracePeriodSeconds") or 0
                rec["deleting"] = ts(dt) - grace
        for uid, rec in self.pods.items():
            if uid not in seen and "gone" not in rec:
                rec["gone"] = round(now, 2)
        sample = {"t": round(now, 2), "ready": {}, "pods": {}}
        for p in pods:
            comp = p["metadata"]["labels"].get("app.kubernetes.io/name")
            sample["pods"][comp] = sample["pods"].get(comp, 0) + 1
            ready = any(
                c["type"] == "Ready" and c["status"] == "True"
                for c in p.get("status", {}).get("conditions", [])
            )
            terminating = bool(p["metadata"].get("deletionTimestamp"))
            if ready and not terminating:
                sample["ready"][comp] = sample["ready"].get(comp, 0) + 1
        # The api's HPA and the ones KEDA made for its ScaledObjects (keda-hpa-<name>): what
        # they want and what they see (CPU utilisation in %, or KEDA's lag per replica).
        hpas = kubectl_json("get", "hpa")["items"]
        sample["desired"], sample["metric"] = {}, {}
        for h in hpas:
            target = h["spec"]["scaleTargetRef"]["name"]
            st = h.get("status", {})
            sample["desired"][target] = st.get("desiredReplicas")
            cur = (st.get("currentMetrics") or [{}])[0]
            value = cur.get("resource", {}).get("current", {}).get("averageUtilization")
            if value is None:
                value = cur.get("external", {}).get("current", {}).get("averageValue")
            sample["metric"][target] = value
        self.samples.append(sample)

    def result(self) -> dict:
        return {
            "pods": sorted(self.pods.values(), key=lambda r: (r["created"] or 0, r["name"])),
            "samples": self.samples,
            "lag": self.lag,
            "poll_errors": self.errors,
        }


def rollout_status(kind_name: str, ns: str = NS, timeout: str = "10m") -> None:
    kubectl("rollout", "status", kind_name, f"--timeout={timeout}", ns=ns, timeout=900)


def replicas(kind_name: str, ns: str = NS) -> tuple[int, int]:
    """(spec.replicas, status.readyReplicas) of a Deployment / StatefulSet."""
    o = kubectl_json("get", kind_name, ns=ns)
    return o["spec"].get("replicas", 1), o.get("status", {}).get("readyReplicas", 0)


def wait_replicas(kind_name: str, want: int, timeout: float = 600, log=print) -> float:
    """Wait until the workload has exactly `want` ready replicas (and wants that many)."""
    t0 = time.time()
    while True:
        spec, ready = replicas(kind_name)
        if spec == want and ready == want:
            return time.time() - t0
        if time.time() - t0 > timeout:
            raise RuntimeError(
                f"{kind_name}: {ready}/{spec} ready after {timeout:.0f} s, want {want}"
            )
        time.sleep(2)


def node_info() -> dict:
    """The node's Kubernetes version, capacity and allocatable (what the scheduler packs)."""
    n = kubectl_json("get", "nodes", ns=None)["items"][0]
    return {
        "name": n["metadata"]["name"],
        "kubelet": n["status"]["nodeInfo"]["kubeletVersion"],
        "container_runtime": n["status"]["nodeInfo"]["containerRuntimeVersion"],
        "capacity": {k: n["status"]["capacity"][k] for k in ("cpu", "memory")},
        "allocatable": {k: n["status"]["allocatable"][k] for k in ("cpu", "memory")},
    }


def containers_resources(ns: str | None = None) -> list[dict]:
    """Every container of the running pods with its image, requests and limits."""
    args = ["get", "pods"] + (["-A"] if ns is None else [])
    out = []
    for p in kubectl_json(*args, ns=ns)["items"]:
        owner = (p["metadata"].get("ownerReferences") or [{}])[0].get("name", "")
        for kind in ("initContainers", "containers"):
            for c in p["spec"].get(kind, []):
                out.append(
                    {
                        "namespace": p["metadata"]["namespace"],
                        "pod": p["metadata"]["name"],
                        "owner": owner,
                        "container": c["name"],
                        "init": kind == "initContainers",
                        "image": c["image"],
                        "requests": c.get("resources", {}).get("requests", {}),
                        "limits": c.get("resources", {}).get("limits", {}),
                    }
                )
    return out
