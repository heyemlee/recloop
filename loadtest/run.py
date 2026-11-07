"""Latency experiments of docs/design.md §5.2 -> artifacts/latency/results.json.

Runs against the bench profile (`make bench-up`): the api and the model server as containers
with fixed CPU and memory limits, k6 on the host (outside the Docker VM, like a client). In the
fault experiment the api reaches the model server and Redis through Toxiproxy (every row, the
no-fault one included); elsewhere it connects directly. Three experiments:

    ladder   offered load (open loop) vs latency: the largest rate that meets the SLO
    cache    the candidate cache on and off at one rate (and on, through Toxiproxy: the proxy's cost)
    onset    the model server is cut (or slowed by 50 ms) in the middle of steady traffic: how
             fast the breaker opens, how many calls failed first, how fast it closes afterwards
    faults   model server 30 % of calls +200 ms (its fault switch), every call +50 ms
             (Toxiproxy latency), model server disconnected, Redis disconnected (Toxiproxy), and
             no fault, each against the naive api (no deadline, no degradation) and this design;
             the candidate cache is off in both so that every request needs the model server

Every run starts from the same state: the consumers have caught up, Redis is reseeded
(`make seed`), Toxiproxy and the model server's fault switch are reset. Then k6 warms up at the
run's rate, the fault is switched on, and k6 measures. Each configuration runs `repeats` times
(the report takes the median); repetitions are interleaved so drift spreads over all of them.

Usage: python -m loadtest.run [--plan full|smoke] [--only ladder,cache,faults,onset]
                              [--resume REASON [--redo KEY#REP ...]] [--note TEXT ...]
       (--resume keeps the finished runs of --out and runs the rest; --redo runs one again)
       python -m loadtest.report      (renders docs/results/latency.md from results.json)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "latency"
K6 = ROOT / ".cache" / "bin" / "k6"
AGGREGATOR = ROOT / ".cache" / "bin" / "aggregator"
WORLD = ROOT / "artifacts" / "sim" / "world.json"
COMPOSE = [
    "docker",
    "compose",
    "-f",
    str(ROOT / "deploy" / "docker-compose.yml"),
    "--profile",
    "bench",
]
API = os.environ.get(
    "BENCH_API_URL", f"http://127.0.0.1:{os.environ.get('RECLOOP_BENCH_API_PORT', '18180')}"
)
TOXIPROXY = os.environ.get("TOXIPROXY_URL", "http://127.0.0.1:18474")
MS_ADMIN = os.environ.get("BENCH_MS_ADMIN_URL", "http://127.0.0.1:18182")
PROMETHEUS = os.environ.get("PROMETHEUS_URL", "http://localhost:19090")
CONTAINERS = [
    "api",
    "model-server",
    "redis",
    "redpanda",
    "postgres",
    "aggregator",
    "log-sink",
    "toxiproxy",
]

FAULTS = {
    "none": "no fault",
    "ms_slow": "model server: 30 % of calls +200 ms",
    "ms_latency": "model server: every call +50 ms",
    "ms_down": "model server disconnected",
    "redis_down": "Redis disconnected",
}


@dataclass(frozen=True)
class ApiConfig:
    resilience: str = "full"  # full | naive
    cache: str = "on"  # on | off
    breaker: str = "time:10s:20:10"  # BREAKER_WINDOW; the design's default
    # Through Toxiproxy (needed to cut or slow a connection) or straight to the model server and
    # Redis. The proxy adds a few ms to the tail (see the cache experiment), so only the fault
    # experiment, all of its rows, goes through it.
    proxied: bool = False

    def env(self) -> dict[str, str]:
        env = {
            "RECLOOP_RESILIENCE": self.resilience,
            "CANDIDATE_CACHE": self.cache,
            "BREAKER_WINDOW": self.breaker,
        }
        if not self.proxied:
            env |= {
                "BENCH_REDIS_URL": "redis://redis:6379/0",
                "BENCH_MODEL_SERVER_ADDR": "model-server:18081",
            }
        return env

    @property
    def mode(self) -> str:
        return "naive" if self.resilience == "naive" else "design"


@dataclass(frozen=True)
class Run:
    experiment: str
    rate: int
    config: ApiConfig
    fault: str = "none"
    rep: int = 1

    @property
    def key(self) -> str:
        c = self.config
        path = "toxiproxy" if c.proxied else "direct"
        return f"{self.experiment}/{self.rate}/{c.mode}/cache-{c.cache}/{c.breaker}/{path}/{self.fault}"


@dataclass
class Plan:
    ladder: list[int]
    cache_rate: int
    fault_rate: int
    repeats: int
    duration: int
    warmup: int
    extra_breaker: str | None = "calls:20"  # also run ms_slow with §5.1's first window
    # onset: seconds of healthy traffic, of fault, and after recovery (>= the 10 s open time)
    onset: tuple[float, float, float] = (5.0, 15.0, 15.0)
    experiments: list[str] = field(default_factory=lambda: ["ladder", "cache", "faults", "onset"])

    def runs(self) -> list[Run]:
        out = []
        for rep in range(1, self.repeats + 1):
            if "ladder" in self.experiments:
                out += [Run("ladder", r, ApiConfig(), rep=rep) for r in self.ladder]
            if "cache" in self.experiments:
                for cfg in (
                    ApiConfig(cache="on"),
                    ApiConfig(cache="off"),
                    ApiConfig(cache="on", proxied=True),
                ):
                    out.append(Run("cache", self.cache_rate, cfg, rep=rep))
            if "faults" in self.experiments:
                naive = ApiConfig(resilience="naive", cache="off", proxied=True)
                design = ApiConfig(cache="off", proxied=True)
                for fault in FAULTS:
                    for cfg in (naive, design):
                        out.append(Run("faults", self.fault_rate, cfg, fault=fault, rep=rep))
                    if fault == "ms_slow" and self.extra_breaker:
                        cfg = ApiConfig(cache="off", breaker=self.extra_breaker, proxied=True)
                        out.append(Run("faults", self.fault_rate, cfg, fault=fault, rep=rep))
            if "onset" in self.experiments:
                for fault in ("ms_down", "ms_latency"):
                    cfg = ApiConfig(cache="off", proxied=True)
                    out.append(Run("onset", self.fault_rate, cfg, fault=fault, rep=rep))
        return out


PLANS = {
    "full": Plan(ladder=[100, 200, 400, 600, 800, 1000, 1200, 1500], cache_rate=400, fault_rate=200,
                 repeats=3, duration=60, warmup=10),
    "smoke": Plan(ladder=[100, 300], cache_rate=200, fault_rate=100, repeats=1, duration=15, warmup=5,
                  onset=(3.0, 8.0, 14.0)),
}  # fmt: skip


# ---------------------------------------------------------------- helpers


def sh(*cmd: str, env: dict | None = None, check: bool = True) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **(env or {})})
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}: {r.stderr.strip()[-500:]}")
    return r.stdout


def http(method: str, url: str, body: dict | None = None, timeout: float = 5.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw else None


def wait_ready(timeout: float = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = http("GET", API + "/readyz", timeout=2)
            if all(v == "ok" for v in r["dependencies"].values()):
                return
        except OSError:
            pass
        time.sleep(0.5)
    raise RuntimeError("the bench api did not become ready")


class Bench:
    def __init__(self, log=print):
        self.config: ApiConfig | None = None
        self.log = log

    def configure(self, cfg: ApiConfig) -> None:
        if cfg == self.config:
            return
        path = "through Toxiproxy" if cfg.proxied else "direct"
        self.log(f"  api: {cfg.mode}, cache {cfg.cache}, breaker {cfg.breaker}, {path}")
        self.reset_faults()
        sh(*COMPOSE, "up", "-d", "--force-recreate", "--no-deps", "api", env=cfg.env())
        wait_ready()
        self.config = cfg

    def reset_faults(self) -> None:
        http("POST", TOXIPROXY + "/reset")
        http("PUT", MS_ADMIN + "/fault", {"delay_ms": 0, "probability": 0})

    def set_fault(self, fault: str) -> None:
        if fault == "ms_slow":
            http("PUT", MS_ADMIN + "/fault", {"delay_ms": 200, "probability": 0.3})
        elif fault == "ms_latency":
            # Toxiproxy delays every response on the connection (all calls, not a share).
            toxic = {
                "name": "slow",
                "type": "latency",
                "stream": "downstream",
                "attributes": {"latency": 50},
            }
            http("POST", TOXIPROXY + "/proxies/model_server/toxics", toxic)
        elif fault == "ms_down":
            http("POST", TOXIPROXY + "/proxies/model_server", {"enabled": False})
        elif fault == "redis_down":
            http("POST", TOXIPROXY + "/proxies/redis", {"enabled": False})

    def reset_state(self) -> float:
        """Consumers caught up, Redis reseeded, faults off. Returns the catch-up time (s)."""
        self.reset_faults()
        t = time.time()
        # The aggregator must be done before Redis is reseeded. The log-sink only writes
        # PostgreSQL, which the api does not read for these requests: a short wait, best effort.
        for group, timeout in (("aggregator", "5m"), ("log-sink", "90s")):
            try:
                sh(str(AGGREGATOR), "wait-lag", group, timeout)
            except RuntimeError as e:
                self.log(f"  warning: {group} has not caught up: {str(e)[-200:]}")
        caught_up = time.time() - t
        sh(
            "uv",
            "run",
            "--project",
            str(ROOT / "ml"),
            "--frozen",
            "python",
            "-m",
            "ml.data.seed_redis",
        )
        return caught_up


def k6(rate: int, duration: int, summary: Path, tag: str) -> dict:
    sh(
        str(K6), "run", "--quiet", "--no-color",
        "-e", f"BASE_URL={API}", "-e", f"RATE={rate}", "-e", f"DURATION={duration}s",
        "-e", f"WORLD={WORLD}", "-e", f"SUMMARY={summary}", "--tag", f"run={tag}",
        str(ROOT / "loadtest" / "recs.js"),
    )  # fmt: skip
    return json.loads(summary.read_text())


def prom(expr: str, at: float) -> list[dict]:
    url = PROMETHEUS + "/api/v1/query?" + urlencode({"query": expr, "time": f"{at:.3f}"})
    try:
        r = http("GET", url, timeout=10)
    except OSError:
        return []
    return r.get("data", {}).get("result", []) if r.get("status") == "success" else []


def prom_value(expr: str, at: float) -> float | None:
    res = prom(expr, at)
    if not res:
        return None
    v = float(res[0]["value"][1])
    return None if v != v else v  # NaN


def prom_by(expr: str, at: float, label: str) -> dict[str, float]:
    out = {}
    for r in prom(expr, at):
        v = float(r["value"][1])
        if v == v and v > 0:
            out[r["metric"].get(label, "")] = round(v, 4)
    return out


def docker_cpu() -> dict[str, dict]:
    names = [f"recloop-{c}-1" for c in CONTAINERS]
    raw = sh("docker", "stats", "--no-stream", "--format", "{{json .}}", *names, check=False)
    out = {}
    for line in raw.splitlines():
        d = json.loads(line)
        name = d["Name"].removeprefix("recloop-").removesuffix("-1")
        out[name] = {
            "cpu_pct": float(d["CPUPerc"].rstrip("%") or 0),
            "mem": d["MemUsage"].split(" / ")[0],
        }
    return out


# ---------------------------------------------------------------- one run


def k6_metrics(s: dict, duration: int) -> dict:
    m = s["metrics"]

    def trend(name: str) -> dict:
        v = m.get(name, {}).get("values", {})
        return {
            k: round(v[k], 3)
            for k in ("med", "p(95)", "p(99)", "p(99.9)", "max", "count")
            if k in v
        }

    def rate(name: str) -> float | None:
        v = m.get(name, {}).get("values", {})
        return round(v["rate"], 6) if "rate" in v else None

    recs = trend("recs_ms")
    return {
        "recs": recs,
        "home": trend("recs_ms{kind:home}"),
        "item": trend("recs_ms{kind:item}"),
        "next_page": trend("recs_ms{kind:next_page}"),
        "events": trend("events_ms"),
        "achieved_rps": round(recs.get("count", 0) / duration, 1),
        "errors": rate("recs_errors"),
        "event_errors": rate("events_errors"),
        "degraded": rate("recs_degraded"),
        "cache_hit_home": rate("cache_hit{kind:home}"),
        "cache_hit_item": rate("cache_hit{kind:item}"),
        "dropped_iterations": int(
            m.get("dropped_iterations", {}).get("values", {}).get("count", 0)
        ),
        "clicks": int(m.get("sim_clicks", {}).get("values", {}).get("count", 0)),
    }


def server_metrics(t0: float, t1: float) -> dict:
    w = f"{max(1, round(t1 - t0))}s"
    sel = 'job="api",mode="bench"'
    total = f'recloop_stage_duration_seconds_bucket{{{sel},stage="total",surface=~"home|item"}}'

    def q(p: float, metric: str, by: str = "") -> str:
        return f"histogram_quantile({p}, sum by (le{by}) (increase({metric}[{w}])))"

    stages = {}
    for r in prom(
        q(0.99, f'recloop_stage_duration_seconds_bucket{{{sel},surface=~"home|item"}}', ", stage"),
        t1,
    ):
        v = float(r["value"][1])
        if v == v:
            stages[r["metric"]["stage"]] = round(v * 1000, 2)
    ms_sel = 'job="model-server",mode="bench"'
    return {
        "p50_ms": round((prom_value(q(0.5, total), t1) or 0) * 1000, 2),
        "p99_ms": round((prom_value(q(0.99, total), t1) or 0) * 1000, 2),
        "stage_p99_ms": stages,
        "degraded_reasons": prom_by(
            f"sum by (reason) (increase(recloop_degraded_total{{{sel}}}[{w}]))", t1, "reason"
        ),
        "breaker_transitions": prom_by(
            f"sum by (to) (increase(recloop_breaker_transitions_total{{{sel}}}[{w}]))", t1, "to"
        ),
        "model_server_calls": prom_by(
            f"sum by (code) (increase(recloop_ms_requests_total{{{ms_sel}}}[{w}]))", t1, "code"
        ),
        "model_server_aborts": prom_by(
            f"sum by (where) (increase(recloop_ms_deadline_aborts_total{{{ms_sel}}}[{w}]))",
            t1,
            "where",
        ),
        "model_server_injected": prom_value(
            f"sum(increase(recloop_ms_injected_delays_total{{{ms_sel}}}[{w}]))", t1
        ),
        "consumer_lag_max": prom_by(
            f"max by (group) (max_over_time(recloop_consumer_lag[{w}]))", t1, "group"
        ),
        # 410: a next page whose result list expired or was evicted; 503: a page read past its
        # budget (design) or any failed dependency (naive)
        "recs_status": prom_by(
            f"sum by (code) (increase(recloop_http_requests_total{{{sel},"
            f'route="GET /v1/recommendations",code!="200"}}[{w}]))',
            t1,
            "code",
        ),
    }


def run_one(bench: Bench, plan: Plan, run: Run, out: Path) -> dict:
    bench.configure(run.config)
    caught_up = bench.reset_state()
    tag = f"{run.key}#{run.rep}".replace("/", "_")
    if plan.warmup:
        k6(run.rate, plan.warmup, out / "warmup.json", tag + "_warmup")
    bench.set_fault(run.fault)
    time.sleep(0.5 if run.fault == "none" else 2)  # let a disconnect take effect
    cpu: dict = {}
    sampler = threading.Timer(plan.duration / 2, lambda: cpu.update(docker_cpu()))
    load_before = os.getloadavg()
    t0 = time.time()
    sampler.start()
    summary = k6(run.rate, plan.duration, out / "summary.json", tag)
    t1 = time.time()
    sampler.join()
    load_after = os.getloadavg()
    bench.reset_faults()
    time.sleep(6)  # one more Prometheus scrape (every 5 s) covers the end of the window
    return {
        "key": run.key,
        "experiment": run.experiment,
        "rate": run.rate,
        "mode": run.config.mode,
        "config": asdict(run.config),
        "fault": run.fault,
        "rep": run.rep,
        "started": round(t0, 1),
        "seconds": round(t1 - t0, 1),
        "consumers_caught_up_s": round(caught_up, 1),
        "load_avg_before": [round(x, 2) for x in load_before],
        "load_avg_after": [round(x, 2) for x in load_after],
        "k6": k6_metrics(summary, plan.duration),
        "server": server_metrics(t0, t1),
        "containers_mid_run": cpu,
    }


def overloaded(k: dict, rate: int, duration: int) -> bool:
    """Far past the SLO: higher rates would only queue up more (and stress a shared machine)."""
    p99 = k["recs"].get("p(99)") or 0
    dropped = k["dropped_iterations"] / max(1, rate * duration)
    return p99 > 1000 or (k["errors"] or 0) > 0.05 or dropped > 0.05


# ---------------------------------------------------------------- breaker onset


def api_metrics() -> dict[str, float]:
    """The bench api's breaker state and SASRec failure counters, read straight from /metrics."""
    with urllib.request.urlopen(API + "/metrics", timeout=2) as r:
        text = r.read().decode()
    out = {"state": 0.0, "failed": 0.0}
    for line in text.splitlines():
        if line.startswith('recloop_breaker_state{dependency="model_server"}'):
            out["state"] = float(line.rsplit(" ", 1)[1])
        elif line.startswith("recloop_degraded_total{") and (
            'reason="sasrec_error"' in line or 'reason="sasrec_timeout"' in line
        ):
            out["failed"] += float(line.rsplit(" ", 1)[1])
    return out


def run_onset(bench: Bench, plan: Plan, run: Run, out: Path) -> dict:
    """Traffic at a steady rate; the fault starts and ends in the middle of it. Measures how long
    the breaker takes to open, how many calls failed (each waiting up to its timeout) before it
    did, and how long after the model server is back the breaker closes again."""
    bench.configure(run.config)
    bench.reset_state()
    if plan.warmup:
        k6(run.rate, plan.warmup, out / "warmup.json", run.key + "_warmup")
    before_s, fault_s, after_s = plan.onset
    total = before_s + fault_s + after_s
    summary = out / "onset.json"
    proc = subprocess.Popen(
        [str(K6), "run", "--quiet", "--no-color", "-e", f"BASE_URL={API}", "-e", f"RATE={run.rate}",
         "-e", f"DURATION={int(total)}s", "-e", f"WORLD={WORLD}", "-e", f"SUMMARY={summary}",
         str(ROOT / "loadtest" / "recs.js")],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )  # fmt: skip
    load_before = os.getloadavg()
    t0 = time.time()
    time.sleep(before_s)
    before = api_metrics()
    bench.set_fault(run.fault)
    t_fault = time.time()
    t_open, failed_at_open = None, None
    while time.time() < t_fault + fault_s:
        m = api_metrics()
        if t_open is None and m["state"] == 2:
            t_open, failed_at_open = time.time(), m["failed"]
        time.sleep(0.01)
    bench.reset_faults()
    t_recover = time.time()
    t_closed = None
    while time.time() < t0 + total + 2 and t_closed is None:
        if api_metrics()["state"] == 0:
            t_closed = time.time()
        time.sleep(0.01)
    proc.wait()
    load_after = os.getloadavg()
    s = json.loads(summary.read_text())
    return {
        "key": run.key,
        "experiment": run.experiment,
        "rate": run.rate,
        "mode": run.config.mode,
        "config": asdict(run.config),
        "fault": run.fault,
        "rep": run.rep,
        "started": round(t0, 1),
        "load_avg_before": [round(x, 2) for x in load_before],
        "load_avg_after": [round(x, 2) for x in load_after],
        "open_after_ms": None if t_open is None else round((t_open - t_fault) * 1000, 1),
        "failed_before_open": None if t_open is None else failed_at_open - before["failed"],
        "closed_after_recovery_s": None if t_closed is None else round(t_closed - t_recover, 2),
        "k6": k6_metrics(s, int(total)),
    }


def wait_idle(threshold: float = 2.0, timeout: float = 600.0) -> float:
    """Wait (up to `timeout`) until the 1-minute load average is below `threshold`: the image
    builds that `make load` starts with raise it for a minute or two. Returns the seconds waited."""
    t0 = time.time()
    while os.getloadavg()[0] >= threshold and time.time() - t0 < timeout:
        time.sleep(10)
    return round(time.time() - t0)


# ---------------------------------------------------------------- metadata


def environment() -> dict:
    info = json.loads(sh("docker", "info", "--format", "{{json .}}"))
    limits = {}
    for c in CONTAINERS:
        raw = sh(
            "docker", "inspect", f"recloop-{c}-1", "--format", "{{json .HostConfig}}", check=False
        )
        if raw.strip():
            hc = json.loads(raw)
            limits[c] = {
                "cpus": hc.get("NanoCpus", 0) / 1e9,
                "memory_mb": hc.get("Memory", 0) // (1 << 20),
            }
    others = [
        n
        for n in sh("docker", "ps", "--format", "{{.Names}}").split()
        if not n.startswith("recloop-")
    ]
    images = {}
    for c in CONTAINERS:
        raw = sh(
            "docker", "inspect", f"recloop-{c}-1", "--format", "{{.Config.Image}}", check=False
        )
        images[c] = raw.strip()
    cpu_model = sh("sysctl", "-n", "machdep.cpu.brand_string", check=False).strip()
    mem = sh("sysctl", "-n", "hw.memsize", check=False).strip()
    ncpu = sh("sysctl", "-n", "hw.ncpu", check=False).strip()
    return {
        "host": {
            "cpu": cpu_model or platform.processor(),
            "cores": int(ncpu) if ncpu.isdigit() else os.cpu_count(),
            "memory_gb": round(int(mem) / (1 << 30)) if mem.isdigit() else None,
            "os": f"macOS {platform.mac_ver()[0]}"
            if platform.mac_ver()[0]
            else platform.platform(),
        },
        "docker": {
            "server": info.get("ServerVersion"),
            "vm_cpus": info.get("NCPU"),
            "vm_memory_gb": round(info.get("MemTotal", 0) / (1 << 30), 2),
            "other_containers_running": len(others),
        },
        "limits": limits,
        "images": images,
        "k6": sh(str(K6), "version").strip(),
        "world": json.loads(WORLD.read_text()).get("digest"),
        "load_avg_start": [round(x, 2) for x in os.getloadavg()],
    }


def resume(
    out: Path,
    meta: dict,
    runs: list[Run],
    reason: str,
    waited: float,
    redo: list[str] | tuple[str, ...] = (),
) -> tuple[dict, list[dict], set[tuple[str, int]]]:
    """Continue an interrupted results file: keep its runs that `runs` still has, except those
    named in `redo` ("key#rep"), move the others to results-discarded.json (each tagged with the
    reason), and note the restart in the metadata (the environment stays the one of the first
    start; the restart's own is kept next to it)."""
    if not out.exists():
        raise SystemExit(f"--resume: {out} does not exist")
    old = json.loads(out.read_text())
    planned = {(r.key, r.rep) for r in runs}
    have = {f"{r['key']}#{r['rep']}" for r in old["runs"]}
    if missing := [x for x in redo if x not in have]:
        raise SystemExit(f"--redo: not in {out.name}: {', '.join(missing)}")

    def keep(r: dict) -> bool:
        return (r["key"], r["rep"]) in planned and f"{r['key']}#{r['rep']}" not in redo

    kept = [r for r in old["runs"] if keep(r)]
    dropped = [r for r in old["runs"] if not keep(r)]
    if dropped:
        path = out.parent / "results-discarded.json"
        prior = json.loads(path.read_text()) if path.exists() else {"runs": []}
        for r in prior["runs"]:  # files written before runs carried their own reason
            r.setdefault("discarded_reason", prior.get("reason", ""))
        now = round(time.time())
        new = [r | {"discarded_reason": reason, "discarded_at": now} for r in dropped]
        path.write_text(json.dumps({"runs": prior["runs"] + new}, indent=1))
    env = meta["environment"]
    restart = {
        "at": round(time.time()),
        "reason": reason,
        "idle_wait_s": waited,
        "load_avg_start": env["load_avg_start"],
        "vm_cpus": env["docker"]["vm_cpus"],
        "other_containers_running": env["docker"]["other_containers_running"],
        "kept": len(kept),
        "discarded": [f"{r['key']}#{r['rep']}" for r in dropped],
    }
    meta = old["meta"]
    meta["resumed"] = meta.get("resumed", []) + [restart]
    print(
        f"latency  resuming {out.name}: {len(kept)} runs kept, {len(dropped)} discarded",
        flush=True,
    )
    return meta, kept, {(r["key"], r["rep"]) for r in kept}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plan", choices=sorted(PLANS), default="full")
    p.add_argument("--only", default="", help="comma-separated: ladder,cache,faults,onset")
    p.add_argument("--out", type=Path, default=OUT / "results.json")
    p.add_argument(
        "--resume",
        default="",
        metavar="REASON",
        help="keep the runs in --out that the plan still has, run the rest; runs the plan does "
        "not have (e.g. from a runner bug) move to results-discarded.json",
    )
    p.add_argument(
        "--redo",
        action="append",
        default=[],
        metavar="KEY#REP",
        help="with --resume: discard this finished run too and run it again (repeatable)",
    )
    p.add_argument(
        "--note",
        action="append",
        default=[],
        metavar="TEXT",
        help="a sentence for the report's setup section (repeatable)",
    )
    p.add_argument(
        "--annotate",
        action="store_true",
        help="only add the --note sentences to --out and exit (no runs)",
    )
    args = p.parse_args(argv)
    args.out = args.out.resolve()
    if args.annotate:
        data = json.loads(args.out.read_text())
        data["meta"]["notes"] = data["meta"].get("notes", []) + args.note
        args.out.write_text(json.dumps(data, indent=1))
        print(f"latency  {len(args.note)} note(s) added to {args.out.name}")
        return
    plan = PLANS[args.plan]
    if args.only:
        plan.experiments = [x.strip() for x in args.only.split(",") if x.strip()]
    for path, what in (
        (K6, "make k6"),
        (AGGREGATOR, "make build-aggregator"),
        (WORLD, "make sim-world"),
    ):
        if not path.exists():
            raise SystemExit(f"{path} is missing: run `{what}`")
    wait_ready()
    waited = wait_idle()
    load_now = [round(x, 2) for x in os.getloadavg()]
    print(f"latency  waited {waited}s for a quiet machine: load {load_now}", flush=True)
    work = args.out.parent / "work"
    work.mkdir(parents=True, exist_ok=True)
    runs = plan.runs()
    env = environment()
    meta = {
        "plan": args.plan,
        **asdict(plan),
        "environment": env,
        "started": round(time.time()),
        "idle_wait_s": waited,
    }
    results, done = [], set()
    if args.redo and not args.resume:
        raise SystemExit("--redo needs --resume")
    if args.resume:
        meta, results, done = resume(args.out, meta, runs, args.resume, waited, args.redo)
    meta["notes"] = meta.get("notes", []) + args.note
    print(
        f"latency  {len(runs) - len(done)} of {len(runs)} runs to go ({args.plan}: "
        f"{plan.duration}s each + {plan.warmup}s warm-up), "
        f"Docker VM {env['docker']['vm_cpus']} CPUs, load {env['load_avg_start']}",
        flush=True,
    )
    bench = Bench()
    t_start = time.time()
    ceiling = None  # the ladder stops climbing above a rate that failed badly
    for res in results:
        if res["experiment"] == "ladder" and overloaded(res["k6"], res["rate"], plan.duration):
            ceiling = res["rate"] if ceiling is None else min(ceiling, res["rate"])
    for i, run in enumerate(runs, 1):
        if (run.key, run.rep) in done:
            continue
        if run.experiment == "ladder" and ceiling is not None and run.rate > ceiling:
            print(
                f"  [{i}/{len(runs)}] {run.key} #{run.rep}: skipped (above {ceiling}/s, which failed)",
                flush=True,
            )
            continue
        if run.experiment == "onset":
            res = run_onset(bench, plan, run, work)
            results.append(res)
            print(
                f"  [{i}/{len(runs)}] {run.key} #{run.rep}: open after {res['open_after_ms']} ms, "
                f"{res['failed_before_open']} failed calls before, closed "
                f"{res['closed_after_recovery_s']} s after recovery, "
                f"{(time.time() - t_start) / 60:.0f} min",
                flush=True,
            )
        else:
            res = run_one(bench, plan, run, work)
            results.append(res)
            if run.experiment == "ladder" and overloaded(res["k6"], run.rate, plan.duration):
                ceiling = run.rate if ceiling is None else min(ceiling, run.rate)
            k = res["k6"]
            print(
                f"  [{i}/{len(runs)}] {run.key} #{run.rep}: p99 {k['recs'].get('p(99)')} ms, "
                f"degraded {k['degraded'] or 0:.3f}, errors {k['errors'] or 0:.4f}, "
                f"dropped {k['dropped_iterations']}, {(time.time() - t_start) / 60:.0f} min",
                flush=True,
            )
        meta["finished"] = round(time.time())
        meta["load_avg_end"] = [round(x, 2) for x in os.getloadavg()]
        args.out.write_text(json.dumps({"meta": meta, "runs": results}, indent=1))
    bench.configure(ApiConfig())
    bench.reset_faults()
    print(
        f"latency  done in {(time.time() - t_start) / 60:.0f} min -> {args.out.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
