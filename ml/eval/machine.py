"""What a timing was measured on: chip, CPU / GPU cores, memory, OS, power source, load; and the
quiet-machine protocol of the cost runs (measure_quietly).

Standard library only: imported by both the ml and the minionerec environments. Only these
fields are kept from system_profiler, which also reports serial numbers and UUIDs. (ml/ann/run.py
has the same probes for the ANN report.)
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import threading
import time
from collections.abc import Callable

BUSY_LOAD = 2.0  # 1-minute load average above which a timing is marked preliminary
QUIET_WAIT = 20 * 60  # longest wait for a quiet machine before a measurement, seconds
WATCH_INTERVAL = 10.0  # seconds between `docker ps` checks while a measurement runs


def _run(cmd: list[str], timeout: float = 30) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout if out.returncode == 0 else None


def _sysctl(name: str) -> str | None:
    out = _run(["sysctl", "-n", name], timeout=5)
    return (out.strip() or None) if out else None


def _profiler(data_type: str) -> list[dict]:
    out = _run(["system_profiler", data_type, "-json"])
    try:
        return json.loads(out)[data_type] if out else []
    except (ValueError, KeyError):
        return []


def power_source() -> str | None:
    """'AC Power' or 'Battery Power' (pmset -g ps)."""
    out = _run(["pmset", "-g", "ps"], timeout=5)
    first = out.splitlines()[0] if out else ""
    return first.split("'")[1] if first.count("'") >= 2 else None


def low_power_mode() -> bool | None:
    out = _run(["pmset", "-g"], timeout=5)
    for line in (out or "").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "lowpowermode":
            return parts[1] == "1"
    return None


def info() -> dict:
    mem = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    out = {
        "chip": _sysctl("machdep.cpu.brand_string"),
        "model": None,
        "performance_cores": _sysctl("hw.perflevel0.physicalcpu"),
        "efficiency_cores": _sysctl("hw.perflevel1.physicalcpu"),
        "gpu_cores": None,
        "logical_cpus": os.cpu_count(),
        "memory_gib": round(mem / 2**30, 1),
        "os": platform.platform(),
        "python": platform.python_version(),
        "power": power_source(),
        "low_power_mode": low_power_mode(),
    }
    if sys.platform == "darwin":
        hw = (_profiler("SPHardwareDataType") or [{}])[0]
        out["chip"] = out["chip"] or hw.get("chip_type")
        out["model"] = hw.get("machine_name")
        procs = hw.get("number_processors", "").replace("proc", "").strip().split(":")
        if len(procs) > 2:  # "proc 12:8:4" = total : performance : efficiency
            out["performance_cores"] = out["performance_cores"] or procs[1]
            out["efficiency_cores"] = out["efficiency_cores"] or procs[2]
        for gpu in _profiler("SPDisplaysDataType"):
            if gpu.get("sppci_cores"):
                out["gpu_cores"] = int(gpu["sppci_cores"])
                break
    return out


# ----------------------------------------------------------------------------- quiet machine


def containers() -> list[str] | None:
    """Names of the running Docker containers; None when Docker does not answer."""
    out = _run(["docker", "ps", "--format", "{{.Names}}"], timeout=30)
    return None if out is None else sorted(out.split())


def state() -> dict:
    return {
        "time": time.strftime("%H:%M:%S"),
        "load1": round(os.getloadavg()[0], 2),
        "containers": containers(),
    }


def is_quiet(s: dict) -> bool:
    return s["load1"] < BUSY_LOAD and not s["containers"]


def wait_quiet(
    max_wait: float = QUIET_WAIT, poll: float = 10.0, log: Callable[[str], None] = print
) -> tuple[dict, float]:
    """Wait until the 1-minute load is below BUSY_LOAD and no container runs, for at most
    `max_wait` seconds; return the state when the wait ended and the seconds waited."""
    t0 = time.monotonic()
    s = state()
    if not is_quiet(s):
        log(f"  waiting for a quiet machine: load {s['load1']}, containers {s['containers']}")
    while not is_quiet(s) and time.monotonic() - t0 < max_wait:
        time.sleep(poll)
        s = state()
    return s, time.monotonic() - t0


class ContainerWatch:
    """`docker ps` every `interval` seconds in a thread while a measurement runs, and once more
    at its end; `seen` is every container name seen."""

    def __init__(self, interval: float = WATCH_INTERVAL):
        self.interval = interval
        self.seen: set[str] = set()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while True:
            self.seen.update(containers() or [])
            if self._stop.wait(self.interval):
                return

    def __enter__(self) -> ContainerWatch:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        self.seen.update(containers() or [])


def measure_quietly(
    fn: Callable[[], object],
    *,
    max_wait: float = QUIET_WAIT,
    log: Callable[[str], None] = print,
) -> tuple[object, dict]:
    """Run fn() under the cost runs' protocol: start only when the 1-minute load is below
    BUSY_LOAD and no Docker container runs (waiting at most `max_wait`), record the load and the
    containers at the start and the end, and watch `docker ps` meanwhile. If a container ran
    during fn(), run it once more; if that is disturbed too, keep it and say so. Returns fn's
    last result and the record of every attempt."""
    attempts = []
    for attempt in (1, 2):
        start, waited = wait_quiet(max_wait, log=log)
        with ContainerWatch() as watch:
            result = fn()
        attempts.append(
            {
                "waited_seconds": round(waited, 1),
                "start": start,
                "end": state(),
                "containers_during": sorted(watch.seen),
            }
        )
        if not watch.seen:
            break
        log(
            f"  containers ran during the measurement: {', '.join(sorted(watch.seen))}; "
            + ("running it once more" if attempt == 1 else "kept with a note")
        )
    last = attempts[-1]
    return result, {
        "attempts": attempts,
        "quiet_start": is_quiet(last["start"]),
        "disturbed": bool(last["containers_during"]),
    }
