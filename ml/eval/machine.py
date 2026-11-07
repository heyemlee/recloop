"""What a timing was measured on: chip, CPU / GPU cores, memory, OS, power source, load.

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

BUSY_LOAD = 2.0  # 1-minute load average above which a timing is marked preliminary


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
