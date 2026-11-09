"""Process memory on macOS: physical footprint, which includes the GPU (Metal) allocations.

On Apple Silicon the GPU shares memory with the CPU, and what a process holds on the GPU
(model weights on MPS, the KV cache, MPS allocator caches) is counted in its physical
footprint -- the "Memory" column of Activity Monitor -- but not necessarily in the resident set
size. proc_pid_rusage(RUSAGE_INFO_V4) reports the current footprint and the lifetime maximum of
any process of the same user, so a parent can watch a child that runs on MPS.

Standard library only (ctypes): imported by both the ml and the minionerec environments.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
import time
from dataclasses import dataclass, field

RUSAGE_INFO_V4 = 4


class _RusageInfoV4(ctypes.Structure):
    # struct rusage_info_v4, <sys/resource.h>
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        *[
            (name, ctypes.c_uint64)
            for name in (
                "ri_user_time",
                "ri_system_time",
                "ri_pkg_idle_wkups",
                "ri_interrupt_wkups",
                "ri_pageins",
                "ri_wired_size",
                "ri_resident_size",
                "ri_phys_footprint",
                "ri_proc_start_abstime",
                "ri_proc_exit_abstime",
                "ri_child_user_time",
                "ri_child_system_time",
                "ri_child_pkg_idle_wkups",
                "ri_child_interrupt_wkups",
                "ri_child_pageins",
                "ri_child_elapsed_abstime",
                "ri_diskio_bytesread",
                "ri_diskio_byteswritten",
                "ri_cpu_time_qos_default",
                "ri_cpu_time_qos_maintenance",
                "ri_cpu_time_qos_background",
                "ri_cpu_time_qos_utility",
                "ri_cpu_time_qos_legacy",
                "ri_cpu_time_qos_user_initiated",
                "ri_cpu_time_qos_user_interactive",
                "ri_billed_system_time",
                "ri_serviced_system_time",
                "ri_logical_writes",
                "ri_lifetime_max_phys_footprint",
                "ri_instructions",
                "ri_cycles",
                "ri_billed_energy",
                "ri_serviced_energy",
                "ri_interval_max_phys_footprint",
                "ri_runnable_time",
            )
        ],
    ]


_libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib") if sys.platform == "darwin" else None


@dataclass(frozen=True)
class Footprint:
    current: int  # bytes
    lifetime_max: int  # bytes
    resident: int  # bytes


def read(pid: int | None = None) -> Footprint | None:
    """Footprint of `pid` (default: this process); None if unsupported or the process is gone."""
    if _libc is None:
        return None
    info = _RusageInfoV4()
    rc = _libc.proc_pid_rusage(
        ctypes.c_int(os.getpid() if pid is None else pid),
        ctypes.c_int(RUSAGE_INFO_V4),
        ctypes.byref(info),
    )
    if rc != 0:
        return None
    return Footprint(
        int(info.ri_phys_footprint),
        int(info.ri_lifetime_max_phys_footprint),
        int(info.ri_resident_size),
    )


@dataclass
class Sampler:
    """Polls a process's footprint in a background thread until stop()."""

    pid: int
    interval: float = 0.2
    peak: int = 0
    lifetime_max: int = 0
    samples: int = 0
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)

    def start(self) -> Sampler:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _poll(self) -> None:
        fp = read(self.pid)
        if fp is not None:
            self.peak = max(self.peak, fp.current)
            self.lifetime_max = max(self.lifetime_max, fp.lifetime_max)
            self.samples += 1

    def _run(self) -> None:
        while not self._stop.is_set():
            self._poll()
            time.sleep(self.interval)

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        return {
            "peak_footprint": self.peak,
            "lifetime_max_footprint": self.lifetime_max,
            "samples": self.samples,
            "interval_s": self.interval,
        }
