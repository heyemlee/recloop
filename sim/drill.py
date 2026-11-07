"""Event-pipeline drill (docs/design.md §7.3, §9.3): duplicates and crashes, then verify.

The simulator runs with every delivery fault switched on while the consumers are killed with
SIGKILL (no clean shutdown, no final offset commit) and the broker is restarted. When the
traffic stops and the consumers have caught up, sim.verify checks that every acknowledged event
is in events_raw exactly once and that Redis shows each one applied exactly once.

    t = 20 %  kill -9 the aggregator, start it again 2 s later
    t = 40 %  kill -9 the log-sink, start it again 2 s later
    t = 60 %  restart Redpanda (graceful: the dev-container broker skips fsync, so a kill -9
              could lose acknowledged writes; that is a broker setting, not a pipeline property)
    t = 80 %  kill -9 the aggregator and the log-sink together

Usage: python -m sim.drill [--seconds 60]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

from sim import run as run_mod
from sim import verify as verify_mod

ROOT = Path(__file__).resolve().parent.parent
COMPOSE = ["docker", "compose", "-f", str(ROOT / "deploy" / "docker-compose.yml")]
FAULTS = "duplicate=0.1,resend=0.05,reorder=0.1,late=0.05,double_click=0.05"


def compose(*args: str) -> None:
    subprocess.run([*COMPOSE, *args], check=True, capture_output=True)


def chaos(seconds: float, log: list) -> None:
    def note(what: str) -> None:
        log.append({"t": round(time.time() - t0, 1), "action": what})
        print(f"  t={log[-1]['t']:5.1f}s  {what}", flush=True)

    t0 = time.time()
    steps = [
        (0.2, ["aggregator"], "kill"),
        (0.4, ["log-sink"], "kill"),
        (0.6, ["redpanda"], "restart"),
        (0.8, ["aggregator", "log-sink"], "kill"),
    ]
    for frac, services, how in steps:
        time.sleep(max(0.0, t0 + frac * seconds - time.time()))
        if how == "kill":
            compose("kill", "-s", "SIGKILL", *services)
            note("kill -9 " + " ".join(services))
            time.sleep(2)
            compose("start", *services)
            note("started " + " ".join(services))
        else:
            compose("restart", *services)
            note("restarted " + " ".join(services))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--faults", default=FAULTS)
    p.add_argument("--out", default=str(ROOT / "artifacts" / "sim" / "drill"))
    args = p.parse_args(argv)

    sim_args = run_mod.parser().parse_args(
        [
            "--sessions", "1000000", "--max-seconds", str(args.seconds),
            "--concurrency", str(args.concurrency), "--think-ms", "20,120",
            "--faults", args.faults, "--out", args.out,
        ]
    )  # fmt: skip
    actions: list = []
    chaos_thread = threading.Thread(target=chaos, args=(args.seconds, actions))
    print(f"drill    {args.seconds:.0f}s of simulated traffic with faults {args.faults}")
    summary = run_mod.run_main(sim_args, on_start=chaos_thread.start)
    chaos_thread.join()
    print("  waiting for the consumers to catch up ...", flush=True)
    t = time.time()
    verify_mod.wait_for_consumers()
    caught_up = time.time() - t
    out = Path(args.out).resolve()
    rep = verify_mod.verify(out, *verify_mod_urls())
    rep.update({"chaos": actions, "sim": summary, "caught_up_seconds": round(caught_up, 1)})
    (out / "verify.json").write_text(json.dumps(rep, indent=2))
    verify_mod.print_report(rep)
    raise SystemExit(0 if rep["ok"] else 1)


def verify_mod_urls() -> tuple[str, str]:
    import os

    return (
        os.environ.get(
            "DATABASE_URL", "postgres://recloop:recloop@localhost:15432/recloop?sslmode=disable"
        ),
        os.environ.get("REDIS_URL", "redis://localhost:16390/0"),
    )


if __name__ == "__main__":
    main()
