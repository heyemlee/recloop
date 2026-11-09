"""The billing drill of docs/design.md §9.3 on the kind cluster (§23.2): sim.billing_drill with
its chaos acting on pods. Same traffic, delivery faults, campaigns, chaos schedule (seeded) and
checks as `make billing-drill`; the differences:

    billing    a billing pod is deleted (8 times), alternately graceful (SIGTERM, 30 s grace) and
               forced (grace 0: removed at once, as close as Kubernetes comes to kill -9). Its
               StatefulSet recreates it at once under the same name, i.e. as the same static
               group member; the schedule's "down" times do not apply. With KEDA scaling
               billing, the pod is picked at random among the running ones.
    redpanda   the broker's pod is deleted (3 times; graceful) and recreated on its volume.
    postgres   +200 ms on every reply to billing through pg-proxy (Toxiproxy), as in compose.

The reconciler's repairs during the run come from Prometheus (a deleted pod's log is gone), the
billing pods' lifetimes from a watcher. Output: artifacts/deploy/drill/ (deploy.md reads
verify.json).

Usage: python -m kube.drill [--minutes 30] [any sim.billing_drill option]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

from kube import cluster as k8s
from kube.cluster import kubectl

OUT = k8s.ROOT / "artifacts" / "deploy" / "drill"
SEED = 20261107  # sim.billing_drill's default: the same chaos schedule as the compose drill


class Kube:
    """sim.billing_drill's chaos platform on the cluster."""

    name = "kube"

    def __init__(self, seed: int = SEED):
        self.rng = random.Random(seed)
        self.deletions = 0
        self.watch: k8s.Watch | None = None

    def kill_billing(self, down: float, note) -> None:
        items = k8s.kubectl_json("get", "pods", "-l", "app.kubernetes.io/name=billing")["items"]
        pods = sorted(
            p["metadata"]["name"] for p in items if not p["metadata"].get("deletionTimestamp")
        )
        if not pods:
            note("no billing pod to delete")
            return
        pod = self.rng.choice(pods)
        self.deletions += 1
        forced = self.deletions % 2 == 0
        args = ["delete", "pod", pod, "--wait=false"]
        if forced:
            args += ["--grace-period=0", "--force"]
        kubectl(*args)
        note(f"deleted pod {pod} ({'forced, grace 0' if forced else 'graceful, SIGTERM'})")

    def restart_redpanda(self, note) -> None:
        kubectl("delete", "pod", "redpanda-0", ns="redpanda", timeout=300)
        note("deleted pod redpanda-0")
        k8s.rollout_status("statefulset/redpanda", ns="redpanda")
        note("redpanda-0 ready again")

    def ensure_billing(self) -> None:
        k8s.rollout_status("statefulset/billing")

    def reconciler_during(self, since: float) -> dict:
        t1 = time.time()
        w = f"{max(1, round(t1 - since))}s"
        repaired = k8s.prom_value(f"sum(increase(recloop_reconcile_repaired_cents_total[{w}]))", t1)
        runs = k8s.prom_by(
            f"sum by (result) (increase(recloop_reconcile_runs_total[{w}]))", t1, "result"
        )
        starts = None
        if self.watch is not None:
            starts = sum(1 for p in self.watch.pods.values() if p["component"] == "billing")
        return {
            "passes_with_repairs": None,  # only the compose log has it per pass
            "repaired_cents": round(repaired or 0),
            "reconcile_runs": {k: round(v) for k, v in runs.items()},
            "billing_starts": starts,
        }


def main(argv: list[str] | None = None) -> None:
    from sim import billing_drill

    os.environ.update(k8s.host_env())
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--out", default=str(OUT))
    known, rest = p.parse_known_args(argv)
    out = Path(known.out).resolve()
    argv = [*rest, "--out", str(out)]
    platform = Kube()
    code = 1
    with k8s.Watch(["billing", "aggregator", "log-sink"]) as watch:
        platform.watch = watch
        try:
            billing_drill.main(argv, platform=platform)
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
    verify = out / "verify.json"
    if verify.exists():
        rep = json.loads(verify.read_text())
        rep["kube"] = {
            "watch": watch.result(),
            "node": k8s.node_info(),
            "billing": k8s.kubectl_json("get", "scaledobject", "billing")["spec"],
        }
        verify.write_text(json.dumps(rep, indent=2))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
