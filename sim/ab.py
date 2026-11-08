"""A/A, then the A/B of the ranker (design §8, §16 M4): RRF against LightGBM, simulated traffic.

Two phases in the rank layer over all users, 50 / 50, one after the other:

  rank-aa-v1   a: rrf        b: rrf          identical arms: a difference is noise or a bug
  rank-ab-v1   rrf: rrf      lightgbm: lightgbm

The A/A runs first, so the analysis is validated on it before the A/B is read (§19, 2026-11-07).
Each phase starts from the same state: the consumers catch up, Redis is reseeded (the dataset
users' training histories, as `make seed`), the experiment is set through the admin api, and the
traffic starts once the api serves the new experiment config. The standing campaigns keep the
sponsored slots running, as in production.

The analysis reads the sample job's attributed impressions: run `make samples` afterwards, then
`make ab-report` (analysis/ab_report.py -> docs/results/ab.md).

Writes artifacts/ab/run.json: the phases with their times, experiments, config versions and the
simulator's summaries.

Usage: python -m sim.ab [--seconds 1200] [--only aa|ab]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from sim import ads as ads_mod
from sim import standing
from sim import traffic as traffic_mod
from sim.admin import Admin

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "ab"
PHASES = {
    "aa": {
        "exp_id": "rank-aa-v1",
        "variants": [
            {"name": "a", "share": 50, "params": {"ranker": "rrf"}},
            {"name": "b", "share": 50, "params": {"ranker": "rrf"}},
        ],
    },
    "ab": {
        "exp_id": "rank-ab-v1",
        "variants": [
            {"name": "rrf", "share": 50, "params": {"ranker": "rrf"}},
            {"name": "lightgbm", "share": 50, "params": {"ranker": "lightgbm"}},
        ],
    },
}


def reset_state() -> None:
    """Consumers caught up, then Redis back to the seed state (as the billing runs do)."""
    agg = ROOT / ".cache" / "bin" / "aggregator"
    for group in ("log-sink", "aggregator", "billing"):
        subprocess.run([str(agg), "wait-lag", group], check=True)
    subprocess.run([sys.executable, "-m", "ml.data.seed_redis"], check=True, cwd=ROOT)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--api", default=os.environ.get("RECLOOP_API_URL", "http://127.0.0.1:18080"))
    p.add_argument("--seconds", type=float, default=1200)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--only", choices=sorted(PHASES))
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args(argv)
    admin = Admin(args.api, actor="ab-run")
    ads = ads_mod.Ads(args.api, os.environ.get("DATABASE_URL", ""))
    campaigns = standing.ensure(ads)
    active = admin.overview()["active"]
    run = {"api": args.api, "bundle": active, "campaigns": len(campaigns), "phases": {}}
    path = args.out / "run.json"
    if path.exists():
        run["phases"] = json.loads(path.read_text()).get("phases", {})
    for name in [args.only] if args.only else list(PHASES):
        spec = PHASES[name]
        admin.stop_layer("rank", keep=spec["exp_id"])
        reset_state()
        version = admin.put_experiment(
            {
                "exp_id": spec["exp_id"],
                "layer": "rank",
                "status": "running",
                "bucket_from": 0,
                "bucket_to": 1000,
                "variants": spec["variants"],
            }
        )
        admin.wait_config(version)
        traffic = traffic_mod.Traffic(args.out / name, args.api, args.concurrency)
        t0 = traffic.start(args.seconds)
        print(
            f"ab       {time.strftime('%H:%M:%S')} {name}: {spec['exp_id']} running, {args.seconds:.0f}s of traffic"
        )
        traffic.thread.join()
        traffic.stop()
        t1 = time.time()
        admin.stop_experiment(spec["exp_id"])
        s = traffic.stats
        run["phases"][name] = {
            "exp_id": spec["exp_id"],
            "variants": spec["variants"],
            "config_version": version,
            "from": t0,
            "to": t1,
            "sessions": s.sessions,
            "page_views": dict(s.page_views),
            "get_status": {str(k): v for k, v in s.status.items()},
            "events": dict(s.events),
            "ads": dict(s.ads),
            "evicted_keys": ads_mod.evicted_keys(),
        }
        args.out.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(run, indent=2) + "\n")
        print(
            f"ab       {name} done: {s.sessions:,} sessions, {sum(s.page_views.values()):,} page views"
        )
    admin.close()


if __name__ == "__main__":
    main()
