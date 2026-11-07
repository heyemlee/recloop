"""Sample ratio mismatch (SRM) check on logged traffic (docs/design.md §8).

For every experiment layer, count the units (users) per variant in the served log and test the
counts against the configured split with a chi-square goodness-of-fit test. A mismatch means
the assignment or the logging is broken (a variant loses events, users switch groups), and the
experiment's result cannot be trusted. The threshold is p < 0.001, the usual SRM alarm level.

Two more checks make the result stronger than the p-value alone: every user has exactly one
variant per layer across all of their served rows, and it is the variant the Python twin of
the bucketing (ml/experiment/bucketing.py) computes for that user.

The rows come from events_raw, i.e. after the api -> Redpanda -> log-sink path; the request
ids of a simulator run (its ledger) select the rows.

Usage: python -m analysis.srm --run artifacts/sim/srm
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import psycopg
from scipy import stats

from ml.experiment import bucketing

ROOT = Path(__file__).resolve().parent.parent
ALPHA = 0.001


def variant_buckets(exp: dict) -> dict[str, int]:
    """Buckets per variant of one experiment: the integer split of serving/internal/experiment
    (variant i covers offsets [width*cum(i-1)/100, width*cum(i)/100))."""
    width = exp["bucket_to"] - exp["bucket_from"]
    out, prev, cum = {}, 0, 0
    for v in exp["variants"]:
        cum += v["share"]
        end = width * cum // 100
        out[v["name"]] = out.get(v["name"], 0) + end - prev
        prev = end
    return out


def expected_shares(experiments: list[dict]) -> dict[str, dict[str, float]]:
    """layer -> variant -> expected share of units (uncovered buckets: "default")."""
    out = {}
    for layer in bucketing.LAYERS:
        counts: Counter = Counter()
        for e in experiments:
            if e["layer"] == layer:
                counts.update(variant_buckets(e))
        covered = sum(counts.values())
        if covered < bucketing.BUCKETS:
            counts["default"] += bucketing.BUCKETS - covered
        out[layer] = {v: n / bucketing.BUCKETS for v, n in counts.items()}
    return out


def assign(experiments: list[dict], layer: str, unit: str) -> str:
    """Variant of a unit, computed independently of the Go code."""
    b = bucketing.bucket(bucketing.SALTS[layer], unit)
    for e in experiments:
        if e["layer"] == layer and e["bucket_from"] <= b < e["bucket_to"]:
            offset, width, cum = b - e["bucket_from"], e["bucket_to"] - e["bucket_from"], 0
            for v in e["variants"]:
                cum += v["share"]
                if offset < width * cum // 100:
                    return v["name"]
            return e["variants"][-1]["name"]
    return "default"


def chi_square(observed: dict[str, int], shares: dict[str, float]) -> dict:
    variants = sorted(shares)
    n = sum(observed.get(v, 0) for v in variants)
    unexpected = sorted(set(observed) - set(variants))
    obs = [observed.get(v, 0) for v in variants]
    exp = [n * shares[v] for v in variants]
    if len(variants) < 2:  # a layer without experiments: everyone is "default"
        p = 1.0 if not unexpected and n == obs[0] else 0.0
        chi2 = 0.0
    else:
        chi2, p = stats.chisquare(obs, exp)
    return {
        "units": n,
        "observed": dict(zip(variants, obs, strict=True)),
        "expected": {v: round(e, 1) for v, e in zip(variants, exp, strict=True)},
        "observed_share": {
            v: round(o / n, 4) if n else 0 for v, o in zip(variants, obs, strict=True)
        },
        "configured_share": shares,
        "unexpected_variants": unexpected,
        "chi2": round(float(chi2), 4),
        "p": float(p),
        "srm": bool(p < ALPHA or unexpected),
    }


def analyse(rows: list[tuple[int, dict]], experiments: list[dict]) -> dict:
    """rows: (user_id, served payload's exp) per served event."""
    shares = expected_shares(experiments)
    per_user: dict[str, dict[int, set]] = defaultdict(lambda: defaultdict(set))
    for uid, exp in rows:
        for layer in bucketing.LAYERS:
            per_user[layer][uid].add((exp.get(layer) or {}).get("variant"))
    layers, problems = {}, []
    for layer in bucketing.LAYERS:
        users = per_user[layer]
        switched = [u for u, vs in users.items() if len(vs) != 1]
        mismatched = [
            u
            for u, vs in users.items()
            if len(vs) == 1 and next(iter(vs)) != assign(experiments, layer, str(u))
        ]
        counts = Counter(next(iter(vs)) for vs in users.values() if len(vs) == 1)
        res = chi_square(dict(counts), shares[layer])
        res["users_in_several_variants"] = len(switched)
        res["users_not_matching_python_bucketing"] = len(mismatched)
        layers[layer] = res
        if res["srm"] or switched or mismatched:
            problems.append(layer)
    return {
        "served_rows": len(rows),
        "alpha": ALPHA,
        "layers": layers,
        "ok": not problems,
        "failed_layers": problems,
    }


def load_rows(conn, request_ids: list[str]) -> list[tuple[int, dict]]:
    conn.execute("CREATE TEMP TABLE srm_requests (request_id uuid PRIMARY KEY)")
    with conn.cursor().copy("COPY srm_requests (request_id) FROM STDIN") as cp:
        for rid in request_ids:
            cp.write_row((rid,))
    return conn.execute(
        """SELECT r.user_id, r.payload -> 'exp' FROM events_raw AS r
           JOIN srm_requests USING (request_id) WHERE r.type = 'served'"""
    ).fetchall()


def load_experiments(conn) -> list[dict]:
    rows = conn.execute(
        """SELECT exp_id, layer, bucket_from, bucket_to, variants FROM experiments
           WHERE status = 'running' AND (start_at IS NULL OR start_at <= now())
             AND (end_at IS NULL OR end_at > now())"""
    ).fetchall()
    return [
        {"exp_id": r[0], "layer": r[1], "bucket_from": r[2], "bucket_to": r[3], "variants": r[4]}
        for r in rows
    ]


def print_report(rep: dict) -> None:
    print(f"srm      {rep['served_rows']:,} served rows, alpha {rep['alpha']}")
    for layer, r in rep["layers"].items():
        flag = "SRM" if r["srm"] else "ok "
        print(
            f"  {flag} {layer:<6} {r['units']:>5} users  observed {r['observed']}  "
            f"expected {r['expected']}  chi2 {r['chi2']:.3f}  p {r['p']:.3f}  "
            f"switched {r['users_in_several_variants']}  "
            f"≠python {r['users_not_matching_python_bucketing']}"
        )
    print("  SRM check passed" if rep["ok"] else f"  SRM CHECK FAILED: {rep['failed_layers']}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", type=Path, default=ROOT / "artifacts" / "sim" / "srm")
    args = p.parse_args(argv)
    request_ids = sorted(
        {
            json.loads(line)["request_id"]
            for line in (args.run / "ledger.jsonl").open()
            if line.startswith('{"kind":"page"')
        }
    )
    url = os.environ.get(
        "DATABASE_URL", "postgres://recloop:recloop@localhost:15432/recloop?sslmode=disable"
    )
    with psycopg.connect(url) as conn:
        rows = load_rows(conn, request_ids)
        experiments = load_experiments(conn)
    rep = analyse(rows, experiments)
    rep["requests"] = len(request_ids)
    rep["experiments"] = experiments
    (args.run / "srm.json").write_text(json.dumps(rep, indent=2))
    print_report(rep)
    raise SystemExit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
