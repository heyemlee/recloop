"""A/A and A/B report of the ranker experiment (design §8) -> docs/results/ab.md.

Inputs: the run (artifacts/ab/run.json: phases, experiments, times), the sample job's
attributed impressions (artifacts/samples/impressions: every impression joined to its served
record and labelled from the clicks of the next 30 minutes) and the simulator's ledgers
(artifacts/ab/<phase>/ledger.jsonl: per-request latency and degradation as the client saw them).

The unit is the user. CTR and the other per-impression rates are ratio metrics whose
impressions are correlated within a user, so each user's sums are the observations:
    rate = sum(clicks) / sum(impressions)
with the variance of the ratio from the delta method,
    Var(R) ~ (Var(C) / m_I^2 - 2 m_C Cov(C, I) / m_I^3 + m_C^2 Var(I) / m_I^4) / n,
and a user-level bootstrap interval next to it (B = 2000, fixed seed). Two-sided z-test on the
difference; 95 % intervals.

Guardrails: SRM (chi-square of users per arm against the configured split; every user in one
arm only, the arm the Python twin of the bucketing gives), degraded share and client latency.
A/A: the same analysis on two identical arms, then on the A/A's logs re-split 100 times with
other salts; the p-values should be uniform (Kolmogorov-Smirnov) with ~5 % below 0.05.

Usage: python -m analysis.ab_report
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analysis import srm as srm_mod
from ml.experiment import bucketing

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "artifacts" / "ab" / "run.json"
IMPRESSIONS = ROOT / "artifacts" / "samples" / "impressions"
OFFLINE = ROOT / "artifacts" / "eval" / "offline.json"
OUT = ROOT / "docs" / "results" / "ab.md"
B = 2000
SEED = 20261108
RESALTS = 100
SRM_ALPHA = 0.001
METRICS = (
    ("ctr", "CTR (home feed)", "clicks", "impressions"),
    ("cart", "Add-to-cart rate (home feed, per impression)", "carts", "impressions"),
    ("sponsored_ctr", "CTR of sponsored slots", "sp_clicks", "sp_impressions"),
)


def load_impressions(directory: Path = IMPRESSIONS) -> pd.DataFrame:
    days = sorted(directory.glob("day=*"))
    if not days:
        raise SystemExit(f"no impressions under {directory}: run `make samples`")
    cols = [
        "impression_id",
        "request_id",
        "user_id",
        "item_id",
        "pos",
        "surface",
        "slot",
        "price",
        "exp_id_rank",
        "exp_rank",
        "bucket_rank",
        "degraded",
        "clicked",
        "carted",
        "purchased",
        "final",
        "t_impression",
        "ranker",
    ]
    return pd.concat([pd.read_parquet(d, columns=cols) for d in days], ignore_index=True)


def user_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Per user and arm: impressions, clicks, carts on the home feed; sponsored impressions and
    clicks; ad revenue (the price of clicked sponsored slots)."""
    home = df[df.surface == "home"]
    sp = home[home.slot == "sponsored"]
    g = home.groupby(["exp_rank", "user_id"])
    out = pd.DataFrame(
        {"impressions": g.size(), "clicks": g.clicked.sum(), "carts": g.carted.sum()}
    )
    gs = sp.groupby(["exp_rank", "user_id"])
    out["sp_impressions"] = gs.size()
    out["sp_clicks"] = gs.clicked.sum()
    out["revenue"] = sp[sp.clicked == 1].groupby(["exp_rank", "user_id"]).price.sum()
    return out.fillna(0.0).reset_index()


def ratio_delta(c: np.ndarray, i: np.ndarray) -> tuple[float, float]:
    """sum(c) / sum(i) and the delta-method variance of that ratio (users as the unit)."""
    n = len(c)
    mc, mi = c.mean(), i.mean()
    if n < 2 or mi == 0:
        return float("nan"), float("nan")
    vc, vi = c.var(ddof=1), i.var(ddof=1)
    cov = np.cov(c, i, ddof=1)[0, 1]
    var = (vc / mi**2 - 2 * mc * cov / mi**3 + mc**2 * vi / mi**4) / n
    return float(mc / mi), float(var)


def compare(a: pd.DataFrame, b: pd.DataFrame, num: str, den: str, rng: np.random.Generator) -> dict:
    ra, va = ratio_delta(a[num].to_numpy(float), a[den].to_numpy(float))
    rb, vb = ratio_delta(b[num].to_numpy(float), b[den].to_numpy(float))
    diff, se = rb - ra, math.sqrt(va + vb) if va == va and vb == vb else float("nan")
    z = diff / se if se and se > 0 else float("nan")
    p = float(2 * stats.norm.sf(abs(z))) if z == z else float("nan")
    xa = a[[num, den]].to_numpy(float)
    xb = b[[num, den]].to_numpy(float)
    reps = np.empty(B)
    for k in range(B):
        sa = xa[rng.integers(0, len(xa), len(xa))].sum(axis=0)
        sb = xb[rng.integers(0, len(xb), len(xb))].sum(axis=0)
        reps[k] = sb[0] / sb[1] - sa[0] / sa[1]
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {
        "a": ra,
        "b": rb,
        "diff": diff,
        "rel": diff / ra if ra else float("nan"),
        "ci_delta": (diff - 1.96 * se, diff + 1.96 * se),
        "ci_boot": (float(lo), float(hi)),
        "p": p,
        "users": (len(a), len(b)),
        "den": (float(a[den].sum()), float(b[den].sum())),
        "num": (float(a[num].sum()), float(b[num].sum())),
    }


def srm(df: pd.DataFrame, variants: list[dict]) -> dict:
    users = df.groupby("user_id").exp_rank.agg(["nunique", "first"])
    counts = users["first"].value_counts()
    names = [v["name"] for v in variants]
    total = int(counts.reindex(names).fillna(0).sum())
    expected = [total * v["share"] / 100 for v in variants]
    observed = [int(counts.get(n, 0)) for n in names]
    chi2, p = stats.chisquare(observed, expected)
    # Every user's arm is the one the bucketing gives (Python twin of serving/internal/experiment).
    exp = {"layer": "rank", "bucket_from": 0, "bucket_to": 1000, "variants": variants}
    mismatch = sum(
        srm_mod.assign([exp], "rank", str(uid)) != row["first"] for uid, row in users.iterrows()
    )
    return {
        "observed": dict(zip(names, observed, strict=True)),
        "chi2": float(chi2),
        "p": float(p),
        "switchers": int((users["nunique"] > 1).sum()),
        "bucketing_mismatch": mismatch,
        "ok": p >= SRM_ALPHA,
    }


def resalt(totals: pd.DataFrame, n: int = RESALTS) -> dict:
    """The A/A's users re-split by other salts: one CTR z-test per salt."""
    per_user = totals.groupby("user_id")[["clicks", "impressions"]].sum()
    uids = per_user.index.to_numpy()
    c, i = per_user.clicks.to_numpy(float), per_user.impressions.to_numpy(float)
    ps = []
    for k in range(n):
        arm = np.asarray([bucketing.bucket(f"rank-aa-resalt-{k}", str(u)) < 500 for u in uids])
        ra, va = ratio_delta(c[~arm], i[~arm])
        rb, vb = ratio_delta(c[arm], i[arm])
        z = (rb - ra) / math.sqrt(va + vb)
        ps.append(float(2 * stats.norm.sf(abs(z))))
    ps = np.asarray(ps)
    ks = stats.kstest(ps, "uniform")
    return {
        "n": n,
        "p_values": ps.tolist(),
        "below_005": int((ps < 0.05).sum()),
        "ks_stat": float(ks.statistic),
        "ks_p": float(ks.pvalue),
        "deciles": np.histogram(ps, bins=10, range=(0, 1))[0].tolist(),
    }


def ledger_guardrails(path: Path, exp_id: str) -> dict:
    """Per arm from the simulator's ledger: requests, degraded share, client P50 / P99."""
    by = {}
    if not path.exists():
        return by
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") != "page" or "ms" not in r:
                continue
            arm = r["variants"].get("rank")
            d = by.setdefault(arm, {"ms": [], "degraded": 0, "n": 0})
            d["ms"].append(r["ms"])
            d["degraded"] += bool(r["degraded"])
            d["n"] += 1
    return {
        a: {
            "requests": d["n"],
            "degraded": d["degraded"] / d["n"],
            "p50": float(np.percentile(d["ms"], 50)),
            "p99": float(np.percentile(d["ms"], 99)),
        }
        for a, d in by.items()
    }


def analyse(df: pd.DataFrame, phase: dict, ledger: Path) -> dict:
    sub = df[df.exp_id_rank == phase["exp_id"]]
    names = [v["name"] for v in phase["variants"]]
    totals = user_totals(sub)
    rng = np.random.default_rng(SEED)
    a = totals[totals.exp_rank == names[0]]
    b = totals[totals.exp_rank == names[1]]
    out = {
        "exp_id": phase["exp_id"],
        "arms": names,
        "impressions": int(len(sub)),
        "home_impressions": int((sub.surface == "home").sum()),
        "users": int(sub.user_id.nunique()),
        "not_final": int((~sub.final).sum()),
        "srm": srm(sub, phase["variants"]),
        "metrics": {
            key: compare(a, b, num, den, rng) | {"label": label} for key, label, num, den in METRICS
        },
        "clicks_per_user": _welch(a.clicks.to_numpy(float), b.clicks.to_numpy(float)),
        "rpm": compare(a, b, "revenue", "impressions", rng),
        "guardrails": ledger_guardrails(ledger, phase["exp_id"]),
        "rankers": sub.groupby("exp_rank")
        .ranker.agg(lambda s: s.value_counts().to_dict())
        .to_dict(),
    }
    out["rpm"] = {k: (v * 1000 if k in ("a", "b", "diff") else v) for k, v in out["rpm"].items()}
    out["rpm"]["ci_delta"] = tuple(x * 1000 for x in out["rpm"]["ci_delta"])
    out["rpm"]["ci_boot"] = tuple(x * 1000 for x in out["rpm"]["ci_boot"])
    if phase["exp_id"].startswith("rank-aa"):
        out["resalt"] = resalt(totals)
    return out


def _welch(a: np.ndarray, b: np.ndarray) -> dict:
    t = stats.ttest_ind(b, a, equal_var=False)
    return {
        "a": float(a.mean()),
        "b": float(b.mean()),
        "diff": float(b.mean() - a.mean()),
        "p": float(t.pvalue),
    }


# ------------------------------------------------------------------------ rendering


def _f(x: float, d: int = 4) -> str:
    return "n/a" if x != x else f"{x:.{d}f}"


def _ci(ci, d: int = 4) -> str:
    return f"[{ci[0]:+.{d}f}, {ci[1]:+.{d}f}]"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + [
        "| " + " | ".join(r) + " |" for r in rows
    ]


def _metric_rows(res: dict) -> list[list[str]]:
    rows = []
    for m in res["metrics"].values():
        rows.append(
            [
                m["label"],
                _f(m["a"]),
                _f(m["b"]),
                f"{m['diff']:+.4f} ({m['rel']:+.1%})",
                _ci(m["ci_delta"]),
                _ci(m["ci_boot"]),
                _f(m["p"], 3),
            ]
        )
    r = res["rpm"]
    rows.append(
        [
            "Ad revenue per 1,000 home impressions (cents)",
            _f(r["a"], 1),
            _f(r["b"], 1),
            f"{r['diff']:+.1f} ({r['rel']:+.1%})",
            _ci(r["ci_delta"], 1),
            _ci(r["ci_boot"], 1),
            _f(r["p"], 3),
        ]
    )
    c = res["clicks_per_user"]
    rows.append(
        [
            "Home-feed clicks per user",
            _f(c["a"], 2),
            _f(c["b"], 2),
            f"{c['diff']:+.2f}",
            "",
            "",
            _f(c["p"], 3),
        ]
    )
    return rows


def _redis_cells(r: dict | None) -> list[str]:
    """A phase's effect on the two Redis instances (sim/redis_stats.py); the state instance
    must show 0 / 0."""
    if not r:
        return ["n/a", "n/a"]
    st, ca = r["state"], r["cache"]
    cell = f"{st['evicted_keys']:,} / {st['oom_errors']:,}"
    if st["evicted_keys"] or st["oom_errors"]:
        cell = f"**{cell}**"
    return [cell, f"{ca['evicted_keys']:,}"]


def render(run: dict, results: dict, offline: dict | None) -> str:
    aa, ab = results.get("aa"), results.get("ab")
    lines = [
        "# A/B test: LightGBM ranker against RRF",
        "",
        "<!-- Generated by `make ab-report` (analysis/ab_report.py). Do not edit by hand. -->",
        "",
        "Simulated traffic (`sim/`, design §2.2) on the api with bundle "
        f"`{run['bundle']}`, the rank-layer experiment split 50 / 50 over all users, the standing "
        f"campaigns ({run['campaigns']}) filling the sponsored slots. The simulator's users click by "
        "examination × relevance to their test-period items; it validates the system and the experiment "
        "pipeline, while the evidence for the model is the offline evaluation (§2.2). The A/A ran first, "
        "and the analysis was checked on it before the A/B was read.",
        "",
        "## Setup",
        "",
    ]
    rows = []
    for name, ph in run["phases"].items():
        pv = sum(ph["page_views"].values())
        rows.append(
            [
                name.upper(),
                f"`{ph['exp_id']}`",
                ", ".join(
                    f"{v['name']} {v['share']}% ({v['params']['ranker']})" for v in ph["variants"]
                ),
                f"{(ph['to'] - ph['from']) / 60:.0f} min",
                f"{ph['sessions']:,}",
                f"{pv:,}",
                *_redis_cells(ph.get("redis")),
            ]
        )
    lines += _table(
        [
            "Phase",
            "Experiment",
            "Arms",
            "Traffic",
            "Sessions",
            "Page views",
            "State Redis: evicted / refused",
            "Cache Redis: evicted",
        ],
        rows,
    )
    lines += [
        "",
        "Each phase started from the same state: the consumers caught up, the cache Redis flushed and the state Redis "
        "reseeded with the dataset users' training histories. Unit: the user. Ratio metrics (CTR, add-to-cart rate) are sums per user, the variance of the ratio from the "
        f"delta method, with a user-level bootstrap interval next to it (B = {B:,}); two-sided z-test. Impressions come from the "
        "sample job (`ml/samples/job.py`): every impression joined to its served record, labelled by a click within 30 minutes.",
        "",
    ]
    for res, title in ((aa, "A/A"), (ab, "A/B")):
        if res is None:
            continue
        s = res["srm"]
        lines += [
            f"## {title}: `{res['exp_id']}`",
            "",
            f"{res['users']:,} users, {res['home_impressions']:,} home-feed impressions ({res['impressions']:,} in all); "
            f"{res['not_final']:,} impressions still inside their attribution window were left out.",
            "",
            "Sample ratio: "
            + ", ".join(f"{k} {v:,}" for k, v in s["observed"].items())
            + f" users; chi-square p = {s['p']:.3f} ({'no mismatch' if s['ok'] else '**MISMATCH**'} at {SRM_ALPHA}); "
            f"{s['switchers']} users seen in two arms; {s['bucketing_mismatch']} users whose arm differs from the Python bucketing.",
            "",
            *_table(
                [
                    f"Metric ({res['arms'][1]} vs {res['arms'][0]})",
                    res["arms"][0],
                    res["arms"][1],
                    "Difference",
                    "95% CI (delta)",
                    "95% CI (bootstrap)",
                    "p",
                ],
                _metric_rows(res),
            ),
            "",
        ]
        g = res["guardrails"]
        if g:
            lines += _table(
                ["Arm", "Requests", "Degraded", "Client P50 (ms)", "Client P99 (ms)"],
                [
                    [
                        a,
                        f"{v['requests']:,}",
                        f"{v['degraded']:.2%}",
                        f"{v['p50']:.1f}",
                        f"{v['p99']:.1f}",
                    ]
                    for a, v in sorted(g.items())
                ],
            )
            lines += [""]
        if "resalt" in res:
            r = res["resalt"]
            lines += [
                f"Re-split {r['n']} times with other salts (`rank-aa-resalt-<k>`), the A/A's CTR test gives p < 0.05 in "
                f"{r['below_005']} of {r['n']} splits (expected ~5); the p-values against the uniform distribution: "
                f"Kolmogorov-Smirnov D = {r['ks_stat']:.3f}, p = {r['ks_p']:.2f}. P-values per decile: "
                + ", ".join(str(x) for x in r["deciles"])
                + ".",
                "",
            ]
    if ab is not None:
        m = ab["metrics"]["ctr"]
        verdict = (
            "higher"
            if m["ci_delta"][0] > 0
            else "lower"
            if m["ci_delta"][1] < 0
            else "not significantly different"
        )
        lines += [
            "## Reading",
            "",
            f"- Home-feed CTR with the LightGBM ranker is {verdict} than with RRF: {m['diff']:+.4f} "
            f"({m['rel']:+.1%}), 95% CI {_ci(m['ci_delta'])}, p = {m['p']:.3g}.",
        ]
        if offline:
            lr = offline["pairs"]["LambdaRank - RRF"]["item"]
            lines.append(
                f"- Offline (`docs/results/offline.md`, test split): NDCG@10 {lr['NDCG@10']['value']:+.4f} "
                f"{_ci(lr['NDCG@10']['ci'])}, HR@10 {lr['HR@10']['value']:+.4f} {_ci(lr['HR@10']['ci'])}."
            )
        lines += [
            "- Both arms make the same calls (Rank runs for every user; the arm only decides which score orders the list), "
            "so the latency and degradation columns compare like with like.",
            "",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    if not RUN.exists():
        raise SystemExit(f"{RUN} is missing: run `make ab` first")
    run = json.loads(RUN.read_text())
    df = load_impressions()
    results = {
        name: analyse(df, ph, RUN.parent / name / "ledger.jsonl")
        for name, ph in run["phases"].items()
    }
    offline = json.loads(OFFLINE.read_text()) if OFFLINE.exists() else None
    OUT.write_text(render(run, results, offline))
    (RUN.parent / "report.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
    for name, res in results.items():
        m = res["metrics"]["ctr"]
        print(
            f"ab       {name}: CTR {m['a']:.4f} vs {m['b']:.4f}, diff {m['diff']:+.4f} {_ci(m['ci_delta'])}, p {m['p']:.3g}; SRM p {res['srm']['p']:.3f}"
        )
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
