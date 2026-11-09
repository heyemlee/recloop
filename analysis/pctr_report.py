"""pCTR with and without the item CTR columns (design §9.2 step 3) -> docs/results/pctr.md.

Inputs: the evaluations `make train-pctr` writes (artifacts/models/pctr/eval/<label>.json: both
variants on the same time-ordered split, the paired user bootstrap of the differences), the
shipped model (artifacts/models/pctr/model.json) and the sample job's online / offline check of
the counts (artifacts/samples/meta/<day>.json, `item_ctr`). One section per evaluation, in the
order of their samples' first serving time.

Usage: python -m analysis.pctr_report
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "artifacts" / "models" / "pctr"
META_DIR = ROOT / "artifacts" / "samples" / "meta"
OUT = ROOT / "docs" / "results" / "pctr.md"


def _f(x, d: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{d}f}"


def _signed(x, d: int = 4) -> str:
    return "n/a" if x is None or math.isnan(x) else f"{x:+.{d}f}"


def _ci(c: dict, d: int = 4) -> str:
    lo, hi = c["ci95"]
    return f"[{_signed(lo, d)}, {_signed(hi, d)}]"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *["| " + " | ".join(r) + " |" for r in rows],
    ]


def _metric_rows(base: dict, ctr: dict, comp: dict | None) -> list[list[str]]:
    def diff(k):
        return _signed(ctr[k] - base[k])

    rows = [
        [
            "AUC",
            _f(base["auc"]),
            _f(ctr["auc"]),
            diff("auc"),
            _ci(comp["auc"]) if comp else "",
        ],
        [
            "Weighted log loss",
            _f(base["logloss"], 5),
            _f(ctr["logloss"], 5),
            _signed(ctr["logloss"] - base["logloss"], 5),
            _ci(comp["logloss"], 5) if comp else "",
        ],
        [
            "Log loss of the constant CTR",
            _f(base["logloss_constant"], 5),
            _f(ctr["logloss_constant"], 5),
            "",
            "",
        ],
        [
            "Predicted CTR / observed",
            f"{_f(base['predicted_ctr'])} / {_f(base['observed_ctr'])}",
            f"{_f(ctr['predicted_ctr'])} / {_f(ctr['observed_ctr'])}",
            "",
            "",
        ],
        [
            "Calibration error (deciles)",
            _f(base["calibration_error"]),
            _f(ctr["calibration_error"]),
            diff("calibration_error"),
            "",
        ],
    ]
    return rows


def _check_lines(checks: dict) -> list[str]:
    out = []
    for day, c in sorted(checks.items()):
        if not c:
            continue
        line = f"- {day}: of {c['eligible']:,} eligible impressions (home feed, with features), "
        if c["logged"]:
            share = c["equal"] / c["logged"]
            diffs = ", ".join(
                f"{f} {v}" for f, v in zip(c["fields"], c["max_abs_diff"], strict=True)
            )
            line += (
                f"{c['logged']:,} had their counts logged at serving: {c['equal']:,} "
                f"({share:.2%}) equal the counts the sample job recomputed from the log, "
                f"{c['differ']:,} differ ({c['logged_below_log']:,} logged below the log, "
                f"{c['logged_above_log']:,} above; largest differences per field {diffs})"
            )
        else:
            line += "none had its counts logged at serving"
        if c["recomputed_only"]:
            line += (
                f"{'; ' if c['logged'] else ': '}{c['recomputed_only']:,} were served before the "
                "api logged counts and use the recomputed ones (E = the newest 5-minute boundary "
                "at least 15 s before serving; no seed, as those records name none)"
            )
        out.append(line + ".")
    return out


def _summary(evals: list[dict]) -> list[str]:
    rows = []
    for ev in evals:
        comp = (ev.get("comparison") or {}).get("all")
        src = ev.get("counts_source") or {}
        counts = " + ".join(
            f"{'logged' if k == 'served' else 'recomputed'}" for k in sorted(src, reverse=True)
        )
        rows.append(
            [
                ev["label"],
                f"{ev['rows']:,}",
                counts or "none",
                f"{_signed(comp['auc']['diff'])} {_ci(comp['auc'])}" if comp else "",
                f"{_signed(comp['logloss']['diff'], 5)} {_ci(comp['logloss'], 5)}" if comp else "",
            ]
        )
    return _table(
        ["Evaluation", "Samples", "Counts", "AUC, with − without", "Log loss, with − without"],
        rows,
    )


def _section(ev: dict) -> list[str]:
    v = ev["variants"]
    base = v["base"]["held_out"]
    lines = [f"## {ev['label']}", ""]
    split = ev["split"]
    test = split["test"]
    src = ev.get("counts_source") or {}
    served = _window(ev.get("since"), ev.get("until"))
    lines.append(
        f"{ev['rows']:,} training samples ({ev['positives']:,} clicked) of {ev['users']:,} users, "
        f"{ev['sponsored_rows']:,} of them sponsored slots, from {', '.join(ev['days'])}{served}. "
        f"In time order: the first 64 % fit, the next 16 % choose the settings, the last 20 % "
        f"({test[1] - test[0]:,} samples, {_time(test[2])} to {_time(test[3])} UTC) judge."
        if len(test) == 4
        else f"{ev['rows']:,} training samples from {', '.join(ev['days'])}{served}."
    )
    if src:
        lines.append(
            "Counts: "
            + ", ".join(
                f"{n:,} {'logged at serving' if k == 'served' else 'recomputed from the log'}"
                for k, n in sorted(src.items(), reverse=True)
            )
            + "."
        )
    cov = ev.get("counts_coverage") or {}
    if cov.get("all") and cov.get("test"):
        a, t = cov["all"], cov["test"]
        lines.append(
            f"Of the impressions (weighted), {a['hour']:.1%} were of an item with at least one "
            f"impression in its hour window and {a['day']:.1%} in its day window (last 20 %: "
            f"{t['hour']:.1%} and {t['day']:.1%}); the others' columns sit at the prior."
        )
    lines.append("")
    if "ctr" not in v:
        lines += [
            "These samples carry no counts: only the model without the item CTR columns was "
            f"trained (AUC {_f(base['auc'])}, log loss {_f(base['logloss'], 5)}).",
            "",
        ]
        return lines
    ctr = v["ctr"]["held_out"]
    sel_b, sel_c = v["base"]["selected"], v["ctr"]["selected"]
    lines.append(
        f"Settings chosen on the middle part: without the CTR columns C = {sel_b['C']}; with them "
        f"C = {sel_c['C']} and a prior strength of {sel_c['strength']:g} impressions."
    )
    lines.append("")
    comp = ev.get("comparison") or {}
    lines += _table(
        ["Last 20 %, every sample", "Without", "With item CTR", "Difference", "95% CI (users)"],
        _metric_rows(base, ctr, comp.get("all")),
    )
    lines.append("")
    if base.get("sponsored") and ctr.get("sponsored"):
        lines += _table(
            [
                "Last 20 %, sponsored slots",
                "Without",
                "With item CTR",
                "Difference",
                "95% CI (users)",
            ],
            _metric_rows(base["sponsored"], ctr["sponsored"], comp.get("sponsored")),
        )
        lines.append("")
    a = comp.get("all")
    if a:
        lines.append(
            f"With the item CTR columns the AUC is {_auc_word(a['auc'])} "
            f"({_signed(a['auc']['diff'])}, 95% CI {_ci(a['auc'])}) and the log loss "
            f"{_loss_word(a['logloss'])} ({_signed(a['logloss']['diff'], 5)}, "
            f"{_ci(a['logloss'], 5)}); intervals from {a['resamples']:,} resamples of the last "
            f"20 %'s {a['users']:,} users, the same resample for both models."
        )
        lines.append("")
    lines += [
        "Calibration by decile of the prediction, with the item CTR columns (weighted impressions):",
        "",
    ]
    lines += _table(
        ["Decile", "Impressions", "Predicted", "Observed"],
        [
            [k, f"{c['impressions']:,.0f}", _f(c["predicted"]), _f(c["observed"])]
            for k, c in ctr["by_decile"].items()
        ],
    )
    lines.append("")
    lines += ["Calibration by position (predicted without / with, observed):", ""]
    lines += _table(
        ["Position", "Impressions", "Without", "With", "Observed"],
        [
            [
                k.removeprefix("pos_").replace("_", "-").replace("-up", " and later"),
                f"{c['impressions']:,.0f}",
                _f(base["by_position"].get(k, {}).get("predicted")),
                _f(c["predicted"]),
                _f(c["observed"]),
            ]
            for k, c in ctr["by_position"].items()
        ],
    )
    lines.append("")
    return lines


def _time(iso: str) -> str:
    return iso[:16].replace("T", " ")


def _window(since: str | None, until: str | None) -> str:
    """The samples' serving times when the training run limited them (pctr --since / --until)."""
    if since and until:
        return f" served from {_time(since)} to {_time(until)} UTC"
    if since:
        return f" served from {_time(since)} UTC on"
    if until:
        return f" served before {_time(until)} UTC"
    return ""


def _auc_word(c: dict) -> str:
    lo, hi = c["ci95"]
    return "higher" if lo > 0 else "lower" if hi < 0 else "not significantly different"


def _loss_word(c: dict) -> str:
    lo, hi = c["ci95"]
    return "lower" if hi < 0 else "higher" if lo > 0 else "not significantly different"


def render(evals: list[dict], model: dict | None, checks: dict | None = None) -> str:
    """checks: the sample job's online / offline check per day (meta `item_ctr`); without them,
    the copies the evaluations took when they were trained."""
    if checks is None:
        checks = {}
        for ev in evals:
            checks.update({d: c for d, c in (ev.get("item_ctr_checks") or {}).items() if c})
    lines = [
        "# pCTR: the item CTR columns",
        "",
        "<!-- Generated by `make pctr-report` (analysis/pctr_report.py). Do not edit by hand. -->",
        "",
        "The pCTR of the sponsored slots is a logistic regression (`ml/models/pctr.py`, design "
        "§9.2 step 3) over the ranker's features of the item, the slot and the position (fixed at "
        "the first sponsored slot when serving). This report compares it with and without the "
        "item's recent CTR: per window, (clicks + strength × prior) / (impressions + strength) "
        "in log-odds and log(1 + impressions), the prior being the training rows' CTR.",
        "",
        "How the counts are made (`serving/internal/itemctr`; the sample job recomputes the same):",
        "",
        "- Counted: impression and click events that carry an impression_id, once per event_id "
        "(the aggregator applies them under the event's dedup marker), on every surface, in "
        "5-minute and hourly buckets of the time the api received them (`events_raw.server_ts`). "
        "A client clock cannot move an event into a closed bucket.",
        "- The api rebuilds an in-memory table 15 s after every 5-minute boundary E: the hour = the "
        "12 complete 5-minute buckets before E, the day = the 24 complete hours before E's hour. "
        "So the hour ends 15 s to 5 min 15 s before the request that reads it, and the day up to "
        "1 h 15 s before it. A table older than 15 minutes is not used (the model then uses its "
        "prior). No Redis call on the request path.",
        "- The served event logs each item's counts next to its features, with E and the time of "
        "the last seed (`make seed` empties the buckets; the windows then start at the seed). "
        "Training uses the logged counts (design §7.5); the recomputation serves samples served "
        "before the api logged counts, and the comparison of the two.",
        "",
    ]
    if model:
        lines.append(
            f"Shipped model (`artifacts/models/pctr/model.json`): variant `{model.get('variant', 'base')}`, "
            f"C = {model['C']}"
            + (
                f", prior {model['ctr']['prior']:.4f}, strength {model['ctr']['strength']:g}"
                if "ctr" in model
                else ""
            )
            + f", trained on the samples of {', '.join(model['training']['days'])}"
            + _window(model["training"].get("since"), model["training"].get("until"))
            + " (the last `make train-pctr`)."
        )
        lines.append("")
    if not evals:
        lines += ["No evaluation yet: run `make train-pctr`.", ""]
    elif any(ev.get("comparison") for ev in evals):
        lines += [
            "Each evaluation on the last 20 % of its samples in time, differences with 95% "
            "intervals (paired user bootstrap):",
            "",
            *_summary(evals),
            "",
        ]
    for ev in evals:
        lines += _section(ev)
    check = _check_lines(checks)
    if check:
        lines += [
            "## Online against offline",
            "",
            "The sample job recomputes every eligible impression's counts from the log, for the E "
            "and the seed its served record logged, and compares them with the counts the api "
            "logged (its latest run per day):",
            "",
            *check,
            "",
        ]
    lines += [
        "## Known limitations",
        "",
        "- The counts are not validated. Impression and click events need no token, so forged "
        "events raise an item's counts and with them its pCTR; and a higher pCTR wins more "
        "auctions at a lower GSP price (the price divides the next ad's eCPM by the winner's "
        "pCTR). Possible later: count only the clicks of sponsored slots whose ad_token "
        "verifies, or only impressions joined to a served record. Not in v1.",
        "- Raw CTR, not normalised by position: an item shown mostly near the top has a higher "
        "CTR than the same item shown lower.",
        "- The simulated traffic on this machine spans less than a day, so the day window holds "
        "only the complete hours since the traffic started (or since the last seed, rounded up "
        "to the hour): it is empty for the first one to two hours and never slides; only the "
        "hour window does.",
        "- The simulator's users click by text relevance to their test-period items: an item's "
        "CTR says how many of them find it relevant. The size of the gain is the simulator's.",
        "",
    ]
    return "\n".join(lines)


def load(
    model_dir: Path = MODEL_DIR, meta_dir: Path = META_DIR
) -> tuple[list[dict], dict | None, dict]:
    """The evaluations (in the order of their samples' first serving time, then by name), the
    shipped model and the sample job's online / offline check of each day it ran for."""
    evals = [json.loads(p.read_text()) for p in sorted((model_dir / "eval").glob("*.json"))]
    evals.sort(key=lambda e: (e["split"]["train"][2:3] or [""], e["label"]))
    path = model_dir / "model.json"
    checks = {}
    for p in sorted(meta_dir.glob("*.json")):
        c = json.loads(p.read_text()).get("item_ctr")
        if c:
            checks[p.stem] = c
    return evals, json.loads(path.read_text()) if path.exists() else None, checks


def main() -> None:
    evals, model, checks = load()
    OUT.write_text(render(evals, model, checks))
    print(f"pctr     {len(evals)} evaluation(s) -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
