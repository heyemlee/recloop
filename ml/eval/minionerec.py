"""MiniOneRec under RecLoop's offline protocol (`make minionerec`; no GPU, no database).

Reads what `make minionerec-gen` (artifacts/minionerec/runs/<name>/) and `make minionerec-cost`
(artifacts/minionerec/cost/, artifacts/minionerec/sasrec_cost.json) wrote, and writes
docs/results/minionerec.md (rendered by ml.eval.minionerec_report) and
artifacts/eval/minionerec.json. Re-running gives the same bytes.

  * Reproduction: MiniOneRec's calc.py, unmodified, on our generations, against the paper's
    Table 1 (Office).
  * Fair comparison: the same generations scored with ml/eval's metric code (item rule as the
    main metric, SID rule, popularity tie order, user-level bootstrap with B = 1000 and seed
    20261106) next to Pop, ItemCF and SASRec, fitted and scored exactly as `make eval` does, on
    the same samples, with paired differences on shared bootstrap replicates.
  * From SIDs to items: MiniOneRec generates semantic IDs, and 15 pairs of items share one. Under
    the item rule a generated SID stands for every item that has it, in the tie order (training
    popularity, then item_id), so 50 beams are a ranked list of at least 50 items. Under the SID
    rule the rank is the position of the target's SID among the beams, which is what calc.py
    computes; at every K calc.py prints, our SID-rule numbers must equal its numbers.
  * Segments: history length; target popularity (head = the 20% most-interacted items in
    training); whether the target's category (level 2 of the Amazon category path, e.g. "Paper"
    or "Printer Ink & Toner") appears in the history; target reviewed on the same day as the
    last history item or later. HR@10 and NDCG@10 with bootstrap intervals inside every
    segment, and paired differences against SASRec and ItemCF.
  * Cost: per-request latency (batch 1), throughput (batch 8) and memory at 10 / 20 / 50 beams on
    the Mac's GPU, against SASRec on the CPU.

Usage: python -m ml.eval.minionerec [--main-run test]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ml import paths
from ml.eval import bootstrap, calc_check, metrics, run
from ml.eval.splits import NUM_ITEMS, Split, item_sids, load_split, train_sequences
from ml.models.itemcf import ItemCF
from ml.models.pop import Pop

MR_ARTIFACTS = paths.ARTIFACTS / "minionerec"
RUNS = MR_ARTIFACTS / "runs"
COST = MR_ARTIFACTS / "cost"
SASREC_COST = MR_ARTIFACTS / "sasrec_cost.json"
OUT_JSON = paths.ARTIFACTS / "eval" / "minionerec.json"
REPORT = paths.RESULTS / "minionerec.md"

NAME = "MiniOneRec"
BASELINES = ("Pop", "ItemCF", "SASRec")
REFERENCES = ("SASRec", "ItemCF")  # paired differences MiniOneRec - reference
SEGMENT_METRICS = ("HR@10", "NDCG@10")
HEAD_SHARE = 0.2
CATEGORY_LEVEL = 2
MISS = 10**6  # rank of a target that is not in the generated list
# Reproduced when the paper's HR@10 and NDCG@10 (calc.py) lie in our 95% bootstrap intervals.
REPRO_METRICS = ("HR@10", "NDCG@10")


def normalize(sid: str) -> str:
    """calc.py's normalization of a generated or target SID string."""
    return sid.strip('"\n').strip()


def sid_items(sids: list[str], tie_order: np.ndarray) -> dict[str, list[int]]:
    """SID -> the items that have it, in tie order."""
    out: dict[str, list[int]] = {}
    for item in tie_order.tolist():
        out.setdefault(sids[item], []).append(item)
    return out


def generated_ranks(
    predict: list[list[str]], target: np.ndarray, sids: list[str], by_sid: dict[str, list[int]]
) -> dict[str, np.ndarray]:
    """1-based rank of the target under both hit rules for generated SID lists (MISS if absent).

    item: position of the target item in the list obtained by replacing every generated SID with
          its items (tie order), dropping repeats and SIDs that are not in the catalog;
    sid:  position of the first beam equal to the target's SID (calc.py's minID + 1; beams that
          are not catalog SIDs still take a position, as in calc.py).
    """
    n = len(target)
    item_rank = np.full(n, MISS, dtype=np.int64)
    sid_rank = np.full(n, MISS, dtype=np.int64)
    for i, (beams, t) in enumerate(zip(predict, target.tolist(), strict=True)):
        target_sid = sids[t]
        seen: set[int] = set()
        for b, raw in enumerate(beams):
            sid = normalize(raw)
            if sid_rank[i] == MISS and sid == target_sid:
                sid_rank[i] = b + 1
            for item in by_sid.get(sid, ()):
                if item not in seen:
                    seen.add(item)
                    if item == t:
                        item_rank[i] = len(seen)
            if item_rank[i] != MISS and sid_rank[i] != MISS:
                break
    return {"item": item_rank, "sid": sid_rank}


@dataclass(frozen=True)
class Generation:
    name: str
    rows: np.ndarray  # row numbers in the published test CSV, in result order
    predict: list[list[str]]
    output: list[str]  # target SIDs as evaluate.py wrote them
    meta: dict  # run.json
    timing: dict  # timing.json

    @property
    def beams(self) -> int:
        return int(self.meta["settings"]["num_beams"])


def load_generation(run_dir: Path) -> Generation:
    rows = np.asarray(json.loads((run_dir / "rows.json").read_text()), dtype=np.int64)
    result = json.loads((run_dir / "result.json").read_text())
    if len(result) != len(rows):
        raise SystemExit(f"{run_dir}: {len(result)} results for {len(rows)} rows")
    return Generation(
        name=run_dir.name,
        rows=rows,
        predict=[r["predict"] for r in result],
        output=[r["output"] for r in result],
        meta=json.loads((run_dir / "run.json").read_text()),
        timing=json.loads((run_dir / "timing.json").read_text()),
    )


CHECKPOINT = "data/hf/Office_ckpt"


def real_checkpoint(meta: dict) -> bool:
    """False for smoke runs made with another --base-model (tests only)."""
    return str(meta.get("checkpoint", "")).startswith(CHECKPOINT)


def finished_runs(runs_dir: Path = RUNS, include_test_models: bool = False) -> list[Path]:
    out = []
    for p in sorted(runs_dir.glob("*")):
        if not (p / "run.json").exists():
            continue
        if not include_test_models and not real_checkpoint(
            json.loads((p / "run.json").read_text())
        ):
            print(f"skipping {p.name}: not generated with {CHECKPOINT}")
            continue
        out.append(p)
    return out


# ----------------------------------------------------------------------------- baselines


@dataclass
class Context:
    """Everything shared by all runs: the test split, tie order, baselines' per-sample values."""

    test: Split
    tie_order: np.ndarray
    counts: np.ndarray  # training interactions per item
    sids: list[str]
    by_sid: dict[str, list[int]]
    baselines: dict  # name -> {"item": {metric: (n,)}, "sid": {...}} on the whole test split
    descriptions: dict[str, str]


def build_context() -> Context:
    train, valid, test = (load_split(s) for s in paths.SPLITS)
    sequences = train_sequences(train)
    pop = Pop().fit(sequences, NUM_ITEMS)
    tie_order = np.lexsort((np.arange(NUM_ITEMS), -pop.counts))
    ev = run.Evaluator(tie_order)
    params, _ = run.tune_itemcf(sequences, valid, ev)
    itemcf = ItemCF(params).fit(sequences, NUM_ITEMS)
    sasrec, _ = run.load_sasrec()
    models = (pop, itemcf, sasrec)
    sids = item_sids()
    return Context(
        test=test,
        tie_order=tie_order,
        counts=pop.counts,
        sids=sids,
        by_sid=sid_items(sids, tie_order),
        baselines={m.name: ev.per_sample(m.score(test), test) for m in models},
        descriptions={m.name: m.describe() for m in models},
    )


# ----------------------------------------------------------------------------- segments


def item_categories(level: int = CATEGORY_LEVEL) -> np.ndarray:
    """Level-`level` category of every item (index = item_id); 'unknown' where the path is short."""
    items = pd.read_parquet(paths.CATALOG_PARQUET, columns=["item_id", "categories"])
    items = items.sort_values("item_id")
    assert np.array_equal(items.item_id.to_numpy(), np.arange(NUM_ITEMS))
    return np.array(
        [c[level] if c is not None and len(c) > level else "unknown" for c in items.categories],
        dtype=object,
    )


def head_items(counts: np.ndarray, tie_order: np.ndarray, share: float = HEAD_SHARE) -> tuple:
    """(mask of head items, threshold): items with at least as many training interactions as
    the item at the `share` quantile of the popularity order."""
    k = int(np.ceil(share * len(counts)))
    threshold = counts[tie_order[k - 1]]
    return counts >= threshold, float(threshold)


def segment_masks(
    test: Split, counts: np.ndarray, tie_order: np.ndarray, categories: np.ndarray, same_day
) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    hist_len = np.array([len(h) for h in test.history])
    head, threshold = head_items(counts, tie_order)
    cross = np.array(
        [
            categories[t] not in set(categories[h].tolist())
            for t, h in zip(test.target.tolist(), test.history, strict=True)
        ]
    )
    masks = {
        "history": {"1-3 items": hist_len <= 3, "4-10 items": hist_len >= 4},
        "popularity": {"head": head[test.target], "tail": ~head[test.target]},
        "category": {"cross-category": cross, "same category": ~cross},
        "day": {"same day": same_day, "later day": ~same_day},
    }
    info = {
        "head_share_of_items": float(head.mean()),
        "head_min_train_interactions": threshold,
        "head_share_of_train_interactions": float(counts[head].sum() / counts.sum()),
        "category_level": CATEGORY_LEVEL,
        "categories": int(len(set(categories.tolist()))),
        "targets_without_category": int((categories[test.target] == "unknown").sum()),
        "targets_never_seen_in_training": int((counts[test.target] == 0).sum()),
    }
    return masks, info


# ----------------------------------------------------------------------------- scoring


def _with_ci(values: dict[str, np.ndarray], resampler: bootstrap.Resampler) -> dict:
    return {k: {"value": float(v.mean()), "ci": resampler.interval(v)} for k, v in values.items()}


def score_generation(gen: Generation, ctx: Context) -> dict:
    """Per-sample values of MiniOneRec and the baselines on the generation's rows."""
    test = ctx.test
    target = test.target[gen.rows]
    expected = [ctx.sids[t] for t in target.tolist()]
    if [normalize(o) for o in gen.output] != expected:
        raise SystemExit(f"{gen.name}: result targets do not match the test rows")
    ranks = generated_ranks(gen.predict, target, ctx.sids, ctx.by_sid)
    ks = metrics.KS
    values = {NAME: {rule: metrics.per_sample(r, ks) for rule, r in ranks.items()}}
    for name in BASELINES:
        values[name] = {
            rule: {k: v[gen.rows] for k, v in ctx.baselines[name][rule].items()}
            for rule in ("item", "sid")
        }
    calc_ks = [k for k in metrics.CALC_KS if k <= gen.beams]
    sid_calc = metrics.summarize(metrics.per_sample(ranks["sid"], calc_ks))
    return {"values": values, "ranks": ranks, "sid_calc_ks": sid_calc, "calc_ks": calc_ks}


def run_calc_py(run_dir: Path) -> dict:
    """MiniOneRec's calc.py, unmodified, on a run's result.json (HR / NDCG per K, CC)."""
    return calc_check.run_calc_py(run_dir / "result.json")


def calc_cross_check(scored: dict, calc: dict) -> dict:
    """calc.py's printed numbers against our SID-rule numbers, at every K it prints."""
    if len(calc["HR"]) != len(scored["calc_ks"]):
        raise SystemExit(
            f"calc.py printed {len(calc['HR'])} K values, expected {scored['calc_ks']}"
        )
    rows, ok = [], calc["invalid_predictions"] == 0
    for i, k in enumerate(scored["calc_ks"]):
        for metric in ("HR", "NDCG"):
            ours = scored["sid_calc_ks"][f"{metric}@{k}"]
            theirs = calc[metric][i]
            same = np.format_float_positional(
                ours, precision=8, trim="-"
            ) == np.format_float_positional(theirs, precision=8, trim="-")
            ok &= bool(same) and abs(ours - theirs) <= 5e-9
            rows.append({"k": k, "metric": metric, "ours": ours, "calc_py": theirs, "same": same})
    return {
        "ok": bool(ok),
        "invalid_predictions": calc["invalid_predictions"],
        "HR": calc["HR"],
        "NDCG": calc["NDCG"],
        "rows": rows,
    }


def comparison(values: dict, users: np.ndarray) -> dict:
    resampler = bootstrap.Resampler.build(users)
    models = {
        name: {rule: _with_ci(v, resampler) for rule, v in by_rule.items()}
        for name, by_rule in values.items()
    }
    pairs = {
        f"{NAME} - {ref}": {
            rule: _with_ci(
                {k: values[NAME][rule][k] - values[ref][rule][k] for k in values[ref][rule]},
                resampler,
            )
            for rule in ("item", "sid")
        }
        for ref in REFERENCES
    }
    return {"models": models, "pairs": pairs}


def segments(values: dict, users: np.ndarray, masks: dict) -> dict:
    out = {}
    for seg, groups in masks.items():
        out[seg] = {}
        for label, m in groups.items():
            if not m.any():  # possible on a small subset of users
                out[seg][label] = {"samples": 0, "share": 0.0, "models": None, "pairs": None}
                continue
            resampler = bootstrap.Resampler.build(users[m])
            out[seg][label] = {
                "samples": int(m.sum()),
                "share": float(m.mean()),
                "models": {
                    name: _with_ci({k: by_rule["item"][k][m] for k in SEGMENT_METRICS}, resampler)
                    for name, by_rule in values.items()
                },
                "pairs": {
                    f"{NAME} - {ref}": _with_ci(
                        {
                            k: values[NAME]["item"][k][m] - values[ref]["item"][k][m]
                            for k in SEGMENT_METRICS
                        },
                        resampler,
                    )
                    for ref in REFERENCES
                },
            }
    return out


def reproduction(gen: Generation, comp: dict, calc: dict) -> dict:
    """calc.py's numbers on our generations, and our SID-rule intervals, against the paper."""
    paper = run.PAPER["rows"][NAME]
    calc_ks = [k for k in metrics.CALC_KS if k <= gen.beams]
    own = {}
    for i, k in enumerate(calc_ks):
        if k in metrics.KS:
            own[f"HR@{k}"] = calc["HR"][i]
            own[f"NDCG@{k}"] = calc["NDCG"][i]
    sid = comp["models"][NAME]["sid"]
    inside = {m: sid[m]["ci"][0] <= paper[m] <= sid[m]["ci"][1] for m in REPRO_METRICS}
    return {
        "paper_source": run.PAPER["source"],
        "paper": paper,
        "calc_py": own,
        "difference": {m: own[m] - paper[m] for m in own},
        "paper_inside_interval": inside,
        "reproduced": all(inside.values()) and calc["invalid_predictions"] == 0,
    }


def agreement(gens: dict[str, Generation], scored: dict, main: str, test: Split) -> list[dict]:
    """Every other run against the main run, paired on the test samples both generated for
    (other beam widths or dtypes on a subset of the main run's users)."""
    base = gens[main]
    pos = {r: i for i, r in enumerate(base.rows.tolist())}
    out = []
    for name, gen in gens.items():
        if name == main:
            continue
        other = [i for i, r in enumerate(gen.rows.tolist()) if r in pos]
        if not other:
            continue
        mine = [pos[int(gen.rows[i])] for i in other]
        top_main = [[normalize(x) for x in base.predict[i][:10]] for i in mine]
        top_other = [[normalize(x) for x in gen.predict[i][:10]] for i in other]
        pairs = list(zip(top_main, top_other, strict=True))
        resampler = bootstrap.Resampler.build(test.user[gen.rows[other]])
        v_main = {k: scored[main]["values"][NAME]["item"][k][mine] for k in SEGMENT_METRICS}
        v_other = {k: scored[name]["values"][NAME]["item"][k][other] for k in SEGMENT_METRICS}
        out.append(
            {
                "run": name,
                "settings": gen.meta["settings"],
                "samples": len(other),
                "identical_top10": float(np.mean([a == b for a, b in pairs])),
                "top10_overlap": float(np.mean([len(set(a) & set(b)) / 10 for a, b in pairs])),
                "main": _with_ci(v_main, resampler),
                "run_values": _with_ci(v_other, resampler),
                "difference": _with_ci({k: v_other[k] - v_main[k] for k in v_main}, resampler),
            }
        )
    return out


# ----------------------------------------------------------------------------- cost


def _latency_stats(seconds: list[float], skip: int) -> dict:
    s = np.asarray(seconds[skip:], dtype=np.float64)
    return {
        "calls": len(s),
        "mean": float(s.mean()),
        "p50": float(np.quantile(s, 0.5)),
        "p90": float(np.quantile(s, 0.9)),
        "p99": float(np.quantile(s, 0.99)),
        "max": float(s.max()),
    }


WARMUP_CALLS = {1: 2, 8: 1}  # first calls of a process (Metal kernel compilation) left out


def cost_summary(
    cost_dir: Path = COST, sasrec_json: Path = SASREC_COST, include_test_models: bool = False
) -> dict | None:
    manifest_path = cost_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    if not include_test_models and not real_checkpoint(manifest):
        print(f"skipping {cost_dir}: not measured with {CHECKPOINT}")
        return None
    configs = {}
    for name, cfg in manifest.get("configs", {}).items():
        timing = json.loads((cost_dir / name / "timing.json").read_text())
        batches = timing["batches"]
        seconds = [b["seconds"] for b in batches]
        skip = min(WARMUP_CALLS.get(cfg["batch_size"], 1), len(seconds) - 1)
        sizes = [b["size"] for b in batches]
        kept = slice(skip, None)
        configs[name] = {
            "num_beams": cfg["num_beams"],
            "batch_size": cfg["batch_size"],
            "max_new_tokens": cfg["max_new_tokens"],
            "samples": cfg["samples"],
            "warmup_calls": skip,
            "per_call": _latency_stats(seconds, skip),
            "samples_per_second": float(sum(sizes[kept]) / sum(seconds[kept])),
            "setup_seconds": cfg["setup_seconds"],
            "device": cfg.get("device", "unknown"),
            "peak_footprint": cfg["memory"]["lifetime_max_footprint"],
            "mps_driver_allocated_max": max(b.get("mps_driver_allocated", 0) for b in batches),
            "mps_allocated_max": max(b.get("mps_allocated", 0) for b in batches),
            "loadavg_start": cfg["loadavg_start"][0],
            "invalid_predictions": cfg["invalid_predictions"],
        }
    sasrec = None
    if sasrec_json.exists():
        raw = json.loads(sasrec_json.read_text())
        sasrec = {k: v for k, v in raw.items() if k != "runs"}
        sasrec["runs"] = [
            {
                "batch_size": r["batch_size"],
                "threads": r["threads"],
                "per_call": _latency_stats(r["seconds"], 0),
                "samples_per_second": float(
                    r["batch_size"] * len(r["seconds"]) / sum(r["seconds"])
                ),
            }
            for r in raw["runs"]
        ]
    return {
        "users": manifest["users"],
        "dtype": manifest["dtype"],
        "versions": manifest["versions"],
        "machine": manifest["machine"],
        "configs": configs,
        "sasrec": sasrec,
    }


# ----------------------------------------------------------------------------- main


def run_summary(gen: Generation, test: Split) -> dict:
    meta = gen.meta
    batches = gen.timing["batches"]
    return {
        "name": gen.name,
        "samples": len(gen.rows),
        "users": int(np.unique(test.user[gen.rows]).size),
        "selection": meta["selection"],
        "settings": meta["settings"],
        "device": meta.get("device", "unknown"),
        "versions": meta["versions"],
        "machine": meta["machine"],
        "started": meta["started"],
        "wall_seconds": meta["wall_seconds"],
        "setup_seconds": meta["setup_seconds"],
        "generate_seconds": meta["generate_seconds"],
        "seconds_per_sample": meta["generate_seconds"] / len(gen.rows),
        "batches": len(batches),
        # the highest 1-minute load average at the start of any shard
        "loadavg_start": meta.get("loadavg_max_start", meta["loadavg_start"][0]),
        "peak_footprint": (meta.get("memory") or {}).get("lifetime_max_footprint"),
        "mps_driver_allocated_max": max(b.get("mps_driver_allocated", 0) for b in batches),
    }


def build(
    runs: list[Path],
    main_run: str | None,
    ctx: Context,
    cost: dict | None,
    categories: np.ndarray,
    same_day: np.ndarray,
    calc_fn=run_calc_py,
) -> dict:
    """Everything the report shows. `categories` (per item) and `same_day` (per test sample)
    come from data/processed; `calc_fn(run_dir)` runs calc.py on a run's result.json."""
    if not runs:
        raise SystemExit("no finished MiniOneRec runs: `make minionerec-gen` first")
    gens = {p.name: load_generation(p) for p in runs}
    names = list(gens)
    main = main_run if main_run in gens else max(names, key=lambda n: len(gens[n].rows))
    test = ctx.test
    masks_full, seg_info = segment_masks(test, ctx.counts, ctx.tie_order, categories, same_day)

    out_runs, scored_runs = {}, {}
    for path in runs:
        name, gen = path.name, gens[path.name]
        scored = scored_runs[name] = score_generation(gen, ctx)
        calc = calc_fn(path)
        users = test.user[gen.rows]
        comp = comparison(scored["values"], users)
        entry = {
            "summary": run_summary(gen, test),
            "calc_py": calc_cross_check(scored, calc),
            "comparison": comp,
            "reproduction": reproduction(gen, comp, calc),
        }
        if name == main:
            masks = {s: {g: m[gen.rows] for g, m in gs.items()} for s, gs in masks_full.items()}
            entry["segments"] = segments(scored["values"], users, masks)
        out_runs[name] = entry

    status, reasons = "final", []
    main_summary = out_runs[main]["summary"]
    if main_summary["samples"] < len(test):
        status = "preliminary"
        reasons.append(
            f"the main run covers {main_summary['samples']:,} of {len(test):,} test samples"
        )
    return {
        "status": status,
        "status_reasons": reasons,
        "main_run": main,
        "test_samples": len(test),
        "test_users": int(np.unique(test.user).size),
        "baselines": ctx.descriptions,
        "segment_info": seg_info,
        "bootstrap": {"B": bootstrap.B, "seed": bootstrap.SEED, "alpha": bootstrap.ALPHA},
        "runs": out_runs,
        "agreement": agreement(gens, scored_runs, main, test),
        "cost": cost,
    }


def main(argv: list[str] | None = None) -> None:
    from ml.eval import minionerec_report

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--main-run", default="test", help="run used for the main tables")
    ap.add_argument(
        "--runs", type=Path, default=RUNS, help="generation runs (default: %(default)s)"
    )
    ap.add_argument("--cost", type=Path, default=COST, help="cost runs (default: %(default)s)")
    ap.add_argument("--sasrec-cost", type=Path, default=SASREC_COST)
    ap.add_argument("--report", type=Path, default=REPORT)
    ap.add_argument("--json", type=Path, default=OUT_JSON)
    ap.add_argument(
        "--include-test-models", action="store_true", help="also report smoke runs (tests only)"
    )
    args = ap.parse_args(argv)

    runs = finished_runs(args.runs, args.include_test_models)
    if not runs:
        raise SystemExit("no finished MiniOneRec runs: `make minionerec-gen` first")
    ctx = build_context()
    out = build(
        runs,
        args.main_run,
        ctx,
        cost_summary(args.cost, args.sasrec_cost, args.include_test_models),
        item_categories(),
        run.same_day_as_last(ctx.test),
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, indent=2) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(minionerec_report.render(out))

    for name, r in out["runs"].items():
        c = r["comparison"]["models"][NAME]["item"]["HR@10"]
        repro = r["reproduction"]
        print(
            f"{name}: {r['summary']['samples']:,} samples; MiniOneRec HR@10 (item rule) "
            f"{c['value']:.4f} [{c['ci'][0]:.4f}, {c['ci'][1]:.4f}]; calc.py HR@10 "
            f"{repro['calc_py'].get('HR@10', float('nan')):.4f} (paper "
            f"{repro['paper']['HR@10']:.4f}); calc.py cross-check "
            f"{'match' if r['calc_py']['ok'] else 'MISMATCH'}"
        )
    print(f"wrote {args.report} ({out['status']})")
    if not all(r["calc_py"]["ok"] for r in out["runs"].values()):
        raise SystemExit("calc.py cross-check FAILED")


if __name__ == "__main__":
    main()
