"""ANN benchmark + index-update experiments -> docs/results/ann.md (and artifacts/ann/ann.json).

  1. Vectors: MiniOneRec's text embeddings of the 3,459 items -> PCA to 64 -> a Gaussian
     mixture chosen by cross-validated log-likelihood -> synthetic catalogs of 3,459, 100K and
     1M items (ml/ann/vectors.py). User vectors are recency-weighted history averages.
  2. Benchmark at each size: Flat, IVF-Flat (nlist x nprobe), HNSW (M x efConstruction x
     efSearch) and, at 1M, IVF-PQ (ml/ann/bench.py).
  3. Top-100 vs top-200 (the retrieval stage asks for 200) on the same 1M indexes.
  4. Index updates: incremental adds with distribution drift vs. full rebuilds, HNSW deletions
     (ml/ann/incremental.py).

Latency is measured one query per call on one thread (median of 3 runs of P50 / P99); builds
use --build-threads threads. Run on an otherwise idle machine for final numbers: the report is
marked preliminary when --preliminary is given or the load average is high at the start.

Usage: python -m ml.ann.run [--preliminary] [--quick] [--source-emb PATH] [--render-only]
                            [--only ksweep]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import faiss
import numpy as np
import scipy

from ml import paths
from ml.ann import bench, incremental, indexes, report, vectors

OUT_DIR = paths.ARTIFACTS / "ann"
JSON_PATH = OUT_DIR / "ann.json"
REPORT = paths.RESULTS / "ann.md"
DEFAULT_EMB = paths.PUBLIC / "index" / f"{paths.DATASET}.emb-qwen-td.npy"
SEED = 20261106
BUSY_LOAD = 2.0  # 1-minute load average above which a run is marked preliminary

MIXTURE_KS = (16, 32, 64, 128)
MIXTURE_REGS = (5e-4, 1e-3, 2e-3, 4e-3)


@dataclass(frozen=True)
class Scale:
    name: str
    n: int
    source: str  # "real" (the 3,459 PCA vectors) or "mixture"
    grid: bench.Grid


FULL_SCALES = (
    Scale("real", 3_459, "real", bench.Grid(ivf_nlists=(16, 64))),
    Scale("synthetic", 3_459, "mixture", bench.Grid(ivf_nlists=(16, 64))),
    Scale("100K", 100_000, "mixture", bench.Grid(ivf_nlists=(256, 1024, 2048))),
    Scale(
        "1M",
        1_000_000,
        "mixture",
        bench.Grid(ivf_nlists=(1024, 4096, 16384), ivfpq_nlist=4096),
    ),
)
QUICK_GRID = dict(
    nprobes=(1, 4, 16), hnsw=((16, 40),), ef_searches=(64, 256), ivfpq_m=(16,), ivfpq_nprobes=(16,)
)
QUICK_SCALES = (
    Scale("real", 3_459, "real", bench.Grid(ivf_nlists=(16,), **QUICK_GRID)),
    Scale("20K", 20_000, "mixture", bench.Grid(ivf_nlists=(128,), ivfpq_nlist=128, **QUICK_GRID)),
)
KSWEEP = bench.KSweep()
QUICK_KSWEEP = bench.KSweep(nlist=128, nprobes=(8, 16), hnsw=(16, 40), ef_searches=(128, 256))
QUICK_PLAN = incremental.Plan(
    initial=20_000,
    batches=2,
    batch_size=5_000,
    nlist=128,
    nprobe=8,
    hnsw_m=16,
    hnsw_efc=40,
    ef_search=128,
    queries=200,
    query_pool=20_000,
    delete_fractions=(0.3,),
)


def _sysctl(name: str) -> str | None:
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _power_source() -> str | None:
    try:
        out = subprocess.run(["pmset", "-g", "ps"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    first = out.stdout.splitlines()[0] if out.returncode == 0 and out.stdout else ""
    return first.split("'")[1] if first.count("'") >= 2 else None


def _hardware_overview() -> dict:
    """Chip, core counts and memory from system_profiler (works where sysctl is blocked).
    Only these fields are kept; the overview also has serial numbers and UUIDs."""
    try:
        out = subprocess.run(
            ["system_profiler", "SPHardwareDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        hw = json.loads(out.stdout)["SPHardwareDataType"][0]
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, IndexError):
        return {}
    procs = hw.get("number_processors", "").replace("proc", "").strip().split(":")
    return {
        "cpu": hw.get("chip_type"),
        "model": hw.get("machine_name"),
        "performance_cores": procs[1] if len(procs) > 2 else None,
        "efficiency_cores": procs[2] if len(procs) > 2 else None,
        "memory": hw.get("physical_memory"),
    }


def machine(label: str | None) -> dict:
    mem = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    info = {
        "label": label,
        "cpu": _sysctl("machdep.cpu.brand_string"),
        "model": None,
        "performance_cores": _sysctl("hw.perflevel0.physicalcpu"),
        "efficiency_cores": _sysctl("hw.perflevel1.physicalcpu"),
        "logical_cpus": os.cpu_count(),
        "memory_gib": round(mem / 2**30, 1),
        "os": platform.platform(),
        "power": _power_source(),
    }
    if sys.platform == "darwin":
        hw = _hardware_overview()
        for key in ("cpu", "model", "performance_cores", "efficiency_cores"):
            info[key] = info[key] or hw.get(key)
    return info


def source_info(path: Path) -> dict:
    try:
        shown = str(path.resolve().relative_to(paths.ROOT))
    except ValueError:
        shown = "third_party/MiniOneRec/" + str(path).split("MiniOneRec/", 1)[-1]
    return {"path": shown, "sha256": vectors.sha256_of(path)}


def neighbour_profile(x: np.ndarray, ranks=(1, 10, 100)) -> dict:
    """Percentiles (10/50/90) of each item's cosine to its r-th nearest other item."""
    x = np.asarray(x, dtype=np.float64)
    sims = x @ x.T
    np.fill_diagonal(sims, -np.inf)
    top = -np.sort(-sims, axis=1)[:, : max(ranks)]
    return {str(r): [float(v) for v in np.percentile(top[:, r - 1], (10, 50, 90))] for r in ranks}


def prepare_vectors(args, log) -> tuple[vectors.Mixture, dict, np.ndarray | None]:
    """The fitted mixture, a description for the report, and the real 3,459 vectors."""
    if args.source_emb.exists():
        real, p = vectors.real_vectors(args.source_emb)
        shape = list(np.load(args.source_emb, mmap_mode="r").shape)
        log(f"source embeddings {args.source_emb.name} {shape} -> {real.shape}; fitting mixture")
        ks, regs = (16, 64), (1e-3,)
        if not args.quick:
            ks, regs = MIXTURE_KS, MIXTURE_REGS
        fit, trials = vectors.select_mixture(real, ks, regs, SEED)
        mix = fit.mixture
        best = max(trials, key=lambda t: t["cv_ll"])
        info = {
            "kind": "mixture fitted to MiniOneRec text embeddings",
            "source": {**source_info(args.source_emb), "shape": shape},
            "pca_explained": p.explained,
            "participation_ratio": vectors.participation_ratio(p.spectrum),
            "trials": trials,
            "k": best["k"],
            "reg": best["reg"],
            "em_iterations": fit.iterations,
            "weights": [float(w) for w in np.sort(mix.weights)[::-1]],
        }
        synth, _ = vectors.make_catalog(mix, len(real), SEED + 3459)
        info["profile"] = {"real": neighbour_profile(real), "synthetic": neighbour_profile(synth)}
        return mix, info, real.astype(np.float32)
    log(f"{args.source_emb} not found: using a synthetic clustered Gaussian")
    mix = vectors.synthetic_mixture(seed=SEED)
    return (
        mix,
        {"kind": "synthetic clustered Gaussian (embedding file not found)", "k": mix.k},
        None,
    )


def scale_data(scale: Scale, mix, real, args) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(items, their components, queries) of a catalog; the same seeds give the same arrays."""
    if scale.source == "real":
        data = vectors.normalize(real)
        labels = mix.predict(data)
    else:
        data, labels = vectors.make_catalog(mix, scale.n, SEED + scale.n)
    queries, _ = vectors.make_queries(data, labels, args.queries, SEED + scale.n + 1)
    return data, labels, queries


def run_scale(scale: Scale, mix, real, args, run: bench.Run, log) -> dict:
    if scale.source == "real" and real is None:
        return {"name": scale.name, "skipped": "no source embeddings"}
    data, labels, queries = scale_data(scale, mix, real, args)
    log(f"[{scale.name}] {len(data):,} items, {len(queries):,} queries")
    t0 = time.perf_counter()
    b = bench.Bench(data, queries, run, labels)
    rows = b.all(scale.grid)
    return {
        "name": scale.name,
        "n": len(data),
        "source": scale.source,
        "queries": len(queries),
        "digest": vectors.array_digest(data),
        "query_digest": vectors.array_digest(queries),
        "grid": asdict(scale.grid),
        "span_median": float(np.median(b.span)),
        "rows": rows,
        "seconds": time.perf_counter() - t0,
    }


def run_ksweep(scale: Scale, sweep: bench.KSweep, mix, real, args, run: bench.Run, log) -> dict:
    data, _, queries = scale_data(scale, mix, real, args)
    log(f"[k sweep] {scale.name}: k = {', '.join(map(str, sweep.ks))}")
    load = os.getloadavg()
    t0 = time.perf_counter()
    rows = bench.k_sweep(data, queries, run, sweep)
    return {
        "scale": scale.name,
        "n": len(data),
        "queries": len(queries),
        "digest": vectors.array_digest(data),
        "sweep": asdict(sweep),
        "load_average_start": list(load),
        "rows": rows,
        "seconds": time.perf_counter() - t0,
    }


def merge_ksweep(args, ksweep: dict, log) -> None:
    """Add a separately run k sweep to an existing ann.json and re-render the report."""
    out = json.loads(args.json.read_text())
    same = [x for x in out["scales"] if x.get("name") == ksweep["scale"]]
    if not same or same[0]["digest"] != ksweep["digest"]:
        raise SystemExit(f"{args.json} has no {ksweep['scale']} catalog with the same vectors")
    out["ksweep"] = ksweep
    setup = out["setup"]
    load = ksweep["load_average_start"][0]
    reasons = [r for r in setup["status_reasons"] if not r.startswith("k sweep run separately")]
    if args.preliminary or load > BUSY_LOAD:
        setup["status"] = "preliminary"
        reasons.append(f"k sweep run separately, load average {load:.1f} at its start")
    setup["status_reasons"] = reasons
    args.json.write_text(json.dumps(out, indent=1) + "\n")
    args.report.write_text(report.render(out))
    log(f"merged the k sweep into {args.json} and wrote {args.report}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--source-emb", type=Path, default=DEFAULT_EMB)
    ap.add_argument("--preliminary", action="store_true", help="mark the report preliminary")
    ap.add_argument("--quick", action="store_true", help="small sizes, for a smoke test")
    ap.add_argument("--queries", type=int, default=2000)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--build-threads", type=int, default=8)
    ap.add_argument("--machine", default=None, help="label for the report, e.g. 'M4 Pro'")
    ap.add_argument("--skip-incremental", action="store_true")
    ap.add_argument(
        "--only",
        choices=("ksweep",),
        help="run only this part and merge it into the existing --json (then re-render)",
    )
    ap.add_argument("--render-only", action="store_true", help="re-render ann.md from ann.json")
    ap.add_argument("--json", type=Path, default=JSON_PATH)
    ap.add_argument("--report", type=Path, default=REPORT)
    args = ap.parse_args()

    if args.quick and args.json == JSON_PATH:
        args.json = OUT_DIR / "ann-quick.json"
    if args.quick and args.report == REPORT:
        args.report = OUT_DIR / "ann-quick.md"
    if args.render_only:
        args.report.write_text(report.render(json.loads(args.json.read_text())))
        print(f"wrote {args.report}")
        return

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    load_start = os.getloadavg()
    t_start = time.perf_counter()
    if args.quick:
        args.queries = min(args.queries, 300)
    qos = indexes.prefer_performance_cores()
    run = bench.Run(runs=args.runs, build_threads=args.build_threads, seed=SEED)
    indexes.set_threads(run.search_threads)

    mix, data_info, real = prepare_vectors(args, log)
    scales = QUICK_SCALES if args.quick else FULL_SCALES
    sweep = QUICK_KSWEEP if args.quick else KSWEEP
    if args.only == "ksweep":
        merge_ksweep(args, run_ksweep(scales[-1], sweep, mix, real, args, run, log), log)
        return
    results = [run_scale(s, mix, real, args, run, log) for s in scales]
    ksweep = run_ksweep(scales[-1], sweep, mix, real, args, run, log)

    inc = None
    if not args.skip_incremental:
        plan = QUICK_PLAN if args.quick else incremental.Plan()
        log("incremental updates")
        inc = incremental.run(mix, plan, replace(run, runs=args.runs), log)

    load_end = os.getloadavg()
    reasons = []
    if args.preliminary:
        reasons.append("run with --preliminary")
    if load_start[0] > BUSY_LOAD:
        reasons.append(f"load average {load_start[0]:.1f} at the start (> {BUSY_LOAD})")
    if args.quick:
        reasons.append("--quick (smoke-test sizes)")
    out = {
        "setup": {
            "status": "preliminary" if reasons else "final",
            "status_reasons": reasons,
            "date": time.strftime("%Y-%m-%d"),
            "machine": machine(args.machine),
            "load_average_start": list(load_start),
            "load_average_end": list(load_end),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "faiss": faiss.__version__,
            "faiss_options": faiss.get_compile_options().strip(),
            "build_threads": run.build_threads,
            "search_threads": run.search_threads,
            "qos_user_interactive": qos,
            "runs": run.runs,
            "warmup": run.warmup,
            "k": run.k,
            "queries": args.queries,
            "seed": SEED,
            "query_params": asdict(vectors.QueryParams()),
            "call_overhead_us": indexes.call_overhead_us(vectors.DIM, run.k),
            "seconds": time.perf_counter() - t_start,
        },
        "data": data_info,
        "scales": results,
        "ksweep": ksweep,
        "incremental": inc,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, indent=1) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report.render(out))
    log(f"wrote {args.report} and {args.json} in {out['setup']['seconds'] / 60:.1f} min")


if __name__ == "__main__":
    sys.exit(main())
