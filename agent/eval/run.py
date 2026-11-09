"""Run the evaluation: offline agents by default, language-model agents only when asked.

  python -m agent.eval.run                          # baseline + oracle variants, no API calls
  python -m agent.eval.run --skip-offline --llm claude-opus-5-5 --llm claude-haiku-5-5 \
      --repeats 3 --max-cost-usd claude-opus-5-5=12,claude-haiku-5-5=1 --total-cap-usd 20 \
      --judge claude-sonnet-5-5
  python -m agent.eval.run --report-only

Each agent's run is written to <runs dir>/<agent>.json (artifacts/eval/agent by default), then
docs/results/agent.md is regenerated from all run files present (not with --limit).

Paid steps need ANTHROPIC_API_KEY in the environment and caps: --max-cost-usd per run (one
amount, or model=amount pairs), --total-cap-usd over every paid call of this invocation. A run
stops before a case that could take it past its cap (twice the dearest case so far). --judge
grades the shown items of every run in the runs dir with that model (agent.eval.judge), within
what is left of the total cap.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

from agent import paths
from agent.catalog import PgCatalog
from agent.eval import cases as case_io
from agent.eval import harness, report


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--llm", action="append", default=[], help="model id to evaluate (repeatable)")
    ap.add_argument("--repeats", type=int, default=1, help="runs per model (the plan is 3)")
    ap.add_argument(
        "--max-cost-usd",
        help="cap per model run: one amount, or model=amount pairs separated by commas",
    )
    ap.add_argument("--total-cap-usd", type=float, help="cap over all paid calls of this run")
    ap.add_argument("--judge", help="grade the runs' shown items with this model (paid)")
    ap.add_argument("--limit", type=int, help="only the first N cases (smoke tests)")
    ap.add_argument("--runs-dir", default=str(paths.RUNS), help="where run files go")
    ap.add_argument("--skip-offline", action="store_true", help="do not re-run the offline agents")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if args.report_only:
        report.main()
        return
    caps: dict[str, float] = {}
    if args.llm or args.judge:
        if args.total_cap_usd is None:
            ap.error("paid steps (--llm, --judge) need --total-cap-usd")
        if args.llm:
            try:
                caps = parse_caps(args.max_cost_usd, args.llm)
            except ValueError as e:
                ap.error(str(e))
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY is not set (the user puts it in .env); nothing ran")
    runs_dir = Path(args.runs_dir)

    from agent.encoder import BgeQueryEncoder

    cases = case_io.load()[: args.limit]
    t0 = time.perf_counter()
    encoder = BgeQueryEncoder()
    plain = encoder.without_instruction()
    catalog = PgCatalog.connect(paths.database_url(), encoder)
    print(f"encoder + catalog ready in {time.perf_counter() - t0:.1f}s; {len(cases)} cases")
    runs_dir.mkdir(parents=True, exist_ok=True)

    def save(run: dict) -> None:
        (runs_dir / f"{run['agent']}.json").write_text(
            json.dumps(run, ensure_ascii=False, indent=1)
        )
        s = report.summarize(run["cases"])
        print(
            f"{run['agent']:28s} success {report.pct(s['success'])}  hard {report.pct(s['hard'])}  "
            f"all {report.pct(s['all'])}  parse {report.pct(s['parse'])}  tools {report.pct(s['tools'])}  "
            f"P50 {report._ms(s['p50'])}  cost {report._usd(s['cost'])}"
        )

    if not args.skip_offline:
        from agent.eval.chat_latency import others

        project = os.environ.get("COMPOSE_PROJECT_NAME", "recloop")
        machine = {"load_start": round(os.getloadavg()[0], 2), "compose_project": project}
        seen = others(project)
        for spec in harness.OFFLINE_AGENTS:
            cat = harness.with_encoder(catalog, plain) if not spec.instruction else catalog
            save(harness.run_agent(spec, cases, cat))
            seen |= others(project)
        machine |= {"load_end": round(os.getloadavg()[0], 2), "other_containers": sorted(seen)}
        (runs_dir / "machine.json").write_text(json.dumps(machine, indent=1))

    total = args.total_cap_usd if args.total_cap_usd is not None else math.inf
    spent = 0.0
    for model in args.llm:
        from agent.llm import ClaudeLLM

        llm = ClaudeLLM(model)
        for r in range(1, args.repeats + 1):
            cap = min(caps[model], total - spent)
            if cap <= 0:
                print(f"total cap ${total:g} reached: {model} run {r} not started")
                break
            spec = harness.llm_spec(model, r)

            def progress(n: int, of: int, cost: float, name: str = spec.name) -> None:
                print(f"\r{name}: {n}/{of} cases, ${cost:.3f}", end="", file=sys.stderr)

            run = harness.run_agent(spec, cases, catalog, llm, cap, progress)
            print(file=sys.stderr)
            spent += run["cost_usd"]
            save(run)
            if run["partial"]:
                print(
                    f"{spec.name}: stopped at its cap (${cap:.2f}) after {len(run['cases'])} cases"
                )
                break
    if args.llm:
        print(f"language-model runs: ${spent:.2f} of the ${total:g} total cap")

    if args.judge:
        from agent.eval import judge
        from agent.llm import ClaudeLLM

        runs = report.load_runs(runs_dir)  # the run files, not machine.json
        grades = judge.grade_runs(
            judge.ClaudeJudge(ClaudeLLM(args.judge)),
            runs,
            catalog,
            runs_dir / "judge" / judge.GRADES_FILE,
            total - spent,
        )
        print(
            f"judge {args.judge}: {grades['calls']} calls, ${grades['spent_now']:.2f} now; "
            f"{'stopped at the cap' if grades['partial'] else 'every shown item graded'}"
        )

    if args.limit:
        print("--limit given: report not regenerated")
        return
    report.main()


def parse_caps(spec: str | None, models: list[str]) -> dict[str, float]:
    """'12' -> every model 12; 'a=12,b=6' -> per model (every model in `models` needs one)."""
    if not spec:
        raise ValueError("--llm needs --max-cost-usd")
    if "=" not in spec:
        return dict.fromkeys(models, float(spec))
    caps = {}
    for part in spec.split(","):
        name, _, amount = part.partition("=")
        caps[name.strip()] = float(amount)
    missing = [m for m in models if m not in caps]
    if missing:
        raise ValueError(f"--max-cost-usd has no cap for {', '.join(missing)}")
    return caps


if __name__ == "__main__":
    main()
