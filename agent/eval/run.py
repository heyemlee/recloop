"""Run the evaluation: offline agents by default, language-model agents only when asked.

  python -m agent.eval.run                          # baseline + oracle variants, no API calls
  python -m agent.eval.run --llm claude-opus-5-5 --repeats 3 --max-cost-usd 20
  python -m agent.eval.run --report-only

Each agent's run is written to artifacts/eval/agent/<agent>.json, then docs/results/agent.md is
regenerated from all run files present. A language-model run needs Anthropic credentials in the
environment and stops once its cost passes --max-cost-usd (required).
"""

from __future__ import annotations

import argparse
import json
import sys
import time

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
        "--max-cost-usd", type=float, help="stop a model run once it has cost this much"
    )
    ap.add_argument("--limit", type=int, help="only the first N cases (smoke tests)")
    ap.add_argument("--skip-offline", action="store_true", help="do not re-run the offline agents")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if args.report_only:
        report.main()
        return
    if args.llm and args.max_cost_usd is None:
        ap.error("--llm needs --max-cost-usd")

    from agent.encoder import BgeQueryEncoder

    cases = case_io.load()[: args.limit]
    t0 = time.perf_counter()
    encoder = BgeQueryEncoder()
    plain = encoder.without_instruction()
    catalog = PgCatalog.connect(paths.database_url(), encoder)
    print(f"encoder + catalog ready in {time.perf_counter() - t0:.1f}s; {len(cases)} cases")
    paths.RUNS.mkdir(parents=True, exist_ok=True)

    def save(run: dict) -> None:
        (paths.RUNS / f"{run['agent']}.json").write_text(
            json.dumps(run, ensure_ascii=False, indent=1)
        )
        s = report.summarize(run["cases"])
        print(
            f"{run['agent']:28s} success {report.pct(s['success'])}  hard {report.pct(s['hard'])}  "
            f"all {report.pct(s['all'])}  parse {report.pct(s['parse'])}  tools {report.pct(s['tools'])}  "
            f"P50 {report._ms(s['p50'])}  cost {report._usd(s['cost'])}"
        )

    if not args.skip_offline:
        for spec in harness.OFFLINE_AGENTS:
            cat = harness.with_encoder(catalog, plain) if not spec.instruction else catalog
            save(harness.run_agent(spec, cases, cat))

    for model in args.llm:
        from agent.llm import ClaudeLLM

        llm = ClaudeLLM(model)
        for r in range(1, args.repeats + 1):
            spec = harness.llm_spec(model, r)

            def progress(n: int, total: int, spent: float, name: str = spec.name) -> None:
                print(f"\r{name}: {n}/{total} cases, ${spent:.3f}", end="", file=sys.stderr)

            run = harness.run_agent(spec, cases, catalog, llm, args.max_cost_usd, progress)
            print(file=sys.stderr)
            save(run)
            if run["partial"]:
                print(f"{spec.name}: stopped at the cost cap (${args.max_cost_usd})")
                break

    if args.limit:
        print("--limit given: report not regenerated")
        return
    report.main()


if __name__ == "__main__":
    main()
