"""Run one agent configuration over the cases and score every turn.

An agent is a parser plus the shared pipeline: 'baseline' parses with rules, 'oracle' is handed
the gold constraints (the ceiling of the pipeline when understanding is perfect), 'llm:<model>'
parses, judges compatibility and explains with a language model. The search variants of the
oracle isolate retrieval: BM25 only, vectors only, and the query without bge's instruction.

Gold constraints, satisfiable sets and violations are computed on one snapshot of the items
table taken when the run starts; its fingerprint goes into the run file and the report.
"""

from __future__ import annotations

import copy
import hashlib
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from agent import taxonomy
from agent.agent import AgentConfig, ShoppingAgent
from agent.catalog import Catalog, Item
from agent.eval import metrics
from agent.eval.cases import Case
from agent.llm import LLM
from agent.parsing import FixedParser, LLMParser, Parser, RuleParser


@dataclass(frozen=True)
class AgentSpec:
    name: str
    description: str
    parser: str  # "rules" | "oracle" | "llm"
    search_mode: str = "hybrid"
    instruction: bool = True  # bge query instruction prefix
    model: str | None = None


OFFLINE_AGENTS = [
    AgentSpec("baseline", "rules parser (no LLM) + hybrid search + code checks", "rules"),
    AgentSpec("oracle", "gold constraints + hybrid search + code checks", "oracle"),
    AgentSpec("oracle-bm25", "gold constraints, BM25 only", "oracle", search_mode="text"),
    AgentSpec("oracle-vector", "gold constraints, pgvector only", "oracle", search_mode="vector"),
    AgentSpec("oracle-noprefix", "gold constraints, hybrid, query without bge instruction", "oracle", instruction=False),
]  # fmt: skip


def llm_spec(model: str, repeat: int) -> AgentSpec:
    return AgentSpec(
        f"llm-{model}-r{repeat}", f"{model} parser, judge and reasons", "llm", model=model
    )


def snapshot(items: list[Item]) -> dict[str, Any]:
    h = hashlib.sha256()
    for it in items:
        h.update(f"{it.item_id},{it.stock},{it.price_cents}\n".encode())
    return {
        "items": len(items),
        "in_stock": sum(it.stock > 0 for it in items),
        "priced": sum(it.price_cents is not None for it in items),
        "fingerprint": h.hexdigest()[:12],
    }


class CostCapExceeded(RuntimeError):
    pass


def run_agent(
    spec: AgentSpec,
    cases: list[Case],
    catalog: Catalog,
    llm: LLM | None = None,
    max_cost_usd: float | None = None,
    progress: Callable[[int, int, float], None] | None = None,
) -> dict[str, Any]:
    if spec.parser == "llm" and (llm is None or max_cost_usd is None):
        raise ValueError("an LLM agent needs an LLM client and a cost cap")
    items = catalog.all_items()
    rows = {it.item_id: it for it in items}
    by_category = {
        k: [it for it in items if taxonomy.in_category(it.categories, k)] for k in taxonomy.KEYS
    }
    rule_parser = RuleParser.from_catalog(it.brand for it in items)
    out_cases = []
    spent = 0.0
    partial = False
    for n, case in enumerate(cases, start=1):
        parser: Parser
        if spec.parser == "rules":
            parser = rule_parser
        elif spec.parser == "oracle":
            parser = FixedParser(case.oracle_request())
        else:
            assert llm is not None
            parser = LLMParser(llm)
        agent = ShoppingAgent(
            catalog,
            parser,
            llm if spec.parser == "llm" else None,
            AgentConfig(search_mode=spec.search_mode),
            purchases=case.history,
        )
        result = agent.run(case.message, case.user_id)
        record = result.record()
        history = [
            (p, rows[p.item_id]) for p in case.history(case.user_id, 20) if p.item_id in rows
        ]
        geff = metrics.gold_effective(case.gold, history)
        pool = by_category[case.gold.category] if case.gold.category else items
        sat = metrics.satisfiable(pool, geff)
        score = metrics.score_case(
            case.id, case.template, case.user_id, case.gold, geff, sat, rows, record
        )
        out_cases.append(
            {
                "case_id": case.id,
                "template": case.template,
                "message": case.message,
                "satisfiable": sorted(sat),
                "record": record,
                "score": score.as_dict(),
            }
        )
        spent += score.cost_usd
        if progress:
            progress(n, len(cases), spent)
        if max_cost_usd is not None and spent > max_cost_usd:
            partial = True
            break
    return {
        "agent": spec.name,
        "spec": asdict(spec),
        "encoder": catalog.encoder.name,
        "snapshot": snapshot(items),
        "partial": partial,
        "cost_usd": spent,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cases": out_cases,
    }


def with_encoder(catalog: Catalog, encoder) -> Catalog:
    """The same catalog (connection, BM25 index) with another query encoder."""
    other = copy.copy(catalog)
    other.encoder = encoder
    return other
