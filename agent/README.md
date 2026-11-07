# agent

The shopping assistant (docs/design.md §10). A language model turns the shopper's message into
constraints (validated, one retry), may judge device compatibility from quoted listing text,
and writes one-line reasons; the code searches the catalog (BM25 + pgvector, RRF, SQL filters
on price, stock and category), checks every constraint on the item rows, and renders the reply
from those rows.

`eval/` holds the 150-case evaluation set, the no-LLM baseline and the harness that writes
`docs/results/agent.md` (`make agent-eval`). Language-model runs are opt-in and paid
(`make agent-eval-llm`, with a spending cap). Run everything from the repo root; see
`CLAUDE.md`.
