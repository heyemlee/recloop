# agent

The shopping assistant (docs/design.md §10). A language model turns the shopper's message into
constraints (validated, one retry), may judge device compatibility from quoted listing text,
and writes one-line reasons; the code searches the catalog (BM25 + pgvector, RRF, SQL filters
on price, stock and category), checks every constraint on the item rows, and renders the reply
from those rows. "Last time" means what the store's feed counts as bought: orders plus the
dataset users' training-period reviews (`history.py`).

In the store the assistant is a service (`server.py`, `make serve-agent`): `POST /v1/chat`
streams the turn as server-sent events, and the web app's `/chat` page shows the reply as item
cards. There it also uses the Go api as a client: the chat surface's recommendations re-order
close matches for the user, and the ads module may add one sponsored item chosen among the
items the code has checked; its label and reason come from code (`store_api.py`, `agent.py`).

`eval/` holds the 150-case evaluation set, the no-LLM baseline and the harness that writes
`docs/results/agent.md` (`make agent-eval`), and the chat path's measurement through the
running services (`make agent-latency`). Language-model runs are opt-in and paid
(`make agent-eval-llm-smoke`, `make agent-eval-llm`, with spending caps). Run everything from
the repo root; see `CLAUDE.md`.
