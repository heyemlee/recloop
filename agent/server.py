"""The shopping assistant's HTTP service: POST /v1/chat (server-sent events), GET /healthz.

  uv run --project agent --group encoder python -m agent.server --port 18083   (make serve-agent)

The web app proxies /api/v1/chat here (web/next.config.ts). The conversation itself does not go
through the Go api: the assistant calls it as a client (recommendations, the sponsored item), so
the chat path's latency and cost are measured apart from the feed's (design §10.3 (6)).

A turn runs on a worker thread and its events are streamed as they come (FastAPI's
EventSourceResponse: `event: <name>` plus a JSON `data:` line per event, see `agent.chat`).
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import queue
import threading
from collections.abc import Callable, Iterator

from fastapi import FastAPI
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, ConfigDict, Field

from agent.chat import ChatService

log = logging.getLogger("agent.server")
DEFAULT_PORT = 18083


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    user_id: int = Field(ge=0)
    message: str = Field(min_length=1, max_length=2000)


def create_app(service_factory: Callable[[], ChatService]) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service_factory()
        yield
        app.state.service.close()

    app = FastAPI(title="RecLoop shopping assistant", lifespan=lifespan)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "parser": app.state.service.parser_name}

    @app.post("/v1/chat", response_class=EventSourceResponse)
    def chat(body: ChatRequest) -> Iterator[ServerSentEvent]:
        service: ChatService = app.state.service
        events: queue.SimpleQueue[ServerSentEvent | None] = queue.SimpleQueue()

        def emit(name: str, data: dict) -> None:
            events.put(ServerSentEvent(event=name, data=data))

        def work() -> None:
            try:
                service.turn(body.user_id, body.message, emit)
            except Exception:
                log.exception("chat turn failed (user %s)", body.user_id)
                emit("error", {"message": "The assistant failed on this message."})
            finally:
                events.put(None)

        threading.Thread(target=work, name="chat-turn", daemon=True).start()
        while (ev := events.get()) is not None:
            yield ev

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("AGENT_PORT", DEFAULT_PORT)))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    import uvicorn

    uvicorn.run(create_app(ChatService.from_env), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
