"""docker compose for this checkout's stack (deploy/docker-compose.yml).

The compose file names its project `recloop`: the shared stack. A checkout with a private stack
sets COMPOSE_PROJECT_NAME (and its own ports) in the environment or in the repo's .env. Given
`-f deploy/docker-compose.yml`, docker compose reads neither the repo's .env nor its
COMPOSE_PROJECT_NAME by itself, so a bare `docker compose -f ...` from a private checkout acts on
the shared stack. Every command built here passes the project (-p) and the repo's .env.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "deploy" / "docker-compose.yml"
ENV_FILE = ROOT / ".env"
DEFAULT_PROJECT = "recloop"


def dotenv(path: Path | None = None) -> dict[str, str]:
    """KEY=VALUE lines of a .env file (the repo's by default; unquoted values, as the Makefile
    includes it)."""
    path = path or ENV_FILE
    out: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def project() -> str:
    """The compose project: COMPOSE_PROJECT_NAME from the environment, else from the repo's
    .env, else the shared stack's."""
    return (
        os.environ.get("COMPOSE_PROJECT_NAME")
        or dotenv().get("COMPOSE_PROJECT_NAME")
        or DEFAULT_PROJECT
    )


def command(*profiles: str) -> list[str]:
    """`docker compose` for this checkout's project, ports (.env) and profiles."""
    cmd = ["docker", "compose", "-p", project(), "-f", str(FILE)]
    if ENV_FILE.exists():
        cmd += ["--env-file", str(ENV_FILE)]
    for p in profiles:
        cmd += ["--profile", p]
    return cmd


def run(*args: str) -> None:
    subprocess.run([*command(), *args], check=True, capture_output=True)


def container(service: str) -> str:
    """The container compose runs a service in: {project}-{service}-1."""
    return f"{project()}-{service}-1"


def service_of(container_name: str) -> str:
    """The service of one of this project's containers (container() reversed)."""
    return container_name.removeprefix(project() + "-").removesuffix("-1")


def other_projects_running() -> list[str]:
    """Running containers that do not belong to this project (by compose's project label, so
    a private project `recloop-x` and the shared `recloop` count as different)."""
    raw = subprocess.run(
        ["docker", "ps", "--format", '{{.Names}}\t{{.Label "com.docker.compose.project"}}'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    me = project()
    return [
        name for name, _, proj in (line.partition("\t") for line in raw.splitlines()) if proj != me
    ]
