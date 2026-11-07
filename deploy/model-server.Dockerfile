# Model server image for the bench profile (deploy/docker-compose.yml). Build context:
# model_server/. The bundle is mounted at /bundles; the dependencies come from uv.lock.
FROM python:3.12.14-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.12.6 /uv /usr/local/bin/uv
ENV UV_PYTHON_DOWNLOADS=never UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON=/usr/local/bin/python3.12
WORKDIR /app/model_server
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project
COPY . ./
ENV PATH=/opt/venv/bin:$PATH PYTHONPATH=/app PYTHONUNBUFFERED=1 ORT_DISABLE_TELEMETRY=1
WORKDIR /app
USER 65534:65534
ENTRYPOINT ["python", "-m", "model_server"]
