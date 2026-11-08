# The training-sample job (ml/samples, docs/design.md §7.5): PySpark in local mode on a Linux JVM.
# A container, not a JDK on the host: HotSpot JVMs started from the Claude app's processes on the
# development Mac crash at startup (SIGBUS in CodeHeap::allocate, Temurin 17 and 21), and the
# container gives every machine the same JVM and Python packages. Only the environment is built;
# the repository is mounted at /repo when the job runs (`make samples`, `make test-samples`).
# Build context: deploy/spark/.
FROM eclipse-temurin:21.0.13_9-jre AS jre

FROM python:3.12.14-slim-bookworm
COPY --from=jre /opt/java/openjdk /opt/java/openjdk
COPY --from=ghcr.io/astral-sh/uv:0.12.6 /uv /usr/local/bin/uv
ENV JAVA_HOME=/opt/java/openjdk UV_PYTHON_DOWNLOADS=never UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON=/usr/local/bin/python3.12
WORKDIR /env
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project
# The job runs as the host user (`docker run --user`, its name from a mounted passwd line), with
# no network (`--network none --hostname localhost`): Spark binds to the loopback only.
ENV PATH=/opt/venv/bin:/opt/java/openjdk/bin:$PATH PYTHONPATH=/repo PYTHONUNBUFFERED=1 \
    HOME=/tmp TZ=UTC SPARK_LOCAL_IP=127.0.0.1 SPARK_LOCAL_HOSTNAME=localhost HADOOP_USER_NAME=spark
WORKDIR /repo
ENTRYPOINT ["python"]
