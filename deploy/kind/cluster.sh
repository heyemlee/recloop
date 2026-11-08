#!/usr/bin/env bash
# Local Kubernetes for RecLoop (M5, docs/design.md §23.2): a one-node kind cluster, the
# infrastructure from official charts and operators, RecLoop's own chart (deploy/helm/recloop).
# The Makefile's kind-* / deploy targets call this; run from the repo root.
#
#   cluster.sh tools    kind, kubectl, helm (versions.lock, sha256-checked) -> .cache/bin
#   cluster.sh charts   the third-party charts (versions.lock, sha256-checked) -> .cache/charts
#   cluster.sh up       create the cluster (cluster.yaml) and install the infrastructure
#   cluster.sh infra    (re)install the infrastructure only
#   cluster.sh images   build RecLoop's images (same build definitions as compose) and load
#                       them into the node; also load the infrastructure images found locally
#   cluster.sh deploy   helm upgrade --install RecLoop's chart with the loaded image tags
#   cluster.sh data     migrations, catalog and users, Redis seed (through the NodePorts)
#   cluster.sh status   pods, autoscalers, node usage
#   cluster.sh down     delete the cluster
#   cluster.sh lint     helm lint RecLoop's charts; render them and the infrastructure values
#                       (no cluster needed)
#
# Nothing here touches ~/.kube or ~/.config: the kubeconfig and Helm's state live in .cache/.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
LOCK=deploy/kind/versions.lock
BIN=$ROOT/.cache/bin
CHARTS=$ROOT/.cache/charts
CLUSTER=${KIND_CLUSTER:-recloop}
NS=${KIND_NAMESPACE:-recloop}
export KUBECONFIG=$ROOT/.cache/kind/kubeconfig
export HELM_CACHE_HOME=$ROOT/.cache/helm/cache HELM_CONFIG_HOME=$ROOT/.cache/helm/config \
  HELM_DATA_HOME=$ROOT/.cache/helm/data
KIND=$BIN/kind
KUBECTL=("$BIN/kubectl" --context "kind-$CLUSTER")
HELM=("$BIN/helm" --kube-context "kind-$CLUSTER")

log() { printf '\033[1m==> %s\033[0m\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

platform() {
  local os arch
  os=$(uname -s | tr '[:upper:]' '[:lower:]')
  case $(uname -m) in
    arm64 | aarch64) arch=arm64 ;;
    x86_64 | amd64) arch=amd64 ;;
    *) die "unsupported machine $(uname -m)" ;;
  esac
  echo "$os-$arch"
}

sha256() { shasum -a 256 "$1" | cut -d' ' -f1; }

# lock <kind> <name> [field...]: the line of versions.lock for this tool / chart (and platform).
lock() {
  local kind=$1 name=$2
  if [ "$kind" = tool ]; then
    awk -v n="$name" -v p="$(platform)" '$1 == "tool" && $2 == n && $4 == p' "$LOCK"
  else
    awk -v k="$kind" -v n="$name" '$1 == k && $2 == n' "$LOCK"
  fi
}

# github_asset <url> <dest>: a GitHub release download through the API's asset endpoint (the
# same file; some proxies time out on github.com/.../releases/download/...).
github_asset() {
  local url=$1 dest=$2 repo tag name id
  [[ $url =~ ^https://github\.com/([^/]+/[^/]+)/releases/download/(.+)/([^/]+)$ ]] || return 1
  repo=${BASH_REMATCH[1]} tag=${BASH_REMATCH[2]} name=${BASH_REMATCH[3]}
  id=$(curl -fsSL --retry 3 "https://api.github.com/repos/$repo/releases/tags/$tag" |
    python3 -c 'import json, sys; print(next(a["id"] for a in json.load(sys.stdin)["assets"] if a["name"] == sys.argv[1]))' "$name") ||
    return 1
  curl -fsSL --retry 3 -H "Accept: application/octet-stream" -o "$dest" \
    "https://api.github.com/repos/$repo/releases/assets/$id"
}

# fetch <url> <sha256> <dest>: download to a temporary file, check it, then move it in place.
fetch() {
  local url=$1 want=$2 dest=$3 tmp
  tmp=$(mktemp "${dest}.XXXXXX")
  curl -fsSL --retry 3 -o "$tmp" "$url" 2>/dev/null || github_asset "$url" "$tmp" ||
    { rm -f "$tmp"; die "download failed: $url"; }
  if [ "$(sha256 "$tmp")" != "$want" ]; then
    rm -f "$tmp"
    die "sha256 mismatch for $url (want $want)"
  fi
  mv "$tmp" "$dest"
}

cmd_tools() {
  mkdir -p "$BIN"
  local name version plat sum stamp url
  for name in kind kubectl helm; do
    read -r _ _ version plat sum <<<"$(lock tool "$name")"
    [ -n "${sum:-}" ] || die "no $name for $(platform) in $LOCK"
    stamp=$BIN/.$name.sha256
    if [ -x "$BIN/$name" ] && [ "$(cat "$stamp" 2>/dev/null)" = "$sum" ]; then
      continue
    fi
    log "$name $version ($plat)"
    case $name in
      kind) url=https://github.com/kubernetes-sigs/kind/releases/download/$version/kind-$plat ;;
      kubectl) url=https://dl.k8s.io/release/$version/bin/${plat%-*}/${plat#*-}/kubectl ;;
      helm) url=https://get.helm.sh/helm-$version-$plat.tar.gz ;;
    esac
    if [ "$name" = helm ]; then
      fetch "$url" "$sum" "$BIN/helm.tar.gz"
      tar -xzf "$BIN/helm.tar.gz" -C "$BIN" --strip-components 1 "$plat/helm"
      rm -f "$BIN/helm.tar.gz"
    else
      fetch "$url" "$sum" "$BIN/$name"
    fi
    chmod +x "$BIN/$name"
    echo "$sum" >"$stamp"
  done
}

# chart_file <name>: path of the downloaded chart archive (after cmd_charts).
chart_file() {
  local version
  read -r _ _ version _ <<<"$(lock chart "$1")"
  echo "$CHARTS/$1-$version.tgz"
}

cmd_charts() {
  mkdir -p "$CHARTS"
  local name version url sum dest
  while read -r _ name version url sum; do
    dest=$CHARTS/$name-$version.tgz
    if [ -f "$dest" ] && [ "$(sha256 "$dest")" = "$sum" ]; then
      continue
    fi
    log "chart $name $version"
    fetch "$url" "$sum" "$dest"
  done < <(awk '$1 == "chart"' "$LOCK")
}

exists() { "$KIND" get clusters 2>/dev/null | grep -qx "$CLUSTER"; }

# Images the infrastructure and RecLoop's chart use that compose has usually pulled already:
# loading them saves the node's pulls (Docker Hub's anonymous limit is low). Best effort: an
# image that is not here, or does not load, is pulled by the node instead.
PRELOAD=(redis:8.10.3-alpine redpandadata/redpanda:v26.2.6 redpandadata/connect:4.115.1
  ghcr.io/shopify/toxiproxy:2.13.0 grafana/grafana:13.2.4 oliver006/redis_exporter:v1.94.0)

cmd_preload() {
  local img
  for img in "${PRELOAD[@]}"; do
    if docker image inspect "$img" >/dev/null 2>&1; then
      log "kind load $img"
      "$KIND" load docker-image --name "$CLUSTER" "$img" >/dev/null 2>&1 ||
        echo "  (not loaded; the node pulls it)" >&2
    fi
  done
}

cmd_up() {
  cmd_tools
  cmd_charts
  [ -d artifacts/bundles/current ] || die "no artifacts/bundles/current: run make bundle first"
  if exists; then
    log "cluster $CLUSTER exists"
  else
    local node
    node=$(awk '$1 == "node" {print $2}' "$LOCK")
    mkdir -p .cache/kind
    sed "s|@REPO@|$ROOT|g" deploy/kind/cluster.yaml >.cache/kind/cluster.yaml
    log "kind create cluster $CLUSTER ($node)"
    "$KIND" create cluster --name "$CLUSTER" --image "$node" --config .cache/kind/cluster.yaml \
      --kubeconfig "$KUBECONFIG" --wait 120s
  fi
  cmd_preload
  cmd_infra
}

# helm_install <release> <namespace> <chart> <values file | -> [helm args...]
helm_install() {
  local release=$1 ns=$2 chart=$3 values=$4
  shift 4
  [ "$values" = - ] || set -- --values "$values" "$@"
  log "helm $release ($ns)"
  "${HELM[@]}" upgrade --install "$release" "$chart" --namespace "$ns" --create-namespace \
    --wait --timeout 10m "$@"
}

cmd_infra() {
  exists || die "no cluster $CLUSTER: run cluster.sh up"
  cmd_charts
  # Monitoring first: the others create ServiceMonitors / PodMonitors (its CRDs).
  helm_install kube-prometheus-stack monitoring "$(chart_file kube-prometheus-stack)" \
    deploy/kind/values/kube-prometheus-stack.yaml
  helm_install metrics-server kube-system "$(chart_file metrics-server)" \
    deploy/kind/values/metrics-server.yaml
  helm_install keda keda "$(chart_file keda)" deploy/kind/values/keda.yaml
  helm_install cnpg cnpg-system "$(chart_file cloudnative-pg)" deploy/kind/values/cloudnative-pg.yaml
  helm_install redpanda redpanda "$(chart_file redpanda)" deploy/kind/values/redpanda.yaml
  # PostgreSQL, Redis, the topics (its hook Job needs the broker, installed above).
  helm_install recloop-infra "$NS" deploy/helm/recloop-infra -
  log "waiting for PostgreSQL"
  "${KUBECTL[@]}" -n "$NS" wait cluster.postgresql.cnpg.io/recloop-pg --for=condition=Ready \
    --timeout=10m
}

# RecLoop's images, built from the same definitions as compose's (the Go image of `make images`,
# deploy/model-server.Dockerfile) plus deploy/web.Dockerfile, under names of their own
# (recloop-kind/*) so the shared compose images are never retagged. Each is tagged with its
# image ID: a rebuild that changes nothing keeps its tag, so `deploy` rolls only what changed.
IMAGES=.cache/kind/images.env

tag_and_load() {
  local repo=$1 id tag
  id=$(docker image inspect -f '{{.Id}}' "$repo:build")
  tag=${id#sha256:}
  tag=${tag:0:12}
  docker tag "$repo:build" "$repo:$tag"
  log "kind load $repo:$tag"
  "$KIND" load docker-image --name "$CLUSTER" "$repo:$tag" >/dev/null
  echo "$tag"
}

cmd_images() {
  exists || die "no cluster $CLUSTER: run cluster.sh up"
  log "Go services (make images)"
  RECLOOP_GO_IMAGE=recloop-kind/go:build make --no-print-directory images
  log "model server"
  docker build -q -f deploy/model-server.Dockerfile -t recloop-kind/model-server:build model_server >/dev/null
  log "web"
  docker build -q -f deploy/web.Dockerfile -t recloop-kind/web:build web >/dev/null
  local go ms web
  go=$(tag_and_load recloop-kind/go)
  ms=$(tag_and_load recloop-kind/model-server)
  web=$(tag_and_load recloop-kind/web)
  mkdir -p .cache/kind
  printf 'GO_TAG=%s\nMODEL_SERVER_TAG=%s\nWEB_TAG=%s\n' "$go" "$ms" "$web" >"$IMAGES"
  cat "$IMAGES"
}

# Host-side access to the cluster's stores (kind port mappings, deploy/kind/cluster.yaml).
KIND_DATABASE_URL=${KIND_DATABASE_URL:-postgres://recloop:recloop@localhost:30432/recloop?sslmode=disable}
KIND_REDIS_URL=${KIND_REDIS_URL:-redis://localhost:30379/0}
KIND_CACHE_REDIS_URL=${KIND_CACHE_REDIS_URL:-redis://localhost:30380/0}

cmd_data() {
  exists || die "no cluster $CLUSTER: run cluster.sh up"
  # Command-line variables: they win over a .env the Makefile includes.
  make --no-print-directory ingest DATABASE_URL="$KIND_DATABASE_URL"
  make --no-print-directory seed REDIS_URL="$KIND_REDIS_URL" CACHE_REDIS_URL="$KIND_CACHE_REDIS_URL"
}

cmd_deploy() {
  exists || die "no cluster $CLUSTER: run cluster.sh up"
  [ -f "$IMAGES" ] || die "no $IMAGES: run cluster.sh images"
  # shellcheck disable=SC1090
  . "$IMAGES"
  # compose's Grafana alert rules, where they exist.
  local alerts=deploy/grafana/provisioning/alerting/redis.yml
  [ -f "$alerts" ] && set -- --set-file grafana.alerts="$alerts" "$@"
  helm_install recloop "$NS" deploy/helm/recloop - \
    --set images.go.repository=recloop-kind/go --set images.go.tag="$GO_TAG" \
    --set images.modelServer.repository=recloop-kind/model-server \
    --set images.modelServer.tag="$MODEL_SERVER_TAG" \
    --set images.web.repository=recloop-kind/web --set images.web.tag="$WEB_TAG" \
    $(recloop_files) "$@"
}

cmd_status() {
  exists || die "no cluster $CLUSTER"
  "${KUBECTL[@]}" get pods -A -o wide
  "${KUBECTL[@]}" -n "$NS" get hpa,scaledobjects
  "${KUBECTL[@]}" top pods -A 2>/dev/null || true
}

# The files the recloop chart takes with --set-file (cmd_deploy and cmd_lint pass them).
recloop_files() {
  echo --set-file logSink.config=deploy/connect/log-sink.yaml \
    --set-file grafana.dashboard=deploy/grafana/dashboards/recloop.json
}

cmd_lint() {
  cmd_tools >/dev/null
  cmd_charts
  local kv name version
  kv=$(awk '$1 == "node" {print $2}' "$LOCK" | sed -E 's/.*:v([0-9.]+)@.*/\1/')
  # shellcheck disable=SC2046
  "${HELM[@]}" lint --strict deploy/helm/recloop-infra --kube-version "$kv"
  # shellcheck disable=SC2046
  "${HELM[@]}" lint --strict deploy/helm/recloop --kube-version "$kv" $(recloop_files)
  for ns_chart in kube-prometheus-stack:monitoring metrics-server:kube-system keda:keda \
    cloudnative-pg:cnpg-system redpanda:redpanda; do
    name=${ns_chart%%:*}
    "${HELM[@]}" template "$name" "$(chart_file "$name")" --namespace "${ns_chart#*:}" \
      --values "deploy/kind/values/$name.yaml" --kube-version "$kv" >/dev/null
    echo "rendered $name with deploy/kind/values/$name.yaml"
  done
}

cmd_down() {
  if exists; then
    log "kind delete cluster $CLUSTER"
    "$KIND" delete cluster --name "$CLUSTER" --kubeconfig "$KUBECONFIG"
  fi
}

cmd=${1:-}
[ $# -gt 0 ] && shift
case $cmd in
  tools) cmd_tools ;;
  charts) cmd_charts ;;
  up) cmd_up ;;
  infra) cmd_infra ;;
  preload) cmd_preload ;;
  images) cmd_images ;;
  data) cmd_data ;;
  deploy) cmd_deploy "$@" ;;
  status) cmd_status ;;
  down) cmd_down ;;
  lint) cmd_lint ;;
  *) sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
