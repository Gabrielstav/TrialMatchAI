#!/usr/bin/env bash
set -euo pipefail

# starts ES apptainer that wraps indexed snapshot data,
# runs before pipeline

# config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$(dirname "$SCRIPT_DIR")}"
SIF_FILE="${SIF_FILE:-$BASE_DIR/apptainer/elasticsearch.sif}"
ES_DATA_DIR="${ES_DATA_DIR:-$BASE_DIR/esdata}"
ES_SNAPSHOTS_DIR="${ES_SNAPSHOTS_DIR:-$BASE_DIR/snapshots}"
ES_LOGS_DIR="${ES_LOGS_DIR:-$BASE_DIR/logs}"
ELASTIC_PASSWORD="${ELASTIC_PASSWORD:-trialmatchai}"

GREEN="\033[0;32m"
NC="\033[0m"

info() { echo -e "${GREEN}[INFO]${NC} $*"; }

info "Starting Elasticsearch..."

# ensure required dirs exist
mkdir -p "$ES_DATA_DIR" "$ES_SNAPSHOTS_DIR" "$ES_LOGS_DIR"
if [ ! -f "$SIF_FILE" ]; then
    echo "[ERROR] Elasticsearch SIF not found at: $SIF_FILE"
    exit 1
fi

# check if ES is already running
if pgrep -f "elasticsearch" > /dev/null 2>&1; then
    warn "Elasticsearch may already be running. Checking..."
    if curl -s "http://localhost:9200" > /dev/null 2>&1; then
        info "Elasticsearch is already running on port 9200"
        exit 0
    fi
fi

# start ES in background
info "Starting Elasticsearch container..."
info "  Data dir: $ES_DATA_DIR"
info "  Snapshots dir: $ES_SNAPSHOTS_DIR"
info "  Logs dir: $ES_LOGS_DIR"

apptainer run \
    --env ELASTIC_PASSWORD="$ELASTIC_PASSWORD" \
    --env ES_JAVA_OPTS="-Xms4g -Xmx4g" \
    --env discovery.type=single-node \
    --env xpack.security.enabled=true \
    --env xpack.security.http.ssl.enabled=false \
    --env xpack.security.transport.ssl.enabled=false \
    --env cluster.name=trialmatchai \
    --env path.repo=/usr/share/elasticsearch/snapshots \
    --bind "$ES_DATA_DIR:/usr/share/elasticsearch/data" \
    --bind "$ES_SNAPSHOTS_DIR:/usr/share/elasticsearch/snapshots" \
    --bind "$ES_LOGS_DIR:/usr/share/elasticsearch/logs" \
    "$SIF_FILE" > "$ES_LOGS_DIR/elasticsearch.log" 2>&1 &

ES_PID=$!
echo "$ES_PID" > "$ES_LOGS_DIR/elasticsearch.pid"

info "Elasticsearch started with PID: $ES_PID"
info "Waiting for Elasticsearch to be ready..."

# wait for ES to be ready
MAX_WAIT=160
WAITED=0
until curl -s -u "elastic:$ELASTIC_PASSWORD" "http://localhost:9200/_cluster/health" 2>/dev/null | grep -q '"status"'; do
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "[ERROR] Elasticsearch did not start within ${MAX_WAIT}s"
        echo "Check logs at: $ES_LOGS_DIR/elasticsearch.log"
        exit 1
    fi
    echo -n "."
    sleep 2
    WAITED=$((WAITED + 2))
done
echo ""

info "Elasticsearch is ready at http://localhost:9200"
info "PID file: $ES_LOGS_DIR/elasticsearch.pid"
info "Log file: $ES_LOGS_DIR/elasticsearch.log"