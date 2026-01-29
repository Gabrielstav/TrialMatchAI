#!/usr/bin/env bash
set -euo pipefail

# build ES snapshot from processed trial- & criteria data,
# uses dedicated ES instance to avoid shared state with local_test

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VM_DEV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$VM_DEV_DIR/.." && pwd)"

# config
: "${COMPOSE_FILE:=$SCRIPT_DIR/docker-compose.yml}"
: "${TRIALS_DIR:=$PROJECT_ROOT/local_test/test_data/processed_trials}"
: "${CRITERIA_DIR:=$PROJECT_ROOT/local_test/test_data/processed_criteria}"
: "${OUTPUT_DIR:=$VM_DEV_DIR/deploy/snapshots}"
: "${ES_HOST:=http://localhost:9201}"
: "${ES_USER:=elastic}"
: "${ES_PASSWORD:=trialmatchai}"
: "${ES_CONTAINER:=trialmatchai-snapshot-es}"
: "${MAX_WAIT_SECONDS:=120}"

: "${TRIALS_INDEX:=clinical_trials}"
: "${CRITERIA_INDEX:=eligibility_criteria}"
: "${SNAPSHOT_REPO:=trialmatchai_backup}"
: "${SNAPSHOT_NAME:=trialmatchai_$(date +%Y%m%d_%H%M%S)}"

# store state files in snapshot-builder
PROCESSED_IDS_FILE="$SCRIPT_DIR/.processed_ids.txt"
CONFIG_FILE="$SCRIPT_DIR/.indexer-config.json"

# logging
if [[ -t 1 ]]; then
  GREEN=$'\033[0;32m'; RED=$'\033[0;31m'; NC=$'\033[0m'
else
  GREEN=""; RED=""; NC=""
fi

info() { printf '%s[INFO]%s %s\n' "$GREEN" "$NC" "$*"; }
die()  { printf '%s[ERR ]%s %s\n' "$RED" "$NC" "$*" >&2; exit 1; }

need_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"; }

curl_es() {
  local url="$1"; shift || true
  curl --fail --silent --show-error \
       --connect-timeout 5 --max-time 60 \
       -u "${ES_USER}:${ES_PASSWORD}" \
       "$@" \
       "$url"
}

wait_for_es() {
  info "Waiting for Elasticsearch (${MAX_WAIT_SECONDS}s timeout)..."
  local start now
  start=$(date +%s)

  while true; do
    if curl_es "${ES_HOST}/_cluster/health" 2>/dev/null | grep -q '"status"'; then
      info "Elasticsearch is ready."
      return 0
    fi

    now=$(date +%s)
    if (( now - start >= MAX_WAIT_SECONDS )); then
      die "Elasticsearch did not become ready within ${MAX_WAIT_SECONDS}s"
    fi

    printf '.'
    sleep 2
  done
}

generate_config() {
  info "Generating indexer config for ${ES_HOST}..."
  cat > "$CONFIG_FILE" <<JSON
{
  "elasticsearch": {
    "hosts": ["${ES_HOST}"],
    "username": "${ES_USER}",
    "password": "${ES_PASSWORD}",
    "verify_certs": false,
    "request_timeout": 300,
    "retry_on_timeout": true,
    "max_retries": 3
  }
}
JSON
}

delete_indices() {
  info "Deleting existing indices (if any)..."
  curl_es "${ES_HOST}/${TRIALS_INDEX}" -X DELETE 2>/dev/null || true
  curl_es "${ES_HOST}/${CRITERIA_INDEX}" -X DELETE 2>/dev/null || true

  if [[ -f "$PROCESSED_IDS_FILE" ]]; then
    info "Removing tracking file: ${PROCESSED_IDS_FILE}"
    rm -f "$PROCESSED_IDS_FILE"
  fi
  sleep 1
}

index_trials() {
  info "Indexing trials from: ${TRIALS_DIR}"
  python "$PROJECT_ROOT/utils/Indexer/index_trials.py" \
    --config "$CONFIG_FILE" \
    --processed-folder "$TRIALS_DIR" \
    --index-name "$TRIALS_INDEX" \
    --batch-size 50
}

index_criteria() {
  info "Indexing criteria from: ${CRITERIA_DIR}"
  python "$PROJECT_ROOT/utils/Indexer/index_criteria.py" \
    --config "$CONFIG_FILE" \
    --processed-folder "$CRITERIA_DIR" \
    --index-name "$CRITERIA_INDEX" \
    --batch-size 100 \
    --max-workers 4 \
    --processed-file "$PROCESSED_IDS_FILE"
}

create_snapshot() {
  info "Registering snapshot repository: ${SNAPSHOT_REPO}"
  curl_es "${ES_HOST}/_snapshot/${SNAPSHOT_REPO}" \
    -X PUT -H "Content-Type: application/json" \
    -d @- <<JSON | jq .
{
  "type": "fs",
  "settings": {
    "location": "/usr/share/elasticsearch/snapshots",
    "compress": true
  }
}
JSON

  info "Creating snapshot: ${SNAPSHOT_NAME}"
  curl_es "${ES_HOST}/_snapshot/${SNAPSHOT_REPO}/${SNAPSHOT_NAME}?wait_for_completion=true" \
    -X PUT -H "Content-Type: application/json" \
    -d @- <<JSON | jq .
{
  "indices": "${TRIALS_INDEX},${CRITERIA_INDEX}",
  "ignore_unavailable": true,
  "include_global_state": false
}
JSON

  info "Verifying snapshot..."
  curl_es "${ES_HOST}/_snapshot/${SNAPSHOT_REPO}/${SNAPSHOT_NAME}" | jq .
}

show_summary() {
  info ""
  info "=== Summary ==="
  info "Indices:"
  curl_es "${ES_HOST}/_cat/indices?v" || true
  info ""
  info "Snapshot saved to: ${OUTPUT_DIR}"
  info ""
  info "To deploy to HPC:"
  info "  1. Copy ${OUTPUT_DIR}/* to the HPC snapshots directory"
  info "  2. The sbatch script will restore it automatically"
}

cleanup_es() {
  info "Stopping Elasticsearch..."
  docker compose -f "$COMPOSE_FILE" down || true
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Build ES snapshot from processed trial/criteria data.
Uses a dedicated ES instance (port 9201) to avoid conflicts with local_test.

Options:
  -h, --help      Show this help message
  --skip-index    Skip indexing, only create snapshot (assumes data exists)
  --clean         Delete existing indices and re-index from scratch
  --keep-es       Don't stop ES after completion (useful for debugging)

Environment variables:
  TRIALS_DIR      Processed trials directory (default: local_test/test_data/processed_trials)
  CRITERIA_DIR    Processed criteria directory (default: local_test/test_data/processed_criteria)
  OUTPUT_DIR      Snapshot output directory (default: vm_development/deploy/snapshots)
  ES_HOST         Elasticsearch URL (default: http://localhost:9201)
  ES_PASSWORD     Elasticsearch password (default: trialmatchai)

Examples:
  # Default: use local_test data, output to deploy/snapshots
  ./build-snapshot.sh

  # Clean build (delete indices, re-index everything)
  ./build-snapshot.sh --clean

  # Custom data location
  TRIALS_DIR=/path/to/trials CRITERIA_DIR=/path/to/criteria ./build-snapshot.sh
EOF
}

main() {
  local skip_index=false
  local clean=false
  local keep_es=false

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help) usage; exit 0 ;;
      --skip-index) skip_index=true; shift ;;
      --clean) clean=true; shift ;;
      --keep-es) keep_es=true; shift ;;
      *) die "Unknown option: $1" ;;
    esac
  done

  need_cmd docker
  need_cmd curl
  need_cmd jq
  need_cmd python

  info "Build ES Snapshot"
  info "  Compose file: ${COMPOSE_FILE}"
  info "  Trials dir:   ${TRIALS_DIR}"
  info "  Criteria dir: ${CRITERIA_DIR}"
  info "  Output dir:   ${OUTPUT_DIR}"
  info "  ES host:      ${ES_HOST}"
  info ""

  # vlidate paths
  [[ -f "$COMPOSE_FILE" ]] || die "Compose file not found: ${COMPOSE_FILE}"

  if [[ "$skip_index" == false ]]; then
    [[ -d "$TRIALS_DIR" ]] || die "Trials directory not found: ${TRIALS_DIR}"
    [[ -d "$CRITERIA_DIR" ]] || die "Criteria directory not found: ${CRITERIA_DIR}"
  fi

  mkdir -p "$OUTPUT_DIR"
  generate_config

  info "Starting Elasticsearch..."
  docker compose -f "$COMPOSE_FILE" up -d
  wait_for_es

  if [[ "$skip_index" == false ]]; then
    if [[ "$clean" == true ]]; then
      delete_indices
    fi
    index_trials
    index_criteria
  else
    info "Skipping indexing (--skip-index)"
  fi

  create_snapshot
  show_summary

  if [[ "$keep_es" == false ]]; then
    cleanup_es
  else
    info "ES still running (--keep-es). Stop with: docker compose -f $COMPOSE_FILE down"
  fi

  info "Done!"
}

main "$@"
