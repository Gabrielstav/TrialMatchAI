#!/usr/bin/env bash
set -euo pipefail

# restores ES snapshot of pre-indexed trials & criteria
# run after starting ES (see: trialmatchai.sbatch)

# configs
: "${ES_HOST:=http://localhost:9200}"
# ES auth empty by default as current build doesn't use auth for ES
: "${ES_USER:=}"
: "${ES_PASSWORD:=}"
: "${SNAPSHOT_REPO:=trialmatchai_backup}"
: "${SNAPSHOT_NAME:=}"  # picks latest if empty
: "${RESTORE_INDICES:=clinical_trials,eligibility_criteria}"
: "${SNAPSHOT_LOCATION:=/usr/share/elasticsearch/snapshots}"
: "${MAX_WAIT_SECONDS:=180}"

# logging
if [[ -t 1 ]]; then
  GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; RED=$'\033[0;31m'; NC=$'\033[0m'
else
  GREEN=""; YELLOW=""; RED=""; NC=""
fi

info() { printf '%s[INFO]%s %s\n' "$GREEN" "$NC" "$*"; }
warn() { printf '%s[WARN]%s %s\n' "$YELLOW" "$NC" "$*"; }
die()  { printf '%s[ERR ]%s %s\n' "$RED" "$NC" "$*" >&2; exit 1; }

need_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"; }

# curl wrapper to fail on HTTP errors
curl_es() {
  local url="$1"; shift || true

  local -a auth=()
  if [[ -n "${ES_USER}" && -n "${ES_PASSWORD}" ]]; then
    auth=(-u "${ES_USER}:${ES_PASSWORD}")
  fi

  curl --fail --silent --show-error \
       --connect-timeout 2 --max-time 30 \
       "${auth[@]}" \
       "$@" \
       "$url"
}

wait_for_es() {
  info "Waiting for Elasticsearch with (${MAX_WAIT_SECONDS}s timeout)..."
  local start now
  start=$(date +%s)

  while true; do
    # ES ready when cluster health responds with status field
    if curl_es "${ES_HOST}/_cluster/health" 2>/dev/null | jq -e '.status' >/dev/null 2>&1; then
      info "Elasticsearch is responding."
      return 0
    fi

    now=$(date +%s)
    if (( now - start >= MAX_WAIT_SECONDS )); then
      die "Elasticsearch did not become ready within ${MAX_WAIT_SECONDS}s (host: ${ES_HOST})"
    fi

    printf '.'
    sleep 2
  done
}

# register snapshot repo of pre-indexed trials & criteria made from build-snapshot.sh,
# *must* be read only
register_repo() {
  info "Registering snapshot repository: ${SNAPSHOT_REPO}"
  curl_es "${ES_HOST}/_snapshot/${SNAPSHOT_REPO}" \
    -X PUT -H "Content-Type: application/json" \
    -d @- <<JSON | jq .
{
  "type": "fs",
  "settings": {
    "location": "${SNAPSHOT_LOCATION}",
    "readonly": true
  }
}
JSON
}

fetch_snapshots_json() {
  curl_es "${ES_HOST}/_snapshot/${SNAPSHOT_REPO}/_all"
}

pick_latest_snapshot() {
  local snapshots_json="$1"
  local latest
  latest="$(jq -r '
    (.snapshots // [])
    | sort_by(.start_time)
    | last
    | .snapshot // empty
  ' <<<"$snapshots_json")"

  [[ -n "$latest" ]] || die "No snapshots found in repo '${SNAPSHOT_REPO}'."
  printf '%s' "$latest"
}

delete_indices_if_present() {
  # check indexed trials and criteria
  # and delete them in apptainer if present
  local existing=""
  existing="$(curl_es "${ES_HOST}/_cat/indices/${RESTORE_INDICES}?h=index" 2>/dev/null || true)"

  if [[ -n "$existing" ]]; then
    warn "Indices already exist; deleting before restore: ${existing//$'\n'/ }"
    curl_es "${ES_HOST}/${RESTORE_INDICES}" -X DELETE >/dev/null || true
    sleep 2
  fi
}

restore_snapshot() {
  local snapshot="$1"
  info "Restoring snapshot: ${snapshot}"
  curl_es "${ES_HOST}/_snapshot/${SNAPSHOT_REPO}/${snapshot}/_restore?wait_for_completion=true" \
    -X POST -H "Content-Type: application/json" \
    -d @- <<JSON | jq .
{
  "indices": "${RESTORE_INDICES}",
  "ignore_unavailable": true,
  "include_global_state": false
}
JSON
}

main() {
  need_cmd curl
  need_cmd jq

  info "Restoring Elasticsearch snapshot..."
  info "Host: ${ES_HOST}"
  info "Repository: ${SNAPSHOT_REPO}"

  wait_for_es
  register_repo

  local snapshots_json
  snapshots_json="$(fetch_snapshots_json)"

  info "Available snapshots:"
  # error if empty deferred to pick_latest_snapshot
  jq -r '(.snapshots // [])[]?.snapshot' <<<"$snapshots_json" | sed 's/^/  - /' || true

  if [[ -z "${SNAPSHOT_NAME}" ]]; then
    SNAPSHOT_NAME="$(pick_latest_snapshot "$snapshots_json")"
    info "Using latest snapshot: ${SNAPSHOT_NAME}"
  else
    info "Using provided snapshot: ${SNAPSHOT_NAME}"
  fi

  delete_indices_if_present
  restore_snapshot "${SNAPSHOT_NAME}"

  info "Verifying restored indices:"
  curl_es "${ES_HOST}/_cat/indices/${RESTORE_INDICES}?v" || true

  info "Snapshot restored successfully!"
}

main "$@"
