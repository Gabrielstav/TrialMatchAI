#!/bin/bash
#SBATCH --job-name=trialmatchai
#SBATCH --output=trialmatchai_%j.out
#SBATCH --error=trialmatchai_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=accel
#SBATCH --account=p1753_tsd

# TrialMatchAI SLURM job:
# runs full pipeline: Elasticsearch + TrialMatchAI matcher

module purge
set -o errexit
set -o nounset
set -o pipefail
set -x  # debug: trace commands

# logging
ts()   { date '+%Y-%m-%d %H:%M:%S'; }
info() { echo "[$(ts)] [INFO] $*"; }
warn() { echo "[$(ts)] [WARN] $*"; }
error(){ echo "[$(ts)] [ERROR] $*" >&2; }

# config (override with env vars)
BASE_DIR="${BASE_DIR:-/tsd/p1753/cluster/trialmatch_ai}"

APPTAINER_DIR="$BASE_DIR/apptainer"
CONFIG_DIR="$BASE_DIR/config"
SCRIPTS_DIR="$BASE_DIR/scripts"

MODELS_DIR="${MODELS_DIR:-$BASE_DIR/models}"
INPUT_DATA_DIR="${DATA_DIR:-$BASE_DIR/test_data}"
TRIALS_JSONS_DIR="${TRIALS_JSONS_DIR:-$BASE_DIR/trials/processed_trials}"
RESULTS_DIR="${RESULTS_DIR:-$BASE_DIR/results}"
CACHE_DIR="${CACHE_DIR:-$BASE_DIR/cache}"

ES_DATA_DIR="${ES_DATA_DIR:-$BASE_DIR/esdata}"
ES_SNAPSHOTS_DIR="${ES_SNAPSHOTS_DIR:-$BASE_DIR/snapshots}"
ES_LOGS_DIR="${ES_LOGS_DIR:-$BASE_DIR/logs}"

ES_SIF="$APPTAINER_DIR/elasticsearch.sif"
APP_SIF="$APPTAINER_DIR/trialmatchai.sif"

CONFIG_HPC="$CONFIG_DIR/config.hpc.json"
RESTORE_SCRIPT="$SCRIPTS_DIR/restore-snapshot.sh"

# elasticsearch settings (currently not using user/passwrod in apptainer)
ELASTIC_PASSWORD="${ELASTIC_PASSWORD:-trialmatchai}"
export ES_HOST="http://localhost:9200"
export ES_USER="elastic"
export ES_PASSWORD="$ELASTIC_PASSWORD"

# validation helpers
VALIDATION_FAILED=0

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    error "Missing file: $path"
    VALIDATION_FAILED=1
  else
    info "  [OK] $path"
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    error "Missing directory: $path"
    VALIDATION_FAILED=1
  else
    info "  [OK] $path"
  fi
}

dir_empty_warn() {
  local path="$1"
  [[ -d "$path" ]] && [[ -z "$(ls -A "$path" 2>/dev/null)" ]] && warn "Directory is empty: $path"
}

has_any_json() {
  local dir="$1"
  # avoids SIGPIPE under pipefail by using find -quit then grep
  find "$dir" -maxdepth 1 -name "*.json" -print -quit 2>/dev/null | grep -q .
}

# cleanup stops ES
ES_PID_FILE="$ES_LOGS_DIR/elasticsearch.pid"

cleanup() {
  info "Cleaning up..."
  if [[ -f "$ES_PID_FILE" ]]; then
    local pid
    pid="$(cat "$ES_PID_FILE")"
    if kill -0 "$pid" 2>/dev/null; then
      info "Stopping Elasticsearch (PID: $pid)..."
      kill "$pid" || true
      sleep 5
    fi
  fi
}
trap cleanup EXIT

# header & env info
info "Starting TrialMatchAI SLURM job"
info "Job ID: ${SLURM_JOB_ID:-unknown}"
info "Node: ${SLURM_NODELIST:-unknown}"
info "GPUs: ${CUDA_VISIBLE_DEVICES:-not set}"

info "NVIDIA Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
info "CUDA Version: $(nvidia-smi --query-gpu=cuda_version --format=csv,noheader 2>/dev/null || echo 'N/A')"

# pre-run validation
info "Running pre-run validation..."
require_file "$ES_SIF"
require_file "$APP_SIF"
require_file "$CONFIG_HPC"
require_file "$RESTORE_SCRIPT"

require_dir "$MODELS_DIR"
require_dir "$INPUT_DATA_DIR"
require_dir "$TRIALS_JSONS_DIR"

dir_empty_warn "$MODELS_DIR"

if [[ -d "$TRIALS_JSONS_DIR" ]]; then
  if has_any_json "$TRIALS_JSONS_DIR"; then
    info "  Found trial JSON files in $TRIALS_JSONS_DIR"
  else
    warn "No JSON files found in trials directory: $TRIALS_JSONS_DIR"
  fi
fi

if [[ "$VALIDATION_FAILED" -ne 0 ]]; then
  error "Pre-run validation failed. Fix issues above and retry."
  exit 1
fi
info "Pre-run validation passed."

# prepare output dirs
mkdir -p "$RESULTS_DIR" "$CACHE_DIR" "$ES_DATA_DIR" "$ES_SNAPSHOTS_DIR" "$ES_LOGS_DIR"

ES_LOG_FILE="$ES_LOGS_DIR/elasticsearch.log"

# step 1: start Elasticsearch
info "Starting Elasticsearch..."

apptainer exec \
  --writable-tmpfs \
  --env "ELASTIC_PASSWORD=$ELASTIC_PASSWORD" \
  --env "ES_JAVA_OPTS=-Xms4g -Xmx4g" \
  --bind "$ES_DATA_DIR:/usr/share/elasticsearch/data" \
  --bind "$ES_SNAPSHOTS_DIR:/usr/share/elasticsearch/snapshots" \
  --bind "$ES_LOGS_DIR:/usr/share/elasticsearch/logs" \
  "$ES_SIF" \
  /usr/share/elasticsearch/bin/elasticsearch \
    -Ediscovery.type=single-node \
    -Ecluster.name=trialmatchai \
    -Expack.security.enabled=false \
    -Epath.repo=/usr/share/elasticsearch/snapshots \
    >"$ES_LOG_FILE" 2>&1 &

ES_PID=$!
echo "$ES_PID" > "$ES_PID_FILE"
info "Elasticsearch started with PID: $ES_PID"
info "Elasticsearch log: $ES_LOG_FILE"

# wait for ES
info "Waiting for Elasticsearch to be ready..."
MAX_WAIT=180
SLEEP_SECS=5
WAITED=0

until curl -s  "$ES_HOST/_cluster/health" 2>/dev/null | grep -q '"status"'; do
  if [[ "$WAITED" -ge "$MAX_WAIT" ]]; then
    error "Elasticsearch did not start within ${MAX_WAIT}s"
    cat "$ES_LOG_FILE" || true
    exit 1
  fi
  sleep "$SLEEP_SECS"
  WAITED=$((WAITED + SLEEP_SECS))
done
info "Elasticsearch is ready."

# step 2: Restore snapshot if needed
INDEX_COUNT="$(curl -s "$ES_HOST/_cat/indices?h=index" 2>/dev/null | wc -l)"
if [[ "$INDEX_COUNT" -lt 2 ]]; then
  info "Indices not found. Restoring from snapshot..."
  if [[ -d "$ES_SNAPSHOTS_DIR" ]] && [[ -n "$(ls -A "$ES_SNAPSHOTS_DIR" 2>/dev/null)" ]]; then
    bash "$RESTORE_SCRIPT"
  else
    warn "No snapshots found in $ES_SNAPSHOTS_DIR. Skipping restore."
  fi
else
  info "Indices already exist. Skipping snapshot restore."
fi

# step 3: run TrialMatchAI
info "Running TrialMatchAI pipeline..."

# reduce CUDA allocator fragmentation (helps reserved memory return to driver, reduces memory)
export PYTORCH_ALLOC_CONF=expandable_segments:True

# app loads config from /app/source/Matcher/config/config.json inside container,
# bind the HPC config to override if needed.
apptainer run --nv \
  --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" \
  --env "TRIALMATCHAI_CONFIG=/app/source/Matcher/config/config.json" \
  --bind "$MODELS_DIR:/mnt/models:ro" \
  --bind "$INPUT_DATA_DIR:/mnt/data:ro" \
  --bind "$TRIALS_JSONS_DIR:/mnt/data/trials_jsons:ro" \
  --bind "$RESULTS_DIR:/mnt/results" \
  --bind "$CACHE_DIR:/mnt/cache" \
  --bind "$CONFIG_HPC:/app/source/Matcher/config/config.json:ro" \
  "$APP_SIF"

# step 4: verify output
info "Pipeline completed. Verifying output..."

if [[ -d "$RESULTS_DIR" ]]; then
  RESULT_COUNT="$(find "$RESULTS_DIR" -name "ranked_trials.json" | wc -l)"
  if [[ "$RESULT_COUNT" -gt 0 ]]; then
    info "SUCCESS: Found $RESULT_COUNT patient result(s)"
    info "Output files (up to 20):"
    find "$RESULTS_DIR" -type f -name "*.json" | head -20
  else
    warn "No ranked_trials.json found in results directory"
  fi
else
  error "Results directory not found: $RESULTS_DIR"
fi

info "Results saved to: $RESULTS_DIR"
