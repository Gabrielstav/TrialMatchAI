#!/usr/bin/env bash
set -euo pipefail

# multi-stage build for TrialMatchAI:
# 1. builds slim Docker image for linux/amd64
# 2. exports image to docker-archive tar
# 3. converts docker-archive to Apptainer/Singularity .sif using singularity-in-docker

# helpers
if [[ -t 1 && -z ${NO_COLOR-} ]]; then
  GREEN=$'\033[0;32m'
  YELLOW=$'\033[1;33m'
  RED=$'\033[0;31m'
  NC=$'\033[0m'
else
  GREEN=""; YELLOW=""; RED=""; NC=""
fi

info() { printf '%s[INFO]%s %s\n' "$GREEN" "$NC" "$*"; }
warn() { printf '%s[WARN]%s %s\n' "$YELLOW" "$NC" "$*" >&2; }
die()  { printf '%s[ERR ]%s %s\n' "$RED" "$NC" "$*" >&2; exit 1; }

need_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"; }

START_SECONDS=$SECONDS
fmt_duration() {
  local s="$1"
  local h=$((s / 3600))
  local m=$(((s % 3600) / 60))
  local sec=$((s % 60))
  printf '%02d:%02d:%02d' "$h" "$m" "$sec"
}

# return available GB on the filesystem containing $1
available_gb() {
  local path="$1"
  if df -BG "$path" >/dev/null 2>&1; then
    df -BG "$path" | awk 'NR==2 {gsub(/G/,"",$4); print $4}'
  elif df -g "$path" >/dev/null 2>&1; then
    df -g "$path" | awk 'NR==2 {print $4}'
  else
    # fallback: df -k => KB, convert to GiB
    df -k "$path" | awk 'NR==2 {print int($4/1024/1024)}'
  fi
}

file_size_bytes() {
  local f="$1"
  stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null || die "Cannot stat file: $f"
}

prompt_continue() {
  local msg="$1"
  local reply=""
  read -r -p "$msg (y/N) " reply
  [[ "$reply" =~ ^[Yy]$ ]]
}

# parse args
DOCKER_NO_CACHE=0
if [[ "${1:-}" == "--no-cache" ]]; then
  DOCKER_NO_CACHE=1
  info "Building with --no-cache (full rebuild)"
elif [[ "${1:-}" != "" ]]; then
  die "Unknown argument: $1"
fi

# paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VM_DEV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$VM_DEV_DIR/.." && pwd)"

OUTPUT_DIR="$VM_DEV_DIR/deploy/apptainer"
DOCKER_IMAGE="trialmatchai:slim"
DOCKER_TAR="$SCRIPT_DIR/trialmatchai-docker.tar"
SIF_PATH="$OUTPUT_DIR/trialmatchai.sif"

SING_VOL="trialmatchai_sing_tmp"
SING_IMAGE="quay.io/singularity/singularity:v3.11.4"

cleanup() {
  # captures exit status
  local rc=$?

  info "Cleaning up temporary files..."
  rm -f "$DOCKER_TAR" || true

  local elapsed=$((SECONDS - START_SECONDS))
  if [[ $rc -eq 0 ]]; then
    info "Total build time: $(fmt_duration "$elapsed")"
  else
    warn "Total build time (failed): $(fmt_duration "$elapsed")"
  fi
}
trap cleanup EXIT

# preconditions
need_cmd docker
need_cmd awk
need_cmd bc

info "=== TrialMatchAI Multi-stage Build ==="
info "Project root: $PROJECT_ROOT"
printf '\n'

# need around 20GB
AVAILABLE_GB="$(available_gb "$SCRIPT_DIR")"
info "Available disk space: ${AVAILABLE_GB}GB"

if [[ "$AVAILABLE_GB" -lt 20 ]]; then
  warn "Less than 20GB available. Build may fail."
  warn "maybe run docker system prune -a"
  if ! prompt_continue "Continue anyway?"; then
    exit 1
  fi
fi

info "Step 1: Building Docker image..."

DOCKER_BUILD_ARGS=(
  --platform linux/amd64
  -f "$SCRIPT_DIR/Dockerfile.multistage"
  -t "$DOCKER_IMAGE"
  "$PROJECT_ROOT"
)
if (( DOCKER_NO_CACHE )); then
  DOCKER_BUILD_ARGS=( --no-cache "${DOCKER_BUILD_ARGS[@]}" )
fi

docker build "${DOCKER_BUILD_ARGS[@]}"

printf '\n'
info "Docker image size:"
docker images "$DOCKER_IMAGE" --format "Size: {{.Size}}"

printf '\n'
info "Step 2: Exporting Docker image to tar..."
docker save "$DOCKER_IMAGE" -o "$DOCKER_TAR"
info "Saved to: $DOCKER_TAR"
ls -lh "$DOCKER_TAR"

printf '\n'
info "Step 3: Converting to SIF inside Docker..."

# apptainer needs a real Linux filesystem for overlay and mount operations,
# using a Docker volume puts tmp/cache inside Dockers Linux VM
docker volume create "$SING_VOL" >/dev/null 2>&1 || true

mkdir -p "$OUTPUT_DIR"

DOCKER_RUN_ARGS=(
  run --rm
  --platform linux/amd64
  --privileged
  -v "$SCRIPT_DIR:/work"
  -v "$OUTPUT_DIR:/output"
  -v "$SING_VOL:/sing-tmp"
  -e "SINGULARITY_TMPDIR=/sing-tmp"
  -e "SINGULARITY_CACHEDIR=/sing-tmp/cache"
  "$SING_IMAGE"
  build --tmpdir /sing-tmp
  "/output/$(basename "$SIF_PATH")"
  "docker-archive:/work/$(basename "$DOCKER_TAR")"
)

docker "${DOCKER_RUN_ARGS[@]}"

# clean up volume after build
docker volume rm "$SING_VOL" >/dev/null 2>&1 || true

printf '\n'
info "=== Build Complete ==="
ls -lh "$SIF_PATH"

SIZE_BYTES="$(file_size_bytes "$SIF_PATH")"
SIZE_GB="$(echo "scale=2; $SIZE_BYTES / 1024 / 1024 / 1024" | bc)"

printf '\n'
info "Final size: ${SIZE_GB} GB"

if (( $(echo "$SIZE_GB < 10" | bc -l) )); then
  info "SUCCESS: Image is under 10GB - no splitting needed if using web portal to upload"
else
  info "Image is over 10GB - run ./scripts/split-large-file.sh to split for upload if using web portal"
fi
