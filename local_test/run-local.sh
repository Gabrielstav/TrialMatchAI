#!/usr/bin/env bash
set -euo pipefail

# local testing script for arm64 with test dataset (120 trials, ~45MB)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# config
ELASTIC_PASSWORD="${ELASTIC_PASSWORD:-trialmatchai}"
ES_HOST="http://localhost:9200"
DOCKER_DIR="$SCRIPT_DIR/docker"

# local models run on MPS
DEFAULT_MODEL="vm-deployment/models/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_MODEL_NAME="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
GEMMA_MODEL="vm-deployment/models/gemma-2-2b-it"
GEMMA_MODEL_NAME="google/gemma-2-2b-it"

USE_GEMMA=false
USE_BM25=false
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--gemma" ]]; then
        USE_GEMMA=true
    elif [[ "$arg" == "--bm25" ]]; then
        USE_BM25=true
    else
        ARGS+=("$arg")
    fi
done
set -- "${ARGS[@]}"

if $USE_GEMMA; then
    LOCAL_MODEL="$GEMMA_MODEL"
    LOCAL_MODEL_NAME="$GEMMA_MODEL_NAME"
else
    LOCAL_MODEL="$DEFAULT_MODEL"
    LOCAL_MODEL_NAME="$DEFAULT_MODEL_NAME"
fi

# select config
if $USE_GEMMA && $USE_BM25; then
    LOCAL_CONFIG="source/Matcher/config/config.local.gemma.bm25.json"
elif $USE_GEMMA; then
    LOCAL_CONFIG="source/Matcher/config/config.local.gemma.json"
elif $USE_BM25; then
    LOCAL_CONFIG="source/Matcher/config/config.local.bm25.json"
else
    LOCAL_CONFIG="source/Matcher/config/config.local.json"
fi

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

info() { echo -e "${GREEN}[INFO]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
step() { echo -e "${BLUE}[STEP]${NC} $*"; }

wait_for_es() {
    info "Waiting for Elasticsearch to be ready..."
    local max_wait=120
    local waited=0
    until curl -s -u "elastic:$ELASTIC_PASSWORD" "$ES_HOST/_cluster/health" 2>/dev/null | grep -q '"status"'; do
        if [ $waited -ge $max_wait ]; then
            error "Elasticsearch did not start within ${max_wait}s"
            return 1
        fi
        echo -n "."
        sleep 2
        waited=$((waited + 2))
    done
    echo ""
    info "Elasticsearch is ready!"
}

check_es_running() {
    curl -s -u "elastic:$ELASTIC_PASSWORD" "$ES_HOST/_cluster/health" 2>/dev/null | grep -q '"status"'
}

check_docker_running() {
    if ! docker info > /dev/null 2>&1; then
        error "Docker is not running. Please start Docker Desktop first."
        return 1
    fi
    return 0
}

check_model_available() {
    info "Checking if local model is available: $LOCAL_MODEL"
    cd "$PROJECT_ROOT"

    if [ -f "$LOCAL_MODEL/config.json" ]; then
        info "Model found at $LOCAL_MODEL"
        return 0
    else
        warn "Model not found at $LOCAL_MODEL"
        return 1
    fi
}

download_model() {
    # models should already be in vm-deployment/models/
    # kept for compatibility but just checks the path
    info "Checking model at: $LOCAL_MODEL"
    cd "$PROJECT_ROOT"

    if [ -d "$LOCAL_MODEL" ]; then
        info "Model already exists at $LOCAL_MODEL"
        return 0
    else
        error "Model not found at $LOCAL_MODEL"
        error "You need to download it from HuggingFace: $LOCAL_MODEL_NAME"
        error "Run: huggingface-cli download $LOCAL_MODEL_NAME --local-dir $LOCAL_MODEL"
        return 1
    fi
}

setup_venv() {
    cd "$PROJECT_ROOT"

    # check for uv first
    if command -v uv &> /dev/null; then
        info "Installing dependencies with uv..."
        uv sync
    elif [ -f ".venv/bin/activate" ]; then
        info "Activating existing venv and checking dependencies..."
        source .venv/bin/activate
        pip install -q -e .
    else
        warn "No .venv found and uv not available."
        warn "Please run: uv sync  OR  python -m venv .venv && pip install -e ."
        return 1
    fi
}

# commands
case "${1:-help}" in
    setup)
        step "Setting up local development environment..."

        step "1/3: Checking dependencies..."
        cd "$PROJECT_ROOT"
        setup_venv

        step "2/3: Checking Docker..."
        check_docker_running || exit 1

        step "3/3: Checking/downloading model..."
        if ! check_model_available; then
            read -p "Download model '$LOCAL_MODEL' (~6GB)? [y/N] " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                download_model
            else
                warn "Skipping model download. You can run './run-local.sh download-model' later."
            fi
        fi

        info "Setup complete! Run './run-local.sh test' to run the full pipeline."
        ;;

    download-model)
        download_model
        ;;

    up|start)
        info "Starting local Elasticsearch..."

        check_docker_running || exit 1

        cd "$DOCKER_DIR"
        ELASTIC_PASSWORD="$ELASTIC_PASSWORD" docker-compose up -d

        wait_for_es

        info "Elasticsearch is ready at $ES_HOST"
        info ""
        info "Next step: Build test indices with: $0 build-index"
        ;;

    down|stop)
        info "Stopping local Elasticsearch..."
        cd "$DOCKER_DIR"
        docker-compose down
        info "Elasticsearch stopped."
        ;;

    status)
        info "Checking status..."
        echo ""

        # Docker
        if check_docker_running 2>/dev/null; then
            echo -e "Docker:        ${GREEN}Running${NC}"
        else
            echo -e "Docker:        ${RED}Not running${NC}"
        fi

        # ES
        if check_es_running; then
            echo -e "Elasticsearch: ${GREEN}Running${NC}"
            health=$(curl -s -u "elastic:$ELASTIC_PASSWORD" "$ES_HOST/_cluster/health" 2>/dev/null | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
            echo -e "ES Health:     ${GREEN}$health${NC}"
        else
            echo -e "Elasticsearch: ${YELLOW}Not running${NC}"
        fi

        # model
        if check_model_available 2>/dev/null; then
            echo -e "Local Model:   ${GREEN}Cached${NC} ($LOCAL_MODEL)"
        else
            echo -e "Local Model:   ${YELLOW}Not cached${NC} ($LOCAL_MODEL)"
        fi

        # venv
        cd "$PROJECT_ROOT"
        if [ -f ".venv/bin/activate" ]; then
            echo -e "Virtual Env:   ${GREEN}Found${NC}"
        else
            echo -e "Virtual Env:   ${YELLOW}Not found${NC}"
        fi

        echo ""
        if check_es_running; then
            info "ES Indices:"
            curl -s -u "elastic:$ELASTIC_PASSWORD" "$ES_HOST/_cat/indices?v"
        fi
        ;;

    run)
        info "Running matcher pipeline locally..."

        # check if ES is running
        if ! check_es_running; then
            error "Elasticsearch is not running. Start it first with: $0 up"
            exit 1
        fi

        cd "$PROJECT_ROOT"

        # check model is available
        if ! check_model_available 2>/dev/null; then
            error "Model not found at $LOCAL_MODEL"
            error "Run: huggingface-cli download $LOCAL_MODEL_NAME --local-dir $LOCAL_MODEL"
            exit 1
        fi

        # activate venv if using pip-based setup
        if [ -z "${VIRTUAL_ENV:-}" ] && [ -f ".venv/bin/activate" ]; then
            info "Activating virtual environment..."
            source .venv/bin/activate
        fi

        # use local config (MPS device, HF backend)
        info "Using local config with model: $LOCAL_MODEL_NAME"
        info "Config: $LOCAL_CONFIG"

        # prefer uv run if available
        if command -v uv &> /dev/null; then
            TRIALMATCHAI_CONFIG="$LOCAL_CONFIG" uv run python -m source.Matcher.main
        else
            TRIALMATCHAI_CONFIG="$LOCAL_CONFIG" python -m source.Matcher.main
        fi
        ;;

    test)
        info "Running full local test (ES + build-index + pipeline)..."

        # start ES if not running
        if ! check_es_running; then
            "$0" up
        fi

        # build test indices
        "$0" build-index

        # reconstruct flags to pass through to run command
        RUN_ARGS=(run)
        $USE_GEMMA && RUN_ARGS=(--gemma "${RUN_ARGS[@]}")
        $USE_BM25 && RUN_ARGS=(--bm25 "${RUN_ARGS[@]}")

        # run pipeline with flags
        "$0" "${RUN_ARGS[@]}"
        ;;

    build-index)
        info "Building test indices from local_test/test_data/ (120 trials, ~45MB)..."

        # check if ES is running
        if ! check_es_running; then
            error "Elasticsearch is not running. Start it first with: $0 up"
            exit 1
        fi

        cd "$PROJECT_ROOT"

        # check test data exists
        if [ ! -d "local_test/test_data/processed_trials" ]; then
            error "Test data not found at local_test/test_data/processed_trials"
            error "Make sure the local_test/test_data directory is populated."
            exit 1
        fi

        # activate venv if needed
        if [ -z "${VIRTUAL_ENV:-}" ] && [ -f ".venv/bin/activate" ]; then
            source .venv/bin/activate
        fi

        # use indexing scripts from utils/Indexer/
        info "Indexing trials..."
        if command -v uv &> /dev/null; then
            uv run python utils/Indexer/index_trials.py \
                --config "$LOCAL_CONFIG" \
                --processed-folder local_test/test_data/processed_trials \
                --index-name clinical_trials
        else
            python utils/Indexer/index_trials.py \
                --config "$LOCAL_CONFIG" \
                --processed-folder local_test/test_data/processed_trials \
                --index-name clinical_trials
        fi

        info "Indexing criteria..."
        if command -v uv &> /dev/null; then
            uv run python utils/Indexer/index_criteria.py \
                --config "$LOCAL_CONFIG" \
                --processed-folder local_test/test_data/processed_criteria \
                --index-name eligibility_criteria \
                --processed-file local_test/processed_ids.txt
        else
            python utils/Indexer/index_criteria.py \
                --config "$LOCAL_CONFIG" \
                --processed-folder local_test/test_data/processed_criteria \
                --index-name eligibility_criteria \
                --processed-file local_test/processed_ids.txt
        fi

        info "Test indices built successfully!"
        ;;

    clean)
        info "Cleaning up..."
        cd "$DOCKER_DIR"
        docker-compose down -v 2>/dev/null || true
        rm -f "$PROJECT_ROOT/local_test/processed_ids.txt"
        info "Cleaned up Docker volumes and indexer state"
        ;;

    help|*)
        cat << 'HELP'
Local development helper for TrialMatchAI

Usage: ./run-local.sh [--gemma] [--bm25] <command>

Commands:
  setup             Initial setup: install deps, check Docker, download model
  download-model    Download the local testing model
  up, start         Start Elasticsearch (no data loaded)
  down, stop        Stop Elasticsearch
  status            Check status of all services and dependencies
  build-index       Build test indices from local_test/test_data/ (120 trials, ~45MB)
  run               Run the matcher pipeline (requires ES + indices)
  test              Full test: start ES + build-index + run pipeline
  clean             Remove Docker volumes

Options:
  --gemma         Use Gemma-2-2B (~5GB) instead of TinyLlama (~2GB)
  --bm25          Use BM25-only search (skip vector embeddings, faster)

Quick start (local test data):
  ./run-local.sh setup    # First time: install deps, download model
  ./run-local.sh test     # Run full pipeline (ES + build-index + run)

Manual workflow:
  1. ./run-local.sh up          # Start ES
  2. ./run-local.sh build-index # Build indices from local_test/test_data/ (120 trials)
  3. ./run-local.sh run         # Run pipeline
  4. ./run-local.sh down        # Stop ES when done

Models:
  Default: TinyLlama-1.1B-Chat (~2GB, faster, lower memory)
  Optional: Gemma-2-2B-it (~5GB, better quality) with --gemma flag

Search modes:
  Default: Hybrid (BM25 + vector embeddings)
  Optional: BM25-only with --bm25 flag (faster, no embedding model needed)

Examples:
  ./run-local.sh run              # TinyLlama + hybrid search
  ./run-local.sh --bm25 run       # TinyLlama + BM25-only (fastest)
  ./run-local.sh --gemma run      # Gemma + hybrid search
  ./run-local.sh --gemma --bm25 run  # Gemma + BM25-only

Configuration:
  - Uses MPS (Apple Silicon) for inference
  - NER is disabled for local testing

Requirements:
  - Docker Desktop running
  - ~8GB RAM for TinyLlama, ~16GB for Gemma
HELP
        ;;
esac
