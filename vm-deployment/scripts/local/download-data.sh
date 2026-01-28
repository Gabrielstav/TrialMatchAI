#!/usr/bin/env bash
set -euo pipefail

#=== Download TrialMatchAI data from Zenodo ===
# Robust download with resume support

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
DATA_DIR="$PROJECT_ROOT/data"

#=== URLs ===#
DATA_URL="https://zenodo.org/records/15516900/files/processed_trials.tar.gz?download=1"
RESOURCES_URL="https://zenodo.org/records/15516900/files/resources.tar.gz?download=1"
MODELS_URL="https://zenodo.org/records/15516900/files/models.tar.gz?download=1"
CRITERIA_BASE_URL="https://zenodo.org/records/15516900/files"

#=== COLORS ===#
GREEN='\033[0;32m'
NC='\033[0m'
info() { echo -e "${GREEN}[INFO]${NC} $*"; }

#=== DOWNLOAD FUNCTION WITH RETRY ===#
download_file() {
    local url="$1"
    local output="$2"
    local max_retries=3
    local retry=0

    if [ -f "$output" ]; then
        # Check if file seems complete (basic size check - at least 10MB)
        local size=$(stat -f%z "$output" 2>/dev/null || stat -c%s "$output" 2>/dev/null || echo 0)
        if [ "$size" -gt 10000000 ]; then
            info "$output already exists (${size} bytes), skipping..."
            return 0
        else
            info "Removing incomplete $output..."
            rm -f "$output"
        fi
    fi

    while [ $retry -lt $max_retries ]; do
        info "Downloading $output (attempt $((retry + 1))/$max_retries)..."
        # No resume (-C -) since Zenodo doesn't support it
        if curl -L --retry 3 --retry-delay 10 --connect-timeout 30 -o "$output" "$url"; then
            info "Downloaded $output successfully!"
            return 0
        fi
        retry=$((retry + 1))
        rm -f "$output"  # Remove partial file
        sleep 10
    done
    echo "[ERROR] Failed to download $output after $max_retries attempts"
    return 1
}

#=== MAIN ===#
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

info "Starting downloads to $DATA_DIR"
info "This will take a while (~10GB total)..."
echo ""

# Download main archives
download_file "$DATA_URL" "processed_trials.tar.gz"
download_file "$RESOURCES_URL" "resources.tar.gz"
download_file "$MODELS_URL" "models.tar.gz"

# Download criteria chunks
for i in $(seq 0 5); do
    chunk="criteria_part_${i}.zip"
    download_file "${CRITERIA_BASE_URL}/${chunk}?download=1" "$chunk"
done

info "All downloads complete!"
echo ""

# Extract files
info "Extracting processed_trials..."
tar -xzf processed_trials.tar.gz

info "Extracting resources to source/Parser..."
mkdir -p "$PROJECT_ROOT/source/Parser"
tar -xzf resources.tar.gz -C "$PROJECT_ROOT/source/Parser"

info "Extracting models..."
mkdir -p "$PROJECT_ROOT/models"
tar -xzf models.tar.gz -C "$PROJECT_ROOT/models"

info "Extracting criteria chunks..."
mkdir -p processed_criteria
for i in $(seq 0 5); do
    unzip -q -o "criteria_part_${i}.zip" -d processed_criteria
done

info "Cleaning up archives..."
rm -f processed_trials.tar.gz resources.tar.gz models.tar.gz criteria_part_*.zip

info "Done! Data is ready in $DATA_DIR"