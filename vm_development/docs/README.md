# TrialMatchAI VM Deployment

This directory contains everything needed to deploy TrialMatchAI's Elasticsearch backend to a VM/HPC environment using Apptainer (formerly Singularity).

## Overview

The deployment workflow:
1. **Local**: Index clinical trial data into Elasticsearch (Docker)
2. **Local**: Create a snapshot of the indexed data
3. **Local**: Build an Apptainer SIF image for amd64
4. **VM**: Upload SIF + snapshot, restore data, run ES

## Directory Structure

```
vm-deployment/
├── README.md                 # This file
├── apptainer/
│   ├── build.sh              # Build script for Apptainer images
│   ├── elasticsearch.def     # Apptainer definition for ES
│   ├── elasticsearch.sif     # Built Apptainer image (536MB, amd64) ← OUTPUT
│   └── trialmatchai.def      # Apptainer definition for main app
├── config/
│   └── config.local.json     # Local configuration overrides
├── docker/
│   ├── docker-compose.yml    # Local ES Docker setup
│   └── snapshots/            # ES snapshot data (53GB) ← OUTPUT
│       ├── index-0
│       ├── index.latest
│       ├── indices/          # Actual index data
│       ├── meta-*.dat
│       └── snap-*.dat
├── scripts/
│   ├── create-snapshot.sh    # Create ES snapshot
│   ├── restore-snapshot.sh   # Restore ES snapshot on VM
│   ├── run-local.sh          # Local development commands
│   ├── test_models_local.py  # Test LLM models locally
│   ├── test_search.py        # Test ES search functionality
│   └── ...
└── slurm/
    └── *.sbatch              # SLURM job scripts for HPC
```

## Output Files Created

| File | Size | Description |
|------|------|-------------|
| `apptainer/elasticsearch.sif` | 536MB | Apptainer image for ES 8.13.4 (amd64) |
| `docker/snapshots/` | 53GB | ES snapshot with indexed trial data |

---

## Local Testing Completed

### 1. Elasticsearch + Embeddings
- **Script**: `scripts/test_search.py`
- **Result**: PASS
- **Details**:
  - 109,270 clinical trials indexed
  - 2,131,540 eligibility criteria indexed
  - BM25 keyword search working
  - BGE-M3 vector embeddings working (~2GB)

### 2. Gemma-2 Reranker Model
- **Script**: `scripts/test_models_local.py`
- **Result**: PASS
- **Details**:
  - Gemma-2-2b-it model loads on Apple MPS (M4)
  - LoRA adapter (`models/models/finetuned_gemma2`) loads correctly
  - Inference produces meaningful eligibility evaluations
  - Memory usage: ~4GB in FP16

### 3. Phi-4 Reasoning Model
- **Result**: FAIL (expected on Mac)
- **Details**:
  - Phi-4 (14B params) requires ~28GB in FP16
  - Cannot run on 24GB MacBook
  - Must run on VM with adequate GPU memory (16GB+ with 4-bit quantization)

---

## Scripts Reference

### `scripts/test_search.py`
Tests Elasticsearch connectivity and search functionality without requiring LLMs.

```bash
python vm-deployment/scripts/test_search.py
```

**What it tests:**
- ES connection and cluster health
- Index document counts
- BM25 keyword search
- Vector similarity search (loads BGE-M3)
- Criteria lookup by trial ID

### `scripts/test_models_local.py`
Tests the Gemma-2 reranker model with real patient data.

```bash
python vm-deployment/scripts/test_models_local.py [options]
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--phenopacket` | `example/phenopacket.json` | Patient data file |
| `--num-trials` | 3 | Trials to evaluate |
| `--num-criteria` | 3 | Criteria per trial |
| `--device` | auto | `auto`, `mps`, `cpu`, `cuda` |
| `--dtype` | float16 | `float16`, `float32` |

**What it does:**
1. Loads patient data from phenopacket JSON
2. Searches ES for matching clinical trials
3. Loads Gemma-2 model with LoRA adapter
4. Evaluates patient info against trial criteria
5. Prints model responses (Yes/No with explanation)

### `scripts/create-snapshot.sh`
Creates an Elasticsearch snapshot from the running Docker instance.

```bash
SNAPSHOT_DIR=vm-deployment/docker/snapshots bash vm-deployment/scripts/create-snapshot.sh
```

**What it does:**
1. Registers a snapshot repository in ES
2. Creates a snapshot of `clinical_trials` and `eligibility_criteria` indices
3. Takes ~20 minutes for 52GB of data

### `scripts/restore-snapshot.sh`
Restores an ES snapshot on the VM.

```bash
./restore-snapshot.sh
```

**Environment variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `ES_HOST` | `http://localhost:9200` | ES endpoint |
| `ES_USER` | `elastic` | Username |
| `ES_PASSWORD` | `trialmatchai` | Password |
| `SNAPSHOT_NAME` | (latest) | Specific snapshot to restore |

**What it does:**
1. Waits for ES to be ready
2. Registers the snapshot repository (readonly)
3. Lists available snapshots
4. Deletes existing indices if present
5. Restores the snapshot
6. Verifies restoration

### `scripts/run-local.sh`
Convenience script for local Docker-based development.

```bash
./run-local.sh up      # Start ES Docker container
./run-local.sh down    # Stop ES Docker container
./run-local.sh status  # Check ES health status
./run-local.sh index   # Run data indexing
./run-local.sh run     # Run the full pipeline
```

### `apptainer/build.sh`
Builds Apptainer images for VM deployment.

```bash
cd vm-deployment/apptainer
BUILD_ES=true BUILD_APP=false bash build.sh
```

**Note:** On ARM Mac, the script uses Docker to pull the amd64 image and convert it. If Apptainer isn't installed locally, you can use Docker directly:

```bash
# Pull amd64 ES image and convert using Singularity in Docker
docker run --rm --platform linux/amd64 \
  -v $(pwd)/apptainer:/output \
  --privileged \
  quay.io/singularity/singularity:v3.11.4 \
  build /output/elasticsearch.sif docker://docker.elastic.co/elasticsearch/elasticsearch:8.13.4
```

---

## VM Deployment Instructions

### Files to Upload

| File | Size | Required |
|------|------|----------|
| `apptainer/elasticsearch.sif` | 536MB | Yes |
| `docker/snapshots/` (entire dir) | 53GB | Yes |
| `scripts/restore-snapshot.sh` | 3KB | Yes |

**Total: ~54GB**

### Step 1: Create Directories on VM
```bash
mkdir -p ~/trialmatchai/{esdata,snapshots,scripts}
```

### Step 2: Upload Files
Upload via your VM's GUI or available transfer method:
- `elasticsearch.sif` → `~/trialmatchai/`
- `snapshots/*` → `~/trialmatchai/snapshots/`
- `restore-snapshot.sh` → `~/trialmatchai/scripts/`

### Step 3: Start Elasticsearch
```bash
cd ~/trialmatchai

# Start ES with Apptainer (runs in foreground)
apptainer run \
    --bind ./esdata:/usr/share/elasticsearch/data \
    --bind ./snapshots:/usr/share/elasticsearch/snapshots \
    --env ELASTIC_PASSWORD=trialmatchai \
    --env path.repo=/usr/share/elasticsearch/snapshots \
    --env discovery.type=single-node \
    --env xpack.security.enabled=true \
    --env xpack.security.http.ssl.enabled=false \
    --env "ES_JAVA_OPTS=-Xms4g -Xmx4g" \
    elasticsearch.sif
```

**Note:** ES takes ~30 seconds to fully start. Watch for the "started" message in logs.

To run in background:
```bash
nohup apptainer run ... elasticsearch.sif > es.log 2>&1 &
```

### Step 4: Restore Snapshot
In a separate terminal:
```bash
cd ~/trialmatchai/scripts
chmod +x restore-snapshot.sh
./restore-snapshot.sh
```

### Step 5: Verify
```bash
curl -u elastic:trialmatchai http://localhost:9200/_cat/indices?v
```

**Expected output:**
```
health status index                pri rep docs.count store.size
green  open   clinical_trials        1   0     109270      8.8gb
green  open   eligibility_criteria   1   0    2131540     43.4gb
```

---

## Elasticsearch Details

### Indices

**clinical_trials**
- Documents: 109,270 clinical trials
- Fields: `nct_id`, `brief_title`, `official_title`, `brief_summary`, `conditions`, `eligibility`, etc.
- Size: ~8.8GB

**eligibility_criteria**
- Documents: 2,131,540 individual eligibility criteria
- Fields: `nct_id`, `criterion`, `criterion_type` (inclusion/exclusion)
- Size: ~43.4GB

### Credentials
- **Username**: `elastic`
- **Password**: `trialmatchai`

### Ports
- **HTTP API**: 9200
- **Transport**: 9300 (not needed for single-node)

---

## Memory Requirements

| Component | Local (Mac M4 24GB) | VM Recommended |
|-----------|---------------------|----------------|
| Elasticsearch | 4GB | 8GB+ |
| BGE-M3 Embedder | 2GB | 2GB |
| Gemma-2 Reranker | 4GB (FP16) | 4GB |
| Phi-4 Reasoning | N/A (too large) | 16GB+ GPU (4-bit) |

---

## Troubleshooting

### ES won't start
```bash
# Check the log output
# Common issues:
# - Insufficient memory: increase ES_JAVA_OPTS (-Xms/-Xmx)
# - Permission issues: ensure esdata/ and snapshots/ are writable
# - Port conflict: ensure 9200 isn't already in use
```

### Snapshot restore fails
```bash
# Verify snapshot files exist and are readable
ls -la snapshots/

# Check ES can access the directory
apptainer exec --bind ./snapshots:/usr/share/elasticsearch/snapshots \
    elasticsearch.sif ls /usr/share/elasticsearch/snapshots

# Manual restore via curl
curl -X POST -u elastic:trialmatchai \
    "http://localhost:9200/_snapshot/trialmatchai_backup/snapshot_20260112/_restore?wait_for_completion=true"
```

### Search returns no results
```bash
# Wait for ES to finish initializing after restore
# Check index health
curl -u elastic:trialmatchai localhost:9200/_cat/indices

# Check document count
curl -u elastic:trialmatchai localhost:9200/clinical_trials/_count
```

---

## Files NOT Included (Download Separately)

For the full TrialMatchAI pipeline, you also need:

1. **Base Models** (from HuggingFace, requires license acceptance):
   - `google/gemma-2-2b-it` - Reranker base model (~4GB)
   - `microsoft/phi-4` - Reasoning base model (~28GB)
   - `BAAI/bge-m3` - Embedding model (~2GB)

2. **LoRA Adapters** (in `models/models/`):
   - `finetuned_gemma2` - Reranker fine-tuned adapter
   - `finetuned_phi_reasoning` - Reasoning fine-tuned adapter

3. **Patient Data** (in `example/`):
   - `phenopacket.json` - Example patient for testing

---

## Version Information

| Component | Version |
|-----------|---------|
| Elasticsearch | 8.13.4 |
| Apptainer/Singularity | 3.11.4 |
| Python | 3.11+ |
| PyTorch | 2.x |

---

## Changelog

**2026-01-12**
- Initial VM deployment setup
- Local testing on MacBook Pro M4 (24GB)
- Built amd64 Apptainer image for ES
- Created 53GB snapshot of indexed trial data
- Verified Gemma-2 reranker works locally
- Target: amd64 Linux VM with Apptainer (no Docker)