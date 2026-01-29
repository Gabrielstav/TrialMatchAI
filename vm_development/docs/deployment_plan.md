# TrialMatchAI HPC Deployment Plan

## Overview

Deploy TrialMatchAI on an air-gapped HPC cluster using Apptainer containers orchestrated by Slurm.

### Target Environment (from hpc_docs.txt)
- **Architecture**: amd64 Linux (Colossus HPC)
- **Container Runtime**: Apptainer (installed as system package on compute nodes)
- **Job Scheduler**: Slurm
- **Network**: No internet access
- **Storage**: NFS shared filesystem (`/cluster`, `/gpfs`)
- **GPU**: Nvidia Tesla A100 (4 per node) or V100 (2 per node), via `--partition=accel --gres=gpu:N`

### Constraints
- Cannot build containers on VM/HPC
- Cannot convert Docker → Apptainer on VM
- All containers must be pre-built locally (amd64) and uploaded
- File transfer via `tacl` CLI

---

## Architecture

Two Apptainer containers running in the same Slurm job, communicating via localhost:

```
┌────────────────────────────────── Single Slurm Job ───────────────────────────────────┐
│  Compute Node (GPU)                                                                   │
│                                                                                       │
│  ┌─────────────────────────┐              ┌─────────────────────────────────────┐     │
│  │  elasticsearch.sif      │   localhost  │  trialmatchai.sif                   │     │
│  │  (ES 8.13.4 runtime)    │◄────:9200───►│  (Python + CUDA + source code)      │     │
│  │  536MB                  │              │  ~12GB                              │     │
│  │                         │              │                                     │     │
│  │  Started in background  │              │  Runs main pipeline with --nv       │     │
│  │  by sbatch script       │              │  for GPU access                     │     │
│  └───────────┬─────────────┘              └───────────────────┬─────────────────┘     │
│              │                                                │                       │
│  ┌───────────▼────────────────────────────────────────────────▼────────────────────┐  │
│  │                         NFS Bind Mounts                                         │  │
│  │                                                                                 │  │
│  │  ~/trialmatchai/                                                                │  │
│  │  ├── elasticsearch.sif       # Container image                                  │  │
│  │  ├── trialmatchai.sif        # Container image                                  │  │
│  │  ├── esdata/                 # ES runtime data (persistent between jobs)        │  │
│  │  ├── snapshots/              # 53GB indexed trial data (read-only, just FILES)  │  │
│  │  ├── models/                 # LLM models from HuggingFace (~35GB)              │  │
│  │  ├── input/                  # Patient phenopackets to process                  │  │
│  │  ├── output/                 # Pipeline results                                 │  │
│  │  └── logs/                   # Job logs                                         │  │
│  └─────────────────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

### Key Insight: The Snapshot is Just Files

**The ES snapshot (`docker/snapshots/`) is NOT a Docker image** - it's regular filesystem files that ES uses to restore indexed data. No conversion needed!

```
docker/snapshots/           ← These are just data files, not a container
├── index-0
├── index.latest
├── indices/                ← The actual index data
├── meta-*.dat
└── snap-*.dat
```

These files get bind-mounted into the ES Apptainer at runtime:
```bash
apptainer run --bind ./snapshots:/usr/share/elasticsearch/snapshots elasticsearch.sif
```

### Inter-Container Communication

Apptainer shares the host network namespace by default (per hpc_docs.txt line 627). Both containers see `localhost:9200`:

1. ES container binds to `localhost:9200`
2. App container connects to `localhost:9200`
3. No special port configuration needed

---

## Completed Work

| Task | Status | Output |
|------|--------|--------|
| Elasticsearch indexing | ✓ | 109,270 trials, 2.1M criteria |
| BGE-M3 embedding search | ✓ | Tested locally |
| Gemma-2 reranker | ✓ | Tested with example phenopacket |
| ES Snapshot | ✓ | `docker/snapshots/` (53GB) |
| ES Apptainer | ✓ | `apptainer/elasticsearch.sif` (536MB, amd64) |
| App Apptainer | ⚠ | `apptainer/trialmatchai.sif` - **needs rebuild** |
| Config loader update | ✓ | Added `TRIALMATCHAI_CONFIG` env var support |
| HPC config | ✓ | `config/config.hpc.json` |
| Local configs | ✓ | `config.local.json`, `config.local.gemma.json`, etc. |
| Local testing script | ✓ | `scripts/run-local.sh` (full dev workflow) |
| Slurm script | ✓ | `slurm/trialmatchai.sbatch` (with pre-flight validation) |
| Build script | ✓ | `apptainer/build.sh` (reproducible builds) |
| NER documentation | ✓ | `docs/NER_docs.txt` (deferred, optional) |
| Documentation | ✓ | `README.md`, this file |

---

## Remaining Work

### Phase 1: Rebuild TrialMatchAI Apptainer

**Status**: Needs rebuild to include latest code changes (config loader, NER toggle, etc.)

**Build command** (from project root):
```bash
cd /path/to/TrialMatchAI
docker run --rm --platform linux/amd64 \
    -v $(pwd):/build \
    --privileged \
    quay.io/singularity/singularity:v3.11.4 \
    build /build/vm-deployment/apptainer/trialmatchai.sif \
    /build/vm-deployment/apptainer/trialmatchai.def
```

**Tasks**:
- [ ] Rebuild `trialmatchai.sif` for amd64 (~12GB)
- [ ] Split large SIF for transfer (see "Large File Transfer" section)
- [ ] Verify container runs locally

### Phase 2: Prepare Models ✓ COMPLETE

**Strategy**: Keep models on NFS, bind-mount into container.

| Model | Size | Notes |
|-------|------|-------|
| BAAI/bge-m3 | 4.3GB | Embeddings |
| google/gemma-2-2b-it | 4.9GB | Reranker base |
| microsoft/phi-4 | 27GB | Reasoning base |
| finetuned/ | 6.4GB | LoRA adapters (gemma2 + phi_reasoning) |
| biosyn-sapbert-bc5cdr-disease | 418MB | Neural normalizer (disease/symptom) |
| biosyn-sapbert-bc5cdr-chemical | 418MB | Neural normalizer (drug) |
| biosyn-sapbert-bc2gn | 419MB | Neural normalizer (gene) |
| neural_norm_caches/ | 4.5GB | FAISS indices for normalization |

**Total models directory: ~48GB**

**Tasks**:
- [x] Download models to `models/` directory (HuggingFace format)
- [x] Download neural normalizer models (biosyn-sapbert-*)
- [x] Add neural_norm_caches (FAISS indices)
- [x] Add config support for neural normalizer paths (backwards compatible)

### Phase 3: Configuration for HPC ✓ INVESTIGATED

The app container's entry point is:
```python
python -m source.Matcher.main "$@"
```

**Investigation Results:**

| Question | Answer |
|----------|--------|
| How does it find models? | HuggingFace `from_pretrained()` with paths from `config.json` |
| How does it connect to ES? | Hardcoded from `config.json`, no env var support |
| What config format? | JSON only (`source/Matcher/config/config.json`) |
| Environment variables? | Almost none - only `VLLM_ALLOW_LONG_MAX_MODEL_LEN` |

**Key Config Settings (from `source/Matcher/config/config.json`):**
```json
{
  "elasticsearch": {
    "host": "https://localhost:9200",
    "username": "elastic",
    "password": "...",
    "index_trials": "clinical_trials",
    "index_trials_eligibility": "eligibility_criteria"
  },
  "model": {
    "base_model": "microsoft/phi-4",
    "cot_adapter_path": "models/finetuned_phi_reasoning",
    "reranker_model_path": "google/gemma-2-2b-it",
    "reranker_adapter_path": "models/finetuned_gemma2"
  },
  "paths": {
    "patients_dir": "../example/",
    "output_dir": "../results/"
  }
}
```

**HPC Deployment Strategy (NO CODE CHANGES REQUIRED):**

Since the code reads config from a hardcoded path (`Matcher/config/config.json`), we can use **bind mounts** to inject our HPC config:

```bash
apptainer run --nv \
    --bind ./config/config.hpc.json:/app/source/Matcher/config/config.json:ro \
    trialmatchai.sif
```

**Tasks:**
- [x] Investigate 
- [x] Create `config/config.hpc.json` for HPC environment
- [x] Update sbatch script to bind-mount config

### Phase 4: Transfer & Test

**Files to upload via `tacl`**:
| File | Size |
|------|------|
| `elasticsearch.sif` | 536MB |
| `trialmatchai.sif` | ~5-10GB |
| `snapshots/` | 53GB |
| `models/` | ~35GB |
| Scripts + configs | <1MB |
| **Total** | **~95GB** |

**VM Directory Structure**:
```
~/trialmatchai/
├── elasticsearch.sif
├── trialmatchai.sif
├── esdata/                # Created at runtime
├── snapshots/             # Upload from docker/snapshots/
├── models/                # Upload HuggingFace models
├── input/                 # Patient data
├── output/                # Results
├── logs/
└── scripts/
    ├── trialmatchai.sbatch
    └── restore-snapshot.sh
```

---

## Slurm Job Workflow

The existing `slurm/trialmatchai.sbatch` already implements this:

```
1. Start ES in background
   └── Bind: esdata/, snapshots/
   └── Wait for ES ready (up to 180s)

2. Restore snapshot (first run only)
   └── Check if indices exist
   └── Run restore-snapshot.sh if needed

3. Run TrialMatchAI pipeline
   └── --nv for GPU
   └── Bind: models/, data/, results/, cache/

4. Cleanup on exit
   └── Kill ES process
```

**Slurm parameters** (from hpc_docs.txt):
```bash
#SBATCH --account=YourProject      # Required
#SBATCH --time=4:00:00             # Required
#SBATCH --mem-per-cpu=8G           # ~64GB total with 8 CPUs
#SBATCH --partition=accel          # GPU partition
#SBATCH --gres=gpu:1               # Request 1 GPU
```

---

## Codebase Changes Made

The following changes were made to the main codebase to support flexible deployment:

### Config Loader Update

**File**: `source/Matcher/config/config_loader.py`

The config loader now supports environment variable configuration:
```python
def load_config(config_path: str | None = None) -> Dict[str, Any]:
    """Load configuration from a JSON file.

    Config path resolution:
    1. Explicit config_path argument
    2. TRIALMATCHAI_CONFIG environment variable
    3. Default: Matcher/config/config.json
    """
    if config_path is None:
        config_path = os.environ.get("TRIALMATCHAI_CONFIG", DEFAULT_CONFIG_PATH)
```

**Benefits**:
- Default behavior unchanged (backward compatible)
- Local testing can use `TRIALMATCHAI_CONFIG=config.local.json`
- HPC deployment can still use bind mounts OR env var

### NER Toggle

**File**: `source/Matcher/config/config.json` (and all config variants)

Added `bio_med_ner.enabled` flag to control NER services:
```json
{
  "bio_med_ner": {
    "enabled": false,  // Set to true to enable NER services
    ...
  }
}
```

When disabled, the pipeline skips synonym extraction and runs without NER services.
See `docs/NER_docs.txt` for full NER documentation.

---

## Local Testing Infrastructure

### run-local.sh Script

**File**: `vm-deployment/scripts/run-local.sh`

Complete local development script for macOS with the following commands:

| Command | Description |
|---------|-------------|
| `setup` | Install deps, check Docker, download model |
| `up` / `start` | Start Elasticsearch and restore snapshot |
| `down` / `stop` | Stop Elasticsearch |
| `status` | Check status of all services |
| `run` | Run the matcher pipeline |
| `test` | Full test: start ES + run pipeline |
| `clean` | Remove Docker volumes |

**Flags**:
- `--gemma` - Use Gemma-2-2B (~5GB) instead of TinyLlama (~2GB)
- `--bm25` - Use BM25-only search (skip vector embeddings, faster)

**Example usage**:
```bash
./run-local.sh setup           # First time setup
./run-local.sh test            # Run full pipeline (default: TinyLlama + hybrid)
./run-local.sh --gemma run     # Run with Gemma model
./run-local.sh --bm25 run      # Run with BM25-only search (fastest)
```

### Local Config Files

| Config | Model | Search Mode | Use Case |
|--------|-------|-------------|----------|
| `config.local.json` | TinyLlama | Hybrid | Quick testing |
| `config.local.bm25.json` | TinyLlama | BM25-only | Fastest testing |
| `config.local.gemma.json` | Gemma-2-2B | Hybrid | Better quality |
| `config.local.gemma.bm25.json` | Gemma-2-2B | BM25-only | Quality + fast |

All local configs:
- Use MPS device (Apple Silicon)
- Disable NER (`bio_med_ner.enabled: false`)
- Point to `vm-deployment/models/` for local models
- Use `vm-deployment/trials_jsons_lite/` (subset of trials)

---

## Codebase Configuration Details

This section documents the configuration system. Config can be injected via bind mounts OR environment variable.

### Config File Location
- **Production**: `source/Matcher/config/config.json`
- **HPC**: `vm-deployment/config/config.hpc.json` (bind-mounted)
- **Local dev**: `source/Matcher/config/config.local*.json`
- **Loader**: `source/Matcher/config/config_loader.py`

Config resolution order:
1. Explicit `config_path` argument
2. `TRIALMATCHAI_CONFIG` environment variable
3. Default: `Matcher/config/config.json`

### Model Path Resolution

Models are loaded via HuggingFace `from_pretrained()`. The config specifies:

| Config Key | Example Value | Description |
|------------|---------------|-------------|
| `model.base_model` | `microsoft/phi-4` | Main reasoning LLM |
| `model.cot_adapter_path` | `models/finetuned_phi_reasoning` | LoRA adapter for reasoning |
| `model.reranker_model_path` | `google/gemma-2-2b-it` | Reranker base model |
| `model.reranker_adapter_path` | `models/finetuned_gemma2` | LoRA adapter for reranker |
| `embedder.model_name` | `BAAI/bge-m3` | Embedding model |

**Important**: Model paths can be:
- HuggingFace model IDs (e.g., `microsoft/phi-4`) - downloaded to HF cache
- Local paths (e.g., `models/finetuned_gemma2`) - relative to working directory

For HPC, we need to:
1. Pre-download models to NFS
2. Update config to use absolute paths pointing to bind-mounted locations

### Elasticsearch Connection

From `source/Matcher/main.py`:
```python
es_client = Elasticsearch(
    hosts=[config["elasticsearch"]["host"]],
    basic_auth=(config["elasticsearch"]["username"], config["elasticsearch"]["password"]),
    ca_certs=config["paths"]["docker_certs"],  # Only for HTTPS
)
```

For HPC (HTTP, no SSL):
```json
{
  "elasticsearch": {
    "host": "http://localhost:9200",
    "username": "elastic",
    "password": "trialmatchai",
    "verify_certs": false
  }
}
```

---

## NER Status

**Current Status**: DISABLED (deferred to post-deployment)

NER (Named Entity Recognition) provides synonym expansion for better search recall. It is **optional** - the pipeline works without it.

**Why disabled**:
- Requires 4 background services (~48GB memory)
- Requires Java JRE for gene/disease normalizers
- Complex startup sequence
- Not critical for phenopacket-based workflows

**Impact of disabling NER**:
- Pipeline still works normally
- Slightly reduced search recall (no synonym expansion)
- Simpler deployment and lower memory footprint

**To enable later**: See `docs/NER_docs.txt` for full documentation on:
- Required components and space (~17GB additional)
- Apptainer modifications (Java JRE, GLiNER model)
- Memory requirements (~48GB for NER services)
- Configuration changes

---

## Excluded from Container (Build Optimization)

To keep the container size under 10GB, the following are **excluded** via `.dockerignore`:

### Excluded Directories

| Directory | Size | Reason | Runtime Access |
|-----------|------|--------|----------------|
| `source/Parser/resources/` | 9.8GB | NER model data | Not needed (NER disabled) |
| `source/Parser/logs/` | 1.1MB | NER logs | Not needed |
| `source/Parser/input/` | 80KB | NER input | Not needed |
| `source/Parser/output/` | 72KB | NER output | Not needed |
| `vm-deployment/` | 104GB | Deployment files | Not code |
| `data/` | 72GB | Trial data | Bind-mounted |
| `models/` | 6.8GB | ML models | Bind-mounted |

### To Enable NER Later

1. **Edit `.dockerignore`** - Remove these lines:
   ```
   source/Parser/resources/
   source/Parser/logs/
   source/Parser/input/
   source/Parser/output/
   ```

2. **Edit `Dockerfile.multistage`** - Add NER packages:
   ```dockerfile
   RUN uv pip install --no-cache-dir \
       spacy gliner nltk medspacy rapidfuzz
   ```

3. **Edit `Dockerfile.multistage`** - Add Java JRE:
   ```dockerfile
   RUN apt-get update && apt-get install -y --no-install-recommends \
       openjdk-17-jre-headless \
       ...
   ```

4. **Rebuild** with `./build-multistage.sh`

5. **Update config** - Set `bio_med_ner.enabled: true`

**Note**: This will increase the container size by ~10-12GB and require ~48GB additional RAM at runtime.

---

## Large File Transfer

The `tacl` CLI may have file size limits. For large files like `trialmatchai.sif` (~12GB), use split/cat:

### Splitting (on local machine)
```bash
# Split into 2GB chunks
split -b 2G trialmatchai.sif trialmatchai.sif.part.

# Creates: trialmatchai.sif.part.aa, .ab, .ac, .ad, .ae, .af
```

### Transfer
```bash
tacl upload trialmatchai.sif.part.*
```

### Reassemble (on HPC)
```bash
cat trialmatchai.sif.part.* > trialmatchai.sif
rm trialmatchai.sif.part.*

# Verify integrity
sha256sum trialmatchai.sif
```

### Verify before splitting (local)
```bash
sha256sum trialmatchai.sif > trialmatchai.sif.sha256
# Upload this file too, then verify on HPC
```

---

## Pre-deployment Checklist

Before uploading to HPC, verify:

### Files Ready
- [ ] `elasticsearch.sif` exists (536MB)
- [ ] `trialmatchai.sif` rebuilt with latest code (~12GB)
- [ ] `models/` directory complete (~48GB)
- [ ] `snapshots/` directory complete (~53GB)
- [ ] `config/config.hpc.json` exists

### Configuration
- [ ] Update `--account=YOUR_PROJECT_HERE` in `trialmatchai.sbatch` with actual project ID
- [ ] Verify `BASE_DIR` path in sbatch script matches HPC directory structure
- [ ] Test phenopacket file in `data/` directory

### Large File Handling
- [ ] Split `trialmatchai.sif` if needed for transfer
- [ ] Generate checksums for verification

### Post-upload Verification
- [ ] Reassemble split files on HPC
- [ ] Verify checksums match
- [ ] Run `sbatch --test-only trialmatchai.sbatch` to validate script
- [ ] Submit test job with small phenopacket

---

## Fallback Options

If the recommended approach doesn't work:

1. **Models in container**: Bake models into `trialmatchai.sif` (~40GB image, slower to transfer but self-contained)

2. **Separate ES job**: Run ES in a dedicated job instead of background process (more complex, two jobs to manage)

3. **Data in container**: Bake snapshot into ES image (~54GB image, cannot update data)

---

## File Inventory

### Ready to Transfer (~113GB total)

```
vm-deployment/
├── apptainer/                          12GB
│   ├── elasticsearch.sif               536MB (amd64)
│   ├── trialmatchai.sif                11GB (amd64)
│   └── build.sh                        Build script (documented)
├── config/
│   └── config.hpc.json                 HPC-specific config (incl. neural_normalizer paths)
├── data/                               <1MB (test input)
│   └── phenopacket.json                Example patient phenopacket for testing
├── models/                             48GB
│   ├── bge-m3/                         4.3GB (embedding model)
│   ├── gemma-2-2b-it/                  4.9GB (reranker base)
│   ├── phi-4/                          27GB (reasoning base)
│   ├── finetuned/                      6.4GB (LoRA adapters)
│   ├── biosyn-sapbert-bc5cdr-disease/  418MB (neural normalizer - disease/symptom)
│   ├── biosyn-sapbert-bc5cdr-chemical/ 418MB (neural normalizer - drug)
│   ├── biosyn-sapbert-bc2gn/           419MB (neural normalizer - gene)
│   └── neural_norm_caches/             4.5GB (FAISS indices for normalization)
├── scripts/
│   ├── restore-snapshot.sh             ES snapshot restore
│   ├── test_search.py                  Local testing
│   └── test_models_local.py            Local testing
├── slurm/
│   └── trialmatchai.sbatch             Main job script (with output verification)
└── snapshots/                          53GB (ES indexed data)
    ├── index-0
    ├── index.latest
    ├── indices/
    ├── meta-*.dat
    └── snap-*.dat
```

### Transfer Summary

| Component | Size | Description |
|-----------|------|-------------|
| Apptainer images | 12GB | ES + App containers |
| Models | 48GB | HuggingFace models + adapters + neural normalizers |
| Snapshots | 53GB | Indexed clinical trial data |
| Data | <1MB | Test phenopacket for validation |
| Config/Scripts | <1MB | Configuration and slurm scripts |
| **Total** | **~113GB** | |

### Pipeline Output

For each patient phenopacket, the pipeline generates:
```
results/{patient_id}/
├── keywords.json              # Extracted keywords from phenopacket
├── nct_ids.txt                # First-level search results (trial IDs)
├── first_level_scores.json    # First-level ranking scores
├── top_trials.txt             # Second-level filtered trials
├── rag_output.json            # RAG processing status
├── ranked_trials.json         # FINAL OUTPUT: ranked trials with scores
├── {nct_id}.json              # Individual trial evaluation (inclusion/exclusion)
└── {nct_id}.txt               # Raw LLM response for each trial
```

---

## Commands Reference

### Build App Apptainer (from project root)
```bash
cd /path/to/TrialMatchAI
docker run --rm --platform linux/amd64 \
    -v $(pwd):/build \
    --privileged \
    quay.io/singularity/singularity:v3.11.4 \
    build /build/vm-deployment/apptainer/trialmatchai.sif \
    /build/vm-deployment/apptainer/trialmatchai.def
```

### Run ES on HPC
```bash
apptainer run \
    --bind ./esdata:/usr/share/elasticsearch/data \
    --bind ./snapshots:/usr/share/elasticsearch/snapshots \
    --env ELASTIC_PASSWORD=trialmatchai \
    --env "ES_JAVA_OPTS=-Xms4g -Xmx4g" \
    --env discovery.type=single-node \
    elasticsearch.sif
```

### Run App on HPC (with GPU)
```bash
apptainer run --nv \
    --bind ./models:/mnt/models:ro \
    --bind ./input:/mnt/data:ro \
    --bind ./output:/mnt/results \
    --bind ./cache:/mnt/cache \
    trialmatchai.sif
```

### Submit Slurm Job
```bash
sbatch trialmatchai.sbatch
```

---

## Next Steps

### Completed
1. ~~**Fix trialmatchai.def**~~ ✓ paths fixed for build context
2. ~~**Investigate config**~~ ✓ documented, added env var support
3. ~~**Create HPC config**~~ ✓ `config/config.hpc.json`
4. ~~**Download models**~~ ✓ 48GB including neural normalizers
5. ~~**Local testing infrastructure**~~ ✓ `run-local.sh` + local configs
6. ~~**NER documentation**~~ ✓ `docs/NER_docs.txt`

### Remaining
7. **Rebuild trialmatchai.sif** - Include latest code changes (~12GB)
8. **Split SIF for transfer** - 5GB chunks for web portal upload
9. **Update sbatch account** - Replace `YOUR_PROJECT_HERE` with actual project ID
10. **Upload to VM** - using web portal 
11. **Test on HPC** - Submit Slurm job with GPU