# Local Testing

Quick local development environment for TrialMatchAI.

## Quick Start

```bash
# One-command test (starts ES, builds indices, runs pipeline)
./local_test/run-local.sh test
```

## Manual Workflow

```bash
# 1. Start Elasticsearch
./local_test/run-local.sh up

# 2. Build test indices (120 trials from test_data/)
./local_test/run-local.sh build-index

# 3. Run the pipeline
./local_test/run-local.sh run

# 4. Stop ES when done
./local_test/run-local.sh down
```

## Directory Structure

```
local_test/
├── run-local.sh            # Main orchestrator script
├── docker/
│   └── docker-compose.yml  # Docker Compose for local ES
├── results/                # Pipeline output (created on run)
├── test_data/              # Test dataset
│   ├── processed_trials/     # 120 trial documents
│   ├── processed_criteria/ # ~1000 eligibility criteria
│   └── phenopacket/        # Example patient
└── README.md
```

## Available Commands

| Command | Description |
|---------|-------------|
| `setup` | Initial setup: install deps, check Docker, download model |
| `up` | Start Elasticsearch |
| `down` | Stop Elasticsearch |
| `build-index` | Build test indices from test_data/ |
| `run` | Run the matcher pipeline |
| `test` | Full test: up + build-index + run |
| `status` | Check status of services |
| `clean` | Remove Docker volumes |

## Options

- `--gemma`: Use Gemma-2-2B (~5GB) instead of TinyLlama (~2GB)
- `--bm25`: Use BM25-only search (faster, no embedding model needed)

## Model Limitations

**TinyLlama (default)** is only suitable for verifying the pipeline runs end-to-end. It produces **invalid results** because it's too small (~1B params) to follow the complex prompts correctly. Instead of extracting actual medical keywords from the patient phenopacket, it copies the template/example from the prompt:

```json
{
  "main_conditions": ["PrimaryCondition", "Synonym1", "Synonym2", "..."],
  "other_conditions": ["AdditionalCondition1", "AdditionalCondition2", "..."]
}
```

This causes searches to fail (queries like "Synonym2" match nothing).

**For meaningful results**, use Gemma-2-2B with the `--gemma` flag:
```bash
./local_test/run-local.sh --gemma test
```
This takes ~1 hour with 24GB RAM running on full processed dataset,
running with subset for local testing takes ~5-10 minutes depending on hardware. 

| Model | RAM | Purpose |
|-------|-----|---------|
| TinyLlama | ~8GB | Verify pipeline runs (bogus output) |
| Gemma-2-2B | ~16GB | Meaningful results |

## Configuration

Local configs are in `source/Matcher/config/`:

- `config.local.json` - TinyLlama + hybrid search
- `config.local.gemma.json` - Gemma-2 + hybrid search
- `config.local.bm25.json` - TinyLlama + BM25 only
- `config.local.gemma.bm25.json` - Gemma-2 + BM25 only

## Test Data

The test dataset in `local_test/test_data/` contains:

- **120 trials** (~42 MB)
  - 30 cardiac/CAD trials (matching example patient)
  - 20 diabetes-related trials
  - 20 lipid/cholesterol trials
  - 20 other cardiac trials
  - 30 non-matching trials (control group)

- **~1,000 eligibility criteria**

The dataset is designed to match the example patient (58yo male with CAD, T2DM, hypercholesterolemia).

## Pipeline Flow

```
1. First-level search (trials index)
   └── Output: first_level_scores.json (ES scores)
   └── Searches trials by patient keywords (BM25 + vector)

2. Second-level search (criteria index)
   └── Retrieves eligibility CRITERIA for matched trials
   └── ~700+ individual criteria documents

3. LLM Reranker (the expensive step)
   └── Scores each (patient query, criterion) pair
   └── Assigns llm_score to each criterion

4. Aggregation (threshold 0.5)
   └── Groups criteria by trial
   └── Only keeps criteria with llm_score >= 0.5
   └── Aggregates to trial score

5. Top 1/3 selection
   └── Output: top_trials.txt

6. RAG evaluation
   └── Detailed eligibility evaluation per trial
   └── Output: NCT*.json, NCT*.txt, ranked_trials.json
```

**Note:** The 0.5 threshold applies to LLM reranker scores on individual **criteria**, not the first-level ES scores on trials. The `first_level_scores.json` contains ES relevance scores (different scale).

## Expected Results

With the test dataset (120 trials) and Gemma model, typical results:

| Stage | Count | Explanation |
|-------|-------|-------------|
| First-level search | ~29 trials | BM25 + vector search matches patient keywords |
| Second-level ranking | ~3 trials | LLM reranker filters criteria (score >= 0.5 threshold) |
| RAG evaluation | ~1 trial | Top 1/3 of ranked trials (`len // 3` formula) |

The small numbers are expected with the test subset. The pipeline takes the **top third** of trials at each stage to reduce noise, so:
- 3 trials from second-level → 3 // 3 = 1 trial for RAG

With the full dataset (~100K trials), you'd see proportionally more trials at each stage.

## Requirements

- Docker Desktop running
- ~8GB RAM for TinyLlama, ~16GB for Gemma
- Models in `vm-deployment/models/` (TinyLlama, Gemma, bge-m3)