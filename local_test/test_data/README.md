# Test Dataset

This directory contains a small subset of the full clinical trials dataset for local testing and development.

## Contents

- **120 trials** in `processed_trials/` (~42 MB total)
  - 30 cardiac/CAD trials (matching the example patient)
  - 20 diabetes-related trials
  - 20 lipid/cholesterol trials
  - 20 other cardiac trials
  - 30 non-matching trials (control group)

- **~1,000 eligibility criteria** in `processed_criteria/`

- **Example patient** in `phenopacket/`

## Directory Structure

```
test_data/
├── processed_trials/       # Trial documents with embeddings (NCT*.json)
├── processed_criteria/   # Eligibility criteria with embeddings (NCT*/criteria*.json)
├── phenopacket/          # Example patient phenopacket
└── README.md
```

## Usage

Use the scripts in `local_test/` to build indices and run the pipeline:

```bash
# Start ES and build indices
./local_test/run-local.sh up
./local_test/run-local.sh build-index

# Run pipeline
./local_test/run-local.sh run
```

## Example Patient

The test dataset is designed to match the example patient in `phenopacket/phenopacket.json`:

- 58-year-old male
- Coronary artery disease (Stage III)
- Type 2 diabetes mellitus
- Hypercholesterolemia
- History of myocardial infarction with CABG
- LDLR mutation (familial hypercholesterolemia)

## Data Selection Criteria

Trials were selected based on keyword matching in condition, title, and eligibility criteria:

| Category | Keywords | Count |
|----------|----------|-------|
| Cardiac/CAD | coronary artery disease, cad, myocardial infarction, heart attack, cabg, bypass | 30 |
| Diabetes | type 2 diabetes, diabetes mellitus, t2dm, metformin, hba1c | 20 |
| Lipid | hypercholesterolemia, cholesterol, ldl, statin, lipid, atorvastatin | 20 |
| Other cardiac | cardiac, heart failure, cardiovascular, cardiomyopathy | 20 |
| Non-matching | (control group - unrelated trials) | 30 |

## Vector Embeddings

The processed documents include pre-computed embeddings:

- **Trials**: 1024-dim vectors for title, summary, condition, eligibility criteria (BAAI/bge-m3)
- **Criteria**: 1024-dim vectors for criterion text