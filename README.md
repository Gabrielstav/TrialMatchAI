# TrialMatchAI

<img src="img/logo.png" alt="Logo" align="right" width="200" height="200">

TrialMatchAI is an AI-driven system for matching patients to relevant clinical trials. It combines retrieval (BM25/vector search), NLP, and LLM-based reasoning to produce transparent, ranked trial recommendations.

## Disclaimer
This software is provided for research and informational purposes only. It is not medical advice and must not replace consultation with qualified healthcare professionals.

## Key Features
- AI-powered matching of patient profiles to clinical trial eligibility criteria.
- Two-stage retrieval (BM25 + vector) with optional reranking.
- Explainable recommendations and criterion-level insights.
- Scalable Elasticsearch-backed indexing.
- Modular pipelines for indexing, retrieval, and reasoning.

## Requirements
- OS: Linux or macOS
- Python: 3.8+
- Java: required for NER/normalization components
- Elasticsearch: Docker Compose or Apptainer
- GPU: NVIDIA recommended for large-scale processing
- Disk space: 100 GB+ for datasets and indices

## Quickstart (uv recommended)
1) Install dependencies
```bash
uv sync
```

2) Ensure Elasticsearch credentials are configured
- If using Apptainer, set `ELASTIC_PASSWORD` in `elasticsearch/.env`.
- For the runtime pipeline, set `TRIALMATCHAI_ES_PASSWORD` in your environment.

3) Start Elasticsearch (auto-start supported)
```bash
trialmatchai-healthcheck --config Matcher/config/config.json --start-es
```

4) Run the pipeline
```bash
python -m Matcher.main
```

Results are written under `results/`.

## One-time Provisioning (data + models + indexing)
You can run the all-in-one bootstrap script:
```bash
./setup.sh
```

Or run steps individually:
```bash
bash scripts/bootstrap_data.sh
bash scripts/start_es.sh
bash scripts/index_data.sh
```

## Package Installation (pip fallback)
```bash
pip install -e .
```

## Configuration Overrides
Use environment variables or a `.env` file to override defaults:
```bash
# Elasticsearch
TRIALMATCHAI_ES_HOST=https://localhost:9200
TRIALMATCHAI_ES_USERNAME=elastic
TRIALMATCHAI_ES_PASSWORD=YourNewPassword
TRIALMATCHAI_ES_AUTO_START=true
TRIALMATCHAI_ES_START_SCRIPT=elasticsearch/apptainer-run-es.sh
TRIALMATCHAI_ES_START_TIMEOUT=600

# Embedder
TRIALMATCHAI_EMBEDDER_MODEL_NAME=BAAI/bge-m3

# Logging
TRIALMATCHAI_LOG_LEVEL=INFO
TRIALMATCHAI_LOG_JSON=0
```

Note: `Matcher/config/config.json` ships with a placeholder password (`CHANGE_ME`). Use env vars for production.

## Healthcheck
```bash
trialmatchai-healthcheck --config Matcher/config/config.json --start-es
```

## Tests
```bash
uv run pytest
```

Integration tests (requires running ES):
```bash
TRIALMATCHAI_RUN_INTEGRATION=1 pytest -m integration
```

## Contributing
We welcome contributions. Please open an issue or submit a PR with tests.

## Support
- Email: abdallahmajd7@gmail.com
- DOI: https://doi.org/10.5281/zenodo.18329084
- arXiv: https://arxiv.org/abs/2505.08508
