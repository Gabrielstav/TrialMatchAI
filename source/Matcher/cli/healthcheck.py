from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

from elasticsearch import Elasticsearch

from Matcher.config.config_loader import load_config
from Matcher.services.elasticsearch_service import ensure_elasticsearch
from Matcher.utils.logging_config import setup_logging

logger = setup_logging(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="TrialMatchAI healthcheck")
    parser.add_argument(
        "--config",
        default="Matcher/config/config.json",
        help="Path to config.json",
    )
    parser.add_argument(
        "--start-es",
        action="store_true",
        help="Attempt to start Elasticsearch if unreachable",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    issues = 0

    _check_paths(config)

    es_cfg = config["elasticsearch"]
    es_client = Elasticsearch(
        hosts=[es_cfg["host"]],
        ca_certs=config["paths"]["docker_certs"],
        basic_auth=(es_cfg["username"], es_cfg["password"]),
        request_timeout=es_cfg["request_timeout"],
        retry_on_timeout=es_cfg["retry_on_timeout"],
    )

    if args.start_es and es_cfg.get("auto_start") is False:
        es_cfg["auto_start"] = True

    if not ensure_elasticsearch(es_client, config):
        logger.error("Elasticsearch healthcheck failed.")
        issues += 1
    else:
        logger.info("Elasticsearch reachable.")

    return 1 if issues else 0


def _check_paths(config: Dict[str, Any]) -> None:
    paths = config["paths"]
    for key in ["patients_dir", "output_dir", "trials_json_folder", "docker_certs"]:
        path = Path(paths[key])
        if not path.exists():
            logger.warning("Path not found: %s=%s", key, path)


if __name__ == "__main__":
    sys.exit(main())
