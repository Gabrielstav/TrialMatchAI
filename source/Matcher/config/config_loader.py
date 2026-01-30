from __future__ import annotations

import json
import os
from typing import Any, Dict

from Matcher.config.settings import TrialMatchSettings, apply_env_overrides
from Matcher.utils.logging_config import setup_logging

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None


logger = setup_logging(__name__)


def load_config(config_path: str = "Matcher/config/config.json") -> Dict[str, Any]:
    """Load and validate configuration from a JSON file with env overrides."""
    if load_dotenv:
        load_dotenv()
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = json.load(f)
    raw = apply_env_overrides(raw)
    settings = TrialMatchSettings.model_validate(raw)
    cfg = settings.to_dict()
    if cfg.get("elasticsearch", {}).get("password") in {"", "CHANGE_ME"}:
        logger.warning(
            "Elasticsearch password is not set. Use TRIALMATCHAI_ES_PASSWORD to supply it."
        )
    return cfg
