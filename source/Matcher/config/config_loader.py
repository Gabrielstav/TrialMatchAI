from __future__ import annotations

import json
import os
from typing import Any, Dict

DEFAULT_CONFIG_PATH = "Matcher/config/config.json"


def load_config(config_path: str | None = None) -> Dict[str, Any]:
    """Load configuration from a JSON file.

    Config path resolution:
    1. Explicit config_path argument
    2. TRIALMATCHAI_CONFIG environment variable
    3. Default: Matcher/config/config.json
    """
    if config_path is None:
        config_path = os.environ.get("TRIALMATCHAI_CONFIG", DEFAULT_CONFIG_PATH)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)
