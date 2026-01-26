from pathlib import Path

from Matcher.config.config_loader import load_config


def test_load_config_from_repo():
    config_path = Path(__file__).resolve().parents[1] / "source/Matcher/config/config.json"
    cfg = load_config(str(config_path))
    assert "elasticsearch" in cfg
    assert "embedder" in cfg
    assert "paths" in cfg
