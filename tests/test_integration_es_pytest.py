import os

import pytest
from elasticsearch import Elasticsearch

from Matcher.config.config_loader import load_config


@pytest.mark.integration
def test_elasticsearch_ping():
    if os.getenv("TRIALMATCHAI_RUN_INTEGRATION") != "1":
        pytest.skip("Set TRIALMATCHAI_RUN_INTEGRATION=1 to enable integration tests.")

    cfg = load_config("Matcher/config/config.json")
    es_cfg = cfg["elasticsearch"]
    client = Elasticsearch(
        hosts=[es_cfg["host"]],
        ca_certs=cfg["paths"]["docker_certs"],
        basic_auth=(es_cfg["username"], es_cfg["password"]),
        request_timeout=es_cfg["request_timeout"],
        retry_on_timeout=es_cfg["retry_on_timeout"],
    )
    assert client.ping()
