from Matcher.pipeline.trial_search.first_level_search import ClinicalTrialSearch


class DummyES:
    def search(self, *args, **kwargs):
        return {"hits": {"hits": [], "max_score": 1.0}}


def test_first_level_query_bm25():
    search = ClinicalTrialSearch(
        es_client=DummyES(),
        embedder=None,
        index_name="index",
        bio_med_ner=None,
    )
    query = search.create_query(
        synonyms=["lung cancer"],
        embeddings={},
        age=45,
        sex="ALL",
        overall_status="Recruiting",
        max_text_score=1.0,
        vector_score_threshold=0.5,
        pre_selected_nct_ids=None,
        other_conditions=None,
        search_mode="bm25",
    )
    assert "bool" in query
    assert "should" in query["bool"]
    assert "filter" in query["bool"]


def test_first_level_query_vector():
    search = ClinicalTrialSearch(
        es_client=DummyES(),
        embedder=None,
        index_name="index",
        bio_med_ner=None,
    )
    embeddings = {"lung cancer": [0.1, 0.2], "smoking": [0.3, 0.4]}
    query = search.create_query(
        synonyms=["lung cancer"],
        embeddings=embeddings,
        age=60,
        sex="MALE",
        overall_status=None,
        max_text_score=1.0,
        vector_score_threshold=0.2,
        pre_selected_nct_ids=None,
        other_conditions=["smoking"],
        search_mode="vector",
    )
    assert "script_score" in query
    params = query["script_score"]["script"]["params"]
    assert len(params["query_vectors"]) == 1
    assert len(params["other_condition_vectors"]) == 1
