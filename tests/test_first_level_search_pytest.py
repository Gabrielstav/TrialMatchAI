from Matcher.pipeline.trial_search.first_level_search import ClinicalTrialSearch


class DummyES:
    def search(self, index, body):
        if body.get("track_total_hits") is False:
            return {"hits": {"max_score": 2.0}}
        return {"hits": {"hits": [{"_source": {"nct_id": "N1"}, "_score": 1.0}]}}


def test_search_trials_bm25_returns_hits():
    search = ClinicalTrialSearch(
        es_client=DummyES(),
        embedder=None,
        index_name="index",
        bio_med_ner=None,
    )
    trials, scores = search.search_trials(
        condition="lung cancer",
        age_input="all",
        sex="ALL",
        size=5,
        synonyms=["lung carcinoma"],
        search_mode="bm25",
    )
    assert len(trials) == 1
    assert scores == [1.0]
