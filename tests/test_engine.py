import pytest

from lexfinder import MODEL_VERSION, QueryError
from lexfinder.engine import normalize_query, stem_tokens


def test_ranking_obvious_queries(engine):
    assert engine.search("ежегодный оплачиваемый отпуск")["results"][0]["article_number"] == "2"
    assert engine.search("работа в ночное время оплата")["results"][0]["article_number"] == "4"
    assert engine.search("как уволиться по собственному желанию")["results"][0]["article_number"] == "3"


def test_response_schema_and_version(engine):
    r = engine.search("дистанционная работа", top_k=3)
    assert r["model_version"] == MODEL_VERSION and r["mode"] == "bm25"
    assert len(r["results"]) == 3
    assert [h["rank"] for h in r["results"]] == [1, 2, 3]
    assert {"article_number", "title", "chapter", "score", "snippet"} <= set(r["results"][0])
    scores = [h["score"] for h in r["results"]]
    assert scores == sorted(scores, reverse=True)


def test_top_k_is_capped_by_article_count(engine):
    assert len(engine.search("работа", top_k=20)["results"]) == 6


@pytest.mark.parametrize("bad", ["", "   ", "ab", "?!?!?!", "я" * 501, None, 123])
def test_invalid_queries_rejected(engine, bad):
    with pytest.raises(QueryError):
        engine.search(bad)


@pytest.mark.parametrize("bad_k", [0, -1, 21, "5", 2.5, True])
def test_invalid_top_k_rejected(engine, bad_k):
    with pytest.raises(QueryError):
        engine.search("отпуск", top_k=bad_k)


def test_query_normalization():
    assert normalize_query("  Как   оплачивается \n работа?  ") == "Как оплачивается работа?"
    assert stem_tokens("Отпуска") == ["отпус"]


def test_query_without_lexical_match_is_flagged(engine):
    r = engine.search("квантовая хромодинамика")
    assert r["no_lexical_match"] is True and len(r["results"]) == 5     # still returns a ranked list


def test_info_describes_model(engine):
    info = engine.info()
    assert info["model_version"] == MODEL_VERSION and info["n_articles"] == 6
    assert info["config"]["dense_model"] == "BAAI/bge-m3"
