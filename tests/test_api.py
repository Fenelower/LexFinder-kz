import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from lexfinder.api import create_app  # noqa: E402
from lexfinder.monitoring import Monitor  # noqa: E402


@pytest.fixture()
def client(engine, tmp_path):
    return TestClient(create_app(engine, Monitor(log_dir=tmp_path)))


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["mode"] == "bm25" and r["degraded"] is True


def test_search_ok(client):
    r = client.post("/search", json={"query": "ежегодный оплачиваемый отпуск", "top_k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["results"][0]["article_number"] == "2" and len(body["results"]) == 2


@pytest.mark.parametrize("payload", [{"query": ""}, {"query": "?!?!"}, {"query": "отпуск", "top_k": 0},
                                     {"query": "отпуск", "top_k": 99}, {}, {"query": 5}])
def test_search_validation_errors_are_422(client, payload):
    assert client.post("/search", json=payload).status_code == 422


def test_model_info_and_ui(client):
    assert client.get("/model-info").json()["model_name"] == "lexfinder-kz-hybrid"
    page = client.get("/")
    assert page.status_code == 200 and "LexFinder" in page.text


def test_metrics_and_log_written(client, tmp_path):
    client.post("/search", json={"query": "дистанционная работа"})
    client.post("/search", json={"query": ""})
    m = client.get("/metrics").json()
    assert m["requests_total"] == 2 and m["rejected_total"] == 1 and m["errors_total"] == 0
    assert m["latency_ms"]["p50"] is not None
    assert "running_in_bm25_fallback_mode" in m["warnings"]
    log = (tmp_path / "requests.jsonl").read_text(encoding="utf-8")
    assert "query_sha1" in log and "дистанционная" not in log       # query text is not logged by default


def test_internal_error_returns_500_and_is_counted(engine, tmp_path):
    class Broken:
        mode, fallback_reason, df = "bm25", None, engine.df
        def search(self, *a, **k): raise RuntimeError("boom")
        def info(self): return {}
    c = TestClient(create_app(Broken(), Monitor(log_dir=tmp_path)))
    assert c.post("/search", json={"query": "отпуск"}).status_code == 500
    assert c.get("/metrics").json()["errors_total"] == 1
