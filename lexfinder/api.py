"""FastAPI service for LexFinder KZ.

Run:  uvicorn lexfinder.api:create_app --factory --host 0.0.0.0 --port 8000
Env:  LEXFINDER_DATA (csv, default data/lexfinder_labour_code.csv), LEXFINDER_INDEX (default index),
      LEXFINDER_DENSE = auto | on | off (default auto), LOG_DIR, LOG_QUERIES=1
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .engine import CONFIG, MODEL_VERSION, LexFinder, QueryError
from .monitoring import Monitor

log = logging.getLogger("lexfinder.api")
STATIC = Path(__file__).parent / "static"


class SearchRequest(BaseModel):
    query: str = Field(..., description="Вопрос пользователя на русском языке (3-500 символов)")
    top_k: int = Field(5, ge=1, le=CONFIG["max_top_k"], description="Сколько статей вернуть")


class Hit(BaseModel):
    rank: int
    article_number: str
    title: str
    chapter: str
    score: float
    snippet: str


class SearchResponse(BaseModel):
    query: str
    results: List[Hit]
    mode: str
    model_version: str
    no_lexical_match: bool
    latency_ms: float


def create_app(engine: Optional[LexFinder] = None, monitor: Optional[Monitor] = None) -> FastAPI:
    if engine is None:
        engine = LexFinder.load(
            os.getenv("LEXFINDER_DATA", "data/lexfinder_labour_code.csv"),
            os.getenv("LEXFINDER_INDEX", "index"),
            dense=os.getenv("LEXFINDER_DENSE", "auto"),
        )
    monitor = monitor or Monitor()
    app = FastAPI(title="LexFinder KZ API", version=MODEL_VERSION,
                  description="Поиск статей Трудового кодекса РК по вопросу пользователя.")
    app.state.engine, app.state.monitor = engine, monitor

    @app.get("/", include_in_schema=False)
    def ui():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health():
        return {"status": "ok", "mode": engine.mode, "model_version": MODEL_VERSION,
                "n_articles": len(engine.df), "degraded": engine.mode == "bm25",
                "fallback_reason": engine.fallback_reason}

    @app.get("/model-info")
    def model_info():
        return engine.info()

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest):
        try:
            resp = engine.search(req.query, req.top_k)
        except QueryError as exc:
            monitor.record_rejected(str(exc))
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:                               # noqa: BLE001
            log.exception("search failed")
            monitor.record_error(f"{type(exc).__name__}: {exc}")
            raise HTTPException(status_code=500, detail="Внутренняя ошибка сервиса.")
        monitor.record_success(resp["query"], resp, req.top_k)
        return resp

    @app.get("/metrics")
    def metrics():
        return monitor.snapshot(engine.mode, MODEL_VERSION)

    return app
