"""Lightweight monitoring: counters, latency percentiles, quality proxies, JSONL request log.

No ground truth is available in production, so quality is tracked with proxies:
- share of queries with no lexical match (query words absent from every article),
- distribution of the top-1 score.
Warning thresholds below are initial values, not calibrated on real traffic.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

WINDOW = 1000
THRESHOLDS = {"p95_latency_ms": 2000.0, "no_lexical_match_rate": 0.30, "error_rate": 0.05}


class Monitor:
    def __init__(self, log_dir=None, log_queries_text=None):
        self.started = time.time()
        self.lock = threading.Lock()
        self.requests = self.rejected = self.errors = 0
        self.latency = deque(maxlen=WINDOW)
        self.top_score = deque(maxlen=WINDOW)
        self.no_match = deque(maxlen=WINDOW)
        self.log_dir = Path(log_dir or os.getenv("LOG_DIR", "logs"))
        # by default the query text is NOT stored (legal questions may contain personal data)
        self.log_text = (os.getenv("LOG_QUERIES") == "1") if log_queries_text is None else log_queries_text
        self._log_ok = True
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            self._log_ok = False

    def _write(self, record: dict):
        if not self._log_ok:
            return
        try:
            with open(self.log_dir / "requests.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            self._log_ok = False

    def record_success(self, query: str, response: dict, top_k: int):
        top = response["results"][0]["score"] if response["results"] else 0.0
        with self.lock:
            self.requests += 1
            self.latency.append(response["latency_ms"])
            self.top_score.append(top)
            self.no_match.append(1 if response["no_lexical_match"] else 0)
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": "ok",
               "mode": response["mode"], "model_version": response["model_version"],
               "top_k": top_k, "query_chars": len(query), "query_sha1": hashlib.sha1(query.encode()).hexdigest()[:12],
               "latency_ms": response["latency_ms"], "top1_article": response["results"][0]["article_number"] if response["results"] else None,
               "top1_score": top, "no_lexical_match": response["no_lexical_match"]}
        if self.log_text:
            rec["query"] = query
        self._write(rec)

    def record_rejected(self, reason: str):
        with self.lock:
            self.requests += 1
            self.rejected += 1
        self._write({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": "rejected", "reason": reason})

    def record_error(self, error: str):
        with self.lock:
            self.requests += 1
            self.errors += 1
        self._write({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": "error", "error": error})

    def snapshot(self, mode: str, model_version: str) -> dict:
        with self.lock:
            lat, top, nm = list(self.latency), list(self.top_score), list(self.no_match)
            total, rej, err = self.requests, self.rejected, self.errors
        p = (lambda q: round(float(np.percentile(lat, q)), 2)) if lat else (lambda q: None)
        snap = {
            "uptime_s": round(time.time() - self.started, 1), "mode": mode, "model_version": model_version,
            "requests_total": total, "rejected_total": rej, "errors_total": err,
            "window_size": len(lat),
            "latency_ms": {"p50": p(50), "p95": p(95), "max": round(max(lat), 2) if lat else None},
            "top1_score_mean": round(float(np.mean(top)), 4) if top else None,
            "no_lexical_match_rate": round(float(np.mean(nm)), 4) if nm else None,
            "error_rate": round(err / total, 4) if total else 0.0,
        }
        flags = []
        if snap["latency_ms"]["p95"] and snap["latency_ms"]["p95"] > THRESHOLDS["p95_latency_ms"]:
            flags.append("high_p95_latency")
        if snap["no_lexical_match_rate"] and len(nm) >= 20 and snap["no_lexical_match_rate"] > THRESHOLDS["no_lexical_match_rate"]:
            flags.append("many_queries_without_lexical_match (possible topic/data drift)")
        if total >= 20 and snap["error_rate"] > THRESHOLDS["error_rate"]:
            flags.append("high_error_rate")
        if mode == "bm25":
            flags.append("running_in_bm25_fallback_mode")
        snap["warnings"] = flags
        return snap
