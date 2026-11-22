"""Quality + latency report for the loaded model -> performance.json (numbers for the 'Performance' slide).
Usage: python scripts/benchmark.py --data data/lexfinder_labour_code.csv --index index --repeats 5"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lexfinder import MODEL_VERSION, LexFinder  # noqa: E402
from lexfinder.golden_queries import QUERIES, SPLIT_SEED  # noqa: E402


def metrics(engine, queries):
    h1, r5, mrr = [], [], []
    for q, rel in queries:
        order = [r["article_number"] for r in engine.search(q, 20)["results"]]
        ranks = [order.index(a) + 1 for a in rel if a in order]
        rank = min(ranks) if ranks else 10 ** 6
        h1.append(rank == 1); r5.append(len(set(rel) & set(order[:5])) / len(rel)); mrr.append(1 / rank if rank <= 5 else 0)
    return {"n": len(queries), "hit@1": round(float(np.mean(h1)), 3), "recall@5": round(float(np.mean(r5)), 3), "mrr@5": round(float(np.mean(mrr)), 3)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lexfinder_labour_code.csv")
    ap.add_argument("--index", default="index")
    ap.add_argument("--dense", default="auto")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="performance.json")
    a = ap.parse_args()

    t0 = time.perf_counter(); eng = LexFinder.load(a.data, a.index, a.dense); load_s = time.perf_counter() - t0
    eng.search(QUERIES[0][0])                                    # warm-up
    lat = []
    for _ in range(a.repeats):
        for q, _rel in QUERIES:
            t = time.perf_counter(); eng.search(q, 5); lat.append((time.perf_counter() - t) * 1000)
    report = {"model_version": MODEL_VERSION, "mode": eng.mode, "platform": platform.platform(), "load_time_s": round(load_s, 1),
              "latency_ms": {"p50": round(float(np.percentile(lat, 50)), 1), "p95": round(float(np.percentile(lat, 95)), 1),
                             "mean": round(float(np.mean(lat)), 1), "n_requests": len(lat)},
              "quality_all_34_queries": metrics(eng, QUERIES)}
    try:
        from sklearn.model_selection import train_test_split
        _, test = train_test_split(QUERIES, test_size=0.30, random_state=SPLIT_SEED, shuffle=True)
        report["quality_test_11_queries"] = metrics(eng, test)
    except ImportError:
        report["quality_test_11_queries"] = "scikit-learn not installed"
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
