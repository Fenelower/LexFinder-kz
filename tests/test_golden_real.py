"""Regression test on the real dataset. Skipped unless data/lexfinder_labour_code.csv (and optionally
index/) is present. Thresholds come from the experiments notebook (BM25 Recall@5: dev 0.87, test 0.91;
hybrid: dev 1.00, test 1.00), lowered by a safety margin."""
import os
from pathlib import Path

import pytest

from lexfinder import LexFinder
from lexfinder.golden_queries import QUERIES

DATA = Path(os.getenv("LEXFINDER_DATA", "data/lexfinder_labour_code.csv"))
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="real dataset not present")


@pytest.fixture(scope="module")
def real_engine():
    return LexFinder.load(DATA, os.getenv("LEXFINDER_INDEX", "index"), dense=os.getenv("LEXFINDER_DENSE", "auto"))


def test_recall_at_5_on_labelled_queries(real_engine):
    hits = 0
    for q, rel in QUERIES:
        top5 = [r["article_number"] for r in real_engine.search(q, 5)["results"]]
        hits += int(bool(set(rel) & set(top5)))
    recall = hits / len(QUERIES)
    assert recall >= (0.95 if real_engine.mode == "hybrid" else 0.80), f"Recall@5={recall:.3f} in {real_engine.mode} mode"
