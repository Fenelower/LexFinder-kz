import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lexfinder import LexFinder  # noqa: E402

MINI = ROOT / "tests" / "data" / "mini_articles.csv"      # tiny SYNTHETIC fixture, not real law text


@pytest.fixture(scope="session")
def engine():
    return LexFinder.load(MINI, dense="off")             # BM25 mode: fast, no model download in CI
