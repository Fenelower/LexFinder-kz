"""Build the dense index (BGE-M3 article embeddings) once, offline.
Usage: python scripts/build_index.py --data data/lexfinder_labour_code.csv --out index
Optional: DENSE_MODEL_REVISION=<hf commit hash> to pin the exact model snapshot."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lexfinder.engine import build_index  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lexfinder_labour_code.csv")
    ap.add_argument("--out", default="index")
    a = ap.parse_args()
    print(json.dumps(build_index(a.data, a.out), ensure_ascii=False, indent=2))
