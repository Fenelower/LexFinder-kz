"""LexFinder KZ final model: hybrid BM25 + BGE-M3 dense retrieval over articles of the
Labour Code of the Republic of Kazakhstan.

Final model = Hybrid BM25 + BGE-M3 dense (zero-shot), weight of the dense part tuned on dev (w = 0.6).
Scoring is identical to the experiments notebook (LexFinder_KZ_ASIK, section 6):
    score = w * z(dense) + (1 - w) * z(bm25),   z(x) = (x - mean) / (std + 1e-9)
The same code can run in BM25-only mode (no neural model) as an explicit fallback.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("lexfinder")

MODEL_NAME = "lexfinder-kz-hybrid"
MODEL_VERSION = "1.0.0"

CONFIG = {
    "dense_model": "BAAI/bge-m3",
    "hybrid_weight_dense": 0.6,      # best w on dev (notebook section 6.1)
    "bm25_k1": 1.5,
    "bm25_b": 0.75,
    "stem_prefix": 5,                # crude stemmer: first 5 characters of a token
    "doc_max_tokens": 512,
    "min_query_chars": 3,
    "max_query_chars": 500,
    "max_top_k": 20,
    "snippet_chars": 300,
}
REQUIRED_COLUMNS = ["article_number", "title", "chapter", "text"]


class QueryError(ValueError):
    """The user query is invalid (empty, too long, no letters or digits, bad top_k)."""


class IndexMismatchError(RuntimeError):
    """Saved index does not match the data file or the configuration."""


# ----------------------------------------------------------------------------- text utils
def normalize_query(query) -> str:
    if not isinstance(query, str):
        raise QueryError("Запрос должен быть строкой.")
    q = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", query)).strip()
    if len(q) < CONFIG["min_query_chars"]:
        raise QueryError(f"Запрос слишком короткий (минимум {CONFIG['min_query_chars']} символа).")
    if len(q) > CONFIG["max_query_chars"]:
        raise QueryError(f"Запрос слишком длинный (максимум {CONFIG['max_query_chars']} символов).")
    if not re.search(r"[A-Za-zА-Яа-яЁё0-9]", q):
        raise QueryError("Запрос должен содержать буквы или цифры.")
    return q


def stem_tokens(text, prefix: int = CONFIG["stem_prefix"]):
    return [t[:prefix] for t in re.findall(r"[а-яёa-z0-9-]+", str(text).lower())]


def zscore(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-9)


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class BM25:
    """Same implementation as in the experiments notebook."""

    def __init__(self, docs, k1=CONFIG["bm25_k1"], b=CONFIG["bm25_b"]):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.dl = np.array([len(d) for d in docs], dtype=float)
        self.avgdl = self.dl.mean()
        n_docs = len(docs)
        doc_freq = Counter()
        for d in docs:
            doc_freq.update(set(d))
        self.idf = {t: np.log(1 + (n_docs - n + 0.5) / (n + 0.5)) for t, n in doc_freq.items()}

    def scores(self, query_tokens):
        s = np.zeros(len(self.tf))
        for t in set(query_tokens):
            if t not in self.idf:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if f:
                    s[i] += self.idf[t] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
        return s


# ----------------------------------------------------------------------------- data / index
def load_articles(data_path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"В данных нет колонок: {missing}")
    df["article_number"] = df["article_number"].astype(str)
    for c in ["title", "chapter", "text"]:
        df[c] = df[c].fillna("").astype(str)
    if df["article_number"].duplicated().any():
        raise ValueError("Номера статей должны быть уникальными.")
    return df.reset_index(drop=True)


def doc_texts_for_dense(df: pd.DataFrame):
    return (df["title"] + ". " + df["text"]).tolist()      # same as in the notebook


def _load_st_model(device=None):
    from sentence_transformers import SentenceTransformer  # heavy import, only when needed
    revision = os.getenv("DENSE_MODEL_REVISION") or None
    kw = {"revision": revision} if revision else {}
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(CONFIG["dense_model"], device=device, **kw)
    model.max_seq_length = CONFIG["doc_max_tokens"]
    if device == "cuda":
        model.half()
    return model


def build_index(data_path, index_dir) -> dict:
    """Encode all articles with BGE-M3 and save embeddings + metadata (model version, data hash)."""
    df = load_articles(data_path)
    model = _load_st_model()
    emb = model.encode(doc_texts_for_dense(df), batch_size=8, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "doc_embeddings.npy", emb)
    import sentence_transformers, torch, transformers
    meta = {
        "model_name": MODEL_NAME, "model_version": MODEL_VERSION,
        "dense_model": CONFIG["dense_model"],
        "dense_model_revision": os.getenv("DENSE_MODEL_REVISION") or "unpinned (latest at build time)",
        "hybrid_weight_dense": CONFIG["hybrid_weight_dense"],
        "data_file": Path(data_path).name, "data_sha256": file_sha256(data_path),
        "n_articles": int(len(df)), "embedding_dim": int(emb.shape[1]),
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "versions": {"sentence-transformers": sentence_transformers.__version__,
                     "transformers": transformers.__version__, "torch": torch.__version__},
    }
    (index_dir / "index_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


# ----------------------------------------------------------------------------- the model
class LexFinder:
    def __init__(self, df, bm25, mode, dense_matrix=None, st_model=None, meta=None, fallback_reason=None):
        self.df, self.bm25, self.mode = df, bm25, mode          # mode: "hybrid" | "bm25"
        self.dense_matrix, self.st_model = dense_matrix, st_model
        self.meta = meta or {}
        self.fallback_reason = fallback_reason
        self.w = CONFIG["hybrid_weight_dense"]

    @classmethod
    def load(cls, data_path, index_dir=None, dense="auto"):
        """dense: 'on' (fail if unavailable) | 'auto' (fall back to BM25 and say so) | 'off' (BM25 only)."""
        df = load_articles(data_path)
        corpus = df["title"] + " " + df["chapter"] + " " + df["text"]
        bm25 = BM25([stem_tokens(x) for x in corpus])
        if dense == "off":
            return cls(df, bm25, "bm25", fallback_reason="dense disabled by configuration")
        try:
            index_dir = Path(index_dir or "index")
            meta = json.loads((index_dir / "index_meta.json").read_text(encoding="utf-8"))
            emb = np.load(index_dir / "doc_embeddings.npy")
            if meta.get("data_sha256") != file_sha256(data_path) or emb.shape[0] != len(df):
                raise IndexMismatchError("Индекс построен по другому файлу данных: пересоберите (scripts/build_index.py).")
            if meta.get("dense_model") != CONFIG["dense_model"]:
                raise IndexMismatchError("Модель в индексе не совпадает с конфигурацией.")
            st_model = _load_st_model()
            return cls(df, bm25, "hybrid", emb, st_model, meta)
        except Exception as exc:                                  # noqa: BLE001
            if dense == "on":
                raise
            log.warning("Dense model unavailable, falling back to BM25: %s", exc)
            return cls(df, bm25, "bm25", fallback_reason=f"{type(exc).__name__}: {exc}")

    # -- inference -------------------------------------------------------------------------
    def scores(self, query: str):
        bm = self.bm25.scores(stem_tokens(query))
        if self.mode == "bm25":
            return bm, bm
        qv = self.st_model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
        dense = self.dense_matrix @ qv
        return self.w * zscore(dense) + (1 - self.w) * zscore(bm), bm

    def search(self, query, top_k: int = 5) -> dict:
        t0 = time.perf_counter()
        q = normalize_query(query)
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= CONFIG["max_top_k"]:
            raise QueryError(f"top_k должен быть целым числом от 1 до {CONFIG['max_top_k']}.")
        score, bm = self.scores(q)
        order = np.argsort(-score, kind="stable")[:top_k]
        results = []
        for rank, i in enumerate(order, 1):
            row = self.df.iloc[i]
            results.append({"rank": rank, "article_number": row["article_number"], "title": row["title"],
                            "chapter": row["chapter"], "score": round(float(score[i]), 4),
                            "snippet": row["text"][:CONFIG["snippet_chars"]]})
        return {"query": q, "results": results, "mode": self.mode, "model_version": MODEL_VERSION,
                "no_lexical_match": bool(bm.max() <= 0),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}

    def info(self) -> dict:
        return {"model_name": MODEL_NAME, "model_version": MODEL_VERSION, "mode": self.mode,
                "fallback_reason": self.fallback_reason, "n_articles": len(self.df),
                "config": CONFIG, "index": self.meta or None}
