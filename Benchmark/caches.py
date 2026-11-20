"""Memoizing wrappers that keep a large config sweep tractable without distorting it.

Many sweep configs retrieve overlapping candidates for the same entity, so the
expensive, backend-independent steps are computed once and reused:

- query embeddings (SapBERT): pre-warmed before timing, so measured retrieval latency
  is the database's own time; embedding cost is reported separately, once.
- cross-encoder scores: memoized per (model, query, description). Rerank COST per
  config is then reported as (#pairs scored) x (measured per-pair compute time), where
  the per-pair time is calibrated from real, uncached batched calls of the same shape
  the pipeline makes (one call per entity).
- LLM selections: memoized per (llm id, formal_term, term, the 5 options shown), and
  persisted, so re-running a sweep never re-bills identical LLM calls.
"""
import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional


class KVCache:
    """Tiny persistent JSON key-value store (stdlib sqlite3)."""

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        self._lock = threading.Lock()

    @staticmethod
    def key(*parts: Any) -> str:
        return hashlib.sha1(json.dumps(parts, ensure_ascii=False).encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            row = self._conn.execute("SELECT v FROM kv WHERE k = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def get_many(self, keys: list) -> dict:
        out = {}
        with self._lock:
            for i in range(0, len(keys), 500):
                chunk = keys[i:i + 500]
                q = f"SELECT k, v FROM kv WHERE k IN ({','.join('?' * len(chunk))})"
                out.update({k: json.loads(v) for k, v in self._conn.execute(q, chunk)})
        return out

    def set_many(self, items: dict) -> None:
        with self._lock:
            self._conn.executemany("INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)",
                                   [(k, json.dumps(v, ensure_ascii=False)) for k, v in items.items()])
            self._conn.commit()

    def set(self, key: str, value: Any) -> None:
        self.set_many({key: value})


class CachedEmbedder:
    """In-memory memo of embed_query; exposes the same interface the stores call."""

    def __init__(self, inner):
        self._inner = inner
        self._memo: dict = {}
        self.compute_s = 0.0
        self.n_computed = 0
        self.model_id = getattr(inner, "model_id", type(inner).__name__)

    def embed_query(self, text: str):
        v = self._memo.get(text)
        if v is None:
            t0 = time.perf_counter()
            v = self._inner.embed_query(text)
            self.compute_s += time.perf_counter() - t0
            self.n_computed += 1
            self._memo[text] = v
        return v

    def embed_documents(self, texts):
        return self._inner.embed_documents(texts)


class _CachedSparseEmbedding:
    def __init__(self, indices, values):
        import numpy as np
        self.indices = np.asarray(indices)
        self.values = np.asarray(values)


class CachedSparseModel:
    """Memo of fastembed's query_embed (Qdrant hybrid's query-side BM25 encoding)."""

    def __init__(self, inner):
        self._inner = inner
        self._memo: dict = {}
        self.model_name = getattr(inner, "model_name", None)

    def query_embed(self, text: str):
        v = self._memo.get(text)
        if v is None:
            e = next(iter(self._inner.query_embed(text)))
            v = _CachedSparseEmbedding(e.indices, e.values)
            self._memo[text] = v
        return [v]

    def embed(self, texts):
        return self._inner.embed(texts)


class CachedReranker:
    """sentence_transformers.CrossEncoder-compatible `predict` with a persistent memo.

    Tracks, per accounting window (see `take_stats`), how many pairs were requested
    (the true per-config cost driver) and calibrates seconds-per-pair from real uncached
    compute.
    """

    def __init__(self, model_id: str, cache: KVCache, device: str = "cpu", **model_kwargs):
        from sentence_transformers import CrossEncoder
        self.model_id = model_id
        self._model = CrossEncoder(model_id, device=device, **model_kwargs)
        if self._model.tokenizer.pad_token is None:  # decoder-based rerankers (Qwen3, mxbai-v2)
            self._model.tokenizer.pad_token = self._model.tokenizer.eos_token
        self._cache = cache
        # Calibration persists across runs: when every pair is already cached, this run
        # computes nothing, yet still needs a per-pair cost to report.
        self._calib_key = KVCache.key("rerank_calibration", self.model_id, device)
        prior = cache.get(self._calib_key) or [0.0, 0]
        self.total_compute_s, self.total_computed_pairs = float(prior[0]), int(prior[1])
        self._window_pairs = 0

    def predict(self, pairs, show_progress_bar: bool = False):
        keys = [KVCache.key("rerank", self.model_id, q, d) for q, d in pairs]
        found = self._cache.get_many(list(set(keys)))
        missing = [(k, p) for k, p in zip(keys, pairs) if k not in found]
        if missing:
            uniq = dict(missing)
            t0 = time.perf_counter()
            scores = self._model.predict(list(uniq.values()), show_progress_bar=False)
            self.total_compute_s += time.perf_counter() - t0
            self.total_computed_pairs += len(uniq)
            new = {k: float(s) for k, s in zip(uniq.keys(), scores)}
            new[self._calib_key] = [self.total_compute_s, self.total_computed_pairs]
            self._cache.set_many(new)
            found.update(new)
        self._window_pairs += len(pairs)
        return [found[k] for k in keys]

    @property
    def seconds_per_pair(self) -> Optional[float]:
        if self.total_computed_pairs == 0:
            return None
        return self.total_compute_s / self.total_computed_pairs

    def take_stats(self) -> int:
        n, self._window_pairs = self._window_pairs, 0
        return n


class CachedLLMSelector:
    """Memoized Src.link.llm_select; keyed on exactly what the LLM is shown."""

    def __init__(self, llm, llm_id: str, cache: KVCache, sleep_s: float = 0.0):
        self._llm = llm
        self.llm_id = llm_id
        self._cache = cache
        self._sleep_s = sleep_s
        self.n_calls = 0
        self.call_s = 0.0

    def select(self, entity: dict, top_docs) -> tuple:
        from Src.link import llm_select
        key = KVCache.key("llm_select", self.llm_id, entity["formal_term"], entity["term"],
                          [d.page_content for d in top_docs[:5]])
        hit = self._cache.get(key)
        if hit is not None:
            return int(hit[0]), hit[1]
        t0 = time.perf_counter()
        choice, reason = llm_select(entity, top_docs, self._llm)
        self.call_s += time.perf_counter() - t0
        self.n_calls += 1
        self._cache.set(key, [choice, reason])
        if self._sleep_s:
            time.sleep(self._sleep_s)
        return choice, reason
