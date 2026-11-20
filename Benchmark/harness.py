"""Benchmark harness: runs linking (stages 3-5) for every sweep config over cached
extractions, writing per config:

  <out>/predictions/<config_id>.json  -- exact Src/run.py prediction schema (Evaluate/*.py-ready)
  <out>/candidates/<config_id>.json   -- raw retrieved concept ids per entity (pre-rerank),
                                         for the candidate-recall ceiling
  <out>/metrics/<config_id>.json      -- config, latency, DB container CPU/mem, rerank/LLM cost

Usage (from repo root):
  python -m Benchmark.harness --gold Minimal_Dataset.json --extractions <cache.json> \
      --out Benchmark/results/run1 --backends postgres qdrant weaviate meilisearch \
      [--load DATA/corpus_cache/...parquet] [--axes k_candidate reranker] [--llm openai:gpt-4.1]
then:
  python -m Benchmark.report --out Benchmark/results/run1 --gold Minimal_Dataset.json
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from Benchmark.caches import CachedEmbedder, CachedLLMSelector, CachedReranker, CachedSparseModel, KVCache
from Benchmark.configs import RERANKERS, RunConfig, build_full_grid, build_sweep
from Benchmark.extraction import load_extractions
from Benchmark.resource_profile import ContainerSampler, latency_summary
from Src.link import (TOP1_REASON, build_prediction, empty_prediction, pad_to_five, rerank_candidates,
                      retrieve_candidates)

WARMUP_QUERIES = 3
RERANK_TOP_N = 20  # same as Src/link.py


def _dedupe_by_concept(docs, top_n: int = RERANK_TOP_N):
    """No-reranker path: same concept-level dedupe as rerank_candidates, retrieval order."""
    seen, out = set(), []
    for d in docs:
        cid = d.metadata.get("concept_id")
        if cid not in seen:
            seen.add(cid)
            out.append(d)
        if len(out) == top_n:
            break
    return out


class Harness:
    def __init__(self, gold_path: str, extractions_path: str, out_dir: str, embedding_model,
                 sparse_model=None, cache_path: Optional[str] = None, reranker_device: str = "cpu",
                 llm_selector: Optional[CachedLLMSelector] = None, latency_repeats: int = 1):
        self.gold = json.loads(Path(gold_path).read_text(encoding="utf-8"))
        self.docs = load_extractions(extractions_path, len(self.gold))
        self.out = Path(out_dir)
        for sub in ("predictions", "candidates", "metrics"):
            (self.out / sub).mkdir(parents=True, exist_ok=True)
        self.cache = KVCache(cache_path or str(self.out / "cache.sqlite"))
        self.embedder = CachedEmbedder(embedding_model)
        self.sparse = CachedSparseModel(sparse_model) if sparse_model is not None else None
        self.reranker_device = reranker_device
        self._rerankers: Dict[str, CachedReranker] = {}
        self.llm_selector = llm_selector
        self.latency_repeats = max(1, latency_repeats)
        # (doc_idx, entity) for every positioned entity
        self.entities = [(i, e) for i, ents in enumerate(self.docs) for e in ents]

    def reranker(self, key: Optional[str]) -> Optional[CachedReranker]:
        if key is None:
            return None
        if key not in self._rerankers:
            self._rerankers[key] = CachedReranker(RERANKERS[key], self.cache, device=self.reranker_device)
        return self._rerankers[key]

    def prewarm_embeddings(self) -> dict:
        """Embed every query once, before any timing, so retrieval latency is DB-only."""
        terms = sorted({e["formal_term"] for _, e in self.entities})
        for t in terms:
            self.embedder.embed_query(t)
            if self.sparse is not None:
                self.sparse.query_embed(t)
        n = max(1, self.embedder.n_computed)
        return {"n_queries": len(terms), "embed_mean_ms": 1000 * self.embedder.compute_s / n,
                "embedding_model": self.embedder.model_id}

    # -- one config ---------------------------------------------------------------------
    def run_config(self, cfg: RunConfig, store, force: bool = False) -> Optional[dict]:
        metrics_path = self.out / "metrics" / f"{cfg.config_id}.json"
        if metrics_path.exists() and not force:
            print(f"  [skip] {cfg.config_id} (done)")
            return None
        if cfg.selection == "llm" and self.llm_selector is None:
            raise ValueError(f"{cfg.config_id} needs an LLM selector (--llm)")

        store.configure(cfg.params)
        use_hybrid = cfg.mode == "hybrid"

        def retrieve(ent):
            return retrieve_candidates(ent, store, cfg.k_candidate, use_hybrid,
                                       min_hits=cfg.min_hits, expand_factor=cfg.expand_factor)

        for _, ent in self.entities[:WARMUP_QUERIES]:
            retrieve(ent)

        # Phase 1: retrieval only (timed, DB container sampled).
        latencies: List[float] = []
        retrieved: List[Optional[list]] = [None] * len(self.entities)
        errors = 0
        with ContainerSampler(cfg.backend) as sampler:
            for _ in range(self.latency_repeats):
                for j, (_, ent) in enumerate(self.entities):
                    t0 = time.perf_counter()
                    try:
                        retrieved[j] = retrieve(ent)
                    except Exception as e:
                        errors += 1
                        print(f"  !!! retrieval error '{ent.get('term')}': {e}")
                        continue
                    latencies.append(time.perf_counter() - t0)

        # Phase 2: rerank + select (client side, cached).
        reranker = self.reranker(cfg.reranker)
        if reranker:
            reranker.take_stats()
        llm_calls_before = self.llm_selector.n_calls if self.llm_selector else 0
        preds = defaultdict(list)
        cands = defaultdict(list)
        for j, (doc_idx, ent) in enumerate(self.entities):
            docs = retrieved[j]
            cands[doc_idx].append({"term": ent.get("term"), "start_index": ent["start_index"],
                                   "end_index": ent["end_index"],
                                   "retrieved": [d.metadata.get("concept_id") for d in docs or []]})
            if docs is None:
                continue
            try:
                if reranker:
                    top = rerank_candidates(reranker=reranker, query_text=ent["formal_term"], docs=docs, top_n=RERANK_TOP_N)
                else:
                    top = _dedupe_by_concept(docs)
                if len(top) < 2:
                    preds[doc_idx].append(empty_prediction(ent))
                    continue
                top = pad_to_five(top)
                if cfg.selection == "llm":
                    choice, reason = self.llm_selector.select(ent, top)
                else:
                    choice, reason = 1, TOP1_REASON
                preds[doc_idx].append(build_prediction(ent, top, choice, reason, store))
            except Exception as e:
                errors += 1
                print(f"  !!! linking error '{ent.get('term')}': {e}")

        n_docs = len(self.docs)
        predictions = [preds.get(i, []) for i in range(n_docs)]
        candidates = [cands.get(i, []) for i in range(n_docs)]
        (self.out / "predictions" / f"{cfg.config_id}.json").write_text(
            json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.out / "candidates" / f"{cfg.config_id}.json").write_text(
            json.dumps(candidates, ensure_ascii=False), encoding="utf-8")

        n_ent = max(1, len(self.entities))
        rerank_pairs = reranker.take_stats() if reranker else 0
        metrics = {
            **cfg.as_dict(),
            "n_entities": len(self.entities),
            "errors": errors,
            **latency_summary(latencies, "retrieval"),
            **sampler.summary(),
            "rerank_pairs_per_entity": rerank_pairs / n_ent,
            "reranker_model": RERANKERS.get(cfg.reranker) if cfg.reranker else None,
            "llm_calls_new": (self.llm_selector.n_calls - llm_calls_before) if self.llm_selector else 0,
            "llm_id": self.llm_selector.llm_id if (self.llm_selector and cfg.selection == "llm") else None,
        }
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"  [done] {cfg.config_id}: p50 {metrics['retrieval_p50_ms'] or float('nan'):.1f} ms, "
              f"{metrics['rerank_pairs_per_entity']:.0f} pairs/entity, errors={errors}")
        return metrics

    def run(self, configs: List[RunConfig], open_store, force: bool = False) -> None:
        meta = {"embedding": self.prewarm_embeddings(), "n_entities": len(self.entities),
                "n_docs": len(self.docs), "latency_repeats": self.latency_repeats}
        by_backend = defaultdict(list)
        for c in configs:
            by_backend[c.backend].append(c)
        for backend, cfgs in by_backend.items():
            print(f"[{backend}] {len(cfgs)} configs")
            store, close = open_store(backend, self.embedder, self.sparse)
            try:
                for cfg in cfgs:
                    self.run_config(cfg, store, force=force)
            finally:
                close()
        meta["reranker_seconds_per_pair"] = {RERANKERS[k]: r.seconds_per_pair for k, r in self._rerankers.items()}
        meta_path = self.out / "run_meta.json"
        old = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        old_spp = old.get("reranker_seconds_per_pair", {})
        old_spp.update({k: v for k, v in meta["reranker_seconds_per_pair"].items() if v is not None})
        meta["reranker_seconds_per_pair"] = old_spp
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


_THINK_RE = None


def strip_reasoning(msg):
    """Drop <think>...</think> blocks emitted by reasoning models (e.g. GLM-Z1, Qwen3)
    so downstream JSON parsers only see the answer. Leaves other output untouched."""
    import re
    from langchain_core.messages import AIMessage
    global _THINK_RE
    _THINK_RE = _THINK_RE or re.compile(r"<think>.*?</think>|^.*?</think>", re.S)
    text = msg.content if hasattr(msg, "content") else str(msg)
    cleaned = _THINK_RE.sub("", text).strip()
    return AIMessage(content=cleaned) if hasattr(msg, "content") else cleaned


def make_llm(spec: str):
    """'openai:<model>' | 'gemini:<model>' | 'lmstudio:<model>' -- keys/URL from .env,
    mirroring 02_main.ipynb (temperature 0)."""
    from dotenv import dotenv_values
    env = dotenv_values(".env")
    provider, model = spec.split(":", 1)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(api_key=env.get("OPENAI_API_KEY"), model=model, temperature=0, max_retries=2)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(api_key=env.get("GEMINI_API_KEY"), model=model, temperature=0, max_retries=2)
    if provider == "lmstudio":
        from langchain_openai import ChatOpenAI
        base = (env.get("LMSTUDIO_URL") or "http://localhost:1234").rstrip("/")
        base = (base if base.endswith("/v1") else base + "/v1") + "/"  # .env may already include /v1
        return ChatOpenAI(model=model, base_url=base, api_key=env.get("LMSTUDIO_API") or "dummy_key",
                          temperature=0, seed=2531)
    raise ValueError(f"unknown LLM provider {provider!r}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--extractions", required=True, help="cache from Benchmark.extraction.run_extraction")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backends", nargs="+", default=["postgres", "qdrant", "weaviate", "meilisearch"])
    ap.add_argument("--load", metavar="PARQUET", help="(re)load these backends from the corpus cache first")
    ap.add_argument("--axes", nargs="+", help="OFAT axes to sweep (default: all in Benchmark/configs.py SWEEP)")
    ap.add_argument("--full-grid", metavar="JSON", help='Cartesian grid instead of OFAT, e.g. \'{"mode":["dense","hybrid"],"k_candidate":[10,50]}\'')
    ap.add_argument("--llm", help="add LLM-selection configs: openai:<model> | gemini:<model> | lmstudio:<model>")
    ap.add_argument("--llm-sleep", type=float, default=5.0, help="seconds between uncached LLM calls (rate limits)")
    ap.add_argument("--device", default="cpu", help="device for SapBERT + rerankers")
    ap.add_argument("--latency-repeats", type=int, default=1)
    ap.add_argument("--cache", help="shared score cache (default <out>/cache.sqlite)")
    ap.add_argument("--force", action="store_true", help="re-run configs that already have metrics")
    args = ap.parse_args(argv)

    # Neo4j credentials live in .env (same as build_indexes); without this open_store("neo4j")
    # falls back to the compose default password and fails auth.
    from dotenv import load_dotenv
    load_dotenv(".env", override=False)

    from Benchmark import backends as B
    from Database.embed_models import AsyncSapBERTEmbedder

    out = Path(args.out)
    if args.load:
        (out / "index").mkdir(parents=True, exist_ok=True)
        for b in args.backends:
            if b in B.LOADABLE:
                stats = B.load_backend(b, args.load)
                (out / "index" / f"{b}.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
                print(f"[{b}] loaded in {stats['load_s']:.1f}s, disk {stats['db_disk_mb']} MB")

    sparse_model = None
    if "qdrant" in args.backends:
        from fastembed import SparseTextEmbedding
        from Src.stores.qdrant_store import SPARSE_MODEL_ID
        sparse_model = SparseTextEmbedding(SPARSE_MODEL_ID)

    if args.full_grid:
        configs = build_full_grid(args.backends, json.loads(args.full_grid), llm=bool(args.llm))
    else:
        configs = build_sweep(args.backends, axes=args.axes, llm=bool(args.llm))
    print(f"{len(configs)} configs across {len(args.backends)} backend(s)")

    harness = Harness(args.gold, args.extractions, args.out, AsyncSapBERTEmbedder(device=args.device),
                      sparse_model=sparse_model, cache_path=args.cache, reranker_device=args.device,
                      latency_repeats=args.latency_repeats)
    if args.llm:
        harness.llm_selector = CachedLLMSelector(make_llm(args.llm), args.llm, harness.cache, sleep_s=args.llm_sleep)
    harness.run(configs, B.open_store, force=args.force)


if __name__ == "__main__":
    main()
