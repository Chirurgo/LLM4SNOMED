"""Freeze full candidate lists (description text + concept id + FSN) for the ranking study.

The ranking study (Benchmark.rank_study) compares rerankers, Jev-style validators and LLM
selection OFFLINE on the same retrieved candidates, so the databases only need to be up
once, here. The harness' candidates/*.json keep only concept ids, which is not enough to
rerank (cross-encoders score the matched description text).

Retrieval is done at k=50 (max ranking depth in the study); shallower depths are taken
as prefixes of this list (rank_study --k), which equals a k-deep retrieval up to
domain-filter edge effects (same expand_factor=2 over-fetch, then truncation).

Usage (one container up at a time; the backend must be running):
  python -m Benchmark.freeze_candidates --backends qdrant \
      --extractions Benchmark/results/gold/extractions_gold.json --out Benchmark/results/rank/frozen
Writes <out>/<config_id>.json: {"config": {...}, "entities": [[doc_idx, ent_idx, [cand...]], ...]}
"""
import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from Benchmark.configs import RunConfig
from Benchmark.extraction import load_extractions
from Src.link import retrieve_candidates

FREEZE_K = 50

# Backends kept after the 2026-11-19 cut (postgres, weaviate dropped).
FREEZE_CONFIGS = {
    "neo4j": [RunConfig(backend="neo4j", mode="dense", k_candidate=FREEZE_K, reranker=None)],
    "qdrant": [RunConfig(backend="qdrant", mode="dense", k_candidate=FREEZE_K, reranker=None),
               RunConfig(backend="qdrant", mode="hybrid", k_candidate=FREEZE_K, reranker=None)],
    "meilisearch": [RunConfig(backend="meilisearch", mode="dense", k_candidate=FREEZE_K, reranker=None),
                    RunConfig(backend="meilisearch", mode="hybrid", k_candidate=FREEZE_K, reranker=None)],
}


def frozen_id(cfg: RunConfig) -> str:
    # The reranker/selection suffix is meaningless for a frozen retrieval list.
    return cfg.config_id.replace("_rr-none_top1", "")


def freeze(cfg: RunConfig, store, docs, out: Path, force: bool = False) -> None:
    path = out / f"{frozen_id(cfg)}.json"
    if path.exists() and not force:
        print(f"  [skip] {path.name}")
        return
    store.configure(cfg.params)
    fsn_cache = {}
    entities, t0 = [], time.perf_counter()
    for i, ents in enumerate(docs):
        for j, ent in enumerate(ents):
            hits = retrieve_candidates(ent, store, cfg.k_candidate, cfg.mode == "hybrid",
                                       min_hits=cfg.min_hits, expand_factor=cfg.expand_factor)
            cands = []
            for d in hits:
                cid = d.metadata.get("concept_id")
                key = str(cid)
                if key not in fsn_cache:
                    fsn_cache[key] = store.get_fsn(cid, hint_doc=d)
                cands.append({"text": d.page_content, "concept_id": key,
                              "semantic_tag": d.metadata.get("semantic_tag"), "fsn": fsn_cache[key]})
            entities.append([i, j, cands])
    path.write_text(json.dumps({"config": cfg.as_dict(), "frozen_id": frozen_id(cfg),
                                "seconds": time.perf_counter() - t0, "entities": entities},
                               ensure_ascii=False), encoding="utf-8")
    print(f"  [done] {path.name}: {len(entities)} entities, {time.perf_counter() - t0:.0f}s")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backends", nargs="+", required=True, choices=sorted(FREEZE_CONFIGS))
    ap.add_argument("--extractions", default="Benchmark/results/gold/extractions_gold.json")
    ap.add_argument("--n-docs", type=int, default=10)
    ap.add_argument("--out", default="Benchmark/results/rank/frozen")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    from dotenv import load_dotenv
    load_dotenv(".env", override=False)
    from Benchmark import backends as B
    from Benchmark.caches import CachedEmbedder, CachedSparseModel
    from Database.embed_models import AsyncSapBERTEmbedder

    docs = load_extractions(args.extractions, args.n_docs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    embedder = CachedEmbedder(AsyncSapBERTEmbedder(device=args.device))
    for backend in args.backends:
        sparse = None
        if backend == "qdrant":
            from fastembed import SparseTextEmbedding
            from Src.stores.qdrant_store import SPARSE_MODEL_ID
            sparse = CachedSparseModel(SparseTextEmbedding(SPARSE_MODEL_ID))
        store, close = B.open_store(backend, embedder, sparse)
        try:
            for cfg in FREEZE_CONFIGS[backend]:
                freeze(cfg, store, docs, out, force=args.force)
        finally:
            close()


if __name__ == "__main__":
    main()
