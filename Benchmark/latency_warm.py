"""Warm-cache retrieval latency, separate from the sweep.

The OFAT sweep times each config's single pass in run order, so the first configs of each
backend pay cold-cache cost (e.g. Meilisearch 5.7 s -> 21 ms p50 as its mmap pages warmed).
Here, per backend and config: one untimed pass over all queries (warms DB pages and the
query-embedding cache), then `--repeats` timed passes. Query embeddings are cached, so the
numbers are DB retrieval (+ domain filter round-trips) only.

Usage (one DB container up at a time):
  python -m Benchmark.latency_warm --gold Minimal_Dataset.json \
      --extractions Benchmark/results/gold/extractions_gold.json \
      --out Benchmark/results/gold --backends qdrant
Writes <out>/latency_warm/<backend>.json.
"""
import argparse
import json
import time
from pathlib import Path

from Benchmark.configs import RunConfig, _supports_hybrid
from Benchmark.harness import Harness
from Benchmark.resource_profile import ContainerSampler, latency_summary
from Src.link import retrieve_candidates

CONFIGS = [("dense", 20), ("hybrid", 20), ("dense", 100)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--extractions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backends", nargs="+", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    from dotenv import load_dotenv
    load_dotenv(".env", override=False)
    from Benchmark import backends as B
    from Database.embed_models import AsyncSapBERTEmbedder

    sparse_model = None
    if "qdrant" in args.backends:
        from fastembed import SparseTextEmbedding
        from Src.stores.qdrant_store import SPARSE_MODEL_ID
        sparse_model = SparseTextEmbedding(SPARSE_MODEL_ID)
    h = Harness(args.gold, args.extractions, args.out, AsyncSapBERTEmbedder(device=args.device),
                sparse_model=sparse_model)
    h.prewarm_embeddings()
    out_dir = Path(args.out) / "latency_warm"
    out_dir.mkdir(parents=True, exist_ok=True)

    for backend in args.backends:
        store, close = B.open_store(backend, h.embedder, h.sparse)
        results = {}
        try:
            for mode, k in CONFIGS:
                if mode == "hybrid" and not _supports_hybrid(backend):
                    continue
                cfg = RunConfig(backend=backend, mode=mode, k_candidate=k)
                store.configure(cfg.params)
                ents = [e for _, e in h.entities]

                def retrieve(ent):
                    return retrieve_candidates(ent, store, cfg.k_candidate, mode == "hybrid",
                                               min_hits=cfg.min_hits, expand_factor=cfg.expand_factor)

                t0 = time.perf_counter()
                for ent in ents:  # warm-up pass, untimed
                    retrieve(ent)
                warm_s = time.perf_counter() - t0
                lat = []
                with ContainerSampler(backend) as sampler:
                    for _ in range(args.repeats):
                        for ent in ents:
                            t = time.perf_counter()
                            retrieve(ent)
                            lat.append(time.perf_counter() - t)
                s = latency_summary(lat, "retrieval")
                results[f"{mode}_k{k}"] = {"warmup_pass_s": warm_s, "n_timed": len(lat), **s,
                                           **sampler.summary()}
                print(f"[{backend}] {mode} k{k}: warm-up {warm_s:.1f}s, "
                      f"p50 {s['retrieval_p50_ms']:.1f} ms, p95 {s['retrieval_p95_ms']:.1f} ms", flush=True)
        finally:
            close()
        (out_dir / f"{backend}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
