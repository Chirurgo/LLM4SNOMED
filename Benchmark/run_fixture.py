"""End-to-end dry run of the full benchmark on the synthetic fixture: build corpus ->
load all 4 backends -> full OFAT sweep -> report. No SNOMED license, LLM, or clinical
data needed. Validates plumbing/bookkeeping only -- fixture quality numbers are not
meaningful (15-row corpus, 6 entities).

    python -m Benchmark.wait_for_services
    python -m Benchmark.run_fixture [--out Benchmark/results/fixture]
"""
import argparse
from pathlib import Path

from Benchmark import harness, report
from Benchmark.fixtures.mini_benchmark import build_fixture_corpus, write_gold_and_extractions


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="Benchmark/results/fixture")
    ap.add_argument("--axes", nargs="+")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from fastembed import SparseTextEmbedding
    from Database.embed_models import AsyncSapBERTEmbedder
    from Src.stores.qdrant_store import SPARSE_MODEL_ID
    parquet = build_fixture_corpus(AsyncSapBERTEmbedder(device="cpu"), SparseTextEmbedding(SPARSE_MODEL_ID),
                                   out_dir=str(out / "corpus"))
    gold, extractions = write_gold_and_extractions(str(out))

    common = ["--gold", gold, "--extractions", extractions, "--out", str(out)]
    harness.main(common + ["--load", parquet, "--latency-repeats", "5", "--force"]
                 + (["--axes", *args.axes] if args.axes else []))
    report.main(["--out", str(out), "--gold", gold, "--extractions", extractions])


if __name__ == "__main__":
    main()
