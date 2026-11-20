"""
Smoke test for Database/build_corpus.py against the synthetic mini-SNOMED fixture
(Benchmark/fixtures/mini_snomed.py) -- no licensed RF2 release needed.

Run directly: python -m tests.test_build_corpus
"""
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq

from Benchmark.fixtures.mini_snomed import mini_snomed_frames, CONCEPTS, DESCRIPTIONS
import Database.build_corpus as build_corpus
from Database.embed_models import AsyncSapBERTEmbedder


def run_all():
    tmpdir = Path(tempfile.mkdtemp(prefix="llm4snomed_corpus_test_"))
    out_path = tmpdir / "mini_corpus.parquet"
    try:
        embedding_model = AsyncSapBERTEmbedder(device="cpu")

        with patch.object(build_corpus, "reformat_snomed", return_value=mini_snomed_frames()):
            result_path = build_corpus.build_and_cache_corpus(
                snomed_release_path="FIXTURE",
                out_path=str(out_path),
                embedding_model=embedding_model,
                sparse_model=None,
                batch_size=4,
            )
        assert result_path == str(out_path)
        assert out_path.exists()

        table = pq.read_table(out_path)
        df = table.to_pandas()

        non_fsn_descriptions = [d for d in DESCRIPTIONS]
        assert len(df) == len(non_fsn_descriptions), (len(df), len(non_fsn_descriptions))

        assert set(df.columns) == {
            "row_id", "concept_id", "text", "name_type", "semantic_tag", "fsn",
            "embedding", "sparse_indices", "sparse_values",
        }

        # row_id is stable/deterministic and unique
        assert df["row_id"].nunique() == len(df)

        # semantic_tag correctly denormalized and lowercased from the owning concept
        tag_by_concept = {cid: tag.lower() for cid, _, tag in CONCEPTS}
        for _, row in df.iterrows():
            assert row["semantic_tag"] == tag_by_concept[row["concept_id"]], row

        # fsn correctly denormalized from the owning concept
        fsn_by_concept = {cid: fsn for cid, fsn, _ in CONCEPTS}
        for _, row in df.iterrows():
            assert row["fsn"] == fsn_by_concept[row["concept_id"]], row

        # dense embeddings present with SapBERT's expected dimensionality, no sparse (not requested)
        assert all(len(e) == 768 for e in df["embedding"])
        assert all(len(s) == 0 for s in df["sparse_indices"])

        # cache hit on rerun with identical args (no sparse_model passed either time)
        with patch.object(build_corpus, "reformat_snomed", return_value=mini_snomed_frames()) as mocked:
            build_corpus.build_and_cache_corpus(
                snomed_release_path="FIXTURE",
                out_path=str(out_path),
                embedding_model=embedding_model,
                sparse_model=None,
                batch_size=4,
            )
            mocked.assert_not_called()  # cache hit -- reformat_snomed never even called

        print(f"build_corpus smoke test PASS ({len(df)} rows, embedding dim {len(df['embedding'][0])})")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    run_all()
