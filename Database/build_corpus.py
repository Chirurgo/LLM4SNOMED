"""
Builds the single shared, denormalized SNOMED CT description corpus used to populate
every non-Neo4j backend (Postgres, Qdrant, Weaviate, Meilisearch) for the multi-backend
retrieval benchmark.

SapBERT embeddings (and, optionally, BM25-style sparse vectors for Qdrant's hybrid mode)
are computed exactly ONCE here and cached to a single Parquet file. Every per-backend
loader (Database/loaders/*.py) streams this file and bulk-inserts the precomputed
vectors directly -- it never re-embeds. This is what makes a 5-way full-SNOMED
benchmark tractable: embedding ~300k-1M description rows is the expensive, GPU/CPU-bound
step, and it must not be repeated per backend.

`semantic_tag` and `fsn` are denormalized onto every row (joined in from the owning
concept) so each backend can domain-filter and look up display names from a single
query, with no second round-trip -- see Src/stores/base.py::ConceptStore.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from Database.transform import reformat_snomed

ARROW_SCHEMA = pa.schema([
    ("row_id", pa.string()),
    ("concept_id", pa.string()),
    ("text", pa.string()),
    ("name_type", pa.string()),
    ("semantic_tag", pa.string()),
    ("fsn", pa.string()),
    ("embedding", pa.list_(pa.float32())),
    ("sparse_indices", pa.list_(pa.int32())),
    ("sparse_values", pa.list_(pa.float32())),
])


def _row_id(concept_id: str, name_type: str, text: str) -> str:
    # Deterministic, order/batch-independent id -- reruns on the same release produce
    # identical row_ids, which every backend uses as its point-ID / primary key.
    digest = hashlib.sha1(f"{concept_id}|{name_type}|{text}".encode("utf-8")).hexdigest()
    return digest[:24]


def build_description_lazyframe(snomed_release_path: str) -> pl.LazyFrame:
    """Join synonym/FSN descriptions to concept metadata (semantic_tag, FSN).

    Mirrors the exact cleaning Database/transform.py::awrite_description_nodes already
    does (strip, dropna, name_type fillna "Synonym"), plus lowercasing semantic_tag so
    every backend's domain filter can compare against Src/domain_mapping.py's
    already-lowercased TYPE_TO_SEMANTIC_TAG values without re-normalizing per query.
    """
    active_concp, active_synon, _relat = reformat_snomed(snomed_release_path)

    concept_meta = (
        active_concp
        .select(["concept_id", "FSN", "semantic_tag"])
        .unique(subset=["concept_id"])
        .with_columns(pl.col("semantic_tag").cast(pl.Utf8).str.to_lowercase())
        .rename({"FSN": "fsn"})
    )

    descriptions = (
        active_synon
        .with_columns([
            pl.col("concept_name").cast(pl.Utf8).str.strip_chars(),
            pl.col("name_type").cast(pl.Utf8).str.strip_chars().fill_null("Synonym"),
        ])
        .drop_nulls(["concept_id", "concept_name"])
        .join(concept_meta, on="concept_id", how="inner")
        .rename({"concept_name": "text"})
        .select(["concept_id", "text", "name_type", "semantic_tag", "fsn"])
        # Deterministic global order is required: build_and_cache_corpus() slices this
        # lazyframe into independent batches via repeated .collect() calls, which is only
        # a safe partition of the rows if the plan is sorted (joins otherwise have no
        # guaranteed stable order across separate executions).
        .sort(["concept_id", "name_type", "text"])
    )
    return descriptions


def _embed_length_sorted(embedding_model, texts):
    """Embed in length order and restore the input order. Grouping similar-length texts
    minimises padding per sub-batch (~1.8x faster on CPU for SapBERT, measured on this
    corpus); with attention masking the vectors are the same up to float noise."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    vecs = embedding_model.embed_documents([texts[i] for i in order])
    out = [None] * len(texts)
    for pos, i in enumerate(order):
        out[i] = vecs[pos]
    return out


@dataclass
class CorpusManifest:
    source_path: str
    embedding_model_id: str
    sparse_model_id: Optional[str]
    row_count: int
    built_at: float

    @classmethod
    def load(cls, path: Path) -> Optional["CorpusManifest"]:
        if not path.exists():
            return None
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")

    def matches(self, source_path: str, embedding_model_id: str, sparse_model_id: Optional[str]) -> bool:
        return (
            self.source_path == source_path
            and self.embedding_model_id == embedding_model_id
            and self.sparse_model_id == sparse_model_id
        )


def build_and_cache_corpus(
    snomed_release_path: str,
    out_path: str,
    embedding_model,
    sparse_model=None,
    batch_size: int = 2000,
    force: bool = False,
) -> str:
    """Compute embeddings once and cache the full corpus to a single Parquet file.

    `embedding_model` must implement `embed_documents(texts: List[str]) -> List[List[float]]`
    (e.g. Database.embed_models.AsyncSapBERTEmbedder -- its sync method is used here since
    this is a one-time offline batch job with no concurrency benefit from the async path).

    `sparse_model`, if given, must implement `embed(texts: List[str]) -> Iterable[SparseEmbedding]`
    (fastembed.SparseTextEmbedding) for Qdrant's hybrid mode. Pass None to skip sparse vectors.

    Idempotent: skips recomputation if `out_path` + its sidecar manifest already match the
    given (source_path, embedding model, sparse model), unless `force=True`.
    """
    out_path = Path(out_path)
    manifest_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
    embedding_model_id = getattr(embedding_model, "model_id", type(embedding_model).__name__)
    sparse_model_id = getattr(sparse_model, "model_name", None) if sparse_model is not None else None

    existing = CorpusManifest.load(manifest_path)
    if not force and existing is not None and existing.matches(snomed_release_path, embedding_model_id, sparse_model_id) and out_path.exists():
        print(f"Corpus cache up to date at {out_path} ({existing.row_count} rows) -- skipping rebuild.")
        return str(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Collected once: re-running the join+sort plan per batch (lf.slice().collect()) costs
    # seconds per batch, i.e. ~an hour of pure overhead over ~900k rows. The text-only
    # frame is small (~100 MB); only embeddings are bounded by batching.
    df_all = build_description_lazyframe(snomed_release_path).collect(engine="streaming")
    total = df_all.height

    # Resumable: each batch is written as its own shard, so a crash mid-build (hours on
    # CPU) resumes at the first missing shard. Shards are merged into out_path at the end.
    # Shards are only reused if they were built with the same (source, models, batch_size).
    parts_dir = out_path.with_suffix(out_path.suffix + ".parts")
    parts_key = {"source_path": snomed_release_path, "embedding_model_id": embedding_model_id,
                 "sparse_model_id": sparse_model_id, "batch_size": batch_size, "total": total}
    parts_key_path = parts_dir / "_key.json"
    if parts_dir.exists() and (force or not parts_key_path.exists()
                               or json.loads(parts_key_path.read_text(encoding="utf-8")) != parts_key):
        for p in parts_dir.glob("*"):
            p.unlink()
    parts_dir.mkdir(parents=True, exist_ok=True)
    parts_key_path.write_text(json.dumps(parts_key), encoding="utf-8")

    row_count = 0
    embedded_now = 0
    start = time.perf_counter()
    for offset in range(0, total, batch_size):
        part_path = parts_dir / f"part-{offset // batch_size:06d}.parquet"
        batch_df = df_all.slice(offset, batch_size)
        if part_path.exists():
            row_count += batch_df.height
            continue

        texts = batch_df["text"].to_list()
        dense = _embed_length_sorted(embedding_model, texts)

        if sparse_model is not None:
            sparse = list(sparse_model.embed(texts))
            sparse_indices = [s.indices.tolist() for s in sparse]
            sparse_values = [s.values.tolist() for s in sparse]
        else:
            sparse_indices = [[] for _ in texts]
            sparse_values = [[] for _ in texts]

        row_ids = [
            _row_id(cid, nt, t)
            for cid, nt, t in zip(batch_df["concept_id"].to_list(), batch_df["name_type"].to_list(), texts)
        ]

        table = pa.table({
            "row_id": row_ids,
            "concept_id": [str(c) for c in batch_df["concept_id"].to_list()],
            "text": texts,
            "name_type": batch_df["name_type"].to_list(),
            "semantic_tag": batch_df["semantic_tag"].to_list(),
            "fsn": batch_df["fsn"].to_list(),
            "embedding": dense,
            "sparse_indices": sparse_indices,
            "sparse_values": sparse_values,
        }, schema=ARROW_SCHEMA)

        tmp = part_path.with_suffix(".tmp")
        pq.write_table(table, tmp)
        tmp.replace(part_path)  # atomic: a crash never leaves a half-written shard
        row_count += batch_df.height
        embedded_now += batch_df.height
        elapsed = time.perf_counter() - start
        rate = embedded_now / elapsed if elapsed else 0.0
        eta = (total - row_count) / rate if rate else float("nan")
        print(f"  embedded {row_count}/{total} rows ({elapsed:.1f}s elapsed, "
              f"{rate:.1f} rows/s, ETA {eta / 60:.1f} min)", flush=True)

    # Merge shards (in order) into the single file every loader streams.
    tmp_out = out_path.with_suffix(out_path.suffix + ".tmp")
    with pq.ParquetWriter(tmp_out, ARROW_SCHEMA) as writer:
        for part_path in sorted(parts_dir.glob("part-*.parquet")):
            writer.write_table(pq.read_table(part_path, schema=ARROW_SCHEMA))
    tmp_out.replace(out_path)
    for p in parts_dir.glob("*"):
        p.unlink()
    parts_dir.rmdir()

    CorpusManifest(
        source_path=snomed_release_path,
        embedding_model_id=embedding_model_id,
        sparse_model_id=sparse_model_id,
        row_count=row_count,
        built_at=time.time(),
    ).save(manifest_path)

    print(f"Corpus cache built: {row_count} rows -> {out_path} ({time.perf_counter() - start:.1f}s total)")
    return str(out_path)
