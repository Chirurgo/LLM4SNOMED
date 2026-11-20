"""Bulk-loads the cached corpus parquet (Database/build_corpus.py) into Qdrant.

The collection is created with a *named* dense vector ("dense") rather than an
unnamed default vector, so Phase 4 can add a named sparse vector alongside it for
hybrid (RRF) search without a schema migration.
"""
import time

import pyarrow.parquet as pq
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from Src.stores.qdrant_store import DENSE_VECTOR_NAME, EMBEDDING_DIM, SPARSE_VECTOR_NAME, QdrantConceptStore

COLLECTION_NAME = "snomed_descriptions"


def connect(url: str, prefer_grpc: bool = True, timeout: int = 300) -> QdrantClient:
    # gRPC (port 6334) by default: on Windows + Docker the REST transport measured
    # 5-27 ms jittery per query vs a steady ~2 ms over gRPC for the same search, which
    # would unfairly inflate Qdrant's latency (Weaviate's v4 client already uses gRPC).
    # timeout: qdrant-client's gRPC default is 5 s, which bulk upserts exceed once the
    # optimizer is busy (crashed the 2026-11-14 full-scale load at 149k rows).
    return QdrantClient(url=url, prefer_grpc=prefer_grpc, timeout=timeout)


def create_collection(client: QdrantClient, collection_name: str = COLLECTION_NAME, drop_existing: bool = True) -> None:
    if drop_existing and client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config={DENSE_VECTOR_NAME: qmodels.VectorParams(size=EMBEDDING_DIM, distance=qmodels.Distance.COSINE)},
        # Qdrant/bm25 sparse vectors carry only term frequencies; the IDF modifier makes
        # Qdrant apply corpus-level IDF at query time, completing BM25 scoring.
        sparse_vectors_config={SPARSE_VECTOR_NAME: qmodels.SparseVectorParams(modifier=qmodels.Modifier.IDF)},
        # No HNSW building during bulk upload; load_corpus re-enables it at the end.
        optimizers_config=qmodels.OptimizersConfigDiff(indexing_threshold=0),
    )
    client.create_payload_index(collection_name, field_name="semantic_tag", field_schema=qmodels.PayloadSchemaType.KEYWORD)
    client.create_payload_index(collection_name, field_name="concept_id", field_schema=qmodels.PayloadSchemaType.KEYWORD)


def _row_id_to_point_id(row_id: str) -> int:
    # Qdrant point IDs must be an unsigned int or UUID. Deterministic int derived from
    # our stable hex row_id keeps re-runs idempotent (upsert on the same id overwrites).
    return int(row_id[:15], 16)


def load_corpus(client: QdrantClient, parquet_path: str, collection_name: str = COLLECTION_NAME, batch_size: int = 256) -> int:
    columns = ["row_id", "concept_id", "text", "name_type", "semantic_tag", "fsn", "embedding",
               "sparse_indices", "sparse_values"]
    pf = pq.ParquetFile(parquet_path)
    total = pf.metadata.num_rows
    loaded = 0
    rows_without_sparse = 0
    start = time.perf_counter()
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        cols = batch.to_pydict()
        points = []
        for row_id, cid, text, nt, tag, fsn, emb, sp_idx, sp_val in zip(
            cols["row_id"], cols["concept_id"], cols["text"], cols["name_type"],
            cols["semantic_tag"], cols["fsn"], cols["embedding"],
            cols["sparse_indices"], cols["sparse_values"],
        ):
            vector = {DENSE_VECTOR_NAME: emb}
            if sp_idx:
                vector[SPARSE_VECTOR_NAME] = qmodels.SparseVector(indices=sp_idx, values=sp_val)
            else:
                rows_without_sparse += 1
            points.append(qmodels.PointStruct(
                id=_row_id_to_point_id(row_id),
                vector=vector,
                payload={"row_id": row_id, "concept_id": cid, "text": text, "name_type": nt, "semantic_tag": tag, "fsn": fsn},
            ))
        for attempt in range(1, 6):
            try:
                client.upsert(collection_name=collection_name, points=points, wait=True)
                break
            except Exception as e:  # idempotent: same point ids overwrite
                if attempt == 5:
                    raise
                print(f"  [qdrant] upsert failed (attempt {attempt}/5): {e!r:.200}; retrying", flush=True)
                time.sleep(10 * attempt)
        loaded += len(points)
        print(f"  [qdrant] loaded {loaded}/{total} rows ({time.perf_counter() - start:.1f}s elapsed)")
    if rows_without_sparse:
        print(f"  [qdrant] !!! {rows_without_sparse}/{loaded} rows have no sparse vector -- "
              f"build the corpus with sparse_model=SparseTextEmbedding('Qdrant/bm25') for hybrid search")
    # Re-enable indexing (Qdrant default threshold) so the HNSW graph is built now.
    client.update_collection(collection_name,
                             optimizers_config=qmodels.OptimizersConfigDiff(indexing_threshold=20000))
    n = client.count(collection_name, exact=True).count
    if n < loaded:
        raise RuntimeError(f"qdrant holds {n} points, expected {loaded} (point-id collision or failed upsert)")
    print(f"[qdrant] load complete: {loaded} rows ({time.perf_counter() - start:.1f}s total)")
    return loaded


def build_store(client: QdrantClient, embedding_model, collection_name: str = COLLECTION_NAME,
                sparse_model=None) -> QdrantConceptStore:
    return QdrantConceptStore(client, embedding_model, collection_name=collection_name, sparse_model=sparse_model)
