"""Bulk-loads the cached corpus parquet (Database/build_corpus.py) into Weaviate.

Collection is created with `Configure.Vectorizer.none()` (bring-your-own-vectors) so
precomputed SapBERT embeddings can be attached directly on insert, never re-embedded.
"""
import time

import pyarrow.parquet as pq
import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.data import DataObject
from weaviate.util import generate_uuid5

from Src.stores.weaviate_store import WeaviateConceptStore

COLLECTION_NAME = "SnomedDescription"


def connect(http_host: str, http_port: int, grpc_port: int):
    # Default init timeout is 2 s; right after a restart Weaviate is still loading the
    # 917k-vector HNSW from disk and the first REST call can take far longer.
    from weaviate.classes.init import AdditionalConfig, Timeout
    return weaviate.connect_to_custom(
        http_host=http_host, http_port=http_port, http_secure=False,
        grpc_host=http_host, grpc_port=grpc_port, grpc_secure=False,
        additional_config=AdditionalConfig(timeout=Timeout(init=180, query=300, insert=300)),
    )


def create_collection(client, collection_name: str = COLLECTION_NAME, drop_existing: bool = True) -> None:
    if drop_existing and client.collections.exists(collection_name):
        client.collections.delete(collection_name)
    client.collections.create(
        name=collection_name,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="row_id", data_type=DataType.TEXT),
            Property(name="concept_id", data_type=DataType.TEXT),
            Property(name="text", data_type=DataType.TEXT),
            Property(name="name_type", data_type=DataType.TEXT),
            Property(name="semantic_tag", data_type=DataType.TEXT),
            Property(name="fsn", data_type=DataType.TEXT),
        ],
    )


def load_corpus(client, parquet_path: str, collection_name: str = COLLECTION_NAME, batch_size: int = 200) -> int:
    collection = client.collections.get(collection_name)
    columns = ["row_id", "concept_id", "text", "name_type", "semantic_tag", "fsn", "embedding"]
    pf = pq.ParquetFile(parquet_path)
    total = pf.metadata.num_rows
    loaded = 0
    start = time.perf_counter()
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        cols = batch.to_pydict()
        objects = [
            DataObject(
                properties={"row_id": row_id, "concept_id": cid, "text": text, "name_type": nt, "semantic_tag": tag, "fsn": fsn},
                uuid=generate_uuid5(row_id),
                vector=emb,
            )
            for row_id, cid, text, nt, tag, fsn, emb in zip(
                cols["row_id"], cols["concept_id"], cols["text"], cols["name_type"],
                cols["semantic_tag"], cols["fsn"], cols["embedding"],
            )
        ]
        response = collection.data.insert_many(objects)
        if response.has_errors:
            errs = list(response.errors.values())
            raise RuntimeError(f"weaviate insert_many: {len(errs)} errors, first: {errs[0]}")
        loaded += len(objects)
        print(f"  [weaviate] loaded {loaded}/{total} rows ({time.perf_counter() - start:.1f}s elapsed)")
    n = collection.aggregate.over_all(total_count=True).total_count
    if n < loaded:
        raise RuntimeError(f"weaviate holds {n} objects, expected {loaded}")
    print(f"[weaviate] load complete: {loaded} rows ({time.perf_counter() - start:.1f}s total)")
    return loaded


def build_store(client, embedding_model, collection_name: str = COLLECTION_NAME) -> WeaviateConceptStore:
    return WeaviateConceptStore(client, embedding_model, collection_name=collection_name)
