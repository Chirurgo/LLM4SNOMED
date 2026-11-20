"""Uniform open/load for every backend, driven by env vars (defaults match
docker-compose.yml). Neo4j is opened from the existing `name_embedding` index built by
01_create_database.ipynb; it is never (re)loaded here."""
import os
import time
from typing import Callable, Dict, Tuple

from Src.stores.base import ConceptStore
from Src.stores.meilisearch_store import MeilisearchConceptStore
from Src.stores.neo4j_store import Neo4jConceptStore
from Src.stores.postgres_store import PostgresConceptStore
from Src.stores.qdrant_store import QdrantConceptStore
from Src.stores.weaviate_store import WeaviateConceptStore
from Benchmark.resource_profile import container_disk_bytes, container_id, container_stats

STORE_CLASSES = {
    "neo4j": Neo4jConceptStore,
    "postgres": PostgresConceptStore,
    "qdrant": QdrantConceptStore,
    "weaviate": WeaviateConceptStore,
    "meilisearch": MeilisearchConceptStore,
}
LOADABLE = ("postgres", "qdrant", "weaviate", "meilisearch")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


POSTGRES_DSN = _env("BENCH_POSTGRES_DSN", "postgresql://llm4snomed:change_this_password@localhost:5433/snomed")
QDRANT_URL = _env("BENCH_QDRANT_URL", "http://localhost:6333")
WEAVIATE_HOST = _env("BENCH_WEAVIATE_HOST", "localhost")
WEAVIATE_HTTP_PORT = int(_env("BENCH_WEAVIATE_HTTP_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(_env("BENCH_WEAVIATE_GRPC_PORT", "50051"))
MEILI_URL = _env("BENCH_MEILI_URL", "http://localhost:7700")
MEILI_API_KEY = _env("BENCH_MEILI_API_KEY", "change_this_master_key")


def _wait_qdrant_indexed(client, collection: str, timeout_s: float = 3600) -> None:
    # Qdrant builds HNSW segments asynchronously after upsert; "green" = fully optimized.
    from qdrant_client.http.models import CollectionStatus
    start = time.perf_counter()
    # Also require the vectors to actually be in HNSW segments: right after indexing is
    # re-enabled the status can still read GREEN before the optimizer has started.
    while True:
        info = client.get_collection(collection)
        indexed = info.indexed_vectors_count or 0
        if info.status == CollectionStatus.GREEN and indexed >= 0.95 * (info.points_count or 0):
            return
        if time.perf_counter() - start > timeout_s:
            raise TimeoutError(f"Qdrant collection {collection} not indexed after {timeout_s}s "
                               f"(status {info.status}, indexed {indexed}/{info.points_count})")
        time.sleep(2)


def load_backend(name: str, parquet_path: str) -> Dict[str, float]:
    """(Re)create the backend's schema and bulk-load the cached corpus. Returns timing and
    footprint: load_s includes index build (Postgres builds indexes after insert;
    Qdrant is waited on until its async HNSW build finishes)."""
    if name not in LOADABLE:
        raise ValueError(f"{name} is not loadable by the benchmark (Neo4j is built by 01_create_database.ipynb)")
    start = time.perf_counter()
    if name == "postgres":
        from Database.loaders import postgres_loader as L
        from Src.stores.postgres_store import connect
        conn = connect(POSTGRES_DSN)
        try:
            L.create_schema(conn, drop_existing=True)
            L.load_corpus(conn, parquet_path)
        finally:
            conn.close()
    elif name == "qdrant":
        from Database.loaders import qdrant_loader as L
        client = L.connect(QDRANT_URL)
        try:
            L.create_collection(client, drop_existing=True)
            L.load_corpus(client, parquet_path)
            _wait_qdrant_indexed(client, L.COLLECTION_NAME)
        finally:
            client.close()
    elif name == "weaviate":
        from Database.loaders import weaviate_loader as L
        client = L.connect(WEAVIATE_HOST, WEAVIATE_HTTP_PORT, WEAVIATE_GRPC_PORT)
        try:
            L.create_collection(client, drop_existing=True)
            L.load_corpus(client, parquet_path)
        finally:
            client.close()
    elif name == "meilisearch":
        from Database.loaders import meilisearch_loader as L
        client = L.connect(MEILI_URL, MEILI_API_KEY)
        L.create_index(client, drop_existing=True)
        L.load_corpus(client, parquet_path)
    load_s = time.perf_counter() - start

    cid = container_id(name)
    idle = container_stats(cid) if cid else None
    disk = container_disk_bytes(name)
    return {
        "backend": name,
        "load_s": load_s,
        "db_disk_mb": disk / 1e6 if disk is not None else None,
        "db_mem_after_load_mb": idle["mem_bytes"] / 1e6 if idle else None,
    }


def open_store(name: str, embedding_model, sparse_model=None) -> Tuple[ConceptStore, Callable[[], None]]:
    """Returns (store, close)."""
    if name == "postgres":
        from Src.stores.postgres_store import connect
        conn = connect(POSTGRES_DSN)
        return PostgresConceptStore(conn, embedding_model), conn.close
    if name == "qdrant":
        from Database.loaders import qdrant_loader as L
        client = L.connect(QDRANT_URL)
        return QdrantConceptStore(client, embedding_model, sparse_model=sparse_model), client.close
    if name == "weaviate":
        from Database.loaders import weaviate_loader as L
        client = L.connect(WEAVIATE_HOST, WEAVIATE_HTTP_PORT, WEAVIATE_GRPC_PORT)
        return WeaviateConceptStore(client, embedding_model), client.close
    if name == "meilisearch":
        from Database.loaders import meilisearch_loader as L
        client = L.connect(MEILI_URL, MEILI_API_KEY)
        return MeilisearchConceptStore(client, embedding_model), (lambda: None)
    if name == "neo4j":
        from langchain_neo4j import Neo4jVector
        url = _env("NEO4J_HOST", "bolt://localhost:7687")
        user = _env("NEO4J_USER", "neo4j")
        password = _env("NEO4J_PASSWORD", "change_this_password")
        # Mirrors 02_main.ipynb exactly, so the baseline stays the published configuration.
        vector = Neo4jVector.from_existing_index(
            url=url, username=user, password=password, embedding=embedding_model,
            database="neo4j", node_label="Concept", text_node_property="text",
            embedding_node_property="embedding", index_name="name_embedding",
        )
        return Neo4jConceptStore(vector, url, user, password), (lambda: None)
    raise ValueError(f"unknown backend {name!r}")
