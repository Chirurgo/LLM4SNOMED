"""Bulk-loads the cached corpus parquet (Database/build_corpus.py) into Postgres+pgvector.

Hybrid search's `tsvector` column and GIN index are created here (Phase 3), but the
Reciprocal Rank Fusion query logic itself is added to Src/stores/postgres_store.py in
Phase 4; the table schema already carries everything the fusion query will need.
"""
import time

import pyarrow.parquet as pq

from Src.stores.postgres_store import EMBEDDING_DIM, connect, PostgresConceptStore

TABLE_NAME = "descriptions"


def create_schema(conn, table_name: str = TABLE_NAME, drop_existing: bool = True) -> None:
    if drop_existing:
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.execute(
        f"""
        CREATE TABLE {table_name} (
            row_id TEXT PRIMARY KEY,
            concept_id TEXT NOT NULL,
            text TEXT NOT NULL,
            name_type TEXT NOT NULL,
            semantic_tag TEXT NOT NULL,
            fsn TEXT NOT NULL,
            embedding VECTOR({EMBEDDING_DIM}) NOT NULL,
            text_search tsvector GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED
        )
        """
    )


# HNSW build tuning. With the default maintenance_work_mem (64 MB) the graph for ~900k
# 768-d vectors does not fit and pgvector falls back to a far slower on-disk build.
# Parallel workers need /dev/shm, hence `shm_size` on the compose service.
INDEX_BUILD_SETTINGS = {"maintenance_work_mem": "3GB", "max_parallel_maintenance_workers": "7"}


def create_indexes(conn, table_name: str = TABLE_NAME) -> None:
    for k, v in INDEX_BUILD_SETTINGS.items():
        conn.execute(f"SET {k} = '{v}'")
    conn.execute(f"CREATE INDEX IF NOT EXISTS {table_name}_semantic_tag_idx ON {table_name} (semantic_tag)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS {table_name}_concept_id_idx ON {table_name} (concept_id)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS {table_name}_text_search_idx ON {table_name} USING GIN (text_search)")
    # Built AFTER bulk load -- building an HNSW index before a large insert is much slower.
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS {table_name}_embedding_idx ON {table_name} "
        f"USING hnsw (embedding vector_cosine_ops)"
    )


def load_corpus(conn, parquet_path: str, table_name: str = TABLE_NAME, batch_size: int = 2000,
                build_indexes: bool = True) -> int:
    columns = ["row_id", "concept_id", "text", "name_type", "semantic_tag", "fsn", "embedding"]
    pf = pq.ParquetFile(parquet_path)
    total = pf.metadata.num_rows
    loaded = 0
    start = time.perf_counter()
    with conn.cursor() as cur:
        for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
            cols = batch.to_pydict()
            rows = list(zip(*(cols[c] for c in columns)))
            # The connection is autocommit; without this every row is its own commit.
            with conn.transaction():
                cur.executemany(
                    f"INSERT INTO {table_name} (row_id, concept_id, text, name_type, semantic_tag, fsn, embedding) "
                    f"VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (row_id) DO NOTHING",
                    rows,
                )
            loaded += len(rows)
            print(f"  [postgres] loaded {loaded}/{total} rows ({time.perf_counter() - start:.1f}s elapsed)")
    n = conn.execute(f"SELECT count(*) FROM {table_name}").fetchone()[0]
    if n < loaded:
        raise RuntimeError(f"postgres holds {n} rows, expected {loaded}")
    if build_indexes:
        create_indexes(conn, table_name)
    print(f"[postgres] load complete: {loaded} rows ({time.perf_counter() - start:.1f}s total)")
    return loaded


def build_store(conn, embedding_model, table_name: str = TABLE_NAME) -> PostgresConceptStore:
    return PostgresConceptStore(conn, embedding_model, table_name=table_name)
