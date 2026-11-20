"""Bulk-loads the cached corpus parquet (Database/build_corpus.py) into Meilisearch.

The index's embedder is configured as "userProvided" so Meilisearch never tries to
compute its own embedding -- every document's precomputed SapBERT vector is attached
directly under `_vectors.<embedder_name>`.
"""
import time

import meilisearch
import pyarrow.parquet as pq

from Src.stores.meilisearch_store import EMBEDDING_DIM, MeilisearchConceptStore

INDEX_NAME = "snomed_descriptions"
EMBEDDER_NAME = "default"


def connect(url: str, api_key: str) -> meilisearch.Client:
    return meilisearch.Client(url, api_key)


def create_index(client: meilisearch.Client, index_name: str = INDEX_NAME, drop_existing: bool = True, embedder_name: str = EMBEDDER_NAME):
    if drop_existing:
        try:  # a crashed earlier load can leave many add_documents tasks queued
            task = client.cancel_tasks({"indexUids": [index_name], "statuses": ["enqueued", "processing"]})
            _wait_ok(client, task.task_uid)
        except Exception:
            pass
        try:
            task = client.index(index_name).delete()
            client.wait_for_task(task.task_uid)
        except Exception:
            pass
    task = client.create_index(index_name, {"primaryKey": "row_id"})
    client.wait_for_task(task.task_uid)

    index = client.index(index_name)
    task = index.update_filterable_attributes(["semantic_tag", "concept_id"])
    client.wait_for_task(task.task_uid)
    task = index.update_embedders({embedder_name: {"source": "userProvided", "dimensions": EMBEDDING_DIM}})
    client.wait_for_task(task.task_uid)
    apply_search_cutoff(client, index_name)
    return index


# Meilisearch's default searchCutoffMs is 1500: a slower search (cold mmap pages on the full
# 917k-vector corpus took ~5 s) silently returns UNRANKED documents in internal id order
# instead of erroring. Raise it so every benchmark query is fully ranked.
SEARCH_CUTOFF_MS = 600_000


def apply_search_cutoff(client: meilisearch.Client, index_name: str = INDEX_NAME) -> None:
    task = client.index(index_name).update_search_cutoff_ms(SEARCH_CUTOFF_MS)
    client.wait_for_task(task.task_uid)


def load_corpus(client: meilisearch.Client, parquet_path: str, index_name: str = INDEX_NAME,
                 embedder_name: str = EMBEDDER_NAME, batch_size: int = 1000, max_in_flight: int = 1,
                 resume: bool = False) -> int:
    """max_in_flight > 1 enqueues several add_documents tasks before waiting, letting
    Meilisearch auto-batch them into one indexing pass (far fewer incremental vector
    index rebuilds on a large corpus). 1 = wait after every batch (old behaviour)."""
    index = client.index(index_name)
    columns = ["row_id", "concept_id", "text", "name_type", "semantic_tag", "fsn", "embedding"]
    pf = pq.ParquetFile(parquet_path)
    total = pf.metadata.num_rows
    skip = 0
    if resume:
        # Rows are sent in Parquet order and Meilisearch applies tasks in order, so once the
        # queue left by an interrupted run drains, the first N rows are exactly the N stored.
        _wait_queue_empty(client, index_name)
        skip = index.get_stats().number_of_documents
        print(f"  [meilisearch] resuming: {skip}/{total} rows already indexed", flush=True)
    loaded = skip
    start = time.perf_counter()
    in_flight = []
    seen = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        seen += batch.num_rows
        if seen <= skip:
            continue
        if seen - batch.num_rows < skip:
            batch = batch.slice(skip - (seen - batch.num_rows))
        cols = batch.to_pydict()
        documents = [
            {
                "row_id": row_id, "concept_id": cid, "text": text, "name_type": nt,
                "semantic_tag": tag, "fsn": fsn,
                "_vectors": {embedder_name: emb},
            }
            for row_id, cid, text, nt, tag, fsn, emb in zip(
                cols["row_id"], cols["concept_id"], cols["text"], cols["name_type"],
                cols["semantic_tag"], cols["fsn"], cols["embedding"],
            )
        ]
        in_flight.append(index.add_documents(documents).task_uid)
        if len(in_flight) >= max_in_flight:
            _wait_ok(client, in_flight.pop(0))
        loaded += len(documents)
        print(f"  [meilisearch] loaded {loaded}/{total} rows ({time.perf_counter() - start:.1f}s elapsed)")
    while in_flight:
        _wait_ok(client, in_flight.pop(0))
    n = index.get_stats().number_of_documents
    if n < loaded:
        raise RuntimeError(f"meilisearch holds {n} documents, expected {loaded}")
    print(f"[meilisearch] load complete: {loaded} rows ({time.perf_counter() - start:.1f}s total)")
    return loaded


def _wait_queue_empty(client: meilisearch.Client, index_name: str, timeout_s: float = 24 * 3600) -> None:
    start = time.perf_counter()
    while True:
        try:
            # raw HTTP: client.get_tasks() fails to parse responses on this client/server pair
            pending = client.http.get(f"tasks?indexUids={index_name}&statuses=enqueued,processing&limit=1")["total"]
        except Exception as e:
            print(f"  [meilisearch] queue poll failed ({e!r:.150}); retrying", flush=True)
            pending = None
        if pending == 0:
            return
        if time.perf_counter() - start > timeout_s:
            raise TimeoutError(f"meilisearch queue for {index_name} not empty after {timeout_s}s")
        if pending:
            print(f"  [meilisearch] waiting for {pending} queued tasks from the previous run", flush=True)
        time.sleep(30)


def _wait_ok(client: meilisearch.Client, task_uid: int, timeout_s: float = 24 * 3600) -> None:
    """Poll a task until it finishes. Transient poll errors (Meilisearch answers 408 / 5xx /
    drops the connection while busy indexing -- crashed the 2026-11-14 load at 700k rows)
    are retried; only a task that ends non-succeeded is fatal."""
    start = time.perf_counter()
    errors = 0
    while True:
        try:
            task = client.get_task(task_uid)
            errors = 0
        except Exception as e:
            errors += 1
            if errors >= 60:
                raise
            print(f"  [meilisearch] polling task {task_uid} failed ({e!r:.150}); retrying", flush=True)
            time.sleep(min(30, 2 * errors))
            continue
        if task.status in ("succeeded", "failed", "canceled"):
            if task.status != "succeeded":
                raise RuntimeError(f"meilisearch task {task_uid} {task.status}: {task.error}")
            return
        if time.perf_counter() - start > timeout_s:
            raise TimeoutError(f"meilisearch task {task_uid} still {task.status} after {timeout_s}s")
        time.sleep(0.5)


def build_store(client: meilisearch.Client, embedding_model, index_name: str = INDEX_NAME,
                 embedder_name: str = EMBEDDER_NAME) -> MeilisearchConceptStore:
    return MeilisearchConceptStore(client, embedding_model, index_name=index_name, embedder_name=embedder_name)
