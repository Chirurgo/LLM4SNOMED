"""Builds the Neo4j baseline graph (Concept / Description / relationships + the
`name_embedding` vector index) for the full-scale benchmark.

Same graph as 01_create_database.ipynb / Database/transform.py produces, with one
deliberate difference: Description embeddings are read from the shared corpus Parquet
(Database/build_corpus.py) instead of being recomputed. They come from the same model and
config (SapBERT, CLS + L2, max_length 128), so the vectors match, and every backend's
indexing time then excludes embedding (which is reported once, for the corpus build).

Faithfulness to the original Neo4jVector.add_embeddings path:
  * Description node key is `id = md5(text)` and rows are MERGEd on it, so descriptions
    whose text is shared by several concepts collapse into one node (the last write wins)
    exactly as before. That affects 2,385 of 917,488 rows on the 20250301 release.
  * Node properties: id, text, embedding, and the metadata the notebook set
    (concept_id as an integer, name, type).
  * Vector index: `name_embedding` on Description.embedding, cosine, 768 dims.

Performance-only additions (no effect on query results): a uniqueness constraint on
Concept.concept_id and a range index on Description.id. Without them every MERGE/MATCH
in the load is a label scan, which is quadratic over ~1M nodes.

The vector index is created after the bulk insert, so its build time is measured on its
own (as with Postgres), rather than being spread across the inserts.
"""
import asyncio
import sys
import time
from hashlib import md5
from typing import Dict

import pyarrow.parquet as pq
from neo4j import GraphDatabase

from Database.transform import (awrite_concept_nodes, awrite_relationships, delete_all_in_batches,
                                reformat_snomed)

DATABASE = "neo4j"
INDEX_NAME = "name_embedding"
EMBEDDING_DIM = 768

# Identical to langchain_community Neo4jVector.add_embeddings' import query.
_IMPORT_QUERY = (
    "UNWIND $data AS row "
    "CALL (row) { WITH row "
    "MERGE (c:`Description` {id: row.id}) "
    "WITH c, row "
    "CALL db.create.setNodeVectorProperty(c, 'embedding', row.embedding) "
    "SET c.`text` = row.text "
    "SET c += row.metadata "
    "} IN TRANSACTIONS OF 1000 ROWS "
)


def wipe(url: str, user: str, password: str) -> Dict[str, int]:
    """Delete every node/relationship and every index/constraint in the `neo4j` database."""
    delete_all_in_batches(uri=url, user=user, password=password, database=DATABASE)
    driver = GraphDatabase.driver(url, auth=(user, password))
    dropped = 0
    try:
        with driver.session(database=DATABASE) as s:
            for r in s.run("SHOW CONSTRAINTS YIELD name").data():
                s.run(f"DROP CONSTRAINT `{r['name']}` IF EXISTS").consume()
                dropped += 1
            for r in s.run("SHOW INDEXES YIELD name, type WHERE type <> 'LOOKUP'").data():
                s.run(f"DROP INDEX `{r['name']}` IF EXISTS").consume()
                dropped += 1
            left = s.run("MATCH (n) RETURN count(n) AS n").single()["n"]
    finally:
        driver.close()
    if left:
        raise RuntimeError(f"wipe left {left} nodes")
    return {"schema_objects_dropped": dropped}


def create_schema(url: str, user: str, password: str) -> None:
    driver = GraphDatabase.driver(url, auth=(user, password))
    try:
        with driver.session(database=DATABASE) as s:
            s.run("CREATE CONSTRAINT concept_id_unique IF NOT EXISTS "
                  "FOR (c:Concept) REQUIRE c.concept_id IS UNIQUE").consume()
            s.run("CREATE INDEX description_id IF NOT EXISTS FOR (d:Description) ON (d.id)").consume()
            s.run("CALL db.awaitIndexes(600)").consume()
    finally:
        driver.close()


def _run_async(coro):
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    return asyncio.run(coro)


def load_concepts_and_relationships(url: str, user: str, password: str, snomed_release_path: str) -> Dict[str, float]:
    """Concept nodes, then typed Concept->Concept relationships, via the original
    Database/transform.py writers (unchanged)."""
    timings = {}
    t = time.perf_counter()
    concepts, _synon, relat = reformat_snomed(snomed_release_path)
    timings["rf2_parse_s"] = time.perf_counter() - t

    t = time.perf_counter()
    _run_async(awrite_concept_nodes(concepts, url, user, password, batch_size=10_000))
    timings["concepts_s"] = time.perf_counter() - t

    t = time.perf_counter()
    _run_async(awrite_relationships(relat, url, user, password, batch_size=10_000))
    timings["relationships_s"] = time.perf_counter() - t
    return timings


def load_descriptions(url: str, user: str, password: str, parquet_path: str, batch_size: int = 5000) -> int:
    """Description nodes (precomputed embeddings) + HAS_NAME links to their Concept."""
    pf = pq.ParquetFile(parquet_path)
    total = pf.metadata.num_rows
    loaded = 0
    start = time.perf_counter()
    driver = GraphDatabase.driver(url, auth=(user, password))
    try:
        with driver.session(database=DATABASE) as s:
            for batch in pf.iter_batches(batch_size=batch_size,
                                         columns=["concept_id", "text", "name_type", "embedding"]):
                cols = batch.to_pydict()
                data, links = [], []
                for cid, text, nt, emb in zip(cols["concept_id"], cols["text"], cols["name_type"], cols["embedding"]):
                    node_id = md5(text.encode("utf-8")).hexdigest()
                    data.append({"id": node_id, "text": text, "embedding": emb,
                                 "metadata": {"concept_id": int(cid), "name": text, "type": nt}})
                    links.append({"concept_id": int(cid), "id": node_id})
                s.run(_IMPORT_QUERY, data=data).consume()
                # The notebook matched on Description.text; with id = md5(text) that is
                # the same node set, but uses the index.
                s.run(
                    "UNWIND $batch AS row "
                    "MATCH (c:Concept {concept_id: row.concept_id}) "
                    "MATCH (d:Description {id: row.id}) "
                    "MERGE (c)-[:HAS_NAME]->(d)",
                    batch=links,
                ).consume()
                loaded += len(data)
                el = time.perf_counter() - start
                print(f"  [neo4j] descriptions {loaded}/{total} ({el:.1f}s elapsed, {loaded / el:.0f} rows/s)", flush=True)
    finally:
        driver.close()
    return loaded


def build_vector_index(url: str, user: str, password: str, poll_s: float = 10.0) -> None:
    """Create `name_embedding` (same statement shape as Neo4jVector.create_new_index) and
    block until it is ONLINE, logging population progress."""
    driver = GraphDatabase.driver(url, auth=(user, password))
    try:
        with driver.session(database=DATABASE) as s:
            s.run(
                f"CREATE VECTOR INDEX {INDEX_NAME} IF NOT EXISTS "
                "FOR (m:`Description`) ON m.`embedding` "
                "OPTIONS { indexConfig: { `vector.dimensions`: toInteger($dim), "
                "`vector.similarity_function`: $sim }}",
                dim=EMBEDDING_DIM, sim="cosine",
            ).consume()
            start = time.perf_counter()
            while True:
                r = s.run("SHOW INDEXES YIELD name, state, populationPercent WHERE name = $n",
                          n=INDEX_NAME).single()
                if r["state"] == "ONLINE":
                    break
                if r["state"] == "FAILED":
                    raise RuntimeError(f"vector index {INDEX_NAME} FAILED")
                print(f"  [neo4j] vector index {r['state']} {r['populationPercent']:.1f}% "
                      f"({time.perf_counter() - start:.0f}s)", flush=True)
                time.sleep(poll_s)
    finally:
        driver.close()


def counts(url: str, user: str, password: str) -> Dict[str, int]:
    driver = GraphDatabase.driver(url, auth=(user, password))
    try:
        with driver.session(database=DATABASE) as s:
            q = lambda c: s.run(c).single()[0]
            return {
                "concepts": q("MATCH (c:Concept) RETURN count(c)"),
                "descriptions": q("MATCH (d:Description) RETURN count(d)"),
                "has_name": q("MATCH ()-[r:HAS_NAME]->() RETURN count(r)"),
                "is_a": q("MATCH ()-[r:IS_A]->() RETURN count(r)"),
                "relationships_total": q("MATCH ()-[r]->() RETURN count(r)"),
            }
    finally:
        driver.close()
