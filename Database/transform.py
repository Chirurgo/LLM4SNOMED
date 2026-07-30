import asyncio
from tqdm.asyncio import tqdm
import polars as pl
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_core.documents import Document
from tqdm.auto import tqdm
from neo4j import AsyncGraphDatabase, GraphDatabase
import re
import unicodedata
from typing import Dict, Any, List, Tuple



def reformat_snomed(data_path: str) -> Tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
        def _scan_active_snomed(filename: str) -> pl.LazyFrame:
            lf = pl.scan_csv(source=filename,
                            encoding="utf8",
                            separator="\t",
                            infer_schema_length=10_000,
                            quote_char=None
                            )
            lf = lf.filter(
                pl.col("active").cast(pl.Int8) == 1
            )
            return lf
        accept_tag = ["finding", "disorder", "situation", "morphologic abnormality", "procedure", "regime/therapy", 
                    "substance", "product", "medicinal product", "clinical drug", "organism", "body structure", 
                    "cell", "event", "observable entity", "record artifact", "physical object", 
                    "staging scale", "assessment scale", "physical force"]
        terms_path = data_path + "/Snapshot/Terminology/sct2_Concept_Snapshot_INT_20250301.txt"
        descs_path = data_path + "/Snapshot/Terminology/sct2_Description_Snapshot-en_INT_20250301.txt"
        relat_path = data_path + "/Snapshot/Terminology/sct2_Relationship_Snapshot_INT_20250301.txt"
        active_terms = _scan_active_snomed(terms_path)
        active_descs = _scan_active_snomed(descs_path)
        active_relat = _scan_active_snomed(relat_path)

        active_concp = active_terms.join(active_descs, 
                                        left_on="id", 
                                        right_on="conceptId", 
                                        how="inner")
        active_concp = active_concp.rename({
                                        "id":"concept_id",
                                        "id_right": "term_id", 
                                        "term": "concept_name", 
                                        "typeId": "name_type"
                                        })

        active_synon = (
                    active_concp
                        .with_columns(
                            pl.col('name_type')
                                .replace_strict({
                                        900000000000003001: "FSN",
                                        900000000000013009: "Synonym"
                                        },
                                        default=None)
                            )
                        .drop_nulls('name_type')
                        .select([
                            "concept_id","name_type","concept_name"
                        ])
        )
        active_concp = (
                    active_synon
                        .filter(pl.col('name_type') == "FSN")
                        .with_columns(
                            pl.col('concept_name')
                            .str.extract(r"\((\w+\s?.?\s?\w+.?\w+.?\w+.?)\)$")
                            .alias('semantic_tag'),
                            pl.lit("concept").alias("LABEL")
                        )
                        .drop('name_type')
                        .rename({"concept_name":"FSN"})
        )
        concepts = (
            pl.Series(active_concp
            .filter(
                pl.col("semantic_tag").is_in(accept_tag)
            )
            .select("concept_id").unique().collect(engine='streaming')).to_list()
        )
        active_synon = active_synon.filter(pl.col("concept_id").is_in(concepts))
        values = pl.Series(active_relat.select('typeId').unique().collect(engine='streaming')).to_list()
        mapping = (
            active_concp
            .filter(pl.col('concept_id').is_in(values))  # Filter by concept_id values
            .with_columns(
                pl.col('concept_id').cast(pl.Int64),  # Cast concept_id to Int64
                pl.col('FSN').cast(pl.Utf8)
                .str.replace(r"\((\w+\s?.?\s?\w+.?\w+.?\w+.?)\)$","")
                .str.strip_chars()
                
            )
            .select(["concept_id", "FSN"])  # Select the columns you need
        )
        mapping = (
            mapping
            .with_columns(
                  pl.col("FSN")
                  .str.replace(" ","_")
                  .str.to_lowercase()
            )   
            .collect(engine='streaming')  # Collect into a DataFrame
            .to_dict()  # Convert to dictionary
        )

        mapping_dict = dict(zip(mapping["concept_id"], mapping["FSN"]))
        active_relat = (
                    active_relat
                        .with_columns(
                            pl.col('typeId')
                            .replace_strict(
                                mapping_dict,
                                default = None
                            )
                            .alias('TYPE')
                        )
                        .drop_nulls('TYPE')               
        )
        relat = active_relat.select(['sourceId','destinationId','TYPE'])
        return active_concp, active_synon, relat



async def awrite_concept_nodes(
    lf: pl.LazyFrame,
    neo4j_url: str,
    neo4j_user: str,
    neo4j_password: str,
    batch_size: int = 10_000,
) -> None:
    """
    Asynchronously writes Concept nodes (with metadata) into Neo4j.
    Each row in `lf` should have: concept_id, FSN, semantic_tag, LABEL.
    """
    df = await asyncio.to_thread(lf.collect)
    total = df.height

    driver = AsyncGraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))
    async with driver:
        async with driver.session() as session:
            pbar = tqdm(total=total, desc="Writing Concept Nodes")
            start = 0

            while start < total:
                batch_df = await asyncio.to_thread(lambda: df[start : start + batch_size])

                rows = batch_df.select([
                    pl.col("concept_id"),
                    pl.col("FSN"),
                    pl.col("semantic_tag"),
                    pl.col("LABEL")
                ]).to_dicts()

                if rows:
                    # 6. Write batch in one UNWIND
                    cypher = """
                    UNWIND $rows AS r
                    MERGE (c:Concept {concept_id: r.concept_id})
                    SET c.FSN = r.FSN,
                        c.semantic_tag = r.semantic_tag,
                        c.LABEL = r.LABEL
                    """
                    await session.run(cypher, rows=rows)

                pbar.update(len(rows))
                start += batch_size

            pbar.close()


async def awrite_description_nodes(
    lf: pl.LazyFrame,
    embedding_model,
    neo4j_url: str,
    neo4j_user: str,
    neo4j_password: str,
    batch_size: int = 10_000,
) -> None:
    """
    Writes Description nodes (with embeddings) and
    links them to Concept nodes in Neo4j, in batches.
    """
    # 1. Clean & collect once
    lf_filtered = (
        lf
        .with_columns([
            pl.col("concept_name").cast(pl.Utf8).str.strip_chars(),
            pl.col("name_type")
              .cast(pl.Utf8)
              .str.strip_chars()
              .fill_null("Synonym")
        ])
        .drop_nulls(["concept_id", "concept_name"])
    )
    df_all = await asyncio.to_thread(lf_filtered.collect)
    total = df_all.height

    # 2. Init vector store once
    vector_store = await Neo4jVector.afrom_documents(
        documents=[],
        embedding=embedding_model,
        url=neo4j_url,
        username=neo4j_user,
        password=neo4j_password,
        database="neo4j",
        node_label="Description",
        embedding_node_property="embedding",
        text_node_property="text",
        index_name="name_embedding",
        pre_delete_collection=True,
    )

    # 3. Open Neo4j driver & session once
    driver = AsyncGraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))
    session = driver.session()

    # 4. Batch loop
    pbar = tqdm(total=total, desc="Writing Description Nodes")
    for start in range(0, total, batch_size):
        # a) slice and collect in thread
        batch_df = await asyncio.to_thread(lambda: df_all[start:start + batch_size])
        rows = batch_df.to_dicts()

        # b) build Document list
        documents = [
            Document(
                page_content=r["concept_name"],
                metadata={
                    "concept_id": r["concept_id"],
                    "name": r["concept_name"],
                    "type": r["name_type"],
                }
            )
            for r in rows
        ]

        # c) upload embeddings
        try:
            if documents:
                await vector_store.aadd_documents(documents)
        except Exception as e:
            print(f"[Batch {start}] Embedding upload error: {e}")

        # d) link descriptions to concepts
        #    use the same `text_node_property` ("text") to match Description nodes
        try:
            params = [
                {"concept_id": r["concept_id"], "text": r["concept_name"]}
                for r in rows
            ]
            await session.run(
                """
                UNWIND $batch AS row
                MATCH (c:Concept {concept_id: row.concept_id})
                MATCH (d:Description {text: row.text})
                MERGE (c)-[:HAS_NAME]->(d)
                """,
                batch=params
            )
        except Exception as e:
            print(f"[Batch {start}] Relationship merge error: {e}")

        # e) update progress
        pbar.update(len(rows))

    # 5. Cleanup
    pbar.close()
    await session.close()
    await driver.close()

def normalize_rel_type(s: str) -> str:
    """Normalize arbitrary strings to a safe Neo4j relationship type."""
    if s is None:
        return ""
    # Unicode fold, strip, lowercase
    s = unicodedata.normalize("NFKC", str(s)).strip().lower()
    # Non-alnum -> underscore
    s = re.sub(r"[^a-z0-9]+", "_", s)
    # Collapse repeats and trim
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        return ""
    if not s[0].isalpha():
        s = "r_" + s
    return s.upper()

async def awrite_relationships(
    lf: pl.LazyFrame,
    neo4j_url: str,
    neo4j_user: str,
    neo4j_password: str,
    batch_size: int = 10_000,
) -> None:
    """
    Write relationships between Concept nodes without APOC.
    Expected columns:
      - sourceId
      - destinationId
      - TYPE (relationship type, e.g., "IS_A", "MAPS_TO")
    Any additional columns become relationship properties.
    """
    REL_TYPE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")  # conservative and safe

    try:
        total_rows = await asyncio.to_thread(lambda: lf.select(pl.len()).collect().item())
        pbar = tqdm(total=total_rows, desc="Writing Relationships")
        skip = 0
        start = 0
        driver = AsyncGraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))
        async with driver:
            async with driver.session() as session:
                while True:
                    batch_lf = lf.slice(offset=start, length=batch_size)
                    batch_df = await asyncio.to_thread(batch_lf.collect)
                    rows = batch_df.height
                    if rows == 0:
                        break

                    batch_dicts: List[Dict[str, Any]] = batch_df.to_dicts()

                    # Group rows by normalized relationship type
                    grouped: Dict[str, List[Dict[str, Any]]] = {}
                    for row in batch_dicts:
                        src = row.get("sourceId")
                        dst = row.get("destinationId")
                        raw_type = row.get("TYPE")

                        if not (src and dst and raw_type):
                            skip += 1
                            continue

                        rel_type = normalize_rel_type(raw_type)
                        if not rel_type or not REL_TYPE_RE.match(rel_type):
                            # Skip truly unsalvageable types
                            skip += 1
                            continue

                        # Relationship properties = all extras + type_raw for traceability
                        props = {
                            k: v for k, v in row.items()
                            if k not in ("sourceId", "destinationId", "TYPE")
                        }
                        props["type_raw"] = raw_type

                        grouped.setdefault(rel_type, []).append(
                            {"sourceId": src, "destinationId": dst, "props": props}
                        )

                    # Execute per relationship type (Cypher must inline the type token)
                    for rel_type, payload in grouped.items():
                        cypher = f"""
                        UNWIND $rows AS row
                        MATCH (source:Concept {{concept_id: row.sourceId}})
                        MATCH (target:Concept {{concept_id: row.destinationId}})
                        MERGE (source)-[r:{rel_type}]->(target)
                        SET r += row.props
                        RETURN count(r) AS upserted
                        """
                        await session.run(cypher, rows=payload)

                    start += batch_size
                    pbar.update(rows)
                    if rows < batch_size:
                        break
        print(f"Skipped {skip} invalid relationship rows.")
        pbar.close()

    except Exception as e:
        print(f"Error writing relationships: {e}")


def delete_all_in_batches(uri, user, password, database="neo4j", batch_size=30000, pause_sec=0.0):
    """
    Repeatedly runs:
        MATCH (n) WITH n LIMIT $batch DETACH DELETE n
    until no more nodes are deleted.

    Returns a dict with total deleted counts and loop iterations.
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))
    total_nodes = 0
    total_rels = 0
    loops = 0

    try:
        with driver.session(database=database) as session:
            while True:
                result = session.run(
                    "MATCH (n) WITH n LIMIT $batch DETACH DELETE n",
                    batch=batch_size,
                )
                summary = result.consume()
                deleted_nodes = summary.counters.nodes_deleted
                deleted_rels = summary.counters.relationships_deleted
                loops += 1
                total_nodes += deleted_nodes
                total_rels += deleted_rels

                if deleted_nodes == 0:
                    break
                if pause_sec > 0:
                    import time
                    time.sleep(pause_sec)
    finally:
        driver.close()

    return {"nodes_deleted": total_nodes, "relationships_deleted": total_rels, "loops": loops}