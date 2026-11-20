from typing import List, Optional, Set

from langchain_core.documents import Document
from langchain_neo4j import Neo4jVector
from neo4j import GraphDatabase

from Src.stores.base import ConceptStore


class Neo4jConceptStore(ConceptStore):
    """Behavior-preserving extraction of the original Neo4j-only retrieval/FSN logic.

    This is a refactor-by-extraction, not new logic: `similarity_search` reproduces
    the exact expand-fetch -> Cypher semantic_tag lookup -> filter -> fallback-if-too-few
    algorithm that used to live in Src/retrieval.py::search_top_k_names, and `get_fsn`
    reproduces the inline Cypher FSN lookup that used to live (duplicated) in
    Src/link.py::link_rerank_entity/link_topone. Baseline results must stay reproducible.
    """

    supports_hybrid = False

    def __init__(self, neo4j_index: Neo4jVector, neo4j_url: str, neo4j_user: str, neo4j_password: str):
        self._index = neo4j_index
        self._url = neo4j_url
        self._user = neo4j_user
        self._password = neo4j_password

    def similarity_search(
        self,
        query_text: str,
        k: int,
        allowed_semantic_tags: Optional[Set[str]] = None,
        min_hits: int = 3,
        expand_factor: float = 2.0,
    ) -> List[Document]:
        if not allowed_semantic_tags:
            print("<!> No domain filter provided — skipping tag filter.")
            return self._index.similarity_search(query_text, k=k)

        fetch_n = int(k * expand_factor)
        raw_results = self._index.similarity_search(query_text, k=fetch_n)

        concept_ids = [
            doc.metadata["concept_id"]
            for doc in raw_results
            if "concept_id" in doc.metadata
        ]
        if not concept_ids:
            return raw_results[:k]

        driver = GraphDatabase.driver(self._url, auth=(self._user, self._password))
        with driver.session() as session:
            records = session.run(
                """
                UNWIND $ids AS id
                MATCH (c:Concept {concept_id: id})
                RETURN c.concept_id AS id, c.semantic_tag AS tag
                """,
                ids=concept_ids,
            ).data()
        driver.close()

        tag_map = {rec["id"]: rec["tag"].lower() for rec in records}

        filtered = [
            doc for doc in raw_results
            if tag_map.get(doc.metadata["concept_id"], "") in allowed_semantic_tags
        ]

        if len(filtered) < min_hits:
            print(f"<!> Only {len(filtered)} filtered hits — returning unfiltered top {k}.")
            return raw_results[:k]

        return filtered[:k]

    def hybrid_search(self, *args, **kwargs) -> List[Document]:
        raise NotImplementedError("Neo4jConceptStore does not support hybrid search (baseline kept unchanged)")

    def get_fsn(self, concept_id: Optional[str], hint_doc: Optional[Document] = None) -> Optional[str]:
        if concept_id is None:
            return None
        driver = GraphDatabase.driver(self._url, auth=(self._user, self._password))
        fsn = None
        with driver.session() as session:
            record = session.run(
                """
                MATCH (c:Concept {concept_id: $id})
                RETURN c.FSN AS fsn
                """,
                id=concept_id,
            ).single()
            if record:
                fsn = record["fsn"]
        driver.close()
        return fsn
