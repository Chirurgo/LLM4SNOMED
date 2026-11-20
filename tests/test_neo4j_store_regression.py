"""
Regression check for the Src/stores refactor (Phase 1 of the multi-backend benchmark).

Verifies that Neo4jConceptStore.similarity_search / get_fsn reproduce, byte-for-byte,
the exact pre-refactor algorithm that used to live in Src/retrieval.py::search_top_k_names
and the inline FSN lookups in Src/link.py. No real Neo4j/SNOMED instance is required:
GraphDatabase.driver is monkeypatched so both the reference copy and the new store see
identical canned Cypher responses.

Run directly: python tests/test_neo4j_store_regression.py
"""
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document
from neo4j import GraphDatabase

from Src.domain_mapping import TYPE_TO_SEMANTIC_TAG, allowed_tags_for_domain
from Src.stores.neo4j_store import Neo4jConceptStore


# ---------------------------------------------------------------------------
# Reference copy of the ORIGINAL (pre-refactor) algorithm, for diffing against.
# ---------------------------------------------------------------------------

def reference_search_top_k_names(
    entity: Dict[str, Any],
    neo4j_index,
    neo4j_url: str,
    neo4j_user: str,
    neo4j_password: str,
    k: int = 10,
    min_hits: int = 3,
    expand_factor: float = 2.0,
) -> List[Document]:
    text = entity["formal_term"]
    domain = entity.get("domain", "")
    allowed = {t.lower() for t in TYPE_TO_SEMANTIC_TAG.get(domain, [])}

    if not allowed:
        print(f"<!> Unknown domain '{domain}' — skipping tag filter.")
        return neo4j_index.similarity_search(text, k=k)

    fetch_n = int(k * expand_factor)
    raw_results = neo4j_index.similarity_search(text, k=fetch_n)

    concept_ids = [
        doc.metadata["concept_id"]
        for doc in raw_results
        if "concept_id" in doc.metadata
    ]
    if not concept_ids:
        return raw_results[:k]

    driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))
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
        if tag_map.get(doc.metadata["concept_id"], "") in allowed
    ]

    if len(filtered) < min_hits:
        print(f"<!> Only {len(filtered)} filtered hits — returning unfiltered top {k}.")
        return raw_results[:k]

    return filtered[:k]


def reference_get_fsn(concept_id, neo4j_url, neo4j_user, neo4j_password):
    if concept_id is None:
        return None
    driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))
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


# ---------------------------------------------------------------------------
# Fakes: a fake vector index + a fake Neo4j driver/session, no real DB needed.
# ---------------------------------------------------------------------------

POOL = [Document(page_content=f"name{i}", metadata={"concept_id": str(i)}) for i in range(1, 7)]
TAG_MAP = {"1": "disorder", "2": "event", "3": "finding", "4": "substance", "5": "situation", "6": "finding"}
FSN_MAP = {"3": "Test finding concept (disorder)"}


class FakeVectorIndex:
    def similarity_search(self, query_text, k):
        return POOL[:k]


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def data(self):
        return self._rows

    def single(self):
        return self._rows[0] if self._rows else None


class FakeSession:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run(self, query, **kwargs):
        if "UNWIND $ids" in query:
            ids = kwargs["ids"]
            return FakeResult([{"id": i, "tag": TAG_MAP[i]} for i in ids if i in TAG_MAP])
        if "RETURN c.FSN" in query:
            cid = kwargs["id"]
            if cid in FSN_MAP:
                return FakeResult([{"fsn": FSN_MAP[cid]}])
            return FakeResult([])
        raise AssertionError(f"Unexpected query: {query}")


class FakeDriver:
    def session(self):
        return FakeSession()

    def close(self):
        pass


def _fake_driver_factory(*args, **kwargs):
    return FakeDriver()


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def _ids(docs):
    return [d.metadata["concept_id"] for d in docs]


def run_all():
    with patch.object(GraphDatabase, "driver", side_effect=_fake_driver_factory):
        index = FakeVectorIndex()
        store = Neo4jConceptStore(index, "bolt://fake", "user", "pw")

        # Scenario 1: normal domain filter, enough hits survive -> filtered + truncated to k
        entity = {"formal_term": "chest pain", "domain": "findings"}
        allowed = allowed_tags_for_domain(entity["domain"])
        old = reference_search_top_k_names(entity, index, "bolt://fake", "user", "pw", k=3)
        new = store.similarity_search(entity["formal_term"], k=3, allowed_semantic_tags=allowed)
        assert _ids(old) == _ids(new) == ["1", "3", "5"], (_ids(old), _ids(new))

        # Scenario 2: domain filter with too few hits -> fallback to unfiltered top-k
        entity = {"formal_term": "incident", "domain": "event"}
        allowed = allowed_tags_for_domain(entity["domain"])
        old = reference_search_top_k_names(entity, index, "bolt://fake", "user", "pw", k=3)
        new = store.similarity_search(entity["formal_term"], k=3, allowed_semantic_tags=allowed)
        assert _ids(old) == _ids(new) == ["1", "2", "3"], (_ids(old), _ids(new))

        # Scenario 3: unknown domain -> no filter, direct similarity_search(k=k) (not k*expand_factor)
        entity = {"formal_term": "widget", "domain": "nonexistent_domain"}
        allowed = allowed_tags_for_domain(entity["domain"]) or None
        old = reference_search_top_k_names(entity, index, "bolt://fake", "user", "pw", k=3)
        new = store.similarity_search(entity["formal_term"], k=3, allowed_semantic_tags=allowed)
        assert _ids(old) == _ids(new) == ["1", "2", "3"], (_ids(old), _ids(new))

        # Scenario 4: FSN lookup, hit and miss, plus None concept_id short-circuit
        assert reference_get_fsn("3", "bolt://fake", "user", "pw") == store.get_fsn("3") == "Test finding concept (disorder)"
        assert reference_get_fsn("99", "bolt://fake", "user", "pw") == store.get_fsn("99") is None
        assert store.get_fsn(None) is None

    print("All Neo4jConceptStore regression scenarios match the pre-refactor reference. PASS")


if __name__ == "__main__":
    run_all()
