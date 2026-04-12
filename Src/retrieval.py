from typing import List, Dict, Any
from neo4j import GraphDatabase
from langchain_core.documents import Document
from langchain_neo4j import Neo4jVector 



TYPE_TO_SEMANTIC_TAG: Dict[str, List[str]] = {
    "findings":    ["finding", "disorder", "situation", "morphologic abnormality"],
    "procedure":   ["procedure", "regime/therapy", "situation"],
    "medication":  ["substance", "product", "medicinal product", "clinical drug", "organism"],
    "laboratory":  ["procedure", "substance"],
    "anatomy":     ["body structure", "cell structure", "morphologic abnormality"],
    "event":       ["event", "physical force"],
    "observable":  ["observable entity", "record artifact", "organism", "physical object"],
    "score":       ["staging scale", "assessment scale", "observable entity"],
}
def extract_context(start_index: int, end_index: int, document: str, range_size: int = 20) -> str:
    before_start = max(0, start_index - range_size)
    after_end = min(len(document), end_index + range_size)
    before = document[before_start:start_index]
    entity = document[start_index:end_index]
    after = document[end_index:after_end]
    return f"{before}**{entity}**{after}"

def search_top_k_names(
    entity: Dict[str, Any],
    neo4j_index: Neo4jVector,
    neo4j_url: str,
    neo4j_user: str,
    neo4j_password: str,
    k: int = 10,
    min_hits: int = 3,
    expand_factor: float = 2.0,
) -> List[Document]:
    """
    1. Vector-search Description nodes for entity["formal_term"] (fetching k*expand_factor).
    2. Look up each parent Concept.semantic_tag via a single blocking Cypher.
    3. Keep only those matching TYPE_TO_SEMANTIC_TAG[entity["domain"]].
    4. If fewer than min_hits survive, return unfiltered top-k.
    """
    text = entity["formal_term"]
    domain = entity.get("domain", "")
    allowed = {t.lower() for t in TYPE_TO_SEMANTIC_TAG.get(domain, [])}
    
    # If domain not recognized, just return raw top-k
    if not allowed:
        print(f"<!> Unknown domain '{domain}' — skipping tag filter.")
        return neo4j_index.similarity_search(text, k=k)
    
    # 1) fetch extra results so we can filter down
    fetch_n = int(k * expand_factor)
    raw_results = neo4j_index.similarity_search(text, k=fetch_n)
    
    # 2) collect concept_ids
    concept_ids = [
        doc.metadata["concept_id"]
        for doc in raw_results
        if "concept_id" in doc.metadata
    ]
    if not concept_ids:
        return raw_results[:k]
    
    # 3) blocking Cypher to fetch semantic_tag for each concept
    driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))
    with driver.session() as session:
        records = session.run(
            """
            UNWIND $ids AS id
            MATCH (c:Concept {concept_id: id})
            RETURN c.concept_id AS id, c.semantic_tag AS tag
            """,
            ids=concept_ids
        ).data()  # List[Dict[str,Any]]
    driver.close()
    
    tag_map = {rec["id"]: rec["tag"].lower() for rec in records}
    
    # 4) filter by allowed tags
    filtered = [
        doc for doc in raw_results
        if tag_map.get(doc.metadata["concept_id"], "") in allowed
    ]
    
    # 5) fallback if too few
    if len(filtered) < min_hits:
        print(f"<!> Only {len(filtered)} filtered hits — returning unfiltered top {k}.")
        return raw_results[:k]
    
    # 6) return best k of the filtered (already in similarity order)
    return filtered[:k]