from typing import Dict, Any, Literal
import json
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from neo4j import GraphDatabase

from Src.retrieval import search_top_k_names

def rerank_candidates(reranker, query_text: str, docs: list[Document], top_n: int = 20) -> list[Document]:
    pairs = [[query_text, d.page_content] for d in docs]
    scores = reranker.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    # dedupe by concept_id while keeping best-scoring description
    seen = set()
    unique = []
    for d, _ in ranked:
        cid = d.metadata.get("concept_id")
        if cid not in seen:
            seen.add(cid)
            unique.append(d)
        if len(unique) == top_n:
            break
    return unique

class Concept(BaseModel):
    """Represents the model selection of SNOMED CT concept."""
    
    choice: Literal[1,2,3,4,5] = Field(
        description="The only choice from 5 options of SNOMED CT description"
    )
                                         
    mapping_reason: str = Field(
        description="The reasoning from the model for deciding the final choice."
    )

def link_rerank_entity(
    entity: Dict[str, Any],
    neo4j_index: Neo4jVector,
    neo4j_url: str,
    neo4j_user: str,
    neo4j_password: str,
    k_candidate: int,
    llm,
    reranker,
) -> Dict[str, Any]:

    # --- 1) set up the LLM chain ---
    link_parser = JsonOutputParser(pydantic_object=Concept, strict=True)
    link_prompt = PromptTemplate(
        input_variables=["entity", "term", "doc1", "doc2", "doc3", "doc4", "doc5"],
        partial_variables={"map_format": link_parser.get_format_instructions()},
        template="""
You are a clinical language model. Your function is to map a medical entity to the most precise SNOMED CT concept.

Given an extracted **Term** and the **Entity**, and three SNOMED CT candidate snippets, choose the one snippet that best matches the semantic meaning of the **Entity**.

Entity: "{entity}" 
Term: "{term}"

Options:
1. {doc1}
2. {doc2}
3. {doc3}
4. {doc4}
5. {doc5}

**RESPONSE INSTRUCTIONS:**
You must respond with a JSON object that strictly adheres to the following format. 
The value for 'choice' must be one of the numbers from 1 to 5.
JSON format:
{map_format}

Respond with only the JSON object. Do not include any other text or explanation outside of the JSON.
""",
    )
    link_chain = link_prompt | llm | link_parser

    # --- 2) retrieve candidates ---
    top_docs = search_top_k_names(
        entity, 
        neo4j_index, 
        neo4j_url, 
        neo4j_user, 
        neo4j_password,
        k_candidate,
    )


    query = entity["formal_term"]
    top_docs = rerank_candidates(reranker=reranker, query_text=query, docs=top_docs, top_n=20)
    
    if len(top_docs) < 2:
        return {
            **entity, 
            "concept_name": None, 
            "matched_name": None,
            "concept_id": None, 
            "tops": None, 
            "llm_choice": None
            }

    # pad to exactly 5
    while len(top_docs) < 5:
        top_docs.append(Document(page_content="(no content)", metadata={"concept_id": None}))

    # --- 3) invoke the LLM ---
    doc_texts = [d.page_content for d in top_docs]
    prompt_inputs = {
        "entity": entity["formal_term"],
        "term": entity["term"],
        **{f"doc{i+1}": doc_texts[i] for i in range(5)}
    }
    idx = 0
    default_concept = Concept(
        choice=1,
        mapping_reason="LLM response failed or could not be parsed as JSON."
    )
    parsed_concept=default_concept
    response=None
    try:
        response = link_chain.invoke(prompt_inputs)
        parsed_concept = Concept(**response)
        idx = parsed_concept.choice - 1 
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        print(f"<!!!> Error mapping '{entity.get('term')}': {e}. Response: {response}")


    chosen = top_docs[idx]
    concept_id = chosen.metadata.get("concept_id")
    fsn = None
    if concept_id is not None:
        driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))
        with driver.session() as session:
            record = session.run(
                """
                MATCH (c:Concept {concept_id: $id})
                RETURN c.FSN AS fsn
                """,
                id=concept_id
            ).single()
            if record:
                fsn = record["fsn"]
        driver.close()
    return {
        **entity,
        "concept_name": fsn, 
        "matched_name": chosen.page_content,
        "concept_id": concept_id,
        "llm_choice": parsed_concept.choice,
        "mapping_reason": parsed_concept.mapping_reason,
        "tops": [
            {
                "content": d.page_content,
                "concept_id": d.metadata.get("concept_id")
            }
            for d in top_docs
        ],
    }


def link_topone(
    entity: Dict[str, Any],
    neo4j_index: Neo4jVector,
    neo4j_url: str,
    neo4j_user: str,
    neo4j_password: str,
    k_candidate: int,
    reranker,
) -> Dict[str, Any]:

    # --- 1) set up the LLM chain --- NO LLM reranking, just pick top 1 from reranker
    link_parser = JsonOutputParser(pydantic_object=Concept, strict=True)

    # --- 2) retrieve candidates ---
    top_docs = search_top_k_names(
        entity, 
        neo4j_index, 
        neo4j_url, 
        neo4j_user, 
        neo4j_password,
        k_candidate,
    )
    query = entity["formal_term"]
    top_docs = rerank_candidates(reranker=reranker, query_text=query, docs=top_docs, top_n=20)
    
    if len(top_docs) < 2:
        return {
            **entity, 
            "concept_name": None, 
            "matched_name": None,
            "concept_id": None, 
            "tops": None, 
            "llm_choice": None
            }

    # pad to exactly 5
    while len(top_docs) < 5:
        top_docs.append(Document(page_content="(no content)", metadata={"concept_id": None}))

    # --- 3) invoke the LLM ---
    chosen = top_docs[0]
    concept_id = chosen.metadata.get("concept_id")
    fsn = None
    if concept_id is not None:
        driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))
        with driver.session() as session:
            record = session.run(
                """
                MATCH (c:Concept {concept_id: $id})
                RETURN c.FSN AS fsn
                """,
                id=concept_id
            ).single()
            if record:
                fsn = record["fsn"]
        driver.close()
    return {
        **entity,
        "concept_name": fsn, 
        "matched_name": chosen.page_content,
        "concept_id": concept_id,
        "llm_choice": 1,
        "mapping_reason": "Top-1 candidate selected without LLM reranking.",
        "tops": [
            {
                "content": d.page_content,
                "concept_id": d.metadata.get("concept_id")
            }
            for d in top_docs
        ],
    }