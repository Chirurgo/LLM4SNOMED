from typing import Dict, Any, Literal
import json
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from Src.domain_mapping import allowed_tags_for_domain
from Src.stores.base import ConceptStore


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

def retrieve_candidates(
    entity: Dict[str, Any],
    store: ConceptStore,
    k_candidate: int,
    use_hybrid: bool = False,
    min_hits: int = 3,
    expand_factor: float = 2.0,
) -> list[Document]:
    """Stage 3: domain-filtered candidate retrieval for one extracted entity."""
    allowed_semantic_tags = allowed_tags_for_domain(entity.get("domain", "")) or None
    search = store.hybrid_search if (use_hybrid and store.supports_hybrid) else store.similarity_search
    return search(entity["formal_term"], k_candidate, allowed_semantic_tags=allowed_semantic_tags,
                  min_hits=min_hits, expand_factor=expand_factor)


def empty_prediction(entity: Dict[str, Any]) -> Dict[str, Any]:
    return {
        **entity,
        "concept_name": None,
        "matched_name": None,
        "concept_id": None,
        "tops": None,
        "llm_choice": None
        }


def pad_to_five(top_docs: list[Document]) -> list[Document]:
    top_docs = list(top_docs)
    while len(top_docs) < 5:
        top_docs.append(Document(page_content="(no content)", metadata={"concept_id": None}))
    return top_docs


def build_prediction(
    entity: Dict[str, Any],
    top_docs: list[Document],
    choice: int,
    mapping_reason: str,
    store: ConceptStore,
) -> Dict[str, Any]:
    """Final prediction record (the schema Evaluate/*.py reads). `choice` is 1-based."""
    chosen = top_docs[choice - 1]
    concept_id = chosen.metadata.get("concept_id")
    fsn = store.get_fsn(concept_id, hint_doc=chosen)
    return {
        **entity,
        "concept_name": fsn,
        "matched_name": chosen.page_content,
        "concept_id": concept_id,
        "llm_choice": choice,
        "mapping_reason": mapping_reason,
        "tops": [
            {
                "content": d.page_content,
                "concept_id": d.metadata.get("concept_id")
            }
            for d in top_docs
        ],
    }


class Concept(BaseModel):
    """Represents the model selection of SNOMED CT concept."""

    choice: Literal[1,2,3,4,5] = Field(
        description="The only choice from 5 options of SNOMED CT description"
    )

    mapping_reason: str = Field(
        description="The reasoning from the model for deciding the final choice."
    )

def llm_select(entity: Dict[str, Any], top_docs: list[Document], llm) -> tuple[int, str]:
    """Stage 5 (LLM mode): ask the LLM to pick 1 of the first 5 candidates.
    Returns (1-based choice, mapping_reason); falls back to choice 1 if parsing fails."""
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

    return idx + 1, parsed_concept.mapping_reason


def link_rerank_entity(
    entity: Dict[str, Any],
    store: ConceptStore,
    k_candidate: int,
    llm,
    reranker,
    use_hybrid: bool = False,
) -> Dict[str, Any]:

    # --- 1) retrieve candidates ---
    query = entity["formal_term"]
    top_docs = retrieve_candidates(entity, store, k_candidate, use_hybrid)

    top_docs = rerank_candidates(reranker=reranker, query_text=query, docs=top_docs, top_n=20)

    if len(top_docs) < 2:
        return empty_prediction(entity)

    top_docs = pad_to_five(top_docs)

    # --- 2) invoke the LLM ---
    choice, reason = llm_select(entity, top_docs, llm)
    return build_prediction(entity, top_docs, choice, reason, store)


def link_topone(
    entity: Dict[str, Any],
    store: ConceptStore,
    k_candidate: int,
    reranker,
    use_hybrid: bool = False,
) -> Dict[str, Any]:

    # --- 1) retrieve candidates --- NO LLM reranking, just pick top 1 from reranker
    query = entity["formal_term"]
    top_docs = retrieve_candidates(entity, store, k_candidate, use_hybrid)

    top_docs = rerank_candidates(reranker=reranker, query_text=query, docs=top_docs, top_n=20)

    if len(top_docs) < 2:
        return empty_prediction(entity)

    top_docs = pad_to_five(top_docs)

    # --- 2) pick top-1 ---
    return build_prediction(entity, top_docs, 1, TOP1_REASON, store)


TOP1_REASON = "Top-1 candidate selected without LLM reranking."
