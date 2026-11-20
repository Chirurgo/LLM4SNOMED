from typing import List, Optional, Set

from langchain_core.documents import Document

from Src.stores.base import ConceptStore, SearchParams

EMBEDDING_DIM = 768


class MeilisearchConceptStore(ConceptStore):
    # Dense = semanticRatio 1.0, hybrid = params.hybrid_weight, on the same native call.
    # Meilisearch exposes no ANN beam-width (ef) knob, no fusion choice, no arm depth.
    supports_hybrid = True
    supported_params = frozenset({"hybrid_weight"})

    def __init__(self, client, embedding_model, index_name: str = "snomed_descriptions",
                 embedder_name: str = "default", params: Optional[SearchParams] = None):
        self._index = client.index(index_name)
        self._embedding_model = embedding_model
        self._embedder_name = embedder_name
        self.configure(params or SearchParams())

    def _embed_query(self, text: str) -> list:
        return self._embedding_model.embed_query(text)

    def _hits_to_documents(self, hits) -> List[Document]:
        return [
            Document(
                page_content=h.get("text", ""),
                metadata={"concept_id": h.get("concept_id"), "semantic_tag": h.get("semantic_tag"), "fsn": h.get("fsn")},
            )
            for h in hits
        ]

    def _query(self, query_text: str, qvec, n: int, semantic_ratio: float) -> List[Document]:
        params = {
            "vector": qvec,
            "hybrid": {"embedder": self._embedder_name, "semanticRatio": semantic_ratio},
            "limit": n,
        }
        result = self._index.search(query_text, params)
        return self._hits_to_documents(result.get("hits", []))

    def similarity_search(
        self,
        query_text: str,
        k: int,
        allowed_semantic_tags: Optional[Set[str]] = None,
        min_hits: int = 3,
        expand_factor: float = 2.0,
    ) -> List[Document]:
        qvec = self._embed_query(query_text)
        return self._expand_filter_fallback(
            lambda n: self._query(query_text, qvec, n, semantic_ratio=1.0),
            k, allowed_semantic_tags, min_hits, expand_factor,
        )

    def hybrid_search(
        self,
        query_text: str,
        k: int,
        allowed_semantic_tags: Optional[Set[str]] = None,
        min_hits: int = 3,
        expand_factor: float = 2.0,
    ) -> List[Document]:
        qvec = self._embed_query(query_text)
        return self._expand_filter_fallback(
            lambda n: self._query(query_text, qvec, n, semantic_ratio=self.params.hybrid_weight),
            k, allowed_semantic_tags, min_hits, expand_factor,
        )

    def get_fsn(self, concept_id: Optional[str], hint_doc: Optional[Document] = None) -> Optional[str]:
        if concept_id is None:
            return None
        if hint_doc is not None:
            fsn = hint_doc.metadata.get("fsn")
            if fsn is not None:
                return fsn
        result = self._index.search("", {"filter": f'concept_id = "{concept_id}"', "limit": 1})
        hits = result.get("hits", [])
        return hits[0].get("fsn") if hits else None
