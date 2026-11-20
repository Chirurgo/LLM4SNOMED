from typing import List, Optional, Set

from langchain_core.documents import Document
from weaviate.classes.config import Reconfigure
from weaviate.classes.query import Filter, HybridFusion

from Src.stores.base import ConceptStore, SearchParams

EMBEDDING_DIM = 768
WEAVIATE_DYNAMIC_EF = -1  # Weaviate's default: ef chosen dynamically from the query limit
FUSIONS = {"ranked": HybridFusion.RANKED, "relative_score": HybridFusion.RELATIVE_SCORE}


class WeaviateConceptStore(ConceptStore):
    supports_hybrid = True
    # arm depth is internal to Weaviate's hybrid; rrf_k is fixed by Weaviate's RANKED fusion.
    supported_params = frozenset({"hybrid_weight", "ef_search", "fusion"})

    def __init__(self, client, embedding_model, collection_name: str = "SnomedDescription",
                 params: Optional[SearchParams] = None):
        self._client = client
        self._embedding_model = embedding_model
        self._collection_name = collection_name
        self._applied_ef: Optional[int] = None
        self.configure(params or SearchParams())

    @property
    def _collection(self):
        return self._client.collections.get(self._collection_name)

    def configure(self, params: SearchParams) -> None:
        if params.fusion not in (None, *FUSIONS):
            raise ValueError(f"Weaviate fusion must be one of {tuple(FUSIONS)}, got {params.fusion!r}")
        self.params = params
        # ef is a collection-level setting in Weaviate (no per-query override), so it is
        # pushed to the server here. Affects every client of this collection.
        ef = params.ef_search if params.ef_search is not None else WEAVIATE_DYNAMIC_EF
        if ef != self._applied_ef:
            self._collection.config.update(vector_index_config=Reconfigure.VectorIndex.hnsw(ef=ef))
            self._applied_ef = ef

    def _embed_query(self, text: str) -> list:
        return self._embedding_model.embed_query(text)

    def _objects_to_documents(self, objects) -> List[Document]:
        docs = []
        for o in objects:
            props = o.properties
            docs.append(Document(
                page_content=props.get("text", ""),
                metadata={
                    "concept_id": props.get("concept_id"),
                    "semantic_tag": props.get("semantic_tag"),
                    "fsn": props.get("fsn"),
                },
            ))
        return docs

    def _dense_query(self, qvec, n: int) -> List[Document]:
        response = self._collection.query.near_vector(near_vector=qvec, limit=n)
        return self._objects_to_documents(response.objects)

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
            lambda n: self._dense_query(qvec, n), k, allowed_semantic_tags, min_hits, expand_factor
        )

    def _hybrid_query(self, query_text: str, qvec, n: int) -> List[Document]:
        # Native BM25F + vector fusion. Default RANKED (rank-based, RRF-style) rather than
        # Weaviate's own default RELATIVE_SCORE, to match the RRF used for Postgres/Qdrant.
        response = self._collection.query.hybrid(
            query=query_text,
            vector=qvec,
            alpha=self.params.hybrid_weight,
            fusion_type=FUSIONS[self.params.fusion or "ranked"],
            query_properties=["text"],
            limit=n,
        )
        return self._objects_to_documents(response.objects)

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
            lambda n: self._hybrid_query(query_text, qvec, n), k, allowed_semantic_tags, min_hits, expand_factor
        )

    def get_fsn(self, concept_id: Optional[str], hint_doc: Optional[Document] = None) -> Optional[str]:
        if concept_id is None:
            return None
        if hint_doc is not None:
            fsn = hint_doc.metadata.get("fsn")
            if fsn is not None:
                return fsn
        response = self._collection.query.fetch_objects(
            filters=Filter.by_property("concept_id").equal(concept_id),
            limit=1,
        )
        return response.objects[0].properties.get("fsn") if response.objects else None
