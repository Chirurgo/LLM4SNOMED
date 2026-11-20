from typing import List, Optional, Set

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from Src.stores.base import ConceptStore, SearchParams

EMBEDDING_DIM = 768
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
SPARSE_MODEL_ID = "Qdrant/bm25"
FUSIONS = ("rrf", "dbsf")


class QdrantConceptStore(ConceptStore):
    """`sparse_model` (fastembed.SparseTextEmbedding("Qdrant/bm25"), the same model that
    produced the corpus' cached sparse vectors) encodes the query for hybrid search;
    without it the store is dense-only.

    Requires Qdrant server >= the version supporting `RrfQuery(Rrf(k, weights))`
    (verified on v1.20); v1.12 rejected it and its built-in RRF used k~2, not 60.
    """

    supported_params = frozenset({"hybrid_weight", "rrf_k", "arm_depth_min", "ef_search", "fusion"})

    def __init__(self, client: QdrantClient, embedding_model, collection_name: str = "snomed_descriptions",
                 sparse_model=None, params: Optional[SearchParams] = None):
        self._client = client
        self._embedding_model = embedding_model
        self._collection = collection_name
        self._sparse_model = sparse_model
        self.supports_hybrid = sparse_model is not None
        self.configure(params or SearchParams())

    def configure(self, params: SearchParams) -> None:
        if params.fusion not in (None, *FUSIONS):
            raise ValueError(f"Qdrant fusion must be one of {FUSIONS}, got {params.fusion!r}")
        self.params = params

    def _search_params(self, n: int) -> Optional[qmodels.SearchParams]:
        if self.params.ef_search is None:
            return None
        return qmodels.SearchParams(hnsw_ef=max(self.params.ef_search, n))

    def _embed_query(self, text: str) -> list:
        return self._embedding_model.embed_query(text)

    def _points_to_documents(self, points) -> List[Document]:
        docs = []
        for p in points:
            payload = p.payload or {}
            docs.append(Document(
                page_content=payload.get("text", ""),
                metadata={
                    "concept_id": payload.get("concept_id"),
                    "semantic_tag": payload.get("semantic_tag"),
                    "fsn": payload.get("fsn"),
                },
            ))
        return docs

    def _dense_query(self, qvec, n: int) -> List[Document]:
        result = self._client.query_points(
            collection_name=self._collection,
            query=qvec,
            using=DENSE_VECTOR_NAME,
            limit=n,
            search_params=self._search_params(n),
            with_payload=True,
        )
        return self._points_to_documents(result.points)

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

    def _fusion_query(self):
        p = self.params
        if (p.fusion or "rrf") == "dbsf":
            # Distribution-Based Score Fusion: score-normalized, takes no per-arm weights,
            # so hybrid_weight is NOT honoured in this mode.
            return qmodels.FusionQuery(fusion=qmodels.Fusion.DBSF)
        # Weights follow prefetch order: [dense, sparse].
        return qmodels.RrfQuery(rrf=qmodels.Rrf(k=p.rrf_k, weights=[p.hybrid_weight, 1.0 - p.hybrid_weight]))

    def _hybrid_query(self, query_text: str, qvec, n: int) -> List[Document]:
        sparse = next(iter(self._sparse_model.query_embed(query_text)))
        depth = self.params.arm_depth(n)
        result = self._client.query_points(
            collection_name=self._collection,
            prefetch=[
                qmodels.Prefetch(query=qvec, using=DENSE_VECTOR_NAME, limit=depth,
                                 params=self._search_params(depth)),
                qmodels.Prefetch(
                    query=qmodels.SparseVector(indices=sparse.indices.tolist(), values=sparse.values.tolist()),
                    using=SPARSE_VECTOR_NAME,
                    limit=depth,
                ),
            ],
            query=self._fusion_query(),
            limit=n,
            with_payload=True,
        )
        return self._points_to_documents(result.points)

    def hybrid_search(
        self,
        query_text: str,
        k: int,
        allowed_semantic_tags: Optional[Set[str]] = None,
        min_hits: int = 3,
        expand_factor: float = 2.0,
    ) -> List[Document]:
        if self._sparse_model is None:
            raise NotImplementedError("QdrantConceptStore needs a sparse_model for hybrid search")
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
        points, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=qmodels.Filter(
                must=[qmodels.FieldCondition(key="concept_id", match=qmodels.MatchValue(value=concept_id))]
            ),
            limit=1,
            with_payload=True,
        )
        return points[0].payload.get("fsn") if points else None
