from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional, Set

from langchain_core.documents import Document


@dataclass(frozen=True)
class SearchParams:
    """Query-time knobs, shared across backends so the benchmark can sweep them uniformly.

    Defaults weight dense and keyword arms equally, so fusion differences come from the
    backend rather than per-backend tuning.

    hybrid_weight: share of the DENSE arm, 0 = keyword only, 1 = dense only. Maps to
        weighted RRF (Postgres, Qdrant), Weaviate `alpha`, Meilisearch `semanticRatio`.
    rrf_k: RRF constant, score = sum_i w_i / (rrf_k + rank_i). Postgres and Qdrant only.
        Pinned explicitly because backends' native defaults differ (Qdrant's built-in
        RRF used k~2 before configurable k existed).
    arm_depth_min: where we control per-arm candidate depth before fusion (Postgres,
        Qdrant prefetch), each arm fetches max(n, this). Arms deeper than the final limit
        let a doc found by both arms outrank one that tops only a single arm.
    ef_search: HNSW query-time beam width (speed vs. ANN recall). None = backend
        default. Applied as max(ef_search, n), since HNSW cannot return more than ef
        results (pgvector silently truncates). No equivalent knob on Meilisearch.
    fusion: backend-specific fusion algorithm, None = our default
        (Qdrant: "rrf" | "dbsf"; Weaviate: "ranked" | "relative_score").
    """
    hybrid_weight: float = 0.5
    rrf_k: int = 60
    arm_depth_min: int = 50
    ef_search: Optional[int] = None
    fusion: Optional[str] = None

    def arm_depth(self, n: int) -> int:
        return max(n, self.arm_depth_min)

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT_PARAMS = SearchParams()


class ConceptStore(ABC):
    """Backend-agnostic access to an indexed corpus of SNOMED CT descriptions.

    Implementations return candidates as `langchain_core.documents.Document`
    (`page_content` = matched description text, `metadata["concept_id"]` required)
    so `Src/link.py::rerank_candidates()` needs no changes across backends.
    """

    supports_hybrid: bool = False
    # Which SearchParams fields this backend actually honours (reported by the benchmark,
    # so a sweep over an unsupported knob is flagged instead of silently being a no-op).
    supported_params: frozenset = frozenset()
    params: SearchParams = DEFAULT_PARAMS

    def configure(self, params: SearchParams) -> None:
        """Apply query-time params. Subclasses override to push server-side settings."""
        self.params = params

    @abstractmethod
    def similarity_search(
        self,
        query_text: str,
        k: int,
        allowed_semantic_tags: Optional[Set[str]] = None,
        min_hits: int = 3,
        expand_factor: float = 2.0,
    ) -> List[Document]:
        ...

    def hybrid_search(
        self,
        query_text: str,
        k: int,
        allowed_semantic_tags: Optional[Set[str]] = None,
        min_hits: int = 3,
        expand_factor: float = 2.0,
    ) -> List[Document]:
        raise NotImplementedError(f"{type(self).__name__} does not support hybrid search")

    @staticmethod
    def _expand_filter_fallback(
        fetch: Callable[[int], List[Document]],
        k: int,
        allowed_semantic_tags: Optional[Set[str]],
        min_hits: int,
        expand_factor: float,
    ) -> List[Document]:
        """Same domain-filter algorithm as Neo4jConceptStore.similarity_search, so
        cross-backend differences reflect the backend, not the filtering strategy:
        fetch k*expand_factor unfiltered nearest neighbours, keep those whose
        semantic_tag is allowed, and fall back to the unfiltered top-k if fewer than
        min_hits survive. `fetch(n)` must return the n nearest docs with
        metadata["semantic_tag"] populated (denormalized at corpus-build time).
        """
        if not allowed_semantic_tags:
            return fetch(k)
        raw = fetch(int(k * expand_factor))
        filtered = [
            d for d in raw
            if (d.metadata.get("semantic_tag") or "").lower() in allowed_semantic_tags
        ]
        if len(filtered) < min_hits:
            return raw[:k]
        return filtered[:k]

    @abstractmethod
    def get_fsn(self, concept_id: Optional[str], hint_doc: Optional[Document] = None) -> Optional[str]:
        ...
