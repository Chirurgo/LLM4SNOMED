from typing import List, Optional, Set

import numpy as np
from langchain_core.documents import Document
from pgvector.psycopg import register_vector

from Src.stores.base import ConceptStore, SearchParams

EMBEDDING_DIM = 768
PGVECTOR_DEFAULT_EF_SEARCH = 40


def connect(dsn: str):
    import psycopg
    conn = psycopg.connect(dsn, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


class PostgresConceptStore(ConceptStore):
    supports_hybrid = True
    supported_params = frozenset({"hybrid_weight", "rrf_k", "arm_depth_min", "ef_search"})

    def __init__(self, conn, embedding_model, table_name: str = "descriptions",
                 params: Optional[SearchParams] = None):
        self._conn = conn
        self._embedding_model = embedding_model
        self._table = table_name
        self._session_ef: Optional[int] = None
        self.configure(params or SearchParams())

    def configure(self, params: SearchParams) -> None:
        self.params = params
        self._session_ef = None  # force re-SET on next query

    def _ensure_ef(self, n: int) -> None:
        # pgvector's HNSW scan returns at most hnsw.ef_search rows (default 40), silently
        # truncating any LIMIT above it -- so always run with ef >= n. Only re-SET when the
        # value changes, to keep the extra round-trip out of steady-state latency.
        ef = max(self.params.ef_search or PGVECTOR_DEFAULT_EF_SEARCH, n)
        if ef != self._session_ef:
            self._conn.execute(f"SET hnsw.ef_search = {int(ef)}")
            self._session_ef = ef

    def _embed_query(self, text: str) -> np.ndarray:
        # pgvector's psycopg adapter only binds numpy arrays as `vector`; a plain
        # list is sent as double precision[] and `<=>` fails to resolve.
        return np.asarray(self._embedding_model.embed_query(text), dtype=np.float32)

    def _rows_to_documents(self, rows) -> List[Document]:
        return [
            Document(
                page_content=text,
                metadata={"concept_id": concept_id, "semantic_tag": tag, "fsn": fsn},
            )
            for concept_id, text, tag, fsn in rows
        ]

    def _dense_query(self, qvec, n: int) -> List[Document]:
        self._ensure_ef(n)
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT concept_id, text, semantic_tag, fsn FROM {self._table} "
                f"ORDER BY embedding <=> %s LIMIT %s",
                (qvec, n),
            )
            return self._rows_to_documents(cur.fetchall())

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
        # pgvector has no native hybrid primitive, so this is hand-rolled weighted
        # Reciprocal Rank Fusion over a cosine-distance arm and a full-text arm.
        # Methodological choices: the 'simple' text config (no stemming/stopwords) since
        # descriptions are mixed-language; and terms are OR-ed (plainto_tsquery AND-s
        # them) to match the any-term BM25 semantics of the other backends' keyword arms.
        p = self.params
        depth = p.arm_depth(n)
        self._ensure_ef(depth)
        sql = f"""
            WITH dense_raw AS (
                SELECT row_id, embedding <=> %(q)s AS dist
                FROM {self._table}
                ORDER BY embedding <=> %(q)s
                LIMIT %(depth)s
            ),
            dense AS (
                SELECT row_id, ROW_NUMBER() OVER (ORDER BY dist) AS r FROM dense_raw
            ),
            kw_raw AS (
                SELECT row_id, ts_rank(text_search, q.tsq) AS score
                FROM {self._table}
                CROSS JOIN (
                    SELECT replace(plainto_tsquery('simple', %(text)s)::text, ' & ', ' | ')::tsquery AS tsq
                ) q
                WHERE text_search @@ q.tsq
                ORDER BY score DESC
                LIMIT %(depth)s
            ),
            kw AS (
                SELECT row_id, ROW_NUMBER() OVER (ORDER BY score DESC) AS r FROM kw_raw
            ),
            fused AS (
                SELECT COALESCE(dense.row_id, kw.row_id) AS row_id,
                       COALESCE(%(w_dense)s / (%(rrf_k)s + dense.r), 0)
                     + COALESCE(%(w_kw)s / (%(rrf_k)s + kw.r), 0) AS score
                FROM dense FULL OUTER JOIN kw ON dense.row_id = kw.row_id
            )
            SELECT t.concept_id, t.text, t.semantic_tag, t.fsn
            FROM fused JOIN {self._table} t ON t.row_id = fused.row_id
            ORDER BY fused.score DESC, t.row_id
            LIMIT %(n)s
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, {
                "q": qvec, "text": query_text, "depth": depth, "n": n, "rrf_k": p.rrf_k,
                "w_dense": float(p.hybrid_weight), "w_kw": float(1.0 - p.hybrid_weight),
            })
            return self._rows_to_documents(cur.fetchall())

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
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT fsn FROM {self._table} WHERE concept_id = %s LIMIT 1", (concept_id,))
            row = cur.fetchone()
        return row[0] if row else None
