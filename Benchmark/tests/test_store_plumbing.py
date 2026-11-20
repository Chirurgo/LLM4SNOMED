"""
Phase 3 smoke suite: validates index/query/domain-filter/FSN-lookup plumbing for the
4 new native ConceptStore backends (Postgres, Qdrant, Weaviate, Meilisearch) against
the synthetic mini-SNOMED fixture (Benchmark/fixtures/mini_snomed.py) -- no licensed
SNOMED RF2 release needed.

Must run against real dockerized services (not mocks): `docker compose up -d postgres
qdrant weaviate meilisearch`, then:

    python -m Benchmark.wait_for_services
    python -m Benchmark.tests.test_store_plumbing

Neo4j's regression (byte-identical to the pre-refactor algorithm) is already covered
deterministically by tests/test_neo4j_store_regression.py via mocks -- not re-tested
here against live infra, since that would just re-prove the same thing less reliably.
"""
import os

from Benchmark.fixtures.mini_benchmark import (
    RARE_KEYWORD_CONCEPT, RARE_KEYWORD_QUERY, build_fixture_corpus,
)
from Src.domain_mapping import allowed_tags_for_domain
from Src.stores.base import SearchParams
from Database.embed_models import AsyncSapBERTEmbedder

POSTGRES_DSN = os.environ.get("TEST_POSTGRES_DSN", "postgresql://llm4snomed:change_this_password@localhost:5433/snomed")
QDRANT_URL = os.environ.get("TEST_QDRANT_URL", "http://localhost:6333")
WEAVIATE_HOST = os.environ.get("TEST_WEAVIATE_HOST", "localhost")
WEAVIATE_HTTP_PORT = int(os.environ.get("TEST_WEAVIATE_HTTP_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.environ.get("TEST_WEAVIATE_GRPC_PORT", "50051"))
MEILI_URL = os.environ.get("TEST_MEILI_URL", "http://localhost:7700")
MEILI_API_KEY = os.environ.get("TEST_MEILI_API_KEY", "change_this_master_key")


def _assert_hybrid(name: str, store):
    assert store.supports_hybrid, f"[{name}] supports_hybrid is False"

    # 5) falsifiable hybrid check: rare-keyword rows are dense-dissimilar (see above), so
    #    dense must NOT put them top-1, while hybrid's keyword arm MUST.
    dense = store.similarity_search(RARE_KEYWORD_QUERY, k=5)
    assert dense and dense[0].metadata.get("concept_id") != RARE_KEYWORD_CONCEPT, \
        f"[{name}] dense search unexpectedly ranked the dense-dissimilar row top-1: {[d.metadata for d in dense]}"
    hybrid = store.hybrid_search(RARE_KEYWORD_QUERY, k=5)
    assert hybrid and hybrid[0].metadata.get("concept_id") == RARE_KEYWORD_CONCEPT, \
        f"[{name}] hybrid did not lift the exact-keyword row to top-1: {[(d.page_content, d.metadata) for d in hybrid]}"

    # 6) domain filtering applies to hybrid results too (same shared helper as dense)
    allowed = allowed_tags_for_domain("findings")
    docs = store.hybrid_search("common cold", k=5, allowed_semantic_tags=allowed)
    assert docs and docs[0].metadata.get("concept_id") == "100001", f"[{name}] hybrid+filter lost target: {[d.metadata for d in docs]}"
    assert all(d.metadata.get("semantic_tag") in allowed for d in docs), f"[{name}] hybrid domain filter leaked: {[d.metadata for d in docs]}"

    print(f"[{name}] hybrid checks PASS")


FUSIONS_BY_BACKEND = {"qdrant": ["rrf", "dbsf"], "weaviate": ["ranked", "relative_score"]}


def _assert_params(name: str, store):
    """Every sweepable knob must actually take effect (a silently ignored knob would make
    a whole sweep axis a no-op)."""
    def top1(**kw):
        store.configure(SearchParams(**kw))
        return store.hybrid_search(RARE_KEYWORD_QUERY, k=5)[0].metadata.get("concept_id")

    try:
        # hybrid_weight is the DENSE share: 1.0 = dense only (rare row ranked last),
        # 0.0 = keyword only (rare row is the only keyword match -> top-1).
        assert top1(hybrid_weight=1.0) != RARE_KEYWORD_CONCEPT, f"[{name}] hybrid_weight=1.0 still keyword-driven"
        assert top1(hybrid_weight=0.0) == RARE_KEYWORD_CONCEPT, f"[{name}] hybrid_weight=0.0 not keyword-driven"

        for fusion in FUSIONS_BY_BACKEND.get(name, []):
            assert top1(fusion=fusion) == RARE_KEYWORD_CONCEPT, f"[{name}] fusion={fusion} lost the keyword hit"

        if "ef_search" in store.supported_params:
            for ef in (8, 400):
                store.configure(SearchParams(ef_search=ef))
                docs = store.similarity_search("common cold", k=5)
                assert len(docs) == 5 and docs[0].metadata.get("concept_id") == "100001", \
                    f"[{name}] ef_search={ef} broke dense search: {[d.metadata for d in docs]}"
    finally:
        store.configure(SearchParams())
    print(f"[{name}] param checks PASS")


def _assert_plumbing(name: str, store):
    # 1) unfiltered dense search -- exact-synonym query should surface the right concept
    docs = store.similarity_search("common cold", k=5)
    assert len(docs) > 0, f"[{name}] empty unfiltered search"
    assert any(d.metadata.get("concept_id") == "100001" for d in docs), f"[{name}] expected concept 100001 in {[d.metadata for d in docs]}"

    # 2) domain filter: "findings" allows disorder (100001, 3 rows) + finding (100004,
    #    2 rows) -- 5 rows >= min_hits=3, so every procedure/substance/event/observable
    #    row must be excluded, while the target concept still surfaces.
    allowed = allowed_tags_for_domain("findings")  # -> {"finding","disorder","situation","morphologic abnormality"}
    docs = store.similarity_search("common cold", k=5, allowed_semantic_tags=allowed)
    assert len(docs) > 0, f"[{name}] empty filtered search"
    assert all(d.metadata.get("semantic_tag") in allowed for d in docs), f"[{name}] domain filter leaked wrong-domain concept: {[d.metadata for d in docs]}"
    assert any(d.metadata.get("concept_id") == "100001" for d in docs), f"[{name}] filtered search lost the target concept: {[d.metadata for d in docs]}"

    # 3) starved-domain fallback: "event" has only 2 description rows (< min_hits=3),
    #    so the store must fall back to unfiltered top-k rather than returning <3 hits.
    allowed = allowed_tags_for_domain("event")  # -> {"event","physical force"}
    docs = store.similarity_search("road traffic accident", k=5, allowed_semantic_tags=allowed)
    assert len(docs) >= 3, f"[{name}] fallback did not broaden beyond the starved domain: {[d.metadata for d in docs]}"
    assert any(d.metadata.get("semantic_tag") != "event" for d in docs), f"[{name}] fallback incorrectly stayed filtered: {[d.metadata for d in docs]}"

    # 4) FSN lookup
    assert store.get_fsn("100001") == "Common cold (disorder)", f"[{name}] wrong FSN"
    assert store.get_fsn(None) is None

    print(f"[{name}] plumbing checks PASS")


def test_postgres(embedding_model, parquet_path, sparse_model):
    from Src.stores.postgres_store import connect
    from Database.loaders import postgres_loader as loader

    conn = connect(POSTGRES_DSN)
    try:
        loader.create_schema(conn, drop_existing=True)
        loader.load_corpus(conn, parquet_path)
        store = loader.build_store(conn, embedding_model)
        _assert_plumbing("postgres", store)
        _assert_hybrid("postgres", store)
        _assert_params("postgres", store)
    finally:
        conn.close()


def test_qdrant(embedding_model, parquet_path, sparse_model):
    from Database.loaders import qdrant_loader as loader

    client = loader.connect(QDRANT_URL)
    try:
        loader.create_collection(client, drop_existing=True)
        loader.load_corpus(client, parquet_path)
        store = loader.build_store(client, embedding_model, sparse_model=sparse_model)
        _assert_plumbing("qdrant", store)
        _assert_hybrid("qdrant", store)
        _assert_params("qdrant", store)
    finally:
        client.close()


def test_weaviate(embedding_model, parquet_path, sparse_model):
    from Database.loaders import weaviate_loader as loader

    client = loader.connect(WEAVIATE_HOST, WEAVIATE_HTTP_PORT, WEAVIATE_GRPC_PORT)
    try:
        loader.create_collection(client, drop_existing=True)
        loader.load_corpus(client, parquet_path)
        store = loader.build_store(client, embedding_model)
        _assert_plumbing("weaviate", store)
        _assert_hybrid("weaviate", store)
        _assert_params("weaviate", store)
    finally:
        client.close()


def test_meilisearch(embedding_model, parquet_path, sparse_model):
    from Database.loaders import meilisearch_loader as loader

    client = loader.connect(MEILI_URL, MEILI_API_KEY)
    loader.create_index(client, drop_existing=True)
    loader.load_corpus(client, parquet_path)
    store = loader.build_store(client, embedding_model)
    _assert_plumbing("meilisearch", store)
    _assert_hybrid("meilisearch", store)
    _assert_params("meilisearch", store)


def run_all():
    from fastembed import SparseTextEmbedding
    from Src.stores.qdrant_store import SPARSE_MODEL_ID

    embedding_model = AsyncSapBERTEmbedder(device="cpu")
    sparse_model = SparseTextEmbedding(SPARSE_MODEL_ID)
    parquet_path = build_fixture_corpus(embedding_model, sparse_model)
    failures = []
    for name, fn in [
        ("postgres", test_postgres),
        ("qdrant", test_qdrant),
        ("weaviate", test_weaviate),
        ("meilisearch", test_meilisearch),
    ]:
        try:
            fn(embedding_model, parquet_path, sparse_model)
        except Exception as e:
            failures.append((name, e))
            print(f"[{name}] FAILED: {e}")

    if failures:
        raise SystemExit(f"{len(failures)}/4 backend(s) failed plumbing checks: {[n for n, _ in failures]}")
    print("All 4 backend plumbing checks PASS")


if __name__ == "__main__":
    run_all()
