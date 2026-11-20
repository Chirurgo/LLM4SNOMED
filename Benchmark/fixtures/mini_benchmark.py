"""
Synthetic end-to-end benchmark fixture: a fixture corpus built from mini_snomed.py,
plus a tiny gold set and matching cached LLM extractions, so the whole
harness -> predictions -> report pipeline can be validated with no licensed SNOMED
release, no LLM calls and no restricted clinical data.

Quality numbers produced on this fixture are meaningless (15-row corpus); only the
plumbing, bookkeeping and metric wiring are being validated.
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import Database.build_corpus as build_corpus
from Benchmark.fixtures.mini_snomed import mini_snomed_frames

RARE_KEYWORD_CONCEPT = "100006"  # "Rare biomarker panel XQ77" in the fixture
RARE_KEYWORD_QUERY = "XQ77"

# (text, [(surface term, gold label, gold concept, extracted formal_term, extracted domain)])
_DOCS = [
    ("Patient has common cold and chest pain since yesterday.", [
        ("common cold", "findings", "100001", "common cold", "findings"),
        ("chest pain", "findings", "100004", "chest pain", "findings"),
    ]),
    ("Status post removal of appendix, given paracetamol for fever.", [
        ("removal of appendix", "procedure", "100002", "appendectomy", "procedure"),
        ("paracetamol", "medication", "100003", "acetaminophen", "medication"),
    ]),
    ("Injured in a road traffic accident; XQ77 panel ordered.", [
        ("road traffic accident", "event", "100005", "road traffic accident", "event"),
        ("XQ77 panel", "observable entity", "100006", "XQ77", "observable"),
    ]),
]


def build_fixture_corpus(embedding_model, sparse_model, poison_rare_keyword: bool = True,
                         out_dir: str = None) -> str:
    out_dir = Path(out_dir or tempfile.mkdtemp(prefix="llm4snomed_fixture_"))
    out_path = out_dir / "mini_corpus.parquet"
    with patch.object(build_corpus, "reformat_snomed", return_value=mini_snomed_frames()):
        build_corpus.build_and_cache_corpus(
            snomed_release_path="FIXTURE",
            out_path=str(out_path),
            embedding_model=embedding_model,
            sparse_model=sparse_model,
            batch_size=4,
            force=True,
        )
    if poison_rare_keyword:
        make_rare_keyword_rows_dense_dissimilar(str(out_path), embedding_model)
    return str(out_path)


def make_rare_keyword_rows_dense_dissimilar(parquet_path: str, embedding_model) -> None:
    """Give the rare-keyword concept's rows the NEGATED embedding of the rare-keyword
    query, so dense search ranks them dead last for that query. Only a working keyword
    arm can then lift them to top-1 -- which makes the hybrid check falsifiable (real
    SapBERT subword overlap on "XQ77" would otherwise let dense search find it too)."""
    anti = (-np.asarray(embedding_model.embed_query(RARE_KEYWORD_QUERY), dtype=np.float32)).tolist()
    table = pq.read_table(parquet_path)
    concept_ids = table.column("concept_id").to_pylist()
    embeddings = table.column("embedding").to_pylist()
    embeddings = [anti if cid == RARE_KEYWORD_CONCEPT else emb for cid, emb in zip(concept_ids, embeddings)]
    idx = table.schema.get_field_index("embedding")
    table = table.set_column(idx, table.schema.field(idx), pa.array(embeddings, type=pa.list_(pa.float32())))
    pq.write_table(table, parquet_path)


def write_gold_and_extractions(out_dir: str) -> tuple[str, str]:
    """Writes mini_gold.json (Minimal_Dataset.json schema) and mini_extractions.json
    (Benchmark.extraction cache schema). Offsets are computed, never hand-typed."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gold, extractions = [], []
    for i, (text, ents) in enumerate(_DOCS):
        gold_ents, ext_ents = [], []
        for term, label, cid, formal, domain in ents:
            start = text.index(term)
            end = start + len(term)
            gold_ents.append({
                "start_index": start, "end_index": end, "label": label, "substring": term,
                "Target code": cid, "Target display": None,
                "Relationship type code": "TARGET_EQUIVALENT", "No map flag": False, "Status": "DRAFT",
            })
            ext_ents.append({
                "term": term, "formal_term": formal, "domain": domain, "status": "present",
                "attributes": {}, "start_index": start, "end_index": end, "match_status": "SUCCESS_EXACT",
            })
        gold.append({"id": f"fixture-{i}", "text": text, "type": "pi", "entity": gold_ents})
        extractions.append(ext_ents)

    gold_path = out_dir / "mini_gold.json"
    ext_path = out_dir / "mini_extractions.json"
    gold_path.write_text(json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8")
    ext_path.write_text(json.dumps({"source": "fixture", "llm": None, "documents": extractions},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    return str(gold_path), str(ext_path)
