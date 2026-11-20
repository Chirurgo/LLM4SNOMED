"""
A tiny, hand-authored fixture mimicking the *output shape* of
Database/transform.py::reformat_snomed() -- used to smoke-test the corpus-build
pipeline and (later) every backend's store/loader plumbing without needing the
licensed SNOMED CT RF2 release.

Spans >= 3 domains used by Minimal_Dataset.json (findings/procedure/medication),
includes a deliberately wrong-domain distractor, a "starved" domain with too few
concepts to satisfy ConceptStore's min_hits fallback, and a rare-keyword row
reserved for hybrid-search testing (Phase 4).
"""
from typing import Tuple

import polars as pl

# (concept_id, FSN, semantic_tag)
CONCEPTS = [
    ("100001", "Common cold (disorder)", "disorder"),
    ("100002", "Appendectomy (procedure)", "procedure"),
    ("100003", "Paracetamol (substance)", "substance"),
    ("100004", "Chest pain (finding)", "finding"),
    ("100005", "Road traffic accident (event)", "event"),
    ("100006", "Rare biomarker panel XQ77 (observable entity)", "observable entity"),
]

# (concept_id, name_type, concept_name)
DESCRIPTIONS = [
    ("100001", "FSN", "Common cold (disorder)"),
    ("100001", "Synonym", "common cold"),
    ("100001", "Synonym", "nasopharyngitis"),
    ("100002", "FSN", "Appendectomy (procedure)"),
    ("100002", "Synonym", "appendectomy"),
    ("100002", "Synonym", "removal of appendix"),
    ("100003", "FSN", "Paracetamol (substance)"),
    ("100003", "Synonym", "paracetamol"),
    ("100003", "Synonym", "acetaminophen"),
    ("100004", "FSN", "Chest pain (finding)"),
    ("100004", "Synonym", "chest pain"),
    ("100005", "FSN", "Road traffic accident (event)"),
    ("100005", "Synonym", "road traffic accident"),
    ("100006", "FSN", "Rare biomarker panel XQ77 (observable entity)"),
    ("100006", "Synonym", "XQ77 rare panel test"),
]

# (sourceId, destinationId, TYPE) -- empty; hierarchy relationships are always
# sourced from the real Neo4j baseline (see Evaluate/evaluate_concept.py), not
# needed for backend-store smoke tests.
RELATIONSHIPS: list = []


def mini_snomed_frames() -> Tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    active_concp = pl.LazyFrame(
        CONCEPTS, schema=["concept_id", "FSN", "semantic_tag"], orient="row"
    )
    active_synon = pl.LazyFrame(
        DESCRIPTIONS, schema=["concept_id", "name_type", "concept_name"], orient="row"
    )
    relat = pl.LazyFrame(
        RELATIONSHIPS, schema=["sourceId", "destinationId", "TYPE"], orient="row"
    )
    return active_concp, active_synon, relat
