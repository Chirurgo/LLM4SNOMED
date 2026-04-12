from __future__ import annotations

import logging
import re
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from rapidfuzz import fuzz
from pythainlp import word_tokenize as th_word_tokenize  # type: ignore
from pythainlp.corpus.common import thai_stopwords as th_stopwords  # type: ignore

MAX_MATCH_CHARS = 8000  # tune as needed


# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# This module: still allow INFO
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ----------------------------------------------------------------------------
# Tokenization helpers (No changes here)
# ----------------------------------------------------------------------------
_WORD_RE = re.compile(r"\w+|\S", flags=re.UNICODE)
_EN_STOP = set(
    "the a an of and or with without in on for to by at from is are be was were this that these those".split()
)


def _tokenize_with_offsets(s: str) -> List[Tuple[str, int, int]]:
    """Greedy tokenization capturing character offsets for span scoring."""
    tokens: List[Tuple[str, int, int]] = []
    for m in _WORD_RE.finditer(s):
        tokens.append((m.group(0), m.start(), m.end()))
    return tokens


def _word_windows(tokens: Sequence[Tuple[str, int, int]], min_len: int, max_len: int) -> Iterable[Tuple[int, int, int, int]]:
    """Yield token-span windows (i, j, start_char, end_char)."""
    n = len(tokens)
    min_len = max(1, min_len)
    max_len = max(min_len, max_len)
    for i in range(n):
        max_L = min(max_len, n - i)
        for L in range(min_len, max_L + 1):
            j = i + L
            start = tokens[i][1]
            end = tokens[j - 1][2]
            yield i, j, start, end


# ----------------------------------------------------------------------------
# Attribute keyword extraction (No changes here)
# ----------------------------------------------------------------------------

def _clean_token(tok: str) -> str:
    tok = tok.lower()
    tok = re.sub(r"^\W+|\W+$", "", tok)
    return tok


def _extract_attribute_keywords(entity: Dict[str, Any]) -> set:
    """Extract a set of cleaned, lowercased keywords from entity attributes."""
    kws = set()
    attrs = entity.get("attribute") or {}
    if not isinstance(attrs, dict):
        return kws

    thai_sw = set(th_stopwords())

    for v in attrs.values():
        raw = str(v)
        toks = th_word_tokenize(raw, engine="newmm")
        for t in toks:
            t2 = _clean_token(t)
            if not t2 or len(t2) <= 1 or t2 in _EN_STOP or t2 in thai_sw:
                continue
            kws.add(t2)
    return kws


# ----------------------------------------------------------------------------
# Window scoring and candidate generation (No changes here)
# ----------------------------------------------------------------------------

def _best_windows_for_term(
    text: str,
    term: str,
    pad: int = 2,
    cutoff: int = 70,
    topk: int = 12,
    max_windows=2000,
) -> List[Tuple[str, int, int, int]]:
    """Return top candidate windows for a term: (candidate_text, score, start, end)."""
    tokens = _tokenize_with_offsets(text)
    term_len = max(1, len(_tokenize_with_offsets(term)))
    min_len = max(1, term_len - pad)
    max_len = term_len + pad

    checked = 0
    scored: List[Tuple[str, int, int, int]] = []
    for _, _, start, end in _word_windows(tokens, min_len, max_len):
        checked += 1
        if checked > max_windows:
            break
        candidate = text[start:end]
        score = fuzz.WRatio(term, candidate)
        if score == 100:
            return [(candidate, score, start, end)]
        if score >= cutoff:
            scored.append((candidate, score, start, end))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:topk]


def _clip_text(text: str) -> str:
    if len(text) <= MAX_MATCH_CHARS:
        return text
    head = text[:MAX_MATCH_CHARS // 2]
    tail = text[-MAX_MATCH_CHARS // 2:]
    return head + tail

def _find_best_candidates_for_entity(
    text: str,
    entity: Dict[str, Any],
    window_pad: int = 2,
    score_cutoff: int = 70,
) -> List[Dict[str, Any]]:
    """Find plausible matches for an entity and score them with context."""
    term = entity.get("term")
    if not term:
        return []
    t = _clip_text(text)
    attribute_keywords = _extract_attribute_keywords(entity)
    windows = _best_windows_for_term(t, term, pad=window_pad, cutoff=score_cutoff, topk=24)
    if not windows:
        return []

    scored: List[Dict[str, Any]] = []
    for mention, term_score, start, end in windows:
        window_start = max(0, start - 80)
        window_end = min(len(text), end + 80)
        context = text[window_start:window_end].lower()

        found = {kw for kw in attribute_keywords if kw and kw in context}
        attr_score = (len(found) / len(attribute_keywords) * 100.0) if attribute_keywords else 0.0

        final_score = term_score * 0.5 + attr_score * 0.5
        scored.append(
            {
                "matched_term": mention,
                "start_index": start,
                "end_index": end,
                "term_score": float(term_score),
                "attr_score": float(attr_score),
                "final_score": float(final_score),
            }
        )

    scored.sort(key=lambda x: (x["final_score"], x["term_score"]), reverse=True)
    return scored


# ----------------------------------------------------------------------------
# Overlap handling and public API (REFACTORED)
# ----------------------------------------------------------------------------

def _overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    """Check if two (start, end) spans overlap."""
    return max(a[0], b[0]) < min(a[1], b[1])


def _find_exact_matches(text: str, term: str) -> List[Tuple[int, int]]:
    """Find all non-overlapping exact match spans for a term."""
    if not term:
        return []
    return [(m.start(), m.end()) for m in re.finditer(re.escape(term), text)]


def position_entities(input_text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Link entities to spans in text using a two-pass approach.

    Pass 1: Find all non-overlapping exact matches.
    Pass 2: Find fuzzy matches for remaining entities in the available text spans.

    Input:
        input_text: source text
        entities: list of dicts, each with at least {"term": str, "attribute": dict}

    Output:
        New list with keys added per entity when matched:
            - matched_term
            - start_index, end_index
            - term_score, attr_score, final_score
            - match_status: SUCCESS_EXACT | SUCCESS_FUZZY | NOT_FOUND | FAILED_OVERLAP | NO_TERM
    """
    if not isinstance(entities, list):
        raise TypeError("entities must be a list of dicts")

    # Prepare entities with original index for stable sorting and initial status
    processed = [
        dict(e, **{"original_index": i, "match_status": "UNPROCESSED"})
        for i, e in enumerate(deepcopy(entities))
    ]
    
    seen_spans: List[Tuple[int, int]] = []
    stats = {"exact": 0, "fuzzy": 0, "not_found": 0, "overlap": 0, "no_term": 0}

    # --- PASS 1: Exact Matching ---
    # Prioritize longer terms to handle nested cases like "apple" and "apple pie"
    processed.sort(key=lambda e: len(str(e.get("term", ""))), reverse=True)
    
    for ent in processed:
        term = ent.get("term", "")
        if not term:
            ent["match_status"] = "NO_TERM"
            stats["no_term"] += 1
            continue

        # Find all possible exact matches
        exact_matches = _find_exact_matches(input_text, term)
        
        # Find the first available (non-overlapping) exact match
        for start, end in exact_matches:
            span = (start, end)
            if not any(_overlaps(span, s) for s in seen_spans):
                ent.update({
                    "matched_term": term,
                    "start_index": start,
                    "end_index": end,
                    "term_score": 100.0,
                    "attr_score": 100.0, # Exact match implies attribute context is perfect
                    "final_score": 100.0,
                    "match_status": "SUCCESS_EXACT",
                })
                seen_spans.append(span)
                stats["exact"] += 1
                break # Move to the next entity

    # --- PASS 2: Fuzzy Matching for Remaining Entities ---
    # The order doesn't strictly matter here, but keeping longest-first is a good heuristic
    for ent in processed:
        # Skip entities that were already matched or had no term
        if ent["match_status"] != "UNPROCESSED":
            continue

        candidates = _find_best_candidates_for_entity(input_text, ent)
        if not candidates:
            ent["match_status"] = "NOT_FOUND"
            stats["not_found"] += 1
            continue

        assigned = False
        for cand in candidates:
            span = (cand["start_index"], cand["end_index"])
            if not any(_overlaps(span, s) for s in seen_spans):
                ent.update(cand)
                ent["match_status"] = "SUCCESS_FUZZY"
                seen_spans.append(span)
                stats["fuzzy"] += 1
                assigned = True
                break
        
        if not assigned:
            ent["match_status"] = "FAILED_OVERLAP"
            stats["overlap"] += 1

    # Restore original order and clean up helper key
    processed.sort(key=lambda e: e["original_index"])
    for e in processed:
        e.pop("original_index", None)

    logger.info(
        "Positioning done. exact=%d, fuzzy=%d, not_found=%d, overlap=%d, no_term=%d, total=%d",
        stats["exact"], stats["fuzzy"], stats["not_found"], stats["overlap"], stats["no_term"], len(processed)
    )
    return processed