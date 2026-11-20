"""Retrieval-only benchmark input: gold spans + fixed canonical English names.

Replaces LLM extraction (Benchmark.extraction) so the sweep measures concept retrieval
and ranking alone, with no generative model anywhere in the loop:

  - span     = gold start_index/end_index/substring (NER is perfect by construction)
  - domain   = gold label (remapped to the extractor's vocabulary, e.g. observable)
  - query    = `formal_term` = a canonical English name translated ONCE from the span
               text (Benchmark/data/canonical_terms_minimal.tsv) and reused by every config

Writes the same cache schema as Benchmark.extraction, so Benchmark.harness and
Benchmark.report run unchanged:

  python -m Benchmark.gold_queries --out Benchmark/results/gold/extractions_gold.json
  python -m Benchmark.harness --gold Minimal_Dataset.json \
      --extractions Benchmark/results/gold/extractions_gold.json --out Benchmark/results/gold ...
"""
import argparse
import json
from pathlib import Path

DEFAULT_TERMS = "Benchmark/data/canonical_terms_minimal.tsv"
QUERY_SOURCE = "gold-span+claude-opus-5-5-translation"
# gold label -> domain key used by Src/domain_mapping.TYPE_TO_SEMANTIC_TAG
LABEL_TO_DOMAIN = {"observable entity": "observable"}


def load_terms(path: str) -> dict:
    terms = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        doc, ent, substring, name = line.split("\t")
        terms[(int(doc), int(ent))] = (substring, name.strip())
    return terms


def build(gold_path: str, terms_path: str) -> dict:
    gold = json.loads(Path(gold_path).read_text(encoding="utf-8"))
    terms = load_terms(terms_path)
    docs, used = [], set()
    for i, rec in enumerate(gold):
        ents = []
        for j, g in enumerate(rec["entity"]):
            if (i, j) not in terms:
                raise KeyError(f"no canonical name for doc {i} entity {j} ({g['substring']!r})")
            substring, name = terms[(i, j)]
            s, e = g["start_index"], g["end_index"]
            if substring != g["substring"] or rec["text"][s:e] != substring:
                raise ValueError(f"doc {i} entity {j}: table has {substring!r}, gold has {g['substring']!r}")
            label = g["label"].lower()
            ents.append({
                "term": substring,
                "formal_term": name,
                "domain": LABEL_TO_DOMAIN.get(label, label),
                "status": None,
                "match_status": "GOLD",
                "start_index": s,
                "end_index": e,
            })
            used.add((i, j))
        docs.append(ents)
    extra = set(terms) - used
    if extra:
        raise ValueError(f"table rows with no gold entity: {sorted(extra)}")
    return {"source": str(gold_path), "llm": QUERY_SOURCE, "terms_table": str(terms_path),
            "documents": docs, "failed": [], "seconds": [None] * len(docs)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default="Minimal_Dataset.json")
    ap.add_argument("--terms", default=DEFAULT_TERMS)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    cache = build(args.gold, args.terms)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    n = sum(len(d) for d in cache["documents"])
    print(f"wrote {out}: {len(cache['documents'])} docs, {n} gold-span queries")


if __name__ == "__main__":
    main()
