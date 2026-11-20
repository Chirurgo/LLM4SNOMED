"""Scores every config produced by Benchmark.harness and assembles one results table.

Quality (all via Evaluate/*.py, unmodified scoring logic):
  link_acc        concept accuracy on entities whose span exactly matches gold
                  (Evaluate/topk.py::concept_accuracy) -- isolates LINKING from NER,
                  since every config shares the same cached extractions
  top1/5/10       Evaluate/topk.py::topk_accuracy over the reranked list
  cand_recall     gold concept anywhere in the raw retrieved candidates (pre-rerank):
                  the ceiling no reranker/LLM can exceed for that config
  strict/loose/hier F1, IoU  end-to-end (span + concept), Evaluate/evaluate_concept.py
Cost:
  retrieval_*_ms  measured DB query latency (query embeddings pre-warmed)
  rerank_est_ms   rerank pairs/entity x calibrated seconds/pair for that cross-encoder
  est_ms_entity   embed + retrieval mean + rerank estimate (LLM time excluded: provider-dependent)
  db_mem_*/cpu    database container, sampled during the retrieval phase
pareto            on the per-backend front of (max link_acc, min est_ms_entity)

Usage: python -m Benchmark.report --out Benchmark/results/run1 --gold Minimal_Dataset.json [--neo4j]
"""
import argparse
import contextlib
import csv
import io
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from Evaluate.evaluate_concept import Evaluator, build_concept_proximity_from_neo4j, extract_gold_concept_ids
from Evaluate.evaluate_domain import DomainEvaluator
from Evaluate.topk import concept_accuracy, topk_accuracy

# Gold labels use "observable entity"; the extractor emits "observable" (known schema
# mismatch documented in README) -- remapped on an in-memory copy, never in Evaluate/.
DOMAIN_REMAP = {"observable": "observable entity"}

COLUMNS = [
    "backend", "mode", "axis", "config_id", "k_candidate", "expand_factor", "p_hybrid_weight", "p_fusion",
    "p_arm_depth_min", "p_rrf_k", "p_ef_search", "reranker", "selection",
    "link_acc", "top1", "top5", "top10", "cand_recall", "strict_f1", "strict_f1_l95", "strict_f1_u95",
    "loose_f1", "hier_f1", "iou",
    "retrieval_mean_ms", "retrieval_p50_ms", "retrieval_p95_ms", "rerank_pairs_per_entity", "rerank_est_ms",
    "est_ms_entity", "db_mem_peak_mb", "db_cpu_mean_pct", "llm_calls_new", "errors", "pareto",
]


def _quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def candidate_recall(gold: list, candidates: list) -> Optional[float]:
    total = hit = 0
    for g, doc in zip(gold, candidates):
        spans = {(e["start_index"], e["end_index"]): str(e["Target code"]) for e in g.get("entity", [])}
        for c in doc:
            span = (c["start_index"], c["end_index"])
            if span in spans:
                total += 1
                hit += spans[span] in {str(x) for x in c["retrieved"]}
    return hit / total if total else None


def ner_summary(gold_path: str, extractions_path: str) -> dict:
    from Benchmark.extraction import load_extractions
    y_true = DomainEvaluator._load_gold_entities(gold_path)
    docs = load_extractions(extractions_path, len(y_true))
    y_pred = [[(DOMAIN_REMAP.get(e.get("domain", ""), e.get("domain", "")), (e["start_index"], e["end_index"]))
               for e in doc] for doc in docs]
    ev = DomainEvaluator(y_true, y_pred)
    return {"ner_strict": _quiet(ev.strict), "ner_loose": _quiet(ev.loose)}


def pareto_flags(rows: List[dict], quality: str = "link_acc", cost: str = "est_ms_entity") -> None:
    by_backend: Dict[str, List[dict]] = {}
    for r in rows:
        by_backend.setdefault(r["backend"], []).append(r)
    for rs in by_backend.values():
        valid = [r for r in rs if r.get(quality) is not None and r.get(cost) is not None]
        for r in rs:
            r["pareto"] = r in valid and not any(
                o is not r and o[quality] >= r[quality] and o[cost] <= r[cost]
                and (o[quality] > r[quality] or o[cost] < r[cost]) for o in valid)


def build_rows(out: Path, gold_path: str, proximity: Optional[dict]) -> List[dict]:
    gold = json.loads(Path(gold_path).read_text(encoding="utf-8"))
    meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    spp = meta.get("reranker_seconds_per_pair", {})
    embed_ms = meta.get("embedding", {}).get("embed_mean_ms") or 0.0
    all_true = Evaluator._load_gold_concept_data(gold_path)

    rows = []
    for mpath in sorted((out / "metrics").glob("*.json")):
        m = json.loads(mpath.read_text(encoding="utf-8"))
        cid = m["config_id"]
        pred_path = out / "predictions" / f"{cid}.json"
        preds = json.loads(pred_path.read_text(encoding="utf-8"))
        cands = json.loads((out / "candidates" / f"{cid}.json").read_text(encoding="utf-8"))
        # Unlinked entities (concept_id None) stay in as predictions of "None", exactly as
        # Evaluator._load_predicted_concepts / 03_evaluation.ipynb score them (always wrong).
        ev = Evaluator(all_true, [[["concept", (p["start_index"], p["end_index"]), str(p["concept_id"])]
                                   for p in doc] for doc in preds], concept_proximity=proximity)
        strict = _quiet(ev.strict, zero_division=0)
        loose = _quiet(ev.loose)
        hier = _quiet(ev.hierarchy_aware) if proximity else None
        _, iou = _quiet(ev.iou)
        topk = _quiet(topk_accuracy, gold, preds)

        model = m.get("reranker_model")
        rerank_ms = (m["rerank_pairs_per_entity"] * spp[model] * 1000) if model and spp.get(model) else (0.0 if not model else None)
        est = (embed_ms + m["retrieval_mean_ms"] + rerank_ms) if (m.get("retrieval_mean_ms") is not None and rerank_ms is not None) else None

        rows.append({
            **{k: m.get(k) for k in COLUMNS if k in m},
            "link_acc": _quiet(concept_accuracy, gold, preds),
            "top1": topk["top1"], "top5": topk["top5"], "top10": topk["top10"],
            "cand_recall": candidate_recall(gold, cands),
            "strict_f1": strict["F1"]["val"] / 100, "strict_f1_l95": strict["F1"]["l95"] / 100,
            "strict_f1_u95": strict["F1"]["u95"] / 100,
            "loose_f1": loose["F1"]["val"] / 100,
            "hier_f1": hier["F1"]["val"] / 100 if hier else None,
            "iou": iou,
            "rerank_est_ms": rerank_ms,
            "est_ms_entity": est,
        })
    pareto_flags(rows)
    return rows


def _fmt(v, nd=3):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "*" if v else ""
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def write_outputs(out: Path, rows: List[dict], ner: dict) -> None:
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    md_cols = ["axis", "mode", "k_candidate", "expand_factor", "p_hybrid_weight", "p_fusion", "p_arm_depth_min",
               "p_rrf_k", "p_ef_search", "reranker", "selection", "link_acc", "top5", "cand_recall", "strict_f1",
               "retrieval_p50_ms", "rerank_est_ms", "est_ms_entity", "db_mem_peak_mb", "pareto"]
    lines = ["# Benchmark results", ""]
    s, l = ner["ner_strict"]["F1"], ner["ner_loose"]["F1"]
    lines += [f"NER (shared by every config): strict F1 {s['val']:.2f}% [{s['l95']:.2f}–{s['u95']:.2f}], "
              f"loose F1 {l['val']:.2f}% [{l['l95']:.2f}–{l['u95']:.2f}]", ""]
    idx_dir = out / "index"
    if idx_dir.exists():
        lines += ["## Index build", "", "| backend | load+index s | disk MB | mem after load MB |", "|---|---|---|---|"]
        for p in sorted(idx_dir.glob("*.json")):
            i = json.loads(p.read_text(encoding="utf-8"))
            lines.append(f"| {i['backend']} | {_fmt(i['load_s'], 1)} | {_fmt(i['db_disk_mb'], 1)} | {_fmt(i['db_mem_after_load_mb'], 1)} |")
        lines.append("")
    for backend in sorted({r["backend"] for r in rows}):
        rs = sorted((r for r in rows if r["backend"] == backend), key=lambda r: (r["mode"], r["axis"], r["config_id"]))
        lines += [f"## {backend}", "", "| " + " | ".join(md_cols) + " |", "|" + "---|" * len(md_cols)]
        for r in rs:
            lines.append("| " + " | ".join(_fmt(r.get(c), 1 if c.endswith(("_ms", "_mb")) else 3) for c in md_cols) + " |")
        lines.append("")
    (out / "results.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--extractions", required=True)
    ap.add_argument("--neo4j", action="store_true",
                    help="hierarchy-aware F1 via IS_A/HAS_ACTIVE_INGREDIENT neighbours (NEO4J_HOST/USER/PASSWORD env)")
    args = ap.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv(".env", override=False)  # NEO4J_* credentials for --neo4j
    out = Path(args.out)

    proximity = None
    if args.neo4j:
        gold_ids = extract_gold_concept_ids(Evaluator._load_gold_concept_data(args.gold))
        proximity = build_concept_proximity_from_neo4j(
            os.environ.get("NEO4J_HOST", "bolt://localhost:7687"), os.environ.get("NEO4J_USER", "neo4j"),
            os.environ.get("NEO4J_PASSWORD", ""), gold_ids)

    rows = build_rows(out, args.gold, proximity)
    write_outputs(out, rows, ner_summary(args.gold, args.extractions))
    print(f"{len(rows)} configs scored -> {out / 'results.csv'}, {out / 'results.md'}")
    try:
        from Benchmark.plots import plot_tradeoffs
        for p in plot_tradeoffs(rows, out):
            print(f"plot -> {p}")
    except ImportError as e:
        print(f"(skipping plots: {e})")


if __name__ == "__main__":
    main()
