"""Offline ranking study over frozen candidate lists (Benchmark.freeze_candidates).

No database is needed: every strategy re-ranks the SAME retrieved candidates, so
differences come from the ranker alone.

  rerank   score candidates with a cross-encoder (or keep retrieval order: --rankers none),
           for each frozen set x depth k x ranker x query variant; writes ranked lists to
           <out>/runs/*.json and the scored table to <out>/results.{csv,md}
  score    rebuild results.{csv,md} from <out>/runs (also scores imported strategies)

Query variants (what the cross-encoder sees as the "query"):
  en    canonical English name (formal_term), as in the paper pipeline
  ctx   the original mention (Thai / mixed) + a window of its source text, no English;
        only meaningful for multilingual rerankers

Usage:
  python -m Benchmark.rank_study rerank --rankers none bge-v2-m3 --queries en ctx --k 20 50
  python -m Benchmark.rank_study score
"""
import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

# BENCH_RESULTS switches every default path to another results root (the GPU rerun uses
# Benchmark/results_gpu, see EXPERIMENT.md), so a fresh run never mixes with old outputs.
RESULTS = os.environ.get("BENCH_RESULTS", "Benchmark/results")
GOLD = os.environ.get("BENCH_GOLD", "Minimal_Dataset.json")
EXTRACTIONS = f"{RESULTS}/gold/extractions_gold.json"
FROZEN = f"{RESULTS}/rank/frozen"
OUT = f"{RESULTS}/rank"
# Shared with the gold sweep: English-query bge-large scores are reused, not recomputed.
CACHE = f"{RESULTS}/gold/cache.sqlite"


def auto_device(device: Optional[str] = None) -> str:
    """'auto' / None -> cuda if available, else cpu."""
    if device and device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
TOP_N = 20          # ranked concepts kept per entity (same as Src/link.py)
CTX_CHARS = 80      # source-text window on each side of the mention for the ctx query

QWEN3_PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
                "and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
                "<|im_start|>user\n")
QWEN3_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
QWEN3_TASK = ("Given a clinical mention from a Thai medical record, judge whether the SNOMED CT "
              "description names the same clinical concept")

# key -> HF id, load kwargs, optional (query, doc) formatter
CROSS_ENCODERS: Dict[str, dict] = {
    "bge-large": {"id": "BAAI/bge-reranker-large"},                       # paper baseline, English
    "bge-v2-m3": {"id": "BAAI/bge-reranker-v2-m3"},                       # multilingual, 568M
    "jina-v2-ml": {"id": "jinaai/jina-reranker-v2-base-multilingual", "kw": {"trust_remote_code": True}},
    "gte-ml": {"id": "Alibaba-NLP/gte-multilingual-reranker-base", "kw": {"trust_remote_code": True}},
    "mxbai-v2": {"id": "mixedbread-ai/mxbai-rerank-base-v2"},
    "qwen3-0.6b": {"id": "tomaarsen/Qwen3-Reranker-0.6B-seq-cls",
                   "fmt": lambda q, d: (f"{QWEN3_PREFIX}<Instruct>: {QWEN3_TASK}\n<Query>: {q}\n",
                                        f"<Document>: {d}{QWEN3_SUFFIX}")},
}


# --- inputs ----------------------------------------------------------------------------
def load_inputs(gold_path=GOLD, extractions_path=EXTRACTIONS):
    gold = json.loads(Path(gold_path).read_text(encoding="utf-8"))
    docs = json.loads(Path(extractions_path).read_text(encoding="utf-8"))["documents"]
    return gold, docs


def context_window(text: str, s: int, e: int, n: int = CTX_CHARS) -> str:
    return " ".join(text[max(0, s - n):min(len(text), e + n)].split())


def make_query(variant: str, ent: dict, text: str) -> str:
    if variant == "en":
        return ent["formal_term"]
    if variant == "ctx":
        return f"Mention: {ent['term']}\nContext: {context_window(text, ent['start_index'], ent['end_index'])}"
    raise ValueError(variant)


def load_frozen(frozen_dir=FROZEN) -> Dict[str, list]:
    sets = {}
    for p in sorted(Path(frozen_dir).glob("*.json")):
        sets[p.stem] = json.loads(p.read_text(encoding="utf-8"))["entities"]
    if not sets:
        raise SystemExit(f"no frozen candidate sets in {frozen_dir} (run Benchmark.freeze_candidates)")
    return sets


# --- ranking ---------------------------------------------------------------------------
def dedupe(scored: List[tuple], top_n: int = TOP_N) -> List[dict]:
    """scored: [(cand, score)] already sorted best-first; keep best description per concept."""
    seen, out = set(), []
    for c, s in scored:
        if c["concept_id"] not in seen:
            seen.add(c["concept_id"])
            out.append({"concept_id": c["concept_id"], "text": c["text"], "fsn": c.get("fsn"),
                        "score": None if s is None else float(s)})
        if len(out) == top_n:
            break
    return out


class FormattedReranker:
    """Applies a model-specific (query, doc) template before the cached cross-encoder."""

    def __init__(self, key: str, cache, device: str = "cpu"):
        from Benchmark.caches import CachedReranker
        spec = CROSS_ENCODERS[key]
        self.key, self.fmt = key, spec.get("fmt")
        self.inner = CachedReranker(spec["id"], cache, device=device, **spec.get("kw", {}))

    def score(self, query: str, docs: List[str]) -> List[float]:
        pairs = [self.fmt(query, d) if self.fmt else (query, d) for d in docs]
        return self.inner.predict([list(p) for p in pairs])

    @property
    def seconds_per_pair(self):
        return self.inner.seconds_per_pair


def rank_set(entities: list, docs: list, gold: list, k: int, ranker, variant: str) -> list:
    out = []
    for i, j, cands in entities:
        ent = docs[i][j]
        pool = cands[:k]
        if ranker is None:
            ranked = dedupe([(c, None) for c in pool])
        else:
            q = make_query(variant, ent, gold[i]["text"])
            scores = ranker.score(q, [c["text"] for c in pool]) if pool else []
            ranked = dedupe(sorted(zip(pool, scores), key=lambda x: x[1], reverse=True))
        out.append({"doc": i, "ent": j, "pool": [c["concept_id"] for c in pool], "ranked": ranked})
    return out


def run_id(set_name: str, k: int, ranker: str, variant: str) -> str:
    return f"{set_name}__k{k}__{ranker}" + ("" if ranker == "none" else f"__{variant}")


# --- scoring ---------------------------------------------------------------------------
def gold_codes(gold: list) -> dict:
    return {(i, j): str(e["Target code"]) for i, rec in enumerate(gold) for j, e in enumerate(rec["entity"])}


def wilson(h: int, n: int, z: float = 1.96):
    if n == 0:
        return None, None
    p = h / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - w, c + w


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar (binomial) on discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, x) for x in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def correctness(run: list, codes: dict, pick=None) -> List[int]:
    """1/0 per entity. pick(rec) -> chosen concept_id (default: ranked[0])."""
    out = []
    for r in run:
        g = codes[(r["doc"], r["ent"])]
        chosen = pick(r) if pick else (r["ranked"][0]["concept_id"] if r["ranked"] else None)
        out.append(int(str(chosen) == g))
    return out


def score_run(run: list, codes: dict, pick=None) -> dict:
    hits = correctness(run, codes, pick)
    n = len(hits)
    top5 = sum(codes[(r["doc"], r["ent"])] in [c["concept_id"] for c in r["ranked"][:5]] for r in run)
    rr = 0.0
    for r in run:
        ids = [c["concept_id"] for c in r["ranked"]]
        g = codes[(r["doc"], r["ent"])]
        rr += 1 / (ids.index(g) + 1) if g in ids else 0.0
    recall = sum(codes[(r["doc"], r["ent"])] in set(r["pool"]) for r in run)
    lo, hi = wilson(sum(hits), n)
    return {"n": n, "acc": sum(hits) / n, "acc_l95": lo, "acc_u95": hi, "top5": top5 / n,
            "mrr": rr / n, "cand_recall": recall / n, "_hits": hits}


def write_table(rows: List[dict], out: Path) -> None:
    cols = ["run_id", "set", "k", "ranker", "query", "acc", "acc_l95", "acc_u95", "top5", "mrr",
            "cand_recall", "vs_none_p", "ms_per_entity", "n"]
    rows = sorted(rows, key=lambda r: (-r["acc"], r["run_id"]))
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    fmt = lambda v: "" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
    lines = ["# Ranking study (offline, frozen candidates)", "",
             "acc = top-1 concept accuracy on gold spans, 95% Wilson CI; vs_none_p = exact McNemar vs "
             "retrieval order on the same set and k; ms_per_entity = pairs x calibrated CPU s/pair.", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def score_all(out: Path, gold: list) -> List[dict]:
    codes = gold_codes(gold)
    runs = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / "runs").glob("*.json"))}
    rows = []
    for rid, blob in runs.items():
        meta, run = blob["meta"], blob["run"]
        s = score_run(run, codes)
        base = runs.get(run_id(meta["set"], meta["k"], "none", ""))
        p = None
        if base is not None and meta["ranker"] != "none":
            b_hits = correctness(base["run"], codes)
            b = sum(x and not y for x, y in zip(s["_hits"], b_hits))
            c = sum(y and not x for x, y in zip(s["_hits"], b_hits))
            p = mcnemar_p(b, c)
        rows.append({"run_id": rid, **meta, **{k: v for k, v in s.items() if not k.startswith("_")},
                     "vs_none_p": p})
    write_table(rows, out)
    return rows


# --- CLI -------------------------------------------------------------------------------
def cmd_rerank(args):
    from Benchmark.caches import KVCache
    gold, docs = load_inputs()
    sets = load_frozen(args.frozen)
    if args.sets:
        sets = {k: v for k, v in sets.items() if k in args.sets}
    out = Path(args.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    cache = KVCache(args.cache)
    for rk in args.rankers:
        ranker = None if rk == "none" else FormattedReranker(rk, cache, device=auto_device(args.device))
        variants = [""] if rk == "none" else args.queries
        for variant in variants:
            for set_name, entities in sets.items():
                for k in args.k:
                    rid = run_id(set_name, k, rk, variant)
                    path = out / "runs" / f"{rid}.json"
                    if path.exists() and not args.force:
                        print(f"  [skip] {rid}")
                        continue
                    t0 = time.perf_counter()
                    pairs = 0
                    if ranker:
                        ranker.inner.take_stats()
                    run = rank_set(entities, docs, gold, k, ranker, variant)
                    if ranker:
                        pairs = ranker.inner.take_stats()
                    spp = ranker.seconds_per_pair if ranker else 0.0
                    meta = {"set": set_name, "k": k, "ranker": rk, "query": variant or None,
                            "model": CROSS_ENCODERS[rk]["id"] if ranker else None,
                            "ms_per_entity": (pairs / max(1, len(run))) * (spp or 0) * 1000}
                    path.write_text(json.dumps({"meta": meta, "run": run}, ensure_ascii=False), encoding="utf-8")
                    print(f"  [done] {rid} ({time.perf_counter() - t0:.0f}s wall)", flush=True)
        del ranker
    score_all(out, gold)
    print(f"wrote {out / 'results.md'}")


def cmd_export_blind(args):
    """Selection-task file with NO gold fields: mention, context, English name, domain,
    and the base run's top-N options. The selector (LLM / human) reads only this file."""
    gold, docs = load_inputs()
    base = json.loads((Path(args.out) / "runs" / f"{args.base}.json").read_text(encoding="utf-8"))
    lines = [f"# base run: {args.base} | options: top-{args.n} | answer: <id>\\t<choice 1..{args.n}>\\t<reason>", ""]
    for r in base["run"]:
        i, j = r["doc"], r["ent"]
        ent = docs[i][j]
        lines.append(f"## {i}.{j}  domain={ent['domain']}")
        lines.append(f"mention: {ent['term']}")
        lines.append(f"english: {ent['formal_term']}")
        lines.append(f"context: {context_window(gold[i]['text'], ent['start_index'], ent['end_index'])}")
        for n, c in enumerate(r["ranked"][:args.n], 1):
            fsn = f"  [FSN: {c['fsn']}]" if c.get("fsn") and c["fsn"] != c["text"] else ""
            lines.append(f"  {n}. {c['text']}{fsn}")
        lines.append("")
    Path(args.file).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.file}: {len(base['run'])} items")


def cmd_import_picks(args):
    """Turn a picks TSV (<doc>.<ent> <choice> <reason>) into a run: chosen option first,
    the rest of the base ranking after it (so top5/MRR stay comparable)."""
    gold, _ = load_inputs()
    out = Path(args.out)
    base = json.loads((out / "runs" / f"{args.base}.json").read_text(encoding="utf-8"))
    picks = {}
    for line in Path(args.file).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        key, choice, *reason = line.split("\t")
        picks[key.strip()] = (int(choice), reason[0] if reason else "")
    run, missing = [], 0
    for r in base["run"]:
        key = f"{r['doc']}.{r['ent']}"
        ranked = list(r["ranked"])
        if key in picks and 1 <= picks[key][0] <= len(ranked):
            c = picks[key][0] - 1
            ranked = [ranked[c]] + ranked[:c] + ranked[c + 1:]
        else:
            missing += 1
        run.append({**r, "ranked": ranked, "reason": picks.get(key, (None, ""))[1]})
    m = base["meta"]
    meta = {**m, "ranker": f"{m['ranker']}+{args.name}", "selector": args.name, "base_run": args.base}
    rid = f"{args.base}__sel-{args.name}"
    (out / "runs" / f"{rid}.json").write_text(json.dumps({"meta": meta, "run": run}, ensure_ascii=False),
                                              encoding="utf-8")
    print(f"wrote run {rid} ({len(run) - missing} picks, {missing} missing -> kept base top-1)")
    score_all(out, gold)


def cmd_llm_select(args):
    """Paper stage 5 on a frozen base run: the LLM sees the English name, the mention and the
    base run's top-5 descriptions (Src.link.llm_select prompt, temperature 0) and picks one.
    Hosted LLMs: only for PUBLIC data (Minimal_Dataset) -- never the restricted dataset."""
    from langchain_core.documents import Document
    from Benchmark.caches import CachedLLMSelector, KVCache
    from Benchmark.harness import make_llm
    gold, docs = load_inputs()
    out = Path(args.out)
    base = json.loads((out / "runs" / f"{args.base}.json").read_text(encoding="utf-8"))
    rid = f"{args.base}__sel-{args.name or args.llm.replace(':', '-').replace('/', '-')}"
    path = out / "runs" / f"{rid}.json"
    if path.exists() and not args.force:
        print(f"  [skip] {rid}")
        return
    sel = CachedLLMSelector(make_llm(args.llm), args.llm, KVCache(args.cache), sleep_s=args.sleep)
    run, t0 = [], time.perf_counter()
    for n, r in enumerate(base["run"], 1):
        ent = docs[r["doc"]][r["ent"]]
        ranked = list(r["ranked"])
        top = [Document(page_content=c["text"], metadata={"concept_id": c["concept_id"]}) for c in ranked[:5]]
        while len(top) < 5:
            top.append(Document(page_content="(no content)", metadata={"concept_id": None}))
        choice, reason = sel.select(ent, top)
        if 1 <= choice <= min(5, len(ranked)):
            c = choice - 1
            ranked = [ranked[c]] + ranked[:c] + ranked[c + 1:]
        run.append({**r, "ranked": ranked, "reason": reason})
        if n % 25 == 0:
            print(f"  {n}/{len(base['run'])} ({sel.n_calls} new LLM calls)", flush=True)
    m = base["meta"]
    meta = {**m, "ranker": f"{m['ranker']}+llm-{args.llm}", "selector": args.llm, "base_run": args.base,
            "ms_per_entity": 1000 * sel.call_s / max(1, sel.n_calls) if sel.n_calls else None}
    path.write_text(json.dumps({"meta": meta, "run": run}, ensure_ascii=False), encoding="utf-8")
    print(f"wrote run {rid} ({time.perf_counter() - t0:.0f}s, {sel.n_calls} new calls)")
    score_all(out, gold)


def cmd_score(args):
    gold, _ = load_inputs()
    score_all(Path(args.out), gold)
    print(f"wrote {Path(args.out) / 'results.md'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rerank")
    r.add_argument("--rankers", nargs="+", required=True, choices=["none", *CROSS_ENCODERS])
    r.add_argument("--queries", nargs="+", default=["en"], choices=["en", "ctx"])
    r.add_argument("--k", nargs="+", type=int, default=[20, 50])
    r.add_argument("--sets", nargs="+", help="frozen set names (default: all)")
    r.add_argument("--frozen", default=FROZEN)
    r.add_argument("--out", default=OUT)
    r.add_argument("--cache", default=CACHE)
    r.add_argument("--device", default="auto", help="auto = cuda if available")
    r.add_argument("--force", action="store_true")
    r.set_defaults(fn=cmd_rerank)
    m = sub.add_parser("llm-select", help="paper's LLM selection over a base run's top-5 (public data only)")
    m.add_argument("--base", required=True, help="run id, e.g. qdrant_hybrid_k50_x2_w0.5_d50_rrf60__k20__none")
    m.add_argument("--llm", required=True, help="openai:<model> | gemini:<model> | lmstudio:<model>")
    m.add_argument("--name", help="selector name in the run id (default from --llm)")
    m.add_argument("--sleep", type=float, default=1.0, help="seconds between uncached calls")
    m.add_argument("--cache", default=f"{OUT}/llm_cache.sqlite")
    m.add_argument("--out", default=OUT)
    m.add_argument("--force", action="store_true")
    m.set_defaults(fn=cmd_llm_select)
    s = sub.add_parser("score")
    s.add_argument("--out", default=OUT)
    s.set_defaults(fn=cmd_score)
    e = sub.add_parser("export-blind")
    e.add_argument("--base", required=True, help="run id whose top-N options are shown")
    e.add_argument("--n", type=int, default=5)
    e.add_argument("--file", required=True)
    e.add_argument("--out", default=OUT)
    e.set_defaults(fn=cmd_export_blind)
    i = sub.add_parser("import-picks")
    i.add_argument("--base", required=True)
    i.add_argument("--file", required=True)
    i.add_argument("--name", required=True, help="selector name, e.g. claude-inline")
    i.add_argument("--out", default=OUT)
    i.set_defaults(fn=cmd_import_picks)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
