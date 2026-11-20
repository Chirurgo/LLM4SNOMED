"""Stages 1-2 (LLM entity extraction + span positioning) run ONCE per document and
cached. These are backend-independent and the slowest/costliest stage, so every sweep
config links the exact same extracted entities -- differences between configs are then
purely retrieval/linking.

Cache schema: {"source": <gold path>, "llm": <llm id>, "documents": [[entity, ...], ...]}
with one entity list per gold record, same order. `null` marks a document whose
extraction failed (scored as no predictions, like Src/run.py's run_batch does).
"""
import json
import time
from pathlib import Path
from typing import List, Optional

from Src.run import extract_and_position


def run_extraction(gold_path: str, llm, llm_id: str, out_path: str, sleep_s: float = 30.0,
                   retry_failed: bool = False) -> str:
    """Resumable: documents already present in `out_path` are skipped, so a rate-limit
    crash mid-run never re-bills completed documents."""
    gold = json.loads(Path(gold_path).read_text(encoding="utf-8"))
    out = Path(out_path)
    if out.exists():
        cache = json.loads(out.read_text(encoding="utf-8"))
        if cache.get("llm") != llm_id or cache.get("source") != str(gold_path):
            raise ValueError(f"{out} was built for llm={cache.get('llm')!r}, source={cache.get('source')!r}; "
                             f"refusing to mix with llm={llm_id!r}, source={gold_path!r}. Use a new out_path.")
        docs: List[Optional[list]] = cache["documents"]
        failed = set(cache.get("failed", []))
        seconds = cache.get("seconds") or []
    else:
        docs, failed, seconds = [], set(), []
    docs += [None] * (len(gold) - len(docs))
    seconds += [None] * (len(gold) - len(seconds))
    # None + not failed = never attempted; failed docs are only retried on request.
    pending = [i for i, d in enumerate(docs) if d is None and (retry_failed or i not in failed)]

    out.parent.mkdir(parents=True, exist_ok=True)
    for n, i in enumerate(pending):
        t0 = time.perf_counter()
        try:
            docs[i] = extract_and_position(gold[i]["text"], llm)
            seconds[i] = round(time.perf_counter() - t0, 2)
            failed.discard(i)
            print(f"[extract] doc {i} ({gold[i].get('id')}): {len(docs[i])} entities in {seconds[i]:.1f}s", flush=True)
        except Exception as e:
            failed.add(i)
            print(f"[extract] doc {i} ({gold[i].get('id')}) FAILED: {e}")
        out.write_text(json.dumps({"source": str(gold_path), "llm": llm_id, "documents": docs,
                                   "failed": sorted(failed), "seconds": seconds}, ensure_ascii=False, indent=2), encoding="utf-8")
        if sleep_s and n < len(pending) - 1:
            time.sleep(sleep_s)
    return str(out)


def load_extractions(path: str, n_docs: int) -> List[list]:
    cache = json.loads(Path(path).read_text(encoding="utf-8"))
    docs = cache["documents"]
    if len(docs) != n_docs:
        raise ValueError(f"extraction cache has {len(docs)} documents but gold has {n_docs} -- "
                         f"evaluation aligns by position, so these must match")
    return [d or [] for d in docs]


def main(argv=None):
    """python -m Benchmark.extraction --llm lmstudio:glm-z1-9b-0414            --out Benchmark/results/full/extractions_glm-z1-9b.json [--gold Minimal_Dataset.json]"""
    import argparse
    from langchain_core.runnables import RunnableLambda
    from Benchmark.harness import make_llm, strip_reasoning
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="Minimal_Dataset.json")
    ap.add_argument("--llm", required=True, help="openai:<model> | gemini:<model> | lmstudio:<model>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sleep", type=float, default=None, help="seconds between documents (default 0 local, 30 hosted)")
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args(argv)
    llm = make_llm(args.llm) | RunnableLambda(strip_reasoning)
    sleep_s = args.sleep if args.sleep is not None else (0.0 if args.llm.startswith("lmstudio:") else 30.0)
    run_extraction(args.gold, llm, args.llm, args.out, sleep_s=sleep_s, retry_failed=args.retry_failed)


if __name__ == "__main__":
    main()
