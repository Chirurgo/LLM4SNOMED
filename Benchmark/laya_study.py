"""Laya (convaiinnovations/laya-multilingual) as reranker, selector and validator.

Runs on the frozen candidate sets of Benchmark.rank_study and writes run files in the
same format, so `python -m Benchmark.rank_study score` puts them in the same table with
McNemar tests against retrieval order. Needs the `jevalt` env (laya needs transformers 5);
this module imports only the stdlib parts of rank_study.

Strategies (zero-shot, no fit/calibration on gold: the 151 gold items are the test set):
  noul    pointwise reranker: one yes/no question per (mention, candidate concept),
          candidates of the top-k pool re-sorted by P(yes)          -> ranker "laya-noul"
  choice  one choice question over the top-N concepts in retrieval order; those N re-sorted
          by choice probability, the rest keep retrieval order      -> ranker "laya-choiceN"

Validation (`validate`): how well each Laya signal tells a correct top-1 from a wrong one
for the retrieval-order baseline (ranker none) of each set: AUROC, accept-precision and
coverage at fixed thresholds, and a gate (keep top-1 if P(yes) >= t, else the best-noul
concept in the top 5). Report -> <out>/laya_validation.md

Query variants (what Laya sees about the mention; the candidate side is always the
description + FSN):
  en    canonical English name + category (no Thai), as the cross-encoders' `en`
  ctx   Thai/mixed mention + +-80-char source window + category, no English

Usage (repo root, jevalt env):
  python -m Benchmark.laya_study run --strategies noul choice --queries en ctx
  python -m Benchmark.laya_study validate
"""
import argparse
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, List

from Benchmark.rank_study import (FROZEN, OUT, context_window, dedupe, gold_codes, load_frozen,
                                  load_inputs, run_id, score_all)

# key -> (cache/meta id, laya.Agent kwargs, run-id prefix, validation report name)
MODELS = {
    "multilingual": ("convaiinnovations/laya-multilingual",
                     {"model_id": "convaiinnovations/laya-multilingual"}, "laya", "laya_validation.md"),
    # vegaml engine: frozen Qwen3.5-0.8B reader (nandakishorm/vega-U); ~0.75 s/pair on CPU
    "vega": ("vegaml:nandakishorm/vega-U", {"vegaml": True}, "vega", "vega_validation.md"),
}
CACHE = f"{OUT}/laya_cache.sqlite"
NOUL_Q = {"same": {"type": "noul", "instructions":
                   "Does the SNOMED CT concept name the same clinical concept as the mention?"}}
CHOICE_INSTR = "Which SNOMED CT concept best matches the clinical mention?"
THRESHOLDS = (0.5, 0.8, 0.9, 0.95, 0.99)


def mention_state(variant: str, ent: dict, text: str) -> dict:
    if variant == "en":
        return {"english name": ent["formal_term"], "category": ent["domain"]}
    if variant == "ctx":
        return {"mention": ent["term"], "category": ent["domain"],
                "context": context_window(text, ent["start_index"], ent["end_index"])}
    raise ValueError(variant)


def concept_label(c: dict) -> str:
    fsn = c.get("fsn")
    return c["text"] if not fsn or fsn == c["text"] else f"{c['text']} | {fsn}"


def unique_concepts(pool: list) -> list:
    seen, out = set(), []
    for c in pool:
        if c["concept_id"] not in seen:
            seen.add(c["concept_id"])
            out.append(c)
    return out


class Laya:
    """A Laya agent (see MODELS) with an on-disk cache of noul / choice answers."""

    def __init__(self, model_key="multilingual", cache_path=CACHE, device="cpu", batch_size=32):
        import laya
        self.model_id, kw, self.prefix, self.report = MODELS[model_key]
        kw = dict(kw)
        mid = kw.pop("model_id", None)
        self.agent = laya.Agent(mid, device=device, **kw) if mid else laya.Agent(device=device, **kw)
        self.batch_size = batch_size
        self.db = sqlite3.connect(cache_path)
        self.db.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")
        self.seconds = {"noul": 0.0, "choice": 0.0}
        self.calls = {"noul": 0, "choice": 0}

    def _key(self, kind, payload) -> str:
        blob = json.dumps([self.model_id, kind, payload], ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _get(self, k):
        row = self.db.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        return None if row is None else json.loads(row[0])

    def _put(self, items):
        self.db.executemany("INSERT OR REPLACE INTO kv VALUES (?, ?)",
                            [(k, json.dumps(v)) for k, v in items])
        self.db.commit()

    def noul(self, states: List[dict]) -> List[float]:
        keys = [self._key("noul", s) for s in states]
        out = [self._get(k) for k in keys]
        miss = [n for n, v in enumerate(out) if v is None]
        if miss:
            t0 = time.perf_counter()
            res = self.agent.predict_batch([states[n] for n in miss], NOUL_Q,
                                           batch_size=self.batch_size, sort_by_length=True)
            self.seconds["noul"] += time.perf_counter() - t0
            self.calls["noul"] += len(miss)
            vals = [float(r["answers"]["same"]["noul"]) for r in res]
            for n, v in zip(miss, vals):
                out[n] = v
            self._put([(keys[n], v) for n, v in zip(miss, vals)])
        return out

    def choice(self, state: dict, labels: List[str]) -> List[float]:
        """Probability per label (same order)."""
        crit = {f"c{n + 1}": lab for n, lab in enumerate(labels)}
        k = self._key("choice", [state, crit, CHOICE_INSTR])
        v = self._get(k)
        if v is None:
            t0 = time.perf_counter()
            a = self.agent.predict(state, {"pick": {"type": "choice", "instructions": CHOICE_INSTR,
                                                    "criteria": crit}})["answers"]["pick"]
            self.seconds["choice"] += time.perf_counter() - t0
            self.calls["choice"] += 1
            v = [float(a["probabilities"][f"c{n + 1}"]) for n in range(len(labels))]
            self._put([(k, v)])
        return v

    def per_call(self, kind) -> float:
        return self.seconds[kind] / self.calls[kind] if self.calls[kind] else float("nan")


# --- strategies ------------------------------------------------------------------------
def run_noul(model: Laya, entities, docs, gold, k, variant):
    recs, states, spans = [], [], []
    for i, j, cands in entities:
        ent = docs[i][j]
        pool = cands[:k]
        uniq = unique_concepts(pool)
        base = mention_state(variant, ent, gold[i]["text"])
        spans.append((len(states), uniq, pool, i, j))
        states += [{**base, "SNOMED CT concept": concept_label(c)} for c in uniq]
    scores = model.noul(states)
    for start, uniq, pool, i, j in spans:
        s = scores[start:start + len(uniq)]
        ranked = dedupe(sorted(zip(uniq, s), key=lambda x: x[1], reverse=True))
        recs.append({"doc": i, "ent": j, "pool": [c["concept_id"] for c in pool], "ranked": ranked})
    return recs, len(states)


def run_choice(model: Laya, entities, docs, gold, k, variant, n_opts):
    recs = []
    for i, j, cands in entities:
        ent = docs[i][j]
        pool = cands[:k]
        uniq = unique_concepts(pool)
        head, tail = uniq[:n_opts], uniq[n_opts:]
        if len(head) > 1:
            probs = model.choice(mention_state(variant, ent, gold[i]["text"]),
                                 [concept_label(c) for c in head])
            head_scored = sorted(zip(head, probs), key=lambda x: x[1], reverse=True)
        else:
            head_scored = [(c, 1.0) for c in head]
        ranked = dedupe(head_scored + [(c, None) for c in tail])
        recs.append({"doc": i, "ent": j, "pool": [c["concept_id"] for c in pool], "ranked": ranked})
    return recs


def cmd_run(args):
    gold, docs = load_inputs()
    sets = load_frozen(args.frozen)
    if args.sets:
        sets = {s: v for s, v in sets.items() if s in args.sets}
    out = Path(args.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    model = Laya(args.model, args.cache, device=args.device, batch_size=args.batch_size)
    for strat in args.strategies:
        ranker = f"{model.prefix}-noul" if strat == "noul" else f"{model.prefix}-choice{args.choice_n}"
        for variant in args.queries:
            for set_name, entities in sets.items():
                for k in args.k:
                    rid = run_id(set_name, k, ranker, variant)
                    path = out / "runs" / f"{rid}.json"
                    if path.exists() and not args.force:
                        print(f"  [skip] {rid}")
                        continue
                    t0 = time.perf_counter()
                    if strat == "noul":
                        run, pairs = run_noul(model, entities, docs, gold, k, variant)
                        ms = pairs / len(run) * model.per_call("noul") * 1000
                    else:
                        run = run_choice(model, entities, docs, gold, k, variant, args.choice_n)
                        ms = model.per_call("choice") * 1000
                    meta = {"set": set_name, "k": k, "ranker": ranker, "query": variant,
                            "model": model.model_id, "ms_per_entity": ms}
                    path.write_text(json.dumps({"meta": meta, "run": run}, ensure_ascii=False),
                                    encoding="utf-8")
                    print(f"  [done] {rid} ({time.perf_counter() - t0:.0f}s wall)", flush=True)
    score_all(out, gold)
    print(f"wrote {out / 'results.md'}")


# --- validation ------------------------------------------------------------------------
def auroc(pos: List[float], neg: List[float]) -> float:
    if not pos or not neg:
        return float("nan")
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def cmd_validate(args):
    gold, docs = load_inputs()
    codes = gold_codes(gold)
    sets = load_frozen(args.frozen)
    if args.sets:
        sets = {s: v for s, v in sets.items() if s in args.sets}
    model = Laya(args.model, args.cache, device=args.device, batch_size=args.batch_size)
    lines = [f"# Laya as validator (zero-shot, {model.model_id})", "",
             "Base = retrieval order (ranker none, k20). Signal = P(yes) from the noul question on "
             "(mention, top-1 concept), or the choice probability of the top-1 among the top 5. "
             "AUROC: correct vs wrong top-1. At threshold t: coverage = share of top-1 accepted, "
             "precision = accuracy of accepted. Gate: keep top-1 if noul >= t, else the "
             "highest-noul concept of the top 5.", ""]
    for variant in args.queries:
        lines += [f"## query `{variant}`", "",
                  "| set | base acc | signal | AUROC | " +
                  " | ".join(f"t={t} cov / prec / gate acc" for t in THRESHOLDS) + " |",
                  "|---|---|---|---|" + "---|" * len(THRESHOLDS)]
        for set_name, entities in sets.items():
            rows = []
            for i, j, cands in entities:
                ent = docs[i][j]
                top5 = unique_concepts(cands[:20])[:5]
                base = mention_state(variant, ent, gold[i]["text"])
                rows.append((codes[(i, j)], top5, base, ent, i))
            states = [{**b, "SNOMED CT concept": concept_label(c)} for _, t5, b, _, _ in rows for c in t5]
            flat = iter(model.noul(states))
            nouls = [[next(flat) for _ in t5] for _, t5, _, _, _ in rows]
            choice_p = []
            for g, t5, b, ent, i in rows:
                choice_p.append(model.choice(b, [concept_label(c) for c in t5])[0]
                                if len(t5) > 1 else 1.0)
            correct = [bool(t5) and str(t5[0]["concept_id"]) == g for g, t5, _, _, _ in rows]
            n = len(rows)
            base_acc = sum(correct) / n
            for sig_name, sig in (("noul", [nl[0] if nl else 0.0 for nl in nouls]),
                                  ("choice-p1", choice_p)):
                auc = auroc([s for s, c in zip(sig, correct) if c], [s for s, c in zip(sig, correct) if not c])
                cells = []
                for t in THRESHOLDS:
                    acc_ = [c for s, c in zip(sig, correct) if s >= t]
                    cov = len(acc_) / n
                    prec = sum(acc_) / len(acc_) if acc_ else float("nan")
                    gate = "–"
                    if sig_name == "noul":
                        hits = 0
                        for (g, t5, _, _, _), nl in zip(rows, nouls):
                            if not t5:
                                continue
                            pick = t5[0] if nl[0] >= t else max(zip(t5, nl), key=lambda x: x[1])[0]
                            hits += str(pick["concept_id"]) == g
                        gate = f"{hits / n:.3f}"
                    cells.append(f"{cov:.2f} / {prec:.3f} / {gate}")
                lines.append(f"| {set_name} | {base_acc:.3f} | {sig_name} | {auc:.3f} | " + " | ".join(cells) + " |")
            # how saturated is the noul signal?
            all_n = [x for nl in nouls for x in nl]
            lines.append(f"| {set_name} | | noul share >= 0.9 over top-5 | {sum(x >= 0.9 for x in all_n) / len(all_n):.2f} |" +
                         " |" * len(THRESHOLDS))
        lines.append("")
    path = Path(args.out) / model.report
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "validate"):
        p = sub.add_parser(name)
        p.add_argument("--queries", nargs="+", choices=["en", "ctx"], default=["en", "ctx"])
        p.add_argument("--sets", nargs="+")
        p.add_argument("--frozen", default=FROZEN)
        p.add_argument("--out", default=OUT)
        p.add_argument("--cache", default=CACHE)
        p.add_argument("--device", default="cpu")
        p.add_argument("--model", choices=sorted(MODELS), default="multilingual")
        p.add_argument("--batch-size", type=int, default=32)
        if name == "run":
            p.add_argument("--strategies", nargs="+", choices=["noul", "choice"], default=["noul", "choice"])
            p.add_argument("--k", nargs="+", type=int, default=[20])
            p.add_argument("--choice-n", type=int, default=5)
            p.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    {"run": cmd_run, "validate": cmd_validate}[args.cmd](args)


if __name__ == "__main__":
    main()
