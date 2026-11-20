"""
SNO-PI-style listwise reranker (arXiv:2610.26425, stage 1): train a cross-encoder on SNOMED CT
itself, no annotated data. A synonym is the query; the options are SapBERT-nearest concepts
(query row excluded) plus structural / lexical hard negatives; loss = softmax cross-entropy
over the options. Each option shows FSN, two shortest synonyms and the parent FSN.

Profiles (see EXPERIMENT.md):
  small   the CPU feasibility setup (4,000 train / 400 dev queries, 50% SapBERT-hard)
  full    the WHOLE SNOMED CT clinical corpus: every eligible concept in the 8 domains
          becomes a query (up to --queries-per-concept synonyms each), natural difficulty
          mix, several epochs, GPU. Dev stays a fixed 50%-hard sample, so dev numbers are
          comparable across profiles.

Hard negatives per query (options = gold + negatives, filled up to N_OPT from SapBERT kNN):
  parents (2) + one sibling each       structural, always on
  --children   2 children of the gold  (fixes "prefer the more specific concept")
  --negation   "No X" / "Absence of X" / "X absent" concepts and the reverse
  --product    substance <-> "Product containing X" medicinal products
  --none       an extra "none of the above" option; with probability --p-none the gold is
               removed and NONE is the target, so the model can abstain at test time

Subcommands (repo root, conda env `thesis`; GPU used automatically when available):
  prep   build <work>/: float16 embedding memmap, concept table with IS_A parents,
         train.jsonl / dev.json query sets with candidate options
  train  fine-tune a cross-encoder (default SapBERT) with listwise CE; checkpoints every
         epoch, dev accuracy per epoch, best epoch kept, resumable (--resume)
  eval   dev accuracy vs SapBERT order + rerank frozen candidate sets -> rank_study runs
         (variants: model / -rrf / -gate / -none) + snopi_<name>.md and .json

Gold codes are read only to EXCLUDE the 151 test concepts from training and dev queries.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

CORPUS = os.environ.get("SNOPI_CORPUS", "DATA/corpus_cache/snomed_20250301_sapbert_bm25.parquet")
IS_A, INFERRED = "116680003", "900000000000011006"
N_OPT = 10
MAX_LEN = 64
NONE_ID = "__NONE__"
NONE_TEXT = "none of the above: no option names the same concept"
NEG_PREFIXES = ("no ", "absence of ", "no history of ", "not ")
NEG_SUFFIXES = (" absent", " not present", " negative")
PRODUCT_PREFIXES = ("product containing ", "product containing only ")


def rf2_relationships() -> str:
    """sct2_Relationship_Snapshot_*.txt under SNOMED_RELEASE_PATH (or SNOPI_RF2_REL)."""
    if os.environ.get("SNOPI_RF2_REL"):
        return os.environ["SNOPI_RF2_REL"]
    root = os.environ.get("SNOMED_RELEASE_PATH")
    if not root:
        from dotenv import load_dotenv
        load_dotenv(".env", override=False)
        root = os.environ.get("SNOMED_RELEASE_PATH")
    if not root:
        raise SystemExit("set SNOMED_RELEASE_PATH in .env (RF2 release folder)")
    hits = glob.glob(os.path.join(root, "**", "sct2_Relationship_Snapshot_*.txt"), recursive=True)
    if not hits:
        raise SystemExit(f"no sct2_Relationship_Snapshot_*.txt under {root}")
    return sorted(hits)[0]


def pick_device(device: str = "auto") -> str:
    import torch
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def base_name(fsn):
    return (fsn or "").rsplit(" (", 1)[0]


# --- prep ------------------------------------------------------------------------------
def build_corpus_arrays(work: Path):
    import pyarrow.parquet as pq
    f = pq.ParquetFile(CORPUS)
    n = f.metadata.num_rows
    emb = np.lib.format.open_memmap(work / "emb_f16.npy", mode="w+", dtype=np.float16, shape=(n, 768))
    cids, texts, ntypes, tags, fsn = [], [], [], [], {}
    off = 0
    for g in range(f.metadata.num_row_groups):
        t = f.read_row_group(g, columns=["concept_id", "text", "name_type", "semantic_tag", "fsn", "embedding"])
        e = t.column("embedding").combine_chunks()
        m = e.values.to_numpy(zero_copy_only=False).reshape(-1, 768)
        emb[off:off + len(m)] = m.astype(np.float16)
        off += len(m)
        cid = [str(x) for x in t.column("concept_id").to_pylist()]
        cids += cid
        texts += t.column("text").to_pylist()
        ntypes += t.column("name_type").to_pylist()
        tags += t.column("semantic_tag").to_pylist()
        for c, s in zip(cid, t.column("fsn").to_pylist()):
            fsn[c] = s
        print(f"  row group {g + 1}/{f.metadata.num_row_groups}", flush=True)
    emb.flush()
    del emb
    assert off == n
    return cids, texts, ntypes, tags, fsn


def load_parents(concepts: set) -> dict:
    import polars as pl
    rel = (pl.scan_csv(rf2_relationships(), separator="\t", quote_char=None, schema_overrides={
                "sourceId": pl.Utf8, "destinationId": pl.Utf8, "typeId": pl.Utf8,
                "characteristicTypeId": pl.Utf8, "active": pl.Int8})
           .filter((pl.col("active") == 1) & (pl.col("typeId") == IS_A)
                   & (pl.col("characteristicTypeId") == INFERRED))
           .select(["sourceId", "destinationId"]).collect())
    parents = defaultdict(list)
    for s, d in zip(rel["sourceId"].to_list(), rel["destinationId"].to_list()):
        if s in concepts:
            parents[s].append(d)
    return {k: sorted(v) for k, v in parents.items()}


class KNN:
    """Top concepts (best row per concept) for query rows, own row excluded.
    GPU: allowed rows resident in fp16, torch.topk. CPU: chunked numpy (slow at full scale)."""

    def __init__(self, emb, cids, allowed_idx, device, k_rows=80):
        self.emb, self.cids, self.idx, self.k = emb, cids, np.asarray(allowed_idx), k_rows
        self.device = device
        if device == "cuda":
            import torch
            self.t_idx = torch.as_tensor(self.idx, device="cuda")
            self.t_emb = torch.as_tensor(np.ascontiguousarray(emb[self.idx]), device="cuda")

    def query(self, q_rows, qbatch=None):
        out = []
        qbatch = qbatch or (1024 if self.device == "cuda" else 1000)
        for a in range(0, len(q_rows), qbatch):
            rows = np.asarray(q_rows[a:a + qbatch])
            s, i = self._gpu(rows) if self.device == "cuda" else self._cpu(rows)
            for row, s_row, i_row in zip(rows, s, i):
                seen, lst = set(), []
                for j in np.argsort(-s_row):
                    if i_row[j] == row or s_row[j] == -np.inf:
                        continue
                    c = self.cids[i_row[j]]
                    if c not in seen:
                        seen.add(c)
                        lst.append(c)
                out.append(lst)
        return out

    def _gpu(self, rows):
        import torch
        q = torch.as_tensor(np.ascontiguousarray(self.emb[rows]), device="cuda")
        S = q @ self.t_emb.T              # own row is dropped in query()
        s, j = torch.topk(S, min(self.k + 1, S.shape[1]), dim=1)
        return s.float().cpu().numpy(), self.t_idx[j].cpu().numpy()

    def _cpu(self, rows, chunk=50_000):
        Q = self.emb[rows].astype(np.float32)
        best_s = np.full((len(rows), self.k), -np.inf, dtype=np.float32)
        best_i = np.zeros((len(rows), self.k), dtype=np.int64)
        for a in range(0, len(self.idx), chunk):
            idx = self.idx[a:a + chunk]
            S = Q @ self.emb[idx].astype(np.float32).T
            S[rows[:, None] == idx[None, :]] = -np.inf
            cat_s = np.concatenate([best_s, S], axis=1)
            cat_i = np.concatenate([best_i, np.broadcast_to(idx, S.shape)], axis=1)
            top = np.argpartition(-cat_s, self.k, axis=1)[:, :self.k]
            best_s = np.take_along_axis(cat_s, top, 1)
            best_i = np.take_along_axis(cat_i, top, 1)
        return best_s, best_i


def lexical_negatives(concepts: dict) -> tuple[dict, dict]:
    """negation[c] / product[c]: concepts whose name differs from c's only by a negation
    marker, or substance <-> 'Product containing X'. Matched on lower-case FSN base name
    and synonyms, across the whole corpus."""
    by_name = defaultdict(set)
    for c, v in concepts.items():
        for n in [base_name(v["fsn"])] + v["syn"]:
            by_name[n.lower()].add(c)
    neg, prod = defaultdict(set), defaultdict(set)
    for c, v in concepts.items():
        b = base_name(v["fsn"]).lower()
        if not b:
            continue
        for x in [p + b for p in NEG_PREFIXES] + [b + s for s in NEG_SUFFIXES]:
            for o in by_name.get(x, ()):
                if o != c:
                    neg[c].add(o)
                    neg[o].add(c)
        for x in [p + b for p in PRODUCT_PREFIXES]:
            for o in by_name.get(x, ()):
                if o != c:
                    prod[c].add(o)
                    prod[o].add(c)
    return ({k: sorted(v) for k, v in neg.items()}, {k: sorted(v) for k, v in prod.items()})


def cmd_prep(args):
    from Src.domain_mapping import TYPE_TO_SEMANTIC_TAG
    from Benchmark.rank_study import load_inputs, gold_codes
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    rng = random.Random(args.seed)
    t0 = time.perf_counter()
    meta_path = work / "corpus_meta.pkl"
    if meta_path.exists() and (work / "emb_f16.npy").exists():
        cids, texts, ntypes, tags, fsn = pickle.loads(meta_path.read_bytes())
    else:
        cids, texts, ntypes, tags, fsn = build_corpus_arrays(work)
        meta_path.write_bytes(pickle.dumps((cids, texts, ntypes, tags, fsn)))
    print(f"corpus arrays {time.perf_counter() - t0:.0f}s ({len(cids)} rows, knn on {device})", flush=True)

    syns = defaultdict(set)
    rows_of = defaultdict(list)
    tag_of = {}
    for r, (c, t, nt, tg) in enumerate(zip(cids, texts, ntypes, tags)):
        rows_of[c].append(r)
        tag_of[c] = tg
        if nt != "FSN":
            syns[c].add(t)
    parents = load_parents(set(rows_of))
    children = defaultdict(list)
    for c, ps in parents.items():
        for p in ps:
            children[p].append(c)
    concepts = {c: {"fsn": fsn.get(c), "tag": tag_of[c], "syn": sorted(syns[c], key=len),
                    "parents": parents.get(c, [])} for c in rows_of}
    neg_of, prod_of = lexical_negatives(concepts) if (args.negation or args.product) else ({}, {})
    print(f"concepts {len(concepts)}, with parents {sum(bool(v['parents']) for v in concepts.values())}, "
          f"with negation partner {len(neg_of)}, with product/substance partner {len(prod_of)}", flush=True)

    gold, _ = load_inputs()
    test_cids = set(gold_codes(gold).values())
    emb = np.load(work / "emb_f16.npy", mmap_mode="r")
    tag_arr = np.array(tags, dtype=object)

    def words(x):
        return set(x.lower().replace(",", " ").replace("-", " ").split())

    def pick_queries(c, n):
        """Up to n query rows: first by --query-mode, the rest random distinct synonyms."""
        base = base_name(concepts[c]["fsn"])
        non_fsn = [r for r in rows_of[c] if ntypes[r] != "FSN"]
        cands = [r for r in non_fsn if texts[r].lower() != base.lower()]
        if args.query_mode == "overlap" and not cands:
            return []
        first = None
        if args.query_mode == "mixed" and rng.random() < 0.5:
            first = rng.choice(non_fsn) if non_fsn else None
        elif cands:
            bw = words(base)
            first = min(cands, key=lambda r: (len(words(texts[r]) & bw) / max(1, len(words(texts[r]) | bw)), r))
        if first is None:
            return []
        rest = [r for r in non_fsn if r != first]
        rng.shuffle(rest)
        return [first] + rest[:n - 1]

    def options(c, nn):
        hard = []
        for p in concepts[c]["parents"][:2]:
            hard.append(p)
            sib = [x for x in children.get(p, []) if x != c]
            rng.shuffle(sib)
            hard += sib[:1]
        if args.children:
            kids = list(children.get(c, []))
            rng.shuffle(kids)
            hard += kids[:2]
        if args.negation:
            hard += neg_of.get(c, [])[:2]
        if args.product:
            hard += prod_of.get(c, [])[:2]
        hard = list(dict.fromkeys(h for h in hard if h in concepts and h != c))
        cap = 3 + 2 * args.children + 2 * args.negation + 2 * args.product
        opts = [c] + hard[:min(cap, N_OPT - 2)]
        for x in nn:
            if len(opts) >= N_OPT:
                break
            if x not in opts:
                opts.append(x)
        return opts

    full = args.profile == "full"
    per_dev = args.n_dev // len(TYPE_TO_SEMANTIC_TAG)
    per_train = None if full else args.n_train // len(TYPE_TO_SEMANTIC_TAG)
    dev_all, n_train = [], 0
    train_f = open(work / f"train{args.data}.jsonl", "w", encoding="utf-8")
    used = set(test_cids)
    stats = {}
    for dom, dtags in TYPE_TO_SEMANTIC_TAG.items():
        td = time.perf_counter()
        allowed = {t.lower() for t in dtags}
        pool = sorted(c for c, v in concepts.items() if v["tag"] in allowed and c not in used)
        rng.shuffle(pool)
        if not full:
            pool = pool[:int(args.pool * 1.5)]
        knn = KNN(emb, cids, np.nonzero(np.isin(tag_arr, list(allowed)))[0], device)
        # dev: 10% of the first `pool` concepts, one query each, then 50% hard / 50% easy
        n_dev_pool = max(per_dev * 4, (min(len(pool), args.pool) // 10))
        dev_pool, rest = pool[:n_dev_pool], pool[n_dev_pool:]
        dq = [(c, rs[0]) for c in dev_pool for rs in [pick_queries(c, 1)] if rs]
        dev_ex = [{"query": texts[r], "gold": c, "domain": dom, "sap_order": nn[:N_OPT],
                   "sap_rank": nn.index(c) if c in nn else None, "options": options(c, nn)}
                  for (c, r), nn in zip(dq, knn.query([r for _, r in dq]))]
        hard_ = [e for e in dev_ex if e["sap_rank"] != 0]
        easy_ = [e for e in dev_ex if e["sap_rank"] == 0]
        h = min(len(hard_), per_dev // 2)
        dev_all += hard_[:h] + easy_[:per_dev - h]
        # train: disjoint concepts
        if full:
            tq = [(c, r) for c in rest for r in pick_queries(c, args.queries_per_concept)]
        else:
            tq = [(c, rs[0]) for c in rest[:args.pool] for rs in [pick_queries(c, 1)] if rs]
        kept = 0
        hard_ex, easy_ex = [], []
        for a in range(0, len(tq), 20_000):
            part = tq[a:a + 20_000]
            for (c, r), nn in zip(part, knn.query([r for _, r in part])):
                ex = {"query": texts[r], "gold": c, "domain": dom,
                      "sap_rank": nn.index(c) if c in nn else None, "options": options(c, nn)}
                if full:
                    train_f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    kept += 1
                else:
                    (easy_ex if ex["sap_rank"] == 0 else hard_ex).append(ex)
            if full:
                print(f"    {dom}: {min(a + 20_000, len(tq))}/{len(tq)} train queries", flush=True)
        if not full:
            h2 = min(len(hard_ex), per_train // 2)
            for ex in hard_ex[:h2] + easy_ex[:per_train - h2]:
                train_f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                kept += 1
        n_train += kept
        used.update(pool)   # a concept never appears in two domains' train/dev sets
        sap1 = sum(e["sap_rank"] == 0 for e in dev_ex) / max(1, len(dev_ex))
        stats[dom] = {"train": kept, "dev": len(hard_[:h]) + len(easy_[:per_dev - h]),
                      "dev_pool_sapbert_top1": sap1, "pool_concepts": len(pool)}
        print(f"  {dom:<11} train {kept:>7}  dev {stats[dom]['dev']}  (pool {len(pool)} concepts, "
              f"dev-pool SapBERT top-1 {sap1:.2f}) {time.perf_counter() - td:.0f}s", flush=True)
        del knn
    train_f.close()
    (work / "concepts.pkl").write_bytes(pickle.dumps(concepts))
    (work / f"dev{args.data}.json").write_text(json.dumps(dev_all, ensure_ascii=False), encoding="utf-8")
    meta = {**vars(args), "n_train": n_train, "n_dev": len(dev_all), "per_domain": stats,
            "seconds": time.perf_counter() - t0, "excluded_test_concepts": len(test_cids)}
    (work / f"prep{args.data}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"prep done: {n_train} train / {len(dev_all)} dev queries, {time.perf_counter() - t0:.0f}s", flush=True)


# --- model -----------------------------------------------------------------------------
def option_text(concepts, cid, exclude=None):
    if cid == NONE_ID:
        return NONE_TEXT
    c = concepts.get(cid)
    if c is None:
        return cid
    base = base_name(c["fsn"]).lower()
    s = [x for x in c["syn"] if x != exclude and x.lower() != base][:2]
    par = ""
    if c["parents"]:
        p = concepts.get(c["parents"][0])
        par = p["fsn"] if p else ""
    return f"{c['fsn']}; {'; '.join(s)} | parent: {par}"


def load_model(path, device="cpu"):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1).to(device)
    return tok, model


def amp_ctx(device):
    import contextlib
    import torch
    if device == "cuda":
        dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dt)
    return contextlib.nullcontext()


def score_lists(tok, model, queries, option_lists, bs=None):
    import torch
    device = next(model.parameters()).device.type
    bs = bs or (512 if device == "cuda" else 64)
    model.eval()
    flat_q, flat_o, sizes = [], [], []
    for q, opts in zip(queries, option_lists):
        flat_q += [q] * len(opts)
        flat_o += opts
        sizes.append(len(opts))
    out = []
    with torch.inference_mode(), amp_ctx(device):
        for a in range(0, len(flat_q), bs):
            enc = tok(flat_q[a:a + bs], flat_o[a:a + bs], truncation="only_second", max_length=MAX_LEN,
                      padding=True, return_tensors="pt").to(device)
            out += model(**enc).logits.float().squeeze(-1).tolist()
    res, a = [], 0
    for n in sizes:
        res.append(out[a:a + n])
        a += n
    return res


def read_train(work: Path, data: str, limit=None):
    rows = []
    with open(work / f"train{data}.jsonl", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def dev_accuracy(tok, model, concepts, dev, use_none=False):
    """Top-1 over the SapBERT top-10 (gold may be missing -> wrong). With use_none the NONE
    option is appended; picking it counts as wrong but is reported as the abstain rate."""
    ol = [[option_text(concepts, o) for o in d["sap_order"]] + ([NONE_TEXT] if use_none else []) for d in dev]
    sc = score_lists(tok, model, [d["query"] for d in dev], ol)
    hits, abst = 0, 0
    for d, s in zip(dev, sc):
        j = int(np.argmax(s))
        if use_none and j == len(d["sap_order"]):
            abst += 1
        elif d["sap_order"][j] == d["gold"]:
            hits += 1
    return hits / len(dev), abst / len(dev), sc


def cmd_train(args):
    import torch
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    if device == "cpu":
        torch.set_num_threads(args.threads)
    rng = random.Random(args.seed)
    work = Path(args.work)
    concepts = pickle.loads((work / "concepts.pkl").read_bytes())
    train = read_train(work, args.data, args.max_queries)
    dev = json.loads((work / f"dev{args.data}.json").read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # one folder per epoch: the loaded checkpoint stays memory-mapped (Windows cannot overwrite it)
    ckpts = sorted(out.glob("checkpoint-epoch*"), key=lambda d: int(d.name.rsplit("epoch", 1)[1]))
    ckpts = [d for d in ckpts if (d / "state.pt").exists()]
    start_ep = 0
    init = args.init
    state = None
    if args.resume and ckpts:
        state = torch.load(ckpts[-1] / "state.pt", map_location="cpu", weights_only=False)
        start_ep, init = state["epoch"] + 1, str(ckpts[-1])
        print(f"resuming after epoch {state['epoch'] + 1} from {ckpts[-1]}", flush=True)
    tok, model = load_model(init, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_ep = math.ceil(len(train) / (args.bs * args.grad_accum))
    steps = steps_per_ep * args.epochs
    warm = max(1, int(args.warmup * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
    scaler = None
    if device == "cuda" and not torch.cuda.is_bf16_supported():
        scaler = torch.amp.GradScaler("cuda")
    best = {"dev_acc": -1.0, "epoch": None}
    history = []
    if state:
        opt.load_state_dict(state["opt"]); sched.load_state_dict(state["sched"])
        best, history = state["best"], state["history"]
    log = open(out / "train_log.jsonl", "a", encoding="utf-8")
    print(f"{len(train)} train queries, {len(dev)} dev, device {device}, {steps} optimizer steps "
          f"({steps_per_ep}/epoch, {args.bs}x{args.grad_accum} queries/step)", flush=True)
    t0 = time.perf_counter()
    step = start_ep * steps_per_ep
    for ep in range(start_ep, args.epochs):
        rng.shuffle(train)
        model.train()
        run_loss, n_loss, micro = 0.0, 0, 0
        for a in range(0, len(train), args.bs):
            batch = train[a:a + args.bs]
            qs, os_, sizes, golds = [], [], [], []
            for ex in batch:
                opts = ex["options"][:N_OPT]
                if args.none:
                    if rng.random() < args.p_none:
                        opts = [o for o in opts if o != ex["gold"]]
                        target = NONE_ID
                    else:
                        target = ex["gold"]
                    opts = opts + [NONE_ID]
                else:
                    target = ex["gold"]
                opts = opts[:]
                rng.shuffle(opts)
                excl = ex["query"] if rng.random() < args.p_exclude else None
                qs += [ex["query"]] * len(opts)
                os_ += [option_text(concepts, o, excl if o == ex["gold"] else None) for o in opts]
                sizes.append(len(opts))
                golds.append(opts.index(target))
            enc = tok(qs, os_, truncation="only_second", max_length=MAX_LEN, padding=True,
                      return_tensors="pt").to(device)
            with amp_ctx(device):
                flat = model(**enc).logits.float().squeeze(-1)
            logits = torch.full((len(batch), max(sizes)), -1e4, device=device)
            mask_i = torch.repeat_interleave(torch.arange(len(batch), device=device),
                                             torch.tensor(sizes, device=device))
            mask_j = torch.cat([torch.arange(n, device=device) for n in sizes])
            logits = logits.index_put((mask_i, mask_j), flat)
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor(golds, device=device)) / args.grad_accum
            (scaler.scale(loss) if scaler else loss).backward()
            run_loss += loss.item() * args.grad_accum
            n_loss += 1
            micro += 1
            if micro % args.grad_accum == 0 or a + args.bs >= len(train):
                if scaler:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler:
                    scaler.step(opt); scaler.update()
                else:
                    opt.step()
                sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0 or step == steps:
                    el = time.perf_counter() - t0
                    done = step - start_ep * steps_per_ep
                    rec = {"epoch": ep + 1, "step": step, "loss": run_loss / n_loss,
                           "lr": sched.get_last_lr()[0], "s_per_step": el / done}
                    log.write(json.dumps(rec) + "\n"); log.flush()
                    print(f"  ep {ep + 1} step {step}/{steps} loss {rec['loss']:.3f} "
                          f"{rec['s_per_step']:.2f} s/step, eta {(steps - step) * el / done / 60:.0f} min", flush=True)
                    run_loss, n_loss = 0.0, 0
        acc, abst, _ = dev_accuracy(tok, model, concepts, dev, use_none=args.none)
        sap = sum(d["sap_rank"] == 0 for d in dev) / len(dev)
        history.append({"epoch": ep + 1, "dev_acc": acc, "dev_none_rate": abst, "dev_sapbert_top1": sap,
                        "minutes": (time.perf_counter() - t0) / 60})
        log.write(json.dumps({"epoch_end": ep + 1, **history[-1]}) + "\n"); log.flush()
        print(f"== epoch {ep + 1}: dev top-1 {acc:.3f} (SapBERT {sap:.3f}, NONE picked {abst:.3f})", flush=True)
        if acc > best["dev_acc"]:
            best = {"dev_acc": acc, "epoch": ep + 1}
            model.save_pretrained(out); tok.save_pretrained(out)
            print(f"   best so far -> saved {out}", flush=True)
        ckpt = out / f"checkpoint-epoch{ep + 1}"
        ckpt.mkdir(exist_ok=True)
        model.save_pretrained(ckpt); tok.save_pretrained(ckpt)
        torch.save({"epoch": ep, "opt": opt.state_dict(), "sched": sched.state_dict(), "best": best,
                    "history": history}, ckpt / "state.pt")
        import shutil
        for old in out.glob("checkpoint-epoch*"):
            if old != ckpt:
                shutil.rmtree(old, ignore_errors=True)
    log.close()
    meta = {**vars(args), "device": device, "steps": steps, "queries": len(train), "dev": len(dev),
            "best": best, "history": history, "none_option": args.none,
            "seconds": time.perf_counter() - t0}
    if device == "cuda":
        meta["gpu"] = torch.cuda.get_device_name(0)
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"done: best dev {best['dev_acc']:.3f} at epoch {best['epoch']} -> {out} "
          f"({(time.perf_counter() - t0) / 60:.1f} min this session)", flush=True)


# --- eval ------------------------------------------------------------------------------
VARIANTS = ("", "-rrf", "-gate", "-none")
RRF_K, GATE_P = 60, 0.8   # fixed before seeing test results (2026-11-20)


def fuse(variant, pool, scores, none_score=None):
    """pool is in retrieval order, scores = model logits for it.
      ''     model order
      -rrf   reciprocal-rank fusion of retrieval and model ranks
      -gate  keep retrieval top-1 unless the model gives another option softmax p >= GATE_P
      -none  (models trained with --none) keep retrieval order if NONE outscores every option,
             else move the model's top option to rank 1, the rest in retrieval order"""
    from Benchmark.rank_study import dedupe
    order = sorted(range(len(pool)), key=lambda i: scores[i], reverse=True)
    if variant == "":
        idx = order
    elif variant == "-rrf":
        mr = {i: r for r, i in enumerate(order)}
        idx = sorted(range(len(pool)), key=lambda i: -(1 / (RRF_K + i + 1) + 1 / (RRF_K + mr[i] + 1)))
    elif variant == "-gate":
        z = np.exp(np.asarray(scores) - max(scores)); pr = z / z.sum()
        top = order[0]
        idx = list(range(len(pool)))
        if top != 0 and pr[top] >= GATE_P:
            idx = [top] + [i for i in idx if i != top]
    else:
        top = order[0]
        idx = list(range(len(pool)))
        if none_score is None or scores[top] > none_score:
            idx = [top] + [i for i in idx if i != top]
    return dedupe([(pool[i], scores[i]) for i in idx])


def cmd_eval(args):
    import torch
    device = pick_device(args.device)
    if device == "cpu":
        torch.set_num_threads(args.threads)
    from Benchmark.rank_study import load_inputs, load_frozen, make_query, dedupe, run_id, score_all, OUT
    work = Path(args.work)
    concepts = pickle.loads((work / "concepts.pkl").read_bytes())
    tok, model = load_model(args.model, device)
    tm_path = Path(args.model) / "train_meta.json"
    tmeta = json.loads(tm_path.read_text()) if tm_path.exists() else {}
    has_none = bool(tmeta.get("none_option"))
    variants = [v for v in VARIANTS if v != "-none" or has_none]
    name = args.name

    dev = json.loads((work / f"dev{args.data}.json").read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    acc, abst, scores = dev_accuracy(tok, model, concepts, dev, use_none=has_none)
    sap = sum(d["sap_rank"] == 0 for d in dev)
    inlist = sum(d["gold"] in d["sap_order"] for d in dev)
    by_dom = defaultdict(lambda: [0, 0, 0])
    for d, s in zip(dev, scores):
        b = by_dom[d["domain"]]
        j = int(np.argmax(s))
        b[0] += 1; b[1] += d["sap_rank"] == 0; b[2] += j < len(d["sap_order"]) and d["sap_order"][j] == d["gold"]
    dev_ms = (time.perf_counter() - t0) / len(dev) * 1000
    summary = {"name": name, "model": args.model, "device": device, "train_meta": tmeta,
               "dev": {"n": len(dev), "gold_in_top10": inlist / len(dev), "sapbert_top1": sap / len(dev),
                       "reranked_top1": acc, "none_rate": abst if has_none else None, "ms_per_query": dev_ms,
                       "by_domain": {k: {"n": n, "sapbert": a / n, "reranked": b / n}
                                     for k, (n, a, b) in sorted(by_dom.items())}},
               "test": []}
    lines = [f"# SNO-PI: {name}", "", f"model `{args.model}` · device {device}"]
    if tmeta:
        lines += [f"trained on {tmeta.get('queries')} queries, {tmeta.get('epochs')} epoch(s), "
                  f"best dev epoch {tmeta.get('best', {}).get('epoch')}, NONE option: {has_none}, "
                  f"{tmeta.get('seconds', 0) / 60:.0f} min"]
    lines += ["", f"## Dev (held-out SNOMED synonym queries, n={len(dev)}, SapBERT top-10 options)", "",
              f"gold in top-10: {inlist / len(dev):.3f} · SapBERT top-1: {sap / len(dev):.3f} · "
              f"reranked top-1: {acc:.3f}" + (f" · NONE picked {abst:.3f}" if has_none else "") +
              f" · {dev_ms:.1f} ms/query", "",
              "| domain | n | SapBERT top-1 | reranked top-1 |", "|---|---|---|---|"]
    for dom, v in summary["dev"]["by_domain"].items():
        lines.append(f"| {dom} | {v['n']} | {v['sapbert']:.3f} | {v['reranked']:.3f} |")
    print("\n".join(lines), flush=True)

    gold, docs = load_inputs()
    sets = load_frozen()
    if args.sets:
        sets = {k: v for k, v in sets.items() if k in args.sets}
    out = Path(OUT)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    for set_name, ents in sets.items():
        for k in args.k:
            t0 = time.perf_counter()
            qs, ol, pools = [], [], []
            for i, j, cands in ents:
                pool = dedupe([(c, None) for c in cands[:k]], top_n=k)
                pools.append((i, j, cands[:k], pool))
                qs.append(make_query("en", docs[i][j], gold[i]["text"]))
                ol.append([option_text(concepts, c["concept_id"]) for c in pool] + ([NONE_TEXT] if has_none else []))
            sc = score_lists(tok, model, qs, ol)
            ms = (time.perf_counter() - t0) / len(pools) * 1000
            runs = {v: [] for v in variants}
            for (i, j, raw, pool), s in zip(pools, sc):
                none_s = s[-1] if has_none else None
                s = s[:-1] if has_none else s
                base = {"doc": i, "ent": j, "pool": [c["concept_id"] for c in raw]}
                for v in variants:
                    runs[v].append({**base, "ranked": fuse(v, pool, s, none_s)})
            for v in variants:
                rk = name + v
                rid = run_id(set_name, k, rk, "en")
                meta = {"set": set_name, "k": k, "ranker": rk, "query": "en", "model": args.model,
                        "ms_per_entity": ms, "device": device}
                (out / "runs" / f"{rid}.json").write_text(json.dumps({"meta": meta, "run": runs[v]}, ensure_ascii=False),
                                                         encoding="utf-8")
                print(f"  [done] {rid} {ms:.1f} ms/entity", flush=True)
            if not (out / "runs" / f"{run_id(set_name, k, 'none', '')}.json").exists():
                none_run = [{"doc": i, "ent": j, "pool": [c["concept_id"] for c in raw], "ranked": dedupe([(c, None) for c in raw])}
                            for i, j, raw, _ in pools]
                (out / "runs" / f"{run_id(set_name, k, 'none', '')}.json").write_text(json.dumps(
                    {"meta": {"set": set_name, "k": k, "ranker": "none", "query": None, "model": None,
                              "ms_per_entity": 0.0}, "run": none_run}, ensure_ascii=False), encoding="utf-8")
    rows = score_all(out, gold)
    acc_ = {(r["set"], r["k"], r["ranker"]): r for r in rows}
    lines += ["", "## Test (151 gold mentions, English query, frozen candidates)", "",
              "| set | k | variant | none acc | reranked acc | top5 | vs none p |", "|---|---|---|---|---|---|---|"]
    for set_name in sets:
        for k in args.k:
            b = acc_.get((set_name, k, "none"))
            for v in variants:
                r = acc_.get((set_name, k, name + v))
                if r:
                    summary["test"].append({"set": set_name, "k": k, "variant": v or "model", "none_acc": b and b["acc"],
                                            "acc": r["acc"], "top5": r["top5"], "p": r["vs_none_p"]})
                    lines.append(f"| {set_name} | {k} | {v or 'model'} | {b['acc'] if b else float('nan'):.3f} | "
                                 f"{r['acc']:.3f} | {r['top5']:.3f} | {r['vs_none_p']:.3f} |")
    rep = out / f"snopi_{name}.md"
    rep.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / f"snopi_{name}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("\n".join(lines[-(len(summary["test"]) + 4):]))
    print(f"wrote {rep}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--profile", choices=["small", "full"], default="small")
    p.add_argument("--n-train", type=int, default=4000, help="small profile only")
    p.add_argument("--n-dev", type=int, default=400)
    p.add_argument("--pool", type=int, default=3000, help="concepts sampled per domain (small profile; dev pool)")
    p.add_argument("--queries-per-concept", type=int, default=2, help="full profile: synonyms used per concept")
    p.add_argument("--data", default="", help="suffix for train/dev files, e.g. _full")
    p.add_argument("--query-mode", choices=["overlap", "mixed"], default="overlap")
    p.add_argument("--children", action="store_true", help="children of the gold as hard negatives")
    p.add_argument("--negation", action="store_true", help="'No X' / 'X absent' concepts as hard negatives")
    p.add_argument("--product", action="store_true", help="substance <-> 'Product containing X' hard negatives")
    p.add_argument("--work", default="DATA/snopi")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=13)
    p = sub.add_parser("train")
    p.add_argument("--init", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    p.add_argument("--out", default="DATA/snopi/model-sapbert")
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--data", default="")
    p.add_argument("--work", default="DATA/snopi")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--bs", type=int, default=8, help="queries per micro-batch (x options pairs)")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--p-exclude", type=float, default=0.0, help="drop the query string from the gold's synonyms")
    p.add_argument("--none", action="store_true", help="add a NONE option (abstain)")
    p.add_argument("--p-none", type=float, default=0.15, help="with --none: share of queries whose gold is removed")
    p.add_argument("--resume", action="store_true", help="continue from <out>/checkpoint")
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=13)
    p = sub.add_parser("eval")
    p.add_argument("--model", default="DATA/snopi/model-sapbert")
    p.add_argument("--name", default="snopi-sapbert")
    p.add_argument("--k", type=int, nargs="+", default=[10, 20])
    p.add_argument("--data", default="")
    p.add_argument("--work", default="DATA/snopi")
    p.add_argument("--sets", nargs="*", default=None, help="frozen set names (default: all)")
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=12)
    args = ap.parse_args(argv)
    {"prep": cmd_prep, "train": cmd_train, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
