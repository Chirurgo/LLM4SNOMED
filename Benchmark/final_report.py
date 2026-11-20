"""Assemble one human-readable report from every stage's outputs.

Reads whatever exists under the results root (missing stages are listed as "not run"):
  pipeline_state.json, env.json           Benchmark.run_all (stage timings, hardware)
  full/indexing/*.json                    Benchmark.build_indexes
  gold/results.csv, gold/latency_warm/    Benchmark.harness + Benchmark.report, Benchmark.latency_warm
  rank/results.csv                        Benchmark.rank_study (rerankers, LLM selection, Laya, SNO-PI runs)
  rank/*_validation.md                    Benchmark.laya_study validate
  rank/snopi_*.json, <model>/train_log.jsonl   Benchmark.snopi_study

Writes <root>/REPORT.md and <root>/figures/*.png.

  python -m Benchmark.final_report --root Benchmark/results_gpu
"""
import argparse
import csv
import json
from pathlib import Path

ALPHA = 0.05


def _f(v, nd=3):
    if v in (None, ""):
        return "–"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def _csv(path: Path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(r, k):
    try:
        return float(r[k])
    except (KeyError, TypeError, ValueError):
        return None


def ranker_family(r):
    rk = r["ranker"]
    if "+llm-" in rk or "+" in rk:
        return "LLM selection"
    if rk.startswith(("laya", "vega")):
        return "Laya"
    if rk.startswith("snopi"):
        return "SNO-PI"
    if rk == "none":
        return "retrieval order"
    return "cross-encoder"


# --- sections ---------------------------------------------------------------------------
def sec_env(root: Path):
    lines = ["## Run environment", ""]
    env = root / "env.json"
    if env.exists():
        e = json.loads(env.read_text(encoding="utf-8"))
        for k in ("started", "host", "os", "python", "torch", "cuda", "gpu", "gpu_memory_gb", "ram_gb",
                  "cpu_count", "git_commit", "git_dirty"):
            if k in e:
                lines.append(f"- **{k}**: {e[k]}")
        lines.append("")
    st = root / "pipeline_state.json"
    if st.exists():
        s = json.loads(st.read_text(encoding="utf-8"))
        lines += ["| stage | status | minutes | finished |", "|---|---|---|---|"]
        for name, v in s.get("stages", {}).items():
            lines.append(f"| {name} | {v.get('status')} | {_f(v.get('seconds', 0) / 60, 1)} | {v.get('finished', '')} |")
        lines.append("")
    return lines


def sec_indexing(root: Path):
    lines = ["## 1. Corpus and indexes", ""]
    d = root / "full" / "indexing"
    if not d.exists():
        return lines + ["Not run.", ""]
    lines += ["| step | minutes | counts |", "|---|---|---|"]
    for p in sorted(d.glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        lines.append(f"| {r['step']} | {_f(r.get('total_s', 0) / 60, 1)} | {r.get('counts', r.get('rows', ''))} |")
    return lines + [""]


def sec_sweep(root: Path):
    lines = ["## 2. Retrieval sweep (gold spans, English query)", "",
             "link_acc = top-1 concept accuracy; cand_recall = gold anywhere in the retrieved candidates "
             "(ceiling for any reranker); est_ms = embed + retrieval + rerank per entity.", ""]
    rows = _csv(root / "gold" / "results.csv")
    if not rows:
        return lines + ["Not run.", ""], None
    best_all = max(rows, key=lambda r: _num(r, "link_acc") or 0)
    lines += ["### Best config per backend", "",
              "| backend | config | link_acc | top5 | cand_recall | est_ms |", "|---|---|---|---|---|---|"]
    for b in sorted({r["backend"] for r in rows}):
        rs = [r for r in rows if r["backend"] == b]
        top = max(rs, key=lambda r: ((_num(r, "link_acc") or 0), -(_num(r, "est_ms_entity") or 1e9)))
        lines.append(f"| {b} | `{top['config_id']}` | {_f(top['link_acc'])} | {_f(top['top5'])} | "
                     f"{_f(top['cand_recall'])} | {_f(top['est_ms_entity'], 1)} |")
    lines.append("")
    lat = root / "gold" / "latency_warm"
    if lat.exists() and any(lat.glob("*.json")):
        lines += ["### Warm retrieval latency (ms, p50 / p95)", "",
                  "| backend | dense k20 | hybrid k20 | dense k100 |", "|---|---|---|---|"]
        for p in sorted(lat.glob("*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            cell = lambda k: (f"{_f(d[k]['retrieval_p50_ms'], 1)} / {_f(d[k]['retrieval_p95_ms'], 1)}"
                              if k in d else "–")
            lines.append(f"| {p.stem} | {cell('dense_k20')} | {cell('hybrid_k20')} | {cell('dense_k100')} |")
        lines.append("")
    lines += [f"Full table: `{(root / 'gold' / 'results.md').as_posix()}`", ""]
    return lines, best_all


def sec_ranking(root: Path, figs: Path):
    lines = ["## 3. Ranking study (offline, frozen candidates)", "",
             "Every strategy re-ranks the same frozen candidate lists. acc = top-1 accuracy (95% Wilson CI); "
             "p = exact McNemar vs retrieval order on the same set and k.", ""]
    rows = _csv(root / "rank" / "results.csv")
    if not rows:
        return lines + ["Not run.", ""], None
    for r in rows:
        r["family"] = ranker_family(r)
    none = {(r["set"], r["k"]): _num(r, "acc") for r in rows if r["ranker"] == "none"}
    best_none = max((r for r in rows if r["ranker"] == "none"), key=lambda r: _num(r, "acc"), default=None)
    # best run per (ranker, query)
    best = {}
    for r in rows:
        key = (r["ranker"], r.get("query") or "")
        if key not in best or (_num(r, "acc") or 0) > (_num(best[key], "acc") or 0):
            best[key] = r
    ordered = sorted(best.values(), key=lambda r: -(_num(r, "acc") or 0))
    lines += ["### Best run per strategy", "",
              "| strategy | family | query | set | k | acc [95% CI] | none acc | top5 | p vs none | ms/entity |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in ordered:
        lines.append(f"| {r['ranker']} | {r['family']} | {r.get('query') or '–'} | {r['set']} | {r['k']} | "
                     f"{_f(r['acc'])} [{_f(r['acc_l95'])}–{_f(r['acc_u95'])}] | {_f(none.get((r['set'], r['k'])))} | "
                     f"{_f(r['top5'])} | {_f(r.get('vs_none_p'))} | {_f(r.get('ms_per_entity'), 1)} |")
    lines.append("")
    better = [r for r in rows if r["ranker"] != "none" and _num(r, "vs_none_p") is not None
              and _num(r, "vs_none_p") < ALPHA and (_num(r, "acc") or 0) > (none.get((r["set"], r["k"])) or 1)]
    verdict = (f"{len(better)} run(s) beat retrieval order significantly (p < {ALPHA}): "
               + ", ".join(f"`{r['run_id']}` ({_f(r['acc'])})" for r in better[:5])) if better else \
        f"No strategy beats retrieval order significantly (p < {ALPHA}) on any set and depth."
    lines += [f"**{verdict}**", ""]
    try:
        _plot_ranking(ordered, best_none, figs / "ranking.png")
        lines += ["![Best run per strategy](figures/ranking.png)", ""]
    except Exception as e:  # plotting is optional
        lines += [f"(figure skipped: {e})", ""]
    lines += [f"Full table: `{(root / 'rank' / 'results.md').as_posix()}`", ""]
    return lines, {"best_none": best_none, "better": better, "best": ordered[0] if ordered else None}


def _plot_ranking(ordered, best_none, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rs = ordered[:25][::-1]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(rs) + 1.2))
    colors = {"retrieval order": "#1f6feb", "cross-encoder": "#6e7781", "SNO-PI": "#bf3989",
              "Laya": "#9a6700", "LLM selection": "#1a7f37"}
    for y, r in enumerate(rs):
        a, lo, hi = _num(r, "acc"), _num(r, "acc_l95"), _num(r, "acc_u95")
        c = colors.get(r["family"], "#6e7781")
        ax.plot([lo, hi], [y, y], color=c, lw=2, alpha=.6)
        ax.plot([a], [y], "o", color=c)
    ax.set_yticks(range(len(rs)))
    ax.set_yticklabels([f"{r['ranker']} ({r.get('query') or '-'}, k{r['k']})" for r in rs], fontsize=8)
    if best_none:
        ax.axvline(_num(best_none, "acc"), color="#1f6feb", ls="--", lw=1)
    ax.set_xlabel("top-1 accuracy (95% CI)")
    ax.set_xlim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sec_laya(root: Path):
    lines = ["## 4. Laya (zero-shot reranker / validator)", ""]
    found = False
    for name in ("laya_validation.md", "vega_validation.md"):
        p = root / "rank" / name
        if p.exists():
            found = True
            body = p.read_text(encoding="utf-8").splitlines()
            lines += [f"### {name}", ""] + [("#" + l if l.startswith("#") else l) for l in body] + [""]
    return lines if found else lines + ["Not run.", ""]


def sec_snopi(root: Path, figs: Path):
    lines = ["## 5. SNO-PI listwise reranker", "",
             "Trained on SNOMED CT only (no annotated mentions). Dev = held-out SNOMED synonym queries; "
             "test = the 151 gold mentions (English query) over frozen candidates. Variants: model order, "
             "`-rrf` (fusion with retrieval rank), `-gate` (switch top-1 only if p >= 0.8), "
             "`-none` (abstain -> retrieval order).", ""]
    files = sorted((root / "rank").glob("snopi_*.json"))
    if not files:
        return lines + ["Not run.", ""], []
    summaries = []
    for p in files:
        s = json.loads(p.read_text(encoding="utf-8"))
        summaries.append(s)
        tm, dv = s.get("train_meta", {}), s["dev"]
        lines += [f"### {s['name']}", "",
                  f"- training: {tm.get('queries', '?')} queries, {tm.get('epochs', '?')} epoch(s), "
                  f"best dev epoch {tm.get('best', {}).get('epoch')}, NONE option {tm.get('none_option', False)}, "
                  f"{_f(tm.get('seconds', 0) / 60, 0)} min on {tm.get('gpu', tm.get('device', '?'))}",
                  f"- dev (n={dv['n']}): SapBERT top-1 {_f(dv['sapbert_top1'])} -> reranked {_f(dv['reranked_top1'])}"
                  + (f", NONE picked {_f(dv['none_rate'])}" if dv.get("none_rate") is not None else ""), "",
                  "| set | k | variant | none acc | acc | top5 | p |", "|---|---|---|---|---|---|---|"]
        for t in s["test"]:
            lines.append(f"| {t['set']} | {t['k']} | {t['variant']} | {_f(t['none_acc'])} | {_f(t['acc'])} | "
                         f"{_f(t['top5'])} | {_f(t['p'])} |")
        lines.append("")
        log = Path(s["model"]) / "train_log.jsonl"
        if log.exists():
            try:
                fn = f"snopi_{s['name']}_training.png"
                _plot_training(log, figs / fn, s["name"])
                lines += [f"![training curve](figures/{fn})", ""]
            except Exception as e:
                lines += [f"(figure skipped: {e})", ""]
    return lines, summaries


def _plot_training(log: Path, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps, loss, ep, dev = [], [], [], []
    for line in log.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if "loss" in r:
            steps.append(r["step"]); loss.append(r["loss"])
        if "epoch_end" in r:
            ep.append(r["epoch_end"]); dev.append(r["dev_acc"])
    fig, axs = plt.subplots(1, 2, figsize=(9, 3))
    axs[0].plot(steps, loss, lw=1, color="#bf3989")
    axs[0].set_xlabel("optimizer step"); axs[0].set_ylabel("train loss")
    axs[1].plot(ep, dev, "o-", color="#1f6feb")
    axs[1].set_xlabel("epoch"); axs[1].set_ylabel("dev top-1"); axs[1].set_ylim(0, 1)
    for a in axs:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summary_bullets(best_sweep, rank, snopi):
    out = ["## Summary", ""]
    if best_sweep:
        out.append(f"- Best retrieval config: `{best_sweep['config_id']}` with link accuracy "
                   f"{_f(best_sweep['link_acc'])} (candidate recall {_f(best_sweep['cand_recall'])}).")
    if rank and rank["best_none"]:
        bn = rank["best_none"]
        out.append(f"- Best retrieval order in the ranking study: `{bn['set']}` k{bn['k']}, top-1 {_f(bn['acc'])}, "
                   f"top-5 {_f(bn['top5'])}, gold among candidates {_f(bn['cand_recall'])}.")
        b = rank["best"]
        if b:
            out.append(f"- Highest top-1 of any strategy: `{b['ranker']}` ({b['family']}) {_f(b['acc'])}"
                       + (f", p = {_f(b.get('vs_none_p'))} vs retrieval order." if b["ranker"] != "none" else "."))
        out.append(f"- {len(rank['better'])} run(s) significantly better than retrieval order."
                   if rank["better"] else "- No strategy significantly beats retrieval order.")
    for s in snopi:
        if s["test"]:
            t = max(s["test"], key=lambda x: x["acc"])
            out.append(f"- SNO-PI `{s['name']}`: dev {_f(s['dev']['sapbert_top1'])} -> {_f(s['dev']['reranked_top1'])}; "
                       f"best test variant {t['variant']} k{t['k']} {_f(t['acc'])} vs {_f(t['none_acc'])} "
                       f"(p = {_f(t['p'])}).")
    out += ["- n = 151 gold mentions (public Minimal_Dataset): differences under ~5 points are within noise.", ""]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="Benchmark/results_gpu")
    args = ap.parse_args(argv)
    root = Path(args.root)
    figs = root / "figures"
    env = sec_env(root)
    idx = sec_indexing(root)
    sweep, best_sweep = sec_sweep(root)
    rank, rinfo = sec_ranking(root, figs)
    laya = sec_laya(root)
    snopi, sums = sec_snopi(root, figs)
    lines = ["# LLM4SNOMED retrieval and ranking benchmark: results", ""]
    lines += summary_bullets(best_sweep, rinfo, sums) + sweep + rank + snopi + laya + idx + env
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {root / 'REPORT.md'}")


if __name__ == "__main__":
    main()
