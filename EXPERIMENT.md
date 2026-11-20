# Rerunning the benchmark on a GPU machine

This guide reruns the whole retrieval and ranking study from scratch on a new computer with an NVIDIA GPU. It also covers training SNO-PI on the whole of SNOMED CT. One command runs every stage and writes a single report:

```bash
python -m Benchmark.run_all
```

The result is `Benchmark/results_gpu/REPORT.md`. The earlier CPU results in `Benchmark/results/` are not touched, so the two runs can be compared.

---

## 1. What the pipeline does

| # | stage | what it does | output (under `Benchmark/results_gpu/`) | rough time on GPU* |
|---|---|---|---|---|
| 0 | `check` | Records hardware and versions; checks `.env`, the RF2 folder, Docker and free disk | `env.json` | seconds |
| 1 | `corpus` | Turns SNOMED CT RF2 into 917,488 descriptions, with SapBERT dense vectors and BM25 sparse vectors | `DATA/corpus_cache/*.parquet`, `full/indexing/corpus.json` | 20–40 min |
| 2 | `index` | Loads Neo4j, Postgres, Qdrant, Weaviate and Meilisearch, one container at a time | `full/indexing/*.json` | 3–6 h (Meilisearch is the slowest) |
| 3 | `queries` | Builds the 151 gold-span queries (gold span + English name from `Benchmark/data/canonical_terms_minimal.tsv`) | `gold/extractions_gold.json` | seconds |
| 4 | `sweep` | Runs 119 retrieval configs across the 5 backends (k, hybrid weight, fusion, reranker…) | `gold/metrics`, `gold/predictions`, `gold/candidates` | 1–2 h |
| 5 | `latency` | Measures warm-cache retrieval latency per backend | `gold/latency_warm/*.json` | 20 min |
| 6 | `sweep_report` | Scores the sweep (strict/loose/hierarchy F1, top-k, candidate recall) and draws Pareto plots | `gold/results.{csv,md}`, `gold/*.png` | 5 min |
| 7 | `freeze` | Saves the top-50 candidates per mention for the Neo4j, Qdrant and Meilisearch dense/hybrid configs | `rank/frozen/*.json` | 10 min |
| 8 | `rerank` | Cross-encoders (bge-large, bge-v2-m3, jina-v2-ml, gte-ml, qwen3-0.6b), English and Thai-context queries, k = 20 and 50 | `rank/runs/*.json`, `rank/results.{csv,md}` | 30–60 min |
| 9 | `laya` | Laya multilingual and vega as zero-shot reranker, selector and validator (separate env, optional) | `rank/laya_validation.md`, `rank/vega_validation.md` | 30–90 min |
| 10 | `llm` | The paper's LLM selection over the top-5 (hosted API, optional, **public data only**) | `rank/runs/*__sel-*.json` | 10 min |
| 11 | `snopi` | SNO-PI listwise reranker: `small` (copy of the CPU v2 run) and `full` (whole SNOMED CT) | `rank/snopi_*.{md,json}`, models in `DATA/snopi/` | `small` 10 min, `full` 3–6 h |
| 12 | `report` | Collects everything into one report with figures | `REPORT.md`, `figures/*.png` | seconds |

\* These times are estimates for one 24 GB GPU, 32 GB RAM and an SSD. Indexing is limited by RAM and disk, not by the GPU.

Every stage writes its log to `Benchmark/results_gpu/logs/<stage>.log` and records its status in `pipeline_state.json`. If a stage fails, fix the cause and **run the same command again**: finished stages and finished sub-steps are skipped.

---

## 2. Hardware

| | minimum | comfortable |
|---|---|---|
| GPU | NVIDIA, 12 GB VRAM | 24 GB (RTX 3090/4090, A5000, L4…) |
| RAM | 16 GB | 32 GB or more. Docker's databases use 6–8 GB while indexing |
| Disk | 60 GB free (SSD) | 100 GB |
| OS | Windows 11 with Docker Desktop (WSL2), or Linux with Docker Engine | |

Without a GPU everything still runs, but slowly (`--device cpu --allow-cpu`). On CPU the full SNO-PI training would take days.

---

## 3. Move the project

Copy these to the new machine:

1. **The repository with all uncommitted work.** `Benchmark/`, `Database/build_corpus.py`, `Database/loaders/`, `Src/stores/`, `Src/domain_mapping.py`, `docker-compose.yml`, `requirement-gpu.txt`, `requirement-laya.txt`, `EXPERIMENT.md` and the modified `Src/`, `Evaluate/` and `Database/` files are not committed yet. Either commit them to a branch and clone that, or copy the whole `LLM4SNOMED_V2` folder (zip it without `DATA/`, `Benchmark/results*/` and `__pycache__/`).
2. **The SNOMED CT RF2 release folder**, e.g. `SnomedCT_InternationalRF2_PRODUCTION_20250301T120000Z/`. The code expects the 20250301 file names.
3. **Your `.env` file**, which you then edit (step 5).

You don't need to copy `DATA/` or `Benchmark/results/`: the pipeline rebuilds everything. If you want to save the 20–40 min corpus step, you can copy `DATA/corpus_cache/` and change `source_path` in `snomed_20250301_sapbert_bm25.parquet.manifest.json` to the new RF2 path. A fresh build is cleaner, though.

> **Data rule.** Only `Minimal_Dataset.json` (public) goes into this pipeline. The restricted clinical dataset must not be sent to hosted APIs (the `llm` stage) without ethics approval.

---

## 4. Install

From the repository root. These examples use Miniconda; any Python 3.12 environment works.

```bash
# main environment
conda create -n thesis python=3.12 -y
conda activate thesis

# 1) CUDA build of PyTorch first. cu128 works for RTX 20xx–50xx with a recent driver;
#    check https://pytorch.org/get-started/locally/ if your driver is older.
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128

# 2) everything else (torch is not listed, so the CUDA build stays)
pip install -r requirement-gpu.txt

# 3) confirm the GPU is visible
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Optional Laya environment. Laya needs transformers 5, which conflicts with the main env:

```bash
conda create -n laya python=3.12 -y
conda activate laya
pip install torch --index-url https://download.pytorch.org/whl/cu128   # pick the index for your driver
pip install -r requirement-laya.txt
python -c "import laya, torch; print(torch.cuda.is_available())"
# note this env's python path for --laya-python, e.g.
#   Windows: C:\Users\<you>\miniconda3\envs\laya\python.exe
#   Linux:   ~/miniconda3/envs/laya/bin/python
conda activate thesis
```

Hugging Face models (about 10 GB in total) download on first use into `~/.cache/huggingface`. Set `HF_HOME` to use another disk.

**Windows:** set `PYTHONIOENCODING=utf-8` in the terminal (`$env:PYTHONIOENCODING="utf-8"` in PowerShell). Thai text crashes the default cp1252 console otherwise. `run_all` sets it for its own subprocesses.

---

## 5. Configure `.env` (repository root)

```ini
NEO4J_HOST=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=change_this_password       # docker compose uses this as the container password
NEO4J_DATABASE=neo4j
SNOMED_RELEASE_PATH=/path/to/SnomedCT_InternationalRF2_PRODUCTION_20250301T120000Z
TEST_DATA_PATH=Minimal_Dataset.json
# optional, only for the llm stage (public data only):
OPENAI_API_KEY=...
GEMINI_API_KEY=...
LMSTUDIO_URL=http://localhost:1234
LMSTUDIO_API=lm-studio
```

---

## 6. Docker

```bash
docker compose pull          # neo4j 5.26, pgvector pg16, qdrant 1.20.0, weaviate 1.27.0, meilisearch 1.13.3
```

You don't start the containers yourself: the pipeline starts **one database at a time** and stops the others. This keeps it inside Docker Desktop's default 8 GB VM.

- **Windows / Docker Desktop:** turn off *Settings → Resources → Advanced → Resource Saver*. While the corpus step runs (all containers stopped), it paused the VM and broke the earlier run. The pipeline refuses to start while it is on, unless you pass `--allow-resource-saver`.
- If the machine has 32 GB or more and you raise Docker's memory limit, `--no-isolate` keeps containers running between stages.
- If you already had a stack from an earlier attempt, `docker compose down -v` wipes the old volumes. Only do this if nothing else uses them.

---

## 7. Run

```bash
conda activate thesis
cd LLM4SNOMED_V2

# see every command without running anything
python -m Benchmark.run_all --dry-run --ignore-check

# the full experiment (Laya and LLM stages are skipped unless configured)
python -m Benchmark.run_all

# with the optional stages
python -m Benchmark.run_all \
    --laya-python ~/miniconda3/envs/laya/bin/python \
    --llm openai:gpt-4.1
```

It takes most of a day, so run it where it survives a closed terminal:

- **Linux:** `nohup python -m Benchmark.run_all > run_all.out 2>&1 &`, or inside `tmux`.
- **Windows:** use a separate PowerShell window that you leave open. `Start-Process` with redirection also works.

Useful options:

| option | meaning |
|---|---|
| `--stages snopi report` | run only these stages |
| `--from rerank` | run this stage and every later one |
| `--force snopi` | redo a stage already marked done (finished runs inside it are still reused) |
| `--backends neo4j qdrant meilisearch` | skip Postgres and Weaviate (saves about 1 h of indexing) |
| `--rerankers bge-large bge-v2-m3` | fewer cross-encoders |
| `--snopi-profiles full` | only the whole-SNOMED SNO-PI run |
| `--laya-models multilingual` | skip vega |
| `--results Benchmark/results_gpu2` | another results root (a clean repeat) |
| `--device cpu --allow-cpu` | no-GPU fallback |

**Progress.** Each stage prints to the console and to `Benchmark/results_gpu/logs/<stage>.log`. Stage status and minutes are in `pipeline_state.json`. SNO-PI training prints loss, seconds per step and an ETA every 200 steps. Its per-step curve is in `DATA/snopi/model-gpu-full/train_log.jsonl`.

---

## 8. The SNO-PI workflow (whole SNOMED CT)

SNO-PI (arXiv:2610.26425), stage 1: a cross-encoder learns to pick the right SNOMED CT concept from a list. It is trained only on SNOMED CT: a synonym is the query and the concept is the answer. No annotated mentions are used, and the 151 test concepts are excluded from both training and dev.

### What the `full` profile does

```bash
# 1. prep: build queries + option lists from the whole clinical corpus (GPU kNN)
python -m Benchmark.snopi_study prep --profile full --n-dev 2000 --queries-per-concept 2 \
    --query-mode mixed --children --negation --product --data _full

# 2. train: listwise cross-entropy, 3 epochs, NONE option, checkpoint + dev eval every epoch
python -m Benchmark.snopi_study train --data _full --epochs 3 --bs 32 --lr 2e-5 \
    --none --p-none 0.15 --resume --out DATA/snopi/model-gpu-full

# 3. eval: dev accuracy + rerank every frozen candidate set at k = 10, 20, 50
python -m Benchmark.snopi_study eval --data _full --model DATA/snopi/model-gpu-full \
    --name snopi-full --k 10 20 50
```

`run_all` runs exactly these commands. The parts:

- **Queries.** Every concept in the 8 clinical domains (findings, procedure, medication, laboratory, anatomy, event, observable, score) gives up to 2 synonym queries, about 0.5 million in total. In `mixed` mode, half of the first queries are a random synonym, often the exact preferred name. This keeps exact matches in training, because run 1 learned to ignore them.
- **Options per query** (10, plus NONE):
  - the gold concept;
  - parents and a sibling, so the model doesn't prefer the general concept;
  - `--children`: two children, so it doesn't prefer the more specific concept (run 1's main failure);
  - `--negation`: "No X" / "X absent" partners (e.g. Pain ↔ No pain (situation));
  - `--product`: substance ↔ "Product containing X" (e.g. Heparin (substance) ↔ Product containing heparin);
  - the rest are SapBERT nearest neighbours.
- **NONE option** (`--none`). In 15% of queries the gold concept is removed and "none of the above" is correct. At test time the `-none` variant keeps the retrieval order when the model picks NONE.
- **Dev set.** 2,000 held-out concepts (250 per domain), 50% of them cases where SapBERT gets it wrong. Dev never touches the gold test items. The best epoch is chosen on dev only.
- **Test variants.** Each is fixed before any test result is seen:
  - `model`: model order;
  - `-rrf`: reciprocal-rank fusion with retrieval order, k = 60;
  - `-gate`: switch the top-1 only if model probability ≥ 0.8;
  - `-none`: abstain → retrieval order.

  The test is reported with McNemar p against retrieval order on the same set and k.
- **Resume.** Every epoch saves `DATA/snopi/model-gpu-full/checkpoint-epochN/`. If training stops, rerun the same command: `--resume` continues from the last finished epoch.

### Tuning knobs (change and rerun the `snopi` stage)

| knob | where | notes |
|---|---|---|
| epochs | `--epochs` | 3 by default. Watch dev accuracy per epoch in the log |
| batch size | `--bs`, `--grad-accum` | `--bs 32` (≈350 pairs) fits in 12 GB at MAX_LEN 64. Use `--bs 64` on 24 GB |
| learning rate | `--lr` | 2e-5 |
| queries per concept | prep `--queries-per-concept` | 1 halves the data and the time |
| hard negatives | prep `--children --negation --product` | drop one flag for an ablation (use a new `--data` suffix) |
| abstain | train `--none --p-none` | 0.15 |
| base model | train `--init` | default SapBERT. Any BERT-style HF model works |

For an ablation, give the prep a new `--data` suffix and the training a new `--out` and `--name`. Every `snopi_<name>` result is then picked up by the report automatically.

### What it does not include

The paper's later stages fine-tune on **annotated clinical mentions**. Here that would mean the restricted dataset, and the 151 public mentions are the test set, so this workflow stops at stage 1. If you later want stage 2 on local restricted data, it must stay on the local GPU (no hosted APIs) and must not share documents with the test set.

---

## 9. Reading the results

Open `Benchmark/results_gpu/REPORT.md`:

1. **Summary.** The best retrieval config, the best ranking strategy, whether anything beats retrieval order significantly, and the SNO-PI results.
2. **Retrieval sweep.** The best config per backend, and warm latency.
3. **Ranking study.** The best run of every strategy with its 95% CI and McNemar p against retrieval order, plus `figures/ranking.png`.
4. **SNO-PI.** Dev and test per profile, with training curves (`figures/snopi_*_training.png`).
5. **Laya.** The validator AUROC tables.
6. **Indexing and run environment.** Timings, GPU, versions and git commit.

The full tables are `gold/results.md` (sweep) and `rank/results.md` (every ranking run). To rebuild only the report: `python -m Benchmark.run_all --stages report --force report`.

To compare with the CPU run: `python -m Benchmark.final_report --root Benchmark/results` writes the same report for the old results.

With n = 151 mentions, differences under about 5 points are within noise. Rely on the McNemar p column, not the raw ranking.

---

## 10. Troubleshooting

| symptom | fix |
|---|---|
| `environment check failed` | Read the PROBLEMS list. Fix it, or pass `--ignore-check` if it doesn't matter (e.g. low disk on a second drive) |
| `Docker Desktop Resource Saver is ON` | Turn it off (section 6), or pass `--allow-resource-saver` |
| Docker HTTP 500 / hangs mid-index | Restart Docker Desktop and rerun. Finished DB steps are skipped |
| Meilisearch load killed / stuck | Free RAM, then `python -m Benchmark.build_indexes --out Benchmark/results_gpu/full --steps meilisearch --no-monitor --resume-meili`. Never use `--drop-caches-every` with Meilisearch |
| Meilisearch accuracy near 0 | `searchCutoffMs` too low. The loader sets 600000 ms. Check with `GET /indexes/snomed_descriptions/settings` |
| Neo4j auth error | `.env` `NEO4J_PASSWORD` must match the password the `neo4j_data` volume was created with (`docker compose down -v` resets it) |
| CUDA out of memory in SNO-PI | Lower `--bs` and raise `--grad-accum` (e.g. `--bs 16 --grad-accum 2`). For prep, use a GPU with ≥12 GB or `--device cpu` (slow) |
| `UnicodeEncodeError` on Windows | `$env:PYTHONIOENCODING="utf-8"` |
| A stage failed after partial work | Rerun the same command. Each module skips finished configs/runs (use `--force <stage>` only to recompute) |

---

## 11. Map of the code

| module | role |
|---|---|
| `Benchmark/run_all.py` | the orchestrator (this guide) |
| `Benchmark/build_indexes.py` | corpus + per-DB indexing |
| `Benchmark/gold_queries.py` | gold-span queries |
| `Benchmark/harness.py`, `configs.py`, `report.py`, `latency_warm.py` | retrieval sweep and its scoring |
| `Benchmark/freeze_candidates.py` | frozen candidate lists for the offline ranking study |
| `Benchmark/rank_study.py` | cross-encoder reranking, `llm-select`, blind export/import, scoring (Wilson CI, McNemar) |
| `Benchmark/laya_study.py` | Laya / vega (separate env) |
| `Benchmark/snopi_study.py` | SNO-PI prep / train / eval |
| `Benchmark/final_report.py` | `REPORT.md` + figures |

Every module also runs alone (`python -m Benchmark.<module> --help`). Set `BENCH_RESULTS=Benchmark/results_gpu` so its default paths point at the GPU results root.
