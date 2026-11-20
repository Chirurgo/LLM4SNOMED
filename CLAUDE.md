# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Research code for *A framework for structuring Thai medical documents using SNOMED CT and large language models* (PLOS Digital Health, 2026). The pipeline uses LLMs to extract clinical entities from Thai / mixed Thai–English medical text, locates each mention's character span in the source text, and links it to a SNOMED CT concept via embedding retrieval + cross-encoder reranking (+ optional LLM selection). This is a research proof-of-concept, not a clinical-grade system.

## Setup and running

There is no package manager config (no `pyproject.toml`/`setup.py`) — dependencies are a flat pinned list in `requirement.txt` (singular filename, not `requirements.txt`).

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
python -m pip install -r requirement.txt
python -m pip install jupyterlab ipykernel
```

Run all commands from the repository root — imports like `Src.run` and `Database.transform` depend on it.

External dependencies that must be running/available before most notebooks work:
- **Neo4j** (5.26+, needs vector-index support) at the URI in `.env` (`NEO4J_HOST`), database name `neo4j`, vector index `name_embedding` on `Description` nodes.
- A **licensed SNOMED CT RF2 release** (International Edition snapshot, filenames hardcoded to date `20250301` in `Database/transform.py` — change there if using a different release).
- Hugging Face models: `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` (embeddings) and `BAAI/bge-reranker-large` (cross-encoder reranker) by default.
- An LLM: hosted (Gemini via `GEMINI_API_KEY`, OpenAI via `OPENAI_API_KEY`) or local via LM Studio's OpenAI-compatible server (`LMSTUDIO_URL`, default `http://localhost:1234`).

Required `.env` keys (repository root): `NEO4J_HOST`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `LMSTUDIO_URL`, `LMSTUDIO_API`, `TEST_DATA_PATH`, `SNOMED_RELEASE_PATH`. See README.md "Configure `.env`" section for exact semantics and gotchas (e.g. `NEO4J_HOST` protocol vs `NEO4J_DATABASE` name are independent; code explicitly selects `database="neo4j"`).

### Workflow order (notebooks are the entry points, no CLI/test suite)

1. `01_create_database.ipynb` — loads SNOMED CT RF2 into Neo4j via `Database/transform.py`, builds `Description` embeddings + vector index. **Destructive**: calls `delete_all_in_batches(...)` and sets `pre_delete_collection=True` — use a dedicated Neo4j database, and re-running can wipe existing data.
2. `02_main.ipynb` — configures models, runs batch extraction + linking over a test set via `Src/run.py` (`run_batch()` or `run_batch_top1()`).
3. `03_evaluation.ipynb` — scores NER and concept-linking predictions against gold annotations using `Evaluate/`.
4. `04_error_analysis.py` — local Flask app (`python 04_error_analysis.py`, serves at `127.0.0.1:5000`) for visually comparing gold vs. predicted entities; expects `DATA/predictions/*.json` and reads `TEST_DATA_PATH` from `.env`.
5. `00_evaluate_encoder.ipynb` / `05_dataset_stats.ipynb` — auxiliary, not required for the main pipeline (encoder comparison and corpus descriptive stats).

No automated test suite exists; validation happens via the evaluation notebook/metrics against the gold corpus.

## Architecture

The pipeline is a 5-stage sequential process per document, implemented across `Src/`:

1. **Entity extraction** (`Src/chain.py`) — a LangChain prompt + `JsonOutputParser` over a Pydantic `Output`/`Entity` schema asks the LLM to extract, per entity: `term` (verbatim substring), `formal_term` (normalized English), `domain` (one of 8 fixed categories: findings/medication/procedure/laboratory/observable/anatomy/event/score), `status` (present/absent/past_history/family_history/no_family_history/hypothetical/planned), and free-form `attributes`. The prompt is few-shot with full worked Thai clinical examples embedded in the template.
2. **Span positioning** (`Src/positioning.py`) — `position_entities()` locates each extracted `term` back in the original text in two passes: exact regex matches first (longest terms prioritized, non-overlapping), then a fuzzy fallback (`rapidfuzz.WRatio` over sliding token windows, boosted by matching attribute keywords extracted via `pythainlp` tokenization). Produces `start_index`/`end_index` plus a `match_status` (`SUCCESS_EXACT`/`SUCCESS_FUZZY`/`NOT_FOUND`/`FAILED_OVERLAP`/`NO_TERM`).
3. **Candidate retrieval** (`Src/retrieval.py`) — `search_top_k_names()` vector-searches `Description` nodes in Neo4j (via `Neo4jVector`) for the `formal_term`, then filters candidates by mapping the entity's `domain` to allowed SNOMED semantic tags (`TYPE_TO_SEMANTIC_TAG`) looked up per-concept via a blocking Cypher query. Falls back to unfiltered top-k if too few results survive the filter.
4. **Candidate ranking** (`Src/link.py`) — `rerank_candidates()` uses a cross-encoder (e.g. `bge-reranker-large`) to score (query, description) pairs, dedupes by `concept_id` keeping the best-scoring description per concept.
5. **Concept selection** (`Src/link.py`) — two modes:
   - `link_topone()` — picks the reranker's #1 candidate directly (no extra LLM call).
   - `link_rerank_entity()` — presents the top 5 reranked candidates to an LLM in a structured prompt and asks it to choose one (`Concept` Pydantic model with `choice: Literal[1..5]` + `mapping_reason`), falling back to choice 1 if the LLM response fails to parse.

`Src/run.py` orchestrates the whole thing: `predict_entities()`/`predict_top1()` run stages 1–5 for one document; `run_batch()`/`run_batch_top1()` loop over a JSON test set, write one prediction list per input record (same order) to an output JSON, and log per-document errors without aborting the whole batch. Batch loops include fixed `time.sleep()` calls between documents/entities to respect LLM rate limits.

**Database layer** (`Database/`): `transform.py` reads SNOMED CT RF2 snapshot files with Polars (lazy frames, active rows only, filtered to clinically relevant semantic tags), reshapes concepts/descriptions/relationships, and writes them to Neo4j as `Concept` nodes, `Description` nodes (with embeddings, linked via `HAS_NAME`), and typed relationships (names dynamically derived from SNOMED relationship type concepts, sanitized via `normalize_rel_type()`). `embed_models.py` wraps several Hugging Face encoders (SapBERT variants, PubMedBERT, GatorTron, BioClinicalBERT, GTE-multilingual) behind a common `Embeddings` interface (sync + async), with CLS or mean pooling and L2 normalization.

**Evaluation layer** (`Evaluate/`): `evaluate_domain.py`'s `DomainEvaluator` scores NER (span + domain label) strict/loose with Wald-interval CIs. `evaluate_concept.py`'s `Evaluator` scores concept linking — strict (exact span + concept_id), loose (overlap + exact concept), hierarchy-aware (overlap + concept_proximity via `IS_A`/`HAS_ACTIVE_INGREDIENT` neighbors fetched from Neo4j), and character-level IoU. `topk.py` measures top-k retrieval accuracy conditioned on exact span matches. Note the inconsistent label naming between gold (`label`, includes `"observable entity"`) and model output (`domain`, uses `"observable"`) — `DomainEvaluator` lowercases but does not remap these; reconcile manually if comparing results.

## Data

`Minimal_Dataset.json` is the only data file in the repo — a small public sample (10 records, 151 annotations) showing the annotation schema: `id`, `text`, `pdx`, `clinic`, `type` (`cc`/`pi`), and `entity` list with `start_index`/`end_index`/`substring`/`label`/`Target code`/`Target display`/`Relationship type code`/`No map flag`/`Status`. The full clinical dataset is restricted (institutional ethics oversight) and is not in this repo; `.gitignore` excludes `data/`, `DATA/`, `outputs/`, `artifacts/`, `models/`, `checkpoints/`, `embeddings/`, and common model/weight file extensions. Preserve exact text and character offsets — whitespace/Unicode normalization changes will invalidate spans.

## Key gotchas carried over from the README

- `01_create_database.ipynb` imports the embedder class but does not instantiate `embedding_model` — must be added manually (e.g. `AsyncSapBERTEmbedder(device="cpu")`) before running, and must match the vector dimensions used at query time.
- `02_main.ipynb`'s local-LLM cells use a hardcoded LM Studio URL/key rather than the `.env` values by default; swap in `neo4j_dict["lmstudio_uri"]`/`["lmstudio_api"]` to use configured values.
- Gemini/GPT model identifiers in the notebooks are snapshots from the original study and may be deprecated — check provider model catalogs before running.
- Evaluation routines match gold/prediction arrays by position; mismatched lengths are truncated to the shorter one with a warning, not an error — verify array alignment before trusting metrics.
