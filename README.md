# LLM4SNOMED: Structuring Thai medical documents with SNOMED CT and large language models

[![Article DOI](https://img.shields.io/badge/DOI-10.1371%2Fjournal.pdig.0001791-blue)](https://doi.org/10.1371/journal.pdig.0001791)
[![License: MIT](https://img.shields.io/badge/Code%20license-MIT-green.svg)](LICENSE)

This repository contains the research code for the article below. The code uses large language models (LLMs). It identifies clinical entities in Thai and mixed Thai–English text. It locates each mention in the source text and links the mention to a SNOMED CT concept.

**Publication:** Tangchitnob N, Ponthongmak W, Kijsanayotin B, et al. (2026). *PLOS Digital Health*, 5(10), e0001791. [Read the article](https://doi.org/10.1371/journal.pdig.0001791).

**Article title:** A framework for structuring Thai medical documents using systematized nomenclature of medicine clinical terms and large language models.

## Study overview

The study evaluated Gemini 2.5 Pro, GPT-4.1, Qwen 3, and MedGemma. Experts annotated 220 Thai medical documents with 6,390 entities. The study compared named entity recognition (NER), NER with concept linking (NER+L), and resource requirements.

Read the article for the study design, dataset split, model settings, results, and limitations.

The implementation supports two concept-selection modes:

| Stage | Implementation |
| --- | --- |
| Entity extraction | An LLM extracts the mention, a normalized English term, clinical domain, status, and attributes. |
| Span positioning | Exact matching and a fuzzy-matching fallback locate extracted mentions in the original text. |
| Candidate retrieval | SapBERT embeddings query descriptions in a Neo4j vector index. The search filters results by semantic tag and uses a fallback when necessary. |
| Candidate ranking | A cross-encoder ranks descriptions. The code retains one candidate per concept identifier. |
| Top-1 selection | `run_batch_top1()` selects the first reranked candidate. The extraction stage still uses an LLM. |
| LLM selection | `run_batch()` asks an LLM to select among the first five reranked candidates. |

This is a proof-of-concept research implementation. Clinical use requires further validation, governance, and human review.

## Repository guide

| File or directory | Purpose |
| --- | --- |
| [Minimal_Dataset.json](Minimal_Dataset.json) | This file contains the public sample of annotated clinical text. See the schema and scope below. |
| [00_evaluate_encoder.ipynb](00_evaluate_encoder.ipynb) | This notebook compares embedding models and retrieval accuracy. It requires additional prediction files. The main pipeline does not require this notebook. |
| [01_create_database.ipynb](01_create_database.ipynb) | Imports SNOMED CT into Neo4j and builds description embeddings and the vector index. |
| [02_main.ipynb](02_main.ipynb) | Configures models and runs batch extraction and linking. |
| [03_evaluation.ipynb](03_evaluation.ipynb) | Evaluates NER and concept linking against the annotated reference data. |
| [04_error_analysis.py](04_error_analysis.py) | Local Flask interface for comparing reference annotations with model predictions. |
| [05_dataset_stats.ipynb](05_dataset_stats.ipynb) | Descriptive statistics and figures for the annotated corpus. |
| [Database/transform.py](Database/transform.py) | Loads SNOMED CT RF2 files and writes concepts, descriptions, relationships, and embeddings to Neo4j. |
| [Database/embed_models.py](Database/embed_models.py) | Hugging Face embedding wrappers, including SapBERT and alternative encoders. |
| [Src/chain.py](Src/chain.py) | Extraction prompts and structured output definitions. |
| [Src/positioning.py](Src/positioning.py) | Character-span positioning and correction. |
| [Src/retrieval.py](Src/retrieval.py) | Vector retrieval and domain-to-semantic-tag filtering. |
| [Src/link.py](Src/link.py) | Cross-encoder reranking, candidate deduplication, and concept selection. |
| [Src/run.py](Src/run.py) | Single-document and batch pipeline functions. |
| [Evaluate/](Evaluate/) | NER, concept-linking, and top-k metrics, plus the comparison-page template. |
| [requirement.txt](requirement.txt) | Recorded Python dependency versions. The filename is singular. |
| [LICENSE](LICENSE) | MIT license for the software. |

## Public sample: `Minimal_Dataset.json`

The public file contains **10 text records**, **five distinct `id` values**, and **151 entity annotations**. Each ID has one chief complaint (`cc`) record and one present illness (`pi`) record. All sample records have clinic code `IN`.

Use the exact filename `Minimal_Dataset.json`. Some operating systems treat uppercase and lowercase letters as different characters in filenames.

The sample shows the annotation format. You can use it for small demonstrations after you configure the code. Its size and clinic composition do not support estimates of performance for the full study. The sample does not contain the complete evaluation corpus or all data necessary to reproduce the published results.

The file is a UTF-8 JSON array. Each array element contains:

| Field | Description |
| --- | --- |
| `id` | This field identifies the source record. Use `id` and `type` together to identify each text record. |
| `text` | Original clinical text, including mixed Thai and English expressions. |
| `pdx` | Primary diagnosis code recorded with the source record. |
| `clinic` | Source clinic code. |
| `type` | Text section: `cc` or `pi` in the released sample. |
| `entity` | List of reference entity annotations. |

Each entity annotation contains:

| Field | Description |
| --- | --- |
| `start_index`, `end_index` | These fields contain character offsets that start at zero. The end offset is exclusive, as in Python slicing. |
| `substring` | Annotated source mention: `text[start_index:end_index]`. |
| `label` | Reference clinical domain/category. |
| `Target code` | Mapped SNOMED CT identifier, stored as a string. |
| `Target display` | This field contains text for the mapped concept. |
| `Relationship type code` | Mapping relationship recorded during annotation. |
| `No map flag` | This flag identifies an annotation with no assigned mapping. |
| `Status` | This field contains the annotation workflow status. It differs from the model's clinical `status` field. |

Spaces and uppercase and lowercase letters are significant in JSON keys. Preserve the original text and character offsets. Changes to whitespace or Unicode normalization can invalidate the spans.

To inspect the sample without downloading models or connecting to a database:

```python
import json
from pathlib import Path

records = json.loads(Path("Minimal_Dataset.json").read_text(encoding="utf-8"))
print("Text records:", len(records))                           # 10
print("Distinct IDs:", len({r["id"] for r in records}))        # 5
print("Annotations:", sum(len(r["entity"]) for r in records))  # 151
```

## Access to the full dataset

The Department of Clinical Epidemiology and Biostatistics holds the complete clinical dataset under institutional ethics oversight. The department belongs to the Faculty of Medicine Ramathibodi Hospital, Mahidol University. Restrictions on public release protect information from patient records.

Researchers should submit data-access requests through the department’s **[website](https://www.rama.mahidol.ac.th/ceb/resource/codes/code4)**.

Include these details in your request:
- Your name and affiliation.
- The research protocol and intended use.
- The data you request.
- Ethics approval documents.
- Your plan for secure storage and access.

Read the [Data Availability Statement in the article](https://doi.org/10.1371/journal.pdig.0001791) for the formal access conditions.

## Setup

### 1. Install and start Neo4j

**Install the Neo4j database server before you run the database import or pipeline notebooks. Start the server before you run these notebooks.** The `neo4j` Python package contains only the client driver. Choose a local installation or Docker.

#### Option A: Local installation with Neo4j Desktop

1. Download **Neo4j Desktop** from the [Neo4j Deployment Center](https://neo4j.com/deployment-center/). Select the installer for your operating system. Follow the [Desktop installation instructions](https://neo4j.com/docs/desktop/current/installation/).
2. Create a local database instance for this project. Set a password for the `neo4j` user. Select a Neo4j version with vector-index support. The Docker example below uses version 5.26 LTS. Keep the database name `neo4j` to match the code.
3. Start the instance. Open its query interface. Check that the database operates. Record the connection URI and password for the `.env` file. Use the ports that Desktop shows for this instance.

For a server installation without Desktop, download **Neo4j Community Edition** from the [Deployment Center](https://neo4j.com/deployment-center/). Follow the [server installation guide](https://neo4j.com/docs/operations-manual/current/installation/).

#### Option B: Run Neo4j with Docker

Install [Docker Desktop or Docker Engine](https://docs.docker.com/get-started/get-docker/). Start Docker. Download the database image from [Docker Hub](https://hub.docker.com/_/neo4j). Read the [Neo4j Docker guide](https://neo4j.com/docs/operations-manual/current/docker/introduction/) for more information.

Replace `change_this_password` with your own password of at least eight characters. Run this command as one line in your terminal or Windows PowerShell:

```bash
docker run -d --name llm4snomed-neo4j -p 127.0.0.1:7474:7474 -p 127.0.0.1:7687:7687 -e NEO4J_AUTH=neo4j/change_this_password -v llm4snomed_neo4j_data:/data -v llm4snomed_neo4j_logs:/logs neo4j:5.26
```

The command uses the official Neo4j 5.26 Community image. It publishes ports on the local machine. Named Docker volumes contain the database files and logs.

The `5.26` tag can receive patch updates. Record the image version or digest for reproducibility. A change to `NEO4J_AUTH` does not reset the password in an existing data volume.

Check startup messages with:

```bash
docker logs llm4snomed-neo4j
```

Open [Neo4j Browser](http://localhost:7474) after the server starts. Use the username `neo4j` and your selected password. Run `RETURN 1 AS connected;` to check the connection.

| Setting | Default for the Docker command above |
| --- | --- |
| Browser interface | `http://localhost:7474` |
| Python connection (`NEO4J_HOST`) | `neo4j://localhost:7687` |
| Username (`NEO4J_USER`) | `neo4j` |
| Password (`NEO4J_PASSWORD`) | The password supplied in `NEO4J_AUTH`. |
| Database name | `neo4j` |

Use the database connection URI in the notebooks. The Browser HTTP address is for the query interface. These addresses assume that Python and Neo4j operate on the same computer. If Python operates elsewhere, replace `localhost` with a hostname that Python can access.

Stop the container with the first command. Start the same container again with the second command:

```bash
docker stop llm4snomed-neo4j
docker start llm4snomed-neo4j
```

The named data volume preserves the database when you recreate the container, if you keep the volume. The notebook reset cells can still delete the database contents.

### 2. Prepare a Python environment

The notebook metadata records **Python 3.12.3**. Use a Python 3.12 environment. Run commands from the repository root. This location lets Python resolve imports such as `Src.run` and `Database.transform`.

```bash
git clone https://github.com/Chirurgo/LLM4SNOMED.git
cd LLM4SNOMED
python -m venv .venv
```

Activate the environment:

- On Linux/macOS, run `source .venv/bin/activate`.
- In Windows PowerShell, run `.venv\Scripts\Activate.ps1`.

Install the dependencies and a notebook interface:

```bash
python -m pip install -r requirement.txt
python -m pip install jupyterlab ipykernel
```

The dependency file records pinned package versions. It is not a verified lockfile for all operating systems. Check installation errors against these versions.

The pin `torch==2.8.0+cpu` may require the PyTorch CPU wheel index. CUDA execution requires a compatible PyTorch installation with CUDA support. The main notebook sets `device="cuda"`. For a CPU build, change this value to `device="cpu"`.

Record changes to dependencies for your experiment.

### 3. Prepare terminology, database, and model access

You will need:

- A licensed SNOMED CT RF2 release. The loader uses International Edition snapshot filenames dated `20250301`. Set the release root in the configuration. Change the filenames in `Database/transform.py` if you use another release.
- An active Neo4j instance with vector-index support. The code uses database `neo4j` and index `name_embedding`.
- Access to the Hugging Face models for embeddings and reranking. The main notebook uses `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` and `BAAI/bge-reranker-large` as its defaults.
- Credentials for your selected hosted LLM provider, or a local server with an OpenAI-compatible API. The local examples use LM Studio at `http://localhost:1234/v1/`.

The repository excludes model weights, SNOMED CT release files, the populated Neo4j database, and prediction files from the full study.

#### Local LLMs: install and start LM Studio

**For a local LLM, install LM Studio before you run the main code. Load the selected model. Start its API server.** The application and API server start separately. Hosted Gemini and GPT runs use their provider APIs.

1. Download [LM Studio](https://lmstudio.ai/download) for your operating system. Install the application.
2. Download the selected local model. Load the model in LM Studio. Choose model size, quantization, and context settings that fit your available memory. Record these settings.
3. Open the **Developer** tab. Select **Start server**. Set port **1234** to match the `.env` example. Keep the server active for the complete batch. Read the [local server instructions](https://lmstudio.ai/docs/developer/core/server) for details.
4. Check the model identifiers at [http://localhost:1234/v1/models](http://localhost:1234/v1/models), or use the command below. Set the notebook's `model` argument to the exact identifier from the server.

```bash
curl http://localhost:1234/v1/models
```

If you installed the LM Studio CLI, use this command to start the server:

```bash
lms server start --port 1234
```

`LMSTUDIO_URL=http://localhost:1234` identifies the server root. The client uses **`http://localhost:1234/v1`** as its API base URL.

Use `LMSTUDIO_API=dummy_key` only when the local server does not require token authentication. If the server requires authentication, supply the actual LM Studio API token. Read the [endpoint documentation](https://lmstudio.ai/docs/developer/openai-compat) and [authentication documentation](https://lmstudio.ai/docs/developer/core/authentication).

#### Gemini and GPT model availability

The Gemini and GPT versions in the study and example code are older releases. These models may now be deprecated, retired, or unavailable to your account.

**Before each use of this code, check model availability in the official Google and OpenAI portals.** The model identifiers in the notebooks may no longer work.

| Provider | Check before running |
| --- | --- |
| Google Gemini | [Google AI Studio](https://aistudio.google.com/), [Gemini model catalog](https://ai.google.dev/gemini-api/docs/models), and [deprecation notices](https://ai.google.dev/gemini-api/docs/deprecations). |
| OpenAI GPT | [OpenAI Platform](https://platform.openai.com/), [API model catalog](https://developers.openai.com/api/docs/models), and [deprecation notices](https://developers.openai.com/api/docs/deprecations). |

Select an available model. Update the model identifier in the notebook. Change API settings as necessary. Check compatibility with the extraction and linking workflow. Record the exact model version and access date.

A model substitution changes the experimental configuration. It may change performance, cost, and runtime. Results from a substitute model do not reproduce the results for the original model.

### 4. Configure `.env` and local paths

Create a local `.env` file in the repository root. Use the keys below. Replace the example password and paths. Supply API keys only for the hosted providers that you use. Set the Neo4j password to the password from Desktop or Docker.

```dotenv
NEO4J_HOST=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=replace_with_your_password
NEO4J_DATABASE=neo4j
OPENAI_API_KEY=replace_if_using_openai
GEMINI_API_KEY=replace_if_using_gemini
LMSTUDIO_URL=http://localhost:1234
LMSTUDIO_API=dummy_key
TEST_DATA_PATH=DATA\test.json
SNOMED_RELEASE_PATH=DATA/path_to_extracted_snomed_release
```

Configuration details:

- `NEO4J_HOST` contains the connection URI. The main notebook explicitly selects `database="neo4j"`. It does not read `NEO4J_DATABASE`. Thus, `NEO4J_DATABASE=bolt` does not change this selection. To select a database named `bolt`, change the code. The database name does not select the connection protocol.
- `TEST_DATA_PATH=DATA\test.json` gives a Windows path for your evaluation file. Use `DATA/test.json` on Linux/macOS. To use the public sample, set `TEST_DATA_PATH=Minimal_Dataset.json`. The repository excludes the full `DATA/test.json` file.
- `SNOMED_RELEASE_PATH` must identify the extracted RF2 release root. This directory must contain `Snapshot/Terminology/`. Replace the placeholder with your actual directory.
- `LMSTUDIO_URL` contains the server root without `/v1`. `LMSTUDIO_API` contains the placeholder or actual token for that server. This key differs from `OPENAI_API_KEY`.

Replace `YOUR_PATH_TO_ENV_FILE` with `.env` in the setup and main notebooks. Set `YOUR_PATH_TO_PREDICTIONS` to an existing output directory. Some cells join paths with Windows backslashes. Change these paths for Linux/macOS, for example with `pathlib.Path`.

The main notebook reads the LM Studio URL and key into `neo4j_dict`. However, the local model cells use a fixed URL and key. To use the `.env` values, replace the local client initialization with this example. Replace `MODEL_ID_FROM_LM_STUDIO` with the exact identifier from LM Studio:

```python
llm = ChatOpenAI(
    model="MODEL_ID_FROM_LM_STUDIO",
    base_url=neo4j_dict["lmstudio_uri"].rstrip("/") + "/v1",
    api_key=neo4j_dict["lmstudio_api"],
    seed=2531,
)
```

Keep credentials and restricted clinical data out of Git commits. Use hosted APIs only when the institution approves the data and processing arrangements.

## Running the workflow

### Build the SNOMED CT index

Open `01_create_database.ipynb`. Set the environment file path. Initialize the embedding model before `await main()`. The notebook imports the embedder but does not assign `embedding_model`. Add a cell such as:

```python
embedding_model = AsyncSapBERTEmbedder(device="cpu")  # or "cuda"
```

Use the same embedding model and vector dimensions for index creation and queries. The importer stores descriptions and embeddings on `Description` nodes. These nodes link to `Concept` nodes.

**Use a dedicated Neo4j database for these experiments.** The database notebooks contain calls to `delete_all_in_batches(...)`. These calls delete all nodes and relationships in the selected database. The description importer also sets `pre_delete_collection=True`.

Check these cells before execution. A repeated import can delete existing content.

After the import completes, select database `neo4j` in Neo4j Browser. Run this query. It does not change the data:

```cypher
SHOW VECTOR INDEXES
YIELD name, state, populationPercent, labelsOrTypes, properties
WHERE name = 'name_embedding'
RETURN name, state, populationPercent, labelsOrTypes, properties;
```

Check for index `name_embedding` on `Description` nodes and property `embedding`. Wait for state `ONLINE`. State `POPULATING` means that Neo4j still builds the index. If the query returns no row, the selected database does not contain this index.

An online index does not prove that the import included all descriptions. Check the import logs for skipped batches and errors.

### Generate predictions

Open `02_main.ipynb`. Set the input path, output directory, embedding device, and model settings. Run only the cells for your selected models.

For local LLM runs, check that the LM Studio API server operates. Check that LM Studio holds the selected model in memory. Follow the [local LLM setup](https://lmstudio.ai/docs/app/basics/download-model).

- `run_batch_top1()` performs extraction, positioning, retrieval, and cross-encoder ranking, then selects the first candidate.
- `run_batch()` adds an LLM selection step over the first five reranked candidates.

The code saves predictions as a JSON array of entity lists. Each input text record has one list in the same order. The entity fields commonly include:

- Mention fields: `term`, `formal_term`, `domain`, `status`, and `attributes`.
- Span fields: `start_index` and `end_index`.
- Linking fields: `concept_id`, `concept_name`, `matched_name`, and `tops`.

The code also records the choice and mapping explanation. An unlinked result may have a null `concept_id`.

The batch code records processing errors in the logs. It can write an empty list for a failed record. Check the logs before you interpret an empty list as a record with no entities. The code overwrites an existing prediction file when you use the same output path.

### Evaluate and inspect errors

Use `03_evaluation.ipynb` after you generate predictions. Set its prediction filenames to match the files from the main notebook. Keep the gold and prediction arrays in the same order. Check that their lengths are equal.

The evaluators match records by array position. Some routines truncate unequal arrays after a warning.

| Evaluation | Matching rule or interpretation |
| --- | --- |
| Strict NER | Exact character span and domain label. |
| Loose NER | Overlapping span and the same domain label. |
| Strict concept linking | Exact character span and SNOMED CT identifier. |
| Hierarchy-aware linking | Spans overlap and the concept matches exactly or is an allowed graph neighbor. The code uses direct `IS_A` or `HAS_ACTIVE_INGREDIENT` relationships. |
| Character-level IoU | Intersection over union of character positions associated with concept identifiers, averaged as implemented in `Evaluator.iou()`. |
| Top-k retrieval accuracy | The reference concept appears among the first k candidates. The metric includes only exact span matches. It measures conditional retrieval accuracy, not end-to-end recall. |

Check label names before NER evaluation. The reference annotations include `observable entity`. The extraction schema uses `observable`. `DomainEvaluator` converts labels to lowercase but does not map these different names.

Use a consistent label mapping for the comparison. Report that mapping.

To inspect errors visually:

1. Create `DATA/predictions/`.
2. Put the prediction JSON files in that directory.
3. Set `TEST_DATA_PATH` to the matching input file.
4. Run this command:

```bash
python 04_error_analysis.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). Select a text record and prediction file. The interface identifies records by `id` and `type`. It reads each prediction at the position of its input record in the array.

To use `05_dataset_stats.ipynb`, replace its local dataset path.

## Reproducibility notes

These notebooks are research working files. Configure them before execution. Their defaults do not automatically reproduce the published experiment.

- Gemini and GPT availability changes over time. Check model availability from respecytive model provider before each run.
- `00_evaluate_encoder.ipynb` uses local `DATA/` files and an additional Claude prediction file. The repository excludes these files. The main pipeline does not require this experimental notebook.
- Model identifiers, quantization, server settings, API availability, dependency versions, terminology releases, and hardware can affect outputs and runtime. Responses may vary even when temperature is zero.
- Full reproduction requires the complete evaluation data and the original model and configuration settings. Use the public sample for inspection and demonstrations.

For each experiment, record:

- The repository commit SHA and dependency versions.
- The input dataset and split.
- The SNOMED CT edition and release.
- The model identifiers, revisions, and prompts.
- The retrieval settings and hardware.
- All differences from the published methods.

## Citation and attribution

Please cite the article when you:

- Use or adapt the code or public sample in research.
- Reproduce, compare, or extend the framework.
- Discuss the study methods or findings in a publication, thesis, presentation, or other scholarly output.

Link to this repository. Identify the commit or release that you used. Cite models and other external resources as appropriate.

> Tangchitnob N, Ponthongmak W, Kijsanayotin B, Pattanaprateep O, Phusanti S, Atiksawedparit P, et al. (2026).
> A framework for structuring Thai medical documents using systematized nomenclature of medicine clinical terms and large language models.
> *PLOS Digital Health* 5(10): e0001791. https://doi.org/10.1371/journal.pdig.0001791

```bibtex
@article{Tangchitnob2026ThaiClinicalDocuments,
  author  = {Tangchitnob, Nat and Ponthongmak, Wanchana and Kijsanayotin, Boonchai and Pattanaprateep, Oraluck and Phusanti, Sithakom and Atiksawedparit, Pongsakorn and Suwanthaweemeesuk, Kamonporn and McKay, Gareth J. and Oami, Takehiko and Thakkinstian, Ammarin},
  title   = {A framework for structuring {Thai} medical documents using systematized nomenclature of medicine clinical terms and large language models},
  journal = {PLOS Digital Health},
  year    = {2026},
  volume  = {5},
  number  = {10},
  pages   = {e0001791},
  doi     = {10.1371/journal.pdig.0001791},
  url     = {https://doi.org/10.1371/journal.pdig.0001791}
}
```

The authors request citation for scholarly attribution. The MIT license governs software reuse. It requires the copyright and permission notices in copies or substantial portions of the software.

## Licensing and third-party resources

The software uses the **[MIT License](LICENSE)**. The article uses **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**. The institutional approval process governs access to the full clinical dataset.

SNOMED CT, model weights, libraries, and APIs have their own licenses or terms. The software license does not grant rights to these external resources. Request terminology access through [SNOMED International](https://www.snomed.org/get-snomed) or the relevant National Release Center. Read the terms for each model and API provider.

## Questions and contributions

For code problems or documentation suggestions, [open a GitHub issue](https://github.com/Chirurgo/LLM4SNOMED/issues). Include the relevant file, environment, model configuration, and a small example without sensitive data. Do not include patient text, restricted data, or API credentials in issues or pull requests.

For study questions or guidance on dataset access, contact **Wanchana Ponthongmak** at **[wanchana.pon@mahidol.ac.th](mailto:wanchana.pon@mahidol.ac.th)** or Nat Tangchitnob at [nat.tag@mahidol.ac.th](mailto:nat.tag@mahidol.ac.th).

