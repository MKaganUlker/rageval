# RAGEval

**RAGEval** is a professional evaluation framework for Retrieval-Augmented Generation systems.

It helps AI engineers benchmark RAG pipelines across retrieval quality, latency, configuration choices, and reproducibility.

The project starts with a clean offline baseline that works without external API keys. It is designed to grow into a full RAG evaluation platform with multiple embedding models, vector stores, retrievers, generation metrics, and experiment comparison tools.

## Why this project exists

Most RAG demos stop at “chat with your PDF.” Real production teams need to answer harder questions:

* Is the retriever finding the right documents?
* Which chunk size works best?
* Does increasing `top_k` improve recall or only increase latency?
* Which embedding model performs better on the same dataset?
* Are evaluation results reproducible across runs?
* Can different RAG configurations be compared fairly?
* Can failures be explained instead of guessed?

RAGEval focuses on measurable RAG quality.

The goal is not only to build a RAG pipeline, but to evaluate one like an engineering system.

## Current project status

RAGEval is currently in early development.

The current version focuses on **retrieval evaluation** and **developer workflow**:

* loading benchmark datasets
* validating configs and datasets
* chunking documents
* embedding chunks with an offline baseline model
* retrieving relevant contexts
* computing retrieval metrics
* generating JSON and Markdown reports
* testing the full CLI workflow in CI

Generation quality metrics, LLM-as-judge evaluation, FAISS/Chroma integrations, experiment sweeps, and dashboard support are planned for later versions.

## Current features

* Installable Python package
* Typed Pydantic schemas
* YAML-based configuration
* JSONL document dataset loader
* JSONL evaluation question loader
* Fixed-size chunking
* Deterministic offline hash embeddings
* In-memory dense vector search
* Retrieval evaluation runner
* Retrieval metrics:

  * Hit Rate@K
  * Recall@K
  * Precision@K
  * MRR
* Markdown report output
* JSON report output
* Typer-based CLI
* Dataset validation command
* Config validation checks
* Starter dataset generator
* Starter config generator
* CLI integration tests
* Pytest test suite
* Ruff linting
* mypy type checking
* GitHub Actions CI for Python 3.10, 3.11, and 3.12
* Changelog
* Contribution guide
* Release checklist

## Installation

Clone the repository:

```bash
git clone https://github.com/MKaganUlker/rageval.git
cd rageval
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

Windows Git Bash:

```bash
source .venv/Scripts/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install the project with development dependencies:

```bash
pip install -e ".[dev]"
```

## Quickstart

Create a starter dataset:

```bash
rageval init-dataset --output data/my_dataset
```

Create a matching config file:

```bash
rageval init-config --output configs/my_config.yaml --dataset-dir data/my_dataset
```

Validate the config and dataset:

```bash
rageval validate --config configs/my_config.yaml
```

Run the evaluation:

```bash
rageval run --config configs/my_config.yaml
```

Generated outputs:

```text
outputs/starter_eval.json
outputs/starter_eval.md
```

## Run the included sample evaluation

The repository includes a small sample dataset and baseline config.

Validate the sample:

```bash
rageval validate --config configs/baseline.yaml
```

Run the sample evaluation:

```bash
rageval run --config configs/baseline.yaml
```

Generated outputs:

```text
outputs/baseline.json
outputs/baseline.md
```

## CLI commands

Show the installed version:

```bash
rageval version
```

Create starter dataset files:

```bash
rageval init-dataset --output data/my_dataset
```

Create a starter config file:

```bash
rageval init-config --output configs/my_config.yaml --dataset-dir data/my_dataset
```

Validate a config and dataset:

```bash
rageval validate --config configs/my_config.yaml
```

Run an evaluation:

```bash
rageval run --config configs/my_config.yaml
```

## Dataset format

RAGEval currently uses JSONL files for documents and evaluation questions.

### Documents

Each line in `documents.jsonl` represents one source document.

```json
{"id":"doc-1","text":"Document text...","metadata":{"source":"sample"}}
```

Required fields:

* `id`: unique document identifier
* `text`: document content

Optional fields:

* `metadata`: extra information such as source, category, author, page, or URL

### Evaluation questions

Each line in `questions.jsonl` represents one evaluation example.

```json
{"id":"q-1","question":"What is RAG?","expected_answer":"RAG combines retrieval with generation.","expected_document_ids":["doc-1"],"metadata":{"difficulty":"easy"}}
```

Required fields:

* `id`: unique question identifier
* `question`: user question
* `expected_answer`: reference answer
* `expected_document_ids`: list of documents expected to be retrieved

Optional fields:

* `metadata`: extra information such as difficulty, topic, category, or source

## Configuration format

RAGEval evaluations are controlled through YAML config files.

Example:

```yaml
run_name: starter_eval

dataset:
  documents_path: data/my_dataset/documents.jsonl
  questions_path: data/my_dataset/questions.jsonl

chunking:
  chunk_size: 500
  chunk_overlap: 50

embedding:
  provider: hash
  model_name: hash
  dimensions: 128

retriever:
  type: dense
  top_k: 3

report:
  output_dir: outputs
```

## Validation

RAGEval validates both the configuration and the dataset before running an evaluation.

It checks for issues such as:

* missing config files
* missing document files
* missing question files
* invalid `chunk_size`
* invalid `chunk_overlap`
* invalid `top_k`
* invalid embedding dimensions
* evaluation questions referencing missing document IDs
* empty document datasets
* empty evaluation datasets

Run validation with:

```bash
rageval validate --config configs/my_config.yaml
```

A valid setup should report that the config and dataset are valid.

## Metrics

RAGEval currently focuses on retrieval metrics.

### Hit Rate@K

Checks whether at least one expected document appears in the top `K` retrieved contexts.

Useful for answering:

> Did the retriever find anything relevant?

### Recall@K

Measures how many expected documents were retrieved in the top `K`.

Useful for answering:

> Did the retriever find all required evidence?

### Precision@K

Measures how many retrieved contexts are expected/relevant documents.

Useful for answering:

> Is the retriever returning mostly useful context or noisy context?

### MRR

Mean Reciprocal Rank rewards systems that retrieve the first relevant document earlier.

Useful for answering:

> How high does the first relevant result appear?

## Reports

Each evaluation run produces:

```text
outputs/<run_name>.json
outputs/<run_name>.md
```

The JSON report is useful for automation and future comparison tools.

The Markdown report is useful for quick inspection, GitHub artifacts, and human-readable summaries.

## Development

Run the full local quality gate:

```bash
ruff check .
mypy src
pytest
```

Run a CLI smoke test:

```bash
rageval init-dataset --output data/demo_dataset
rageval init-config --output configs/demo.yaml --dataset-dir data/demo_dataset
rageval validate --config configs/demo.yaml
rageval run --config configs/demo.yaml
```

Clean generated demo files:

```bash
rm -rf data/demo_dataset configs/demo.yaml outputs/starter_eval.json outputs/starter_eval.md
```

## Project structure

```text
rageval/
├── configs/
│   └── baseline.yaml
├── data/
│   └── sample/
│       ├── documents.jsonl
│       └── questions.jsonl
├── docs/
│   └── release_checklist.md
├── src/
│   └── rageval/
│       ├── chunking/
│       ├── cli/
│       ├── core/
│       ├── datasets/
│       ├── embeddings/
│       ├── evaluators/
│       ├── metrics/
│       ├── reports/
│       ├── retrievers/
│       ├── validation/
│       └── vectorstores/
├── tests/
├── CHANGELOG.md
├── CONTRIBUTING.md
├── pyproject.toml
└── README.md
```

## Roadmap

### v0.1 — Offline retrieval evaluation

* [x] JSONL datasets
* [x] Fixed-size chunking
* [x] Offline hash embeddings
* [x] In-memory dense retrieval
* [x] Retrieval metrics
* [x] Markdown reports
* [x] JSON reports
* [x] CLI
* [x] Tests and CI

### v0.2 — Developer UX and dataset tooling

* [x] Dataset validation command
* [x] Config validation checks
* [x] Starter dataset generator
* [x] Starter config generator
* [x] CLI integration tests
* [x] Changelog
* [x] Contribution guide
* [x] Release checklist
* [x] README update

### v0.3 — More retrieval backends

* [ ] SentenceTransformers embeddings
* [ ] FAISS vector store
* [ ] Chroma vector store
* [ ] BM25 retriever
* [ ] Configurable retriever selection

### v0.4 — Experiment comparison

* [ ] Multiple run comparison
* [ ] Leaderboard output
* [ ] Configurable experiment sweeps
* [ ] Latency comparison
* [ ] Metric comparison reports

### v0.5 — Generation evaluation

* [ ] LLM provider abstraction
* [ ] OpenAI support
* [ ] Ollama support
* [ ] Faithfulness metric
* [ ] Answer relevancy metric
* [ ] Cost and token tracking

### v1.0 — Full RAG evaluation framework

* [ ] Retrieval evaluation
* [ ] Generation evaluation
* [ ] Multiple vector stores
* [ ] Multiple embedding providers
* [ ] Multiple retriever strategies
* [ ] Reproducible reports
* [ ] Experiment comparison
* [ ] CI-tested professional workflow

## Example CV bullet

> Built RAGEval, a modular Python framework for evaluating Retrieval-Augmented Generation pipelines with configurable datasets, chunking, embeddings, vector search, retrieval metrics, validation tooling, reproducible reports, automated tests, and CI/CD.

## License

MIT
