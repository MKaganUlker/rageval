# RAGEval

**RAGEval** is a professional evaluation framework for Retrieval-Augmented Generation systems.

It helps AI engineers benchmark RAG pipelines across retrieval quality, latency, configuration choices, and reproducibility.

This repository starts with a clean offline baseline and is designed to grow into a full RAG evaluation platform.

## Why this project exists

Most RAG demos stop at “chat with your PDF.” Real production teams need to answer harder questions:

- Is the retriever finding the right documents?
- Which chunk size works best?
- Does increasing `top_k` improve recall or only increase latency?
- Which embedding model performs better on the same dataset?
- Are results reproducible across runs?

RAGEval focuses on measurable RAG quality.

## Current features

- Installable Python package
- Typed Pydantic schemas
- YAML configuration
- JSONL dataset loader
- Fixed-size chunking
- Deterministic offline hash embeddings
- In-memory dense vector search
- Retrieval metrics:
  - Hit Rate@K
  - Recall@K
  - Precision@K
  - MRR
- Markdown and JSON reports
- Typer CLI
- Pytest test suite
- Ruff, mypy, GitHub Actions CI

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Run the sample evaluation

```bash
rageval run --config configs/baseline.yaml
```

Outputs are written to:

```text
outputs/baseline.json
outputs/baseline.md
```

## Dataset format

Documents use JSONL:

```json
{"id":"doc-1","text":"Document text...","metadata":{"source":"sample"}}
```

Questions use JSONL:

```json
{"id":"q1","question":"What is RAG?","expected_answer":"...","expected_document_ids":["doc-1"]}
```

## Roadmap

### v0.1 — Offline retrieval evaluation

- [x] JSONL datasets
- [x] Fixed chunking
- [x] Offline embeddings
- [x] Dense retrieval
- [x] Retrieval metrics
- [x] Markdown reports
- [x] CLI
- [x] Tests and CI

### v0.2 — Real embedding and vector DB support

- [ ] SentenceTransformers embeddings
- [ ] FAISS vector store
- [ ] Chroma vector store
- [ ] BM25 retriever
- [ ] Configurable experiment sweeps

### v0.3 — Generation evaluation

- [ ] LLM provider abstraction
- [ ] OpenAI / Ollama support
- [ ] Faithfulness metric
- [ ] Answer relevancy metric
- [ ] Cost and token tracking

### v0.4 — Professional reporting

- [ ] HTML reports
- [ ] Leaderboard comparison
- [ ] Error analysis pages
- [ ] Latency and quality plots

## Example CV bullet

> Built RAGEval, a modular Python framework for evaluating Retrieval-Augmented Generation pipelines with configurable datasets, chunking, embeddings, vector search, retrieval metrics, reproducible reports, automated tests, and CI/CD.

## License

MIT
