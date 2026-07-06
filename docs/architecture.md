# Architecture

RAGEval is organized around explicit interfaces and typed data models.

```text
Dataset -> Documents -> Chunker -> Chunks -> Embeddings -> Vector Store -> Retriever -> Metrics -> Report
```

## Core principles

1. Reproducibility over demos
2. Config-driven experiments
3. Small interfaces
4. Observable metrics
5. Offline-first baseline

## Package layout

```text
src/rageval/
  core/          shared schemas and config models
  datasets/      dataset loaders
  chunking/      chunking strategies
  embeddings/    embedding model abstractions
  vectorstores/  vector database abstractions
  retrievers/    retrieval implementations
  metrics/       retrieval and generation metrics
  evaluators/    evaluation runners
  reports/       report writers
  cli/           command-line interface
```
