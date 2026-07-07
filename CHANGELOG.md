# Changelog

All notable changes to this project will be documented in this file.

This project follows a simple versioned changelog format.

## [Unreleased]

### Added

- Configuration validation checks.
- Dataset validation command.
- Starter dataset generator.
- Starter config generator.
- CLI integration tests.

### Changed

- Improved CLI validation flow with clearer error messages.

## [0.1.0] - Initial Retrieval Evaluation Baseline

### Added

- Installable Python package.
- Typer-based CLI.
- Config-driven evaluation runner.
- JSONL document loader.
- JSONL evaluation example loader.
- Fixed-size chunking.
- Deterministic offline hash embeddings.
- In-memory dense vector search.
- Retrieval metrics:
  - Hit Rate@K
  - Recall@K
  - Precision@K
  - MRR
- JSON report output.
- Markdown report output.
- Ruff, mypy, pytest, and GitHub Actions CI.
