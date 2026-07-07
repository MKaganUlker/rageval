from __future__ import annotations

from pathlib import Path

from rageval.core.config import RagevalConfig
from rageval.validation.dataset import ValidationIssue, ValidationReport


def validate_config_file(config_path: str | Path) -> ValidationReport:
    path = Path(config_path)
    issues: list[ValidationIssue] = []

    if not path.exists():
        issues.append(
            ValidationIssue(
                level="error",
                message=f"Config file does not exist: {path}",
            )
        )

    if path.exists() and not path.is_file():
        issues.append(
            ValidationIssue(
                level="error",
                message=f"Config path is not a file: {path}",
            )
        )

    return ValidationReport(issues=issues)


def validate_config_values(config: RagevalConfig) -> ValidationReport:
    issues: list[ValidationIssue] = []

    if not config.dataset.documents_path.exists():
        issues.append(
            ValidationIssue(
                level="error",
                message=f"Documents file does not exist: {config.dataset.documents_path}",
            )
        )

    if not config.dataset.questions_path.exists():
        issues.append(
            ValidationIssue(
                level="error",
                message=f"Questions file does not exist: {config.dataset.questions_path}",
            )
        )

    if config.chunking.chunk_size <= 0:
        issues.append(
            ValidationIssue(
                level="error",
                message="chunk_size must be greater than 0.",
            )
        )

    if config.chunking.chunk_overlap < 0:
        issues.append(
            ValidationIssue(
                level="error",
                message="chunk_overlap must be greater than or equal to 0.",
            )
        )

    if config.chunking.chunk_overlap >= config.chunking.chunk_size:
        issues.append(
            ValidationIssue(
                level="error",
                message="chunk_overlap must be smaller than chunk_size.",
            )
        )

    if config.embedding.dimensions <= 0:
        issues.append(
            ValidationIssue(
                level="error",
                message="embedding dimensions must be greater than 0.",
            )
        )

    if config.retriever.top_k <= 0:
        issues.append(
            ValidationIssue(
                level="error",
                message="top_k must be greater than 0.",
            )
        )

    return ValidationReport(issues=issues)
