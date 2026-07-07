from __future__ import annotations

from rageval.core.config import (
    ChunkingConfig,
    DatasetConfig,
    EmbeddingConfig,
    RagevalConfig,
    ReportConfig,
    RetrieverConfig,
)
from rageval.validation.config import validate_config_file, validate_config_values


def test_validate_config_file_fails_for_missing_file(tmp_path) -> None:
    report = validate_config_file(tmp_path / "missing.yaml")

    assert not report.is_valid
    assert "does not exist" in report.issues[0].message


def test_validate_config_values_fails_for_missing_dataset_files(tmp_path) -> None:
    config = RagevalConfig(
        run_name="test",
        dataset=DatasetConfig(
            documents_path=tmp_path / "missing-documents.jsonl",
            questions_path=tmp_path / "missing-questions.jsonl",
        ),
        chunking=ChunkingConfig(chunk_size=500, chunk_overlap=50),
        embedding=EmbeddingConfig(provider="hash", model_name="hash", dimensions=128),
        retriever=RetrieverConfig(type="dense", top_k=3),
        report=ReportConfig(output_dir=tmp_path / "outputs"),
    )

    report = validate_config_values(config)

    assert not report.is_valid
    assert len(report.issues) == 2


def test_validate_config_values_fails_for_invalid_chunk_overlap(tmp_path) -> None:
    documents_path = tmp_path / "documents.jsonl"
    questions_path = tmp_path / "questions.jsonl"
    documents_path.write_text("", encoding="utf-8")
    questions_path.write_text("", encoding="utf-8")

    config = RagevalConfig(
        run_name="test",
        dataset=DatasetConfig(
            documents_path=documents_path,
            questions_path=questions_path,
        ),
        chunking=ChunkingConfig(chunk_size=100, chunk_overlap=100),
        embedding=EmbeddingConfig(provider="hash", model_name="hash", dimensions=128),
        retriever=RetrieverConfig(type="dense", top_k=3),
        report=ReportConfig(output_dir=tmp_path / "outputs"),
    )

    report = validate_config_values(config)

    assert not report.is_valid
    assert "chunk_overlap" in report.issues[0].message
