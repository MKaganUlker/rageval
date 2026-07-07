from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from rageval.cli.app import app

runner = CliRunner()


def test_cli_version_command() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.stdout.strip()


def test_cli_validate_command_passes_for_baseline_config() -> None:
    result = runner.invoke(app, ["validate", "--config", "configs/baseline.yaml"])

    assert result.exit_code == 0
    assert "Config and dataset are valid" in result.stdout


def test_cli_run_command_generates_reports(tmp_path: Path) -> None:
    config_path = tmp_path / "test-config.yaml"
    documents_path = tmp_path / "documents.jsonl"
    questions_path = tmp_path / "questions.jsonl"
    output_dir = tmp_path / "outputs"

    document = {
        "id": "doc-1",
        "text": "Python is a programming language.",
        "metadata": {"source": "test"},
    }
    question = {
        "id": "q-1",
        "question": "What is Python?",
        "expected_answer": "Python is a programming language.",
        "expected_document_ids": ["doc-1"],
    }

    documents_path.write_text(json.dumps(document) + "\n", encoding="utf-8")
    questions_path.write_text(json.dumps(question) + "\n", encoding="utf-8")

    config_path.write_text(
        f"""
run_name: cli_test

dataset:
  documents_path: {documents_path.as_posix()}
  questions_path: {questions_path.as_posix()}

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
  output_dir: {output_dir.as_posix()}
""",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["run", "--config", str(config_path)])

    assert result.exit_code == 0
    assert (output_dir / "cli_test.json").exists()
    assert (output_dir / "cli_test.md").exists()


def test_cli_init_dataset_command_creates_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "starter"

    result = runner.invoke(app, ["init-dataset", "--output", str(output_dir)])

    assert result.exit_code == 0
    assert (output_dir / "documents.jsonl").exists()
    assert (output_dir / "questions.jsonl").exists()


def test_cli_init_config_command_creates_config_file(tmp_path: Path) -> None:
    output_path = tmp_path / "config.yaml"
    dataset_dir = tmp_path / "dataset"

    result = runner.invoke(
        app,
        [
            "init-config",
            "--output",
            str(output_path),
            "--dataset-dir",
            str(dataset_dir),
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
