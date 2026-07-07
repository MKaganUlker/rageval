from __future__ import annotations

import pytest

from rageval.core.config import RagevalConfig
from rageval.core.starter_config import create_starter_config


def test_create_starter_config_creates_valid_yaml(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    output_path = tmp_path / "config.yaml"

    config_path = create_starter_config(
        output_path=output_path,
        dataset_dir=dataset_dir,
    )

    assert config_path.exists()

    config = RagevalConfig.from_yaml(config_path)

    assert config.run_name == "starter_eval"
    assert config.dataset.documents_path == dataset_dir / "documents.jsonl"
    assert config.dataset.questions_path == dataset_dir / "questions.jsonl"


def test_create_starter_config_refuses_to_overwrite_existing_file(tmp_path) -> None:
    output_path = tmp_path / "config.yaml"

    create_starter_config(
        output_path=output_path,
        dataset_dir=tmp_path / "dataset",
    )

    with pytest.raises(FileExistsError):
        create_starter_config(
            output_path=output_path,
            dataset_dir=tmp_path / "dataset",
        )
