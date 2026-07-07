from __future__ import annotations

import pytest

from rageval.datasets.jsonl import load_documents, load_examples
from rageval.datasets.starter import create_starter_dataset


def test_create_starter_dataset_creates_valid_jsonl_files(tmp_path) -> None:
    documents_path, questions_path = create_starter_dataset(tmp_path)

    assert documents_path.exists()
    assert questions_path.exists()

    documents = load_documents(documents_path)
    examples = load_examples(questions_path)

    assert documents[0].id == "doc-1"
    assert examples[0].expected_document_ids == ["doc-1"]


def test_create_starter_dataset_refuses_to_overwrite_existing_files(tmp_path) -> None:
    create_starter_dataset(tmp_path)

    with pytest.raises(FileExistsError):
        create_starter_dataset(tmp_path)
