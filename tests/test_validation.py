from __future__ import annotations

from rageval.core.schema import Document, EvaluationExample
from rageval.validation.dataset import validate_dataset


def test_validate_dataset_passes_for_valid_expected_document_ids() -> None:
    documents = [
        Document(id="doc-1", text="Python is a programming language."),
    ]
    examples = [
        EvaluationExample(
            id="q-1",
            question="What is Python?",
            expected_answer="Python is a programming language.",
            expected_document_ids=["doc-1"],
        )
    ]

    report = validate_dataset(documents=documents, examples=examples)

    assert report.is_valid


def test_validate_dataset_fails_for_missing_expected_document_id() -> None:
    documents = [
        Document(id="doc-1", text="Python is a programming language."),
    ]
    examples = [
        EvaluationExample(
            id="q-1",
            question="What is Python?",
            expected_answer="Python is a programming language.",
            expected_document_ids=["missing-doc"],
        )
    ]

    report = validate_dataset(documents=documents, examples=examples)

    assert not report.is_valid
    assert "missing-doc" in report.issues[0].message
