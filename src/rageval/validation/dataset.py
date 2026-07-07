from __future__ import annotations

from dataclasses import dataclass

from rageval.core.schema import Document, EvaluationExample


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    issues: list[ValidationIssue]

    @property
    def is_valid(self) -> bool:
        return not any(issue.level == "error" for issue in self.issues)


def validate_dataset(
    documents: list[Document],
    examples: list[EvaluationExample],
) -> ValidationReport:
    issues: list[ValidationIssue] = []

    if not documents:
        issues.append(ValidationIssue(level="error", message="No documents found."))

    if not examples:
        issues.append(ValidationIssue(level="error", message="No evaluation examples found."))

    document_ids = {document.id for document in documents}

    for example in examples:
        if not example.expected_document_ids:
            issues.append(
                ValidationIssue(
                    level="warning",
                    message=f"Example '{example.id}' has no expected document IDs.",
                )
            )

        missing_ids = [
            document_id
            for document_id in example.expected_document_ids
            if document_id not in document_ids
        ]

        if missing_ids:
            issues.append(
                ValidationIssue(
                    level="error",
                    message=(
                        f"Example '{example.id}' references missing document IDs: "
                        f"{', '.join(missing_ids)}"
                    ),
                )
            )

    return ValidationReport(issues=issues)
