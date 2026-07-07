from __future__ import annotations

from rageval.core.schema import EvaluationExample, MetricResult, RetrievedContext


def hit_rate_at_k(
    example: EvaluationExample,
    contexts: list[RetrievedContext],
    k: int,
) -> MetricResult:
    expected = set(example.expected_document_ids)
    retrieved = {context.document_id for context in contexts[:k]}
    value = 1.0 if expected and bool(expected & retrieved) else 0.0
    return MetricResult(name=f"hit_rate@{k}", value=value)


def recall_at_k(
    example: EvaluationExample,
    contexts: list[RetrievedContext],
    k: int,
) -> MetricResult:
    expected = set(example.expected_document_ids)
    if not expected:
        return MetricResult(name=f"recall@{k}", value=0.0, metadata={"skipped": True})
    retrieved = {context.document_id for context in contexts[:k]}
    value = len(expected & retrieved) / len(expected)
    return MetricResult(name=f"recall@{k}", value=value)


def precision_at_k(
    example: EvaluationExample,
    contexts: list[RetrievedContext],
    k: int,
) -> MetricResult:
    expected = set(example.expected_document_ids)
    if not contexts[:k]:
        return MetricResult(name=f"precision@{k}", value=0.0)
    retrieved = [context.document_id for context in contexts[:k]]
    value = sum(1 for document_id in retrieved if document_id in expected) / len(retrieved)
    return MetricResult(name=f"precision@{k}", value=value)


def mrr(example: EvaluationExample, contexts: list[RetrievedContext]) -> MetricResult:
    expected = set(example.expected_document_ids)
    for context in contexts:
        if context.document_id in expected:
            return MetricResult(name="mrr", value=1.0 / context.rank)
    return MetricResult(name="mrr", value=0.0)
