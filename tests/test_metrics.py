from rageval.core.schema import EvaluationExample, RetrievedContext
from rageval.metrics.retrieval import hit_rate_at_k, mrr, precision_at_k, recall_at_k


def test_retrieval_metrics() -> None:
    example = EvaluationExample(id="q1", question="x", expected_document_ids=["doc-2"])
    contexts = [
        RetrievedContext(chunk_id="c1", document_id="doc-1", text="", score=0.9, rank=1),
        RetrievedContext(chunk_id="c2", document_id="doc-2", text="", score=0.8, rank=2),
    ]

    assert hit_rate_at_k(example, contexts, 2).value == 1.0
    assert recall_at_k(example, contexts, 2).value == 1.0
    assert precision_at_k(example, contexts, 2).value == 0.5
    assert mrr(example, contexts).value == 0.5
