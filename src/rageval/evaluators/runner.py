from __future__ import annotations

import time
from statistics import mean

from rageval.core.schema import EvaluationExample, EvaluationResult, EvaluationRun
from rageval.metrics.retrieval import hit_rate_at_k, mrr, precision_at_k, recall_at_k
from rageval.retrievers.dense import DenseRetriever


class EvaluationRunner:
    def __init__(self, retriever: DenseRetriever, run_name: str = "baseline") -> None:
        self.retriever = retriever
        self.run_name = run_name

    def run(self, examples: list[EvaluationExample]) -> EvaluationRun:
        results: list[EvaluationResult] = []
        top_k = self.retriever.top_k

        for example in examples:
            start = time.perf_counter()
            contexts = self.retriever.retrieve(example.question)
            latency_ms = (time.perf_counter() - start) * 1000
            metrics = [
                hit_rate_at_k(example, contexts, top_k),
                recall_at_k(example, contexts, top_k),
                precision_at_k(example, contexts, top_k),
                mrr(example, contexts),
            ]
            results.append(
                EvaluationResult(
                    example_id=example.id,
                    question=example.question,
                    retrieved_contexts=contexts,
                    metrics=metrics,
                    latency_ms=latency_ms,
                )
            )

        return EvaluationRun(
            name=self.run_name,
            results=results,
            config={
                "top_k": top_k,
                "average_latency_ms": mean([result.latency_ms for result in results]) if results else 0,
            },
        )
