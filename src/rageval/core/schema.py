from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    id: str
    document_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationExample(BaseModel):
    id: str
    question: str
    expected_answer: str | None = None
    expected_document_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedContext(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float
    rank: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class GeneratedAnswer(BaseModel):
    answer: str
    model: str = "unknown"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetricResult(BaseModel):
    name: str
    value: float
    passed: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    example_id: str
    question: str
    retrieved_contexts: list[RetrievedContext]
    generated_answer: GeneratedAnswer | None = None
    metrics: list[MetricResult]
    latency_ms: float
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EvaluationRun(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    results: list[EvaluationResult]
    config: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def save_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
