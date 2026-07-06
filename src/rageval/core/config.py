from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    documents_path: Path
    questions_path: Path


class ChunkingConfig(BaseModel):
    strategy: Literal["fixed"] = "fixed"
    chunk_size: int = 800
    chunk_overlap: int = 120


class EmbeddingConfig(BaseModel):
    provider: Literal["hash"] = "hash"
    model: str = "hash-embedding-v1"
    dimensions: int = 384


class RetrieverConfig(BaseModel):
    top_k: int = 5


class ReportConfig(BaseModel):
    output_dir: Path = Path("outputs")


class RagevalConfig(BaseModel):
    run_name: str = "baseline"
    dataset: DatasetConfig
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RagevalConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(raw)
