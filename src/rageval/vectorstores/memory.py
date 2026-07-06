from __future__ import annotations

import numpy as np

from rageval.core.schema import Chunk, RetrievedContext


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self._chunks = chunks
        self._vectors = vectors.astype(np.float32)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[RetrievedContext]:
        if self._vectors is None:
            raise RuntimeError("Vector store is empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        scores = self._vectors @ query_vector.astype(np.float32)
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        contexts: list[RetrievedContext] = []
        for rank, index in enumerate(ranked_indices, start=1):
            chunk = self._chunks[int(index)]
            contexts.append(
                RetrievedContext(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    score=float(scores[int(index)]),
                    rank=rank,
                    metadata=chunk.metadata,
                )
            )
        return contexts
