from __future__ import annotations

import numpy as np

from rageval.core.schema import Chunk, RetrievedContext
from rageval.embeddings.base import FloatArray


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: FloatArray | None = None

    def add(self, chunks: list[Chunk], vectors: FloatArray) -> None:
        if len(chunks) != len(vectors):
            msg = "Number of chunks must match number of vectors."
            raise ValueError(msg)

        self._chunks.extend(chunks)

        if self._vectors is None:
            self._vectors = vectors
            return

        self._vectors = np.vstack([self._vectors, vectors])

    def search(self, query_vector: FloatArray, top_k: int) -> list[RetrievedContext]:
        if self._vectors is None:
            return []

        scores = self._vectors @ query_vector
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results: list[RetrievedContext] = []
        for rank, index in enumerate(ranked_indices, start=1):
            chunk = self._chunks[int(index)]
            results.append(
                RetrievedContext(
                    document_id=chunk.document_id,
                    chunk_id=chunk.id,
                    text=chunk.text,
                    score=float(scores[int(index)]),
                    rank=rank,
                    metadata=chunk.metadata,
                )
            )

        return results
