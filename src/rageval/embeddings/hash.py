from __future__ import annotations

import hashlib
from typing import cast

import numpy as np

from rageval.embeddings.base import EmbeddingModel, FloatArray


class HashEmbeddingModel(EmbeddingModel):
    """Deterministic offline embedding model.

    This is intentionally simple and dependency-light. It allows examples,
    tests, and CI to run without external API keys.
    """

    def __init__(self, dimensions: int = 128) -> None:
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> FloatArray:
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float64)

        for row_index, text in enumerate(texts):
            tokens = text.lower().split()
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                column_index = int.from_bytes(digest[:4], "big") % self.dimensions
                vectors[row_index, column_index] += 1.0

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_vectors = vectors / norms
        return cast(FloatArray, normalized_vectors)
