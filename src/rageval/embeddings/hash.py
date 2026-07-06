from __future__ import annotations

import hashlib
import re

import numpy as np

from rageval.embeddings.base import EmbeddingModel

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


class HashEmbeddingModel(EmbeddingModel):
    name = "hash-embedding-v1"

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in TOKEN_PATTERN.findall(text.lower()):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimensions
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vectors[row, index] += sign
            norm = float(np.linalg.norm(vectors[row]))
            if norm > 0:
                vectors[row] /= norm
        return vectors
