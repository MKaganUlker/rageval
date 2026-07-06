from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingModel(ABC):
    name: str
    dimensions: int

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]
