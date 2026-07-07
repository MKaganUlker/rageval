from __future__ import annotations

from abc import ABC, abstractmethod
from typing import cast

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


class EmbeddingModel(ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> FloatArray:
        raise NotImplementedError

    def embed_query(self, query: str) -> FloatArray:
        vectors = self.embed_texts([query])
        return cast(FloatArray, vectors[0])
