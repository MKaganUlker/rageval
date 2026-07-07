from __future__ import annotations

from rageval.core.schema import RetrievedContext
from rageval.embeddings.base import EmbeddingModel
from rageval.vectorstores.memory import InMemoryVectorStore


class DenseRetriever:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: InMemoryVectorStore,
        top_k: int = 5,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, question: str) -> list[RetrievedContext]:
        query_vector = self.embedding_model.embed_query(question)
        return self.vector_store.search(query_vector=query_vector, top_k=self.top_k)
