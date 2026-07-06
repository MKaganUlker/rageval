from __future__ import annotations

from rageval.core.schema import Chunk, Document


class FixedSizeChunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be between 0 and chunk_size - 1")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        step = self.chunk_size - self.chunk_overlap
        for document in documents:
            text = document.text.strip()
            if not text:
                continue
            for index, start in enumerate(range(0, len(text), step)):
                chunk_text = text[start : start + self.chunk_size].strip()
                if not chunk_text:
                    continue
                chunks.append(
                    Chunk(
                        id=f"{document.id}::chunk-{index}",
                        document_id=document.id,
                        text=chunk_text,
                        metadata={**document.metadata, "chunk_index": index},
                    )
                )
                if start + self.chunk_size >= len(text):
                    break
        return chunks
