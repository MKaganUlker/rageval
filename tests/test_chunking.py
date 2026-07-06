from rageval.chunking.fixed import FixedSizeChunker
from rageval.core.schema import Document


def test_fixed_chunker_preserves_document_id() -> None:
    documents = [Document(id="doc-1", text="a" * 120)]
    chunks = FixedSizeChunker(chunk_size=50, chunk_overlap=10).split(documents)

    assert chunks
    assert all(chunk.document_id == "doc-1" for chunk in chunks)
    assert chunks[0].id == "doc-1::chunk-0"
