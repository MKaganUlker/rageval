from __future__ import annotations

import json
from pathlib import Path

SAMPLE_DOCUMENT = {
    "id": "doc-1",
    "text": "RAG systems combine retrieval with generation to answer questions using external knowledge.",
    "metadata": {
        "source": "starter",
        "category": "rag",
    },
}


SAMPLE_QUESTION = {
    "id": "q-1",
    "question": "What do RAG systems combine?",
    "expected_answer": "RAG systems combine retrieval with generation.",
    "expected_document_ids": ["doc-1"],
    "metadata": {
        "difficulty": "easy",
        "category": "rag",
    },
}


def create_starter_dataset(output_dir: str | Path, overwrite: bool = False) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    documents_path = output / "documents.jsonl"
    questions_path = output / "questions.jsonl"

    if not overwrite:
        existing_paths = [
            path
            for path in [documents_path, questions_path]
            if path.exists()
        ]
        if existing_paths:
            joined_paths = ", ".join(str(path) for path in existing_paths)
            msg = f"Refusing to overwrite existing files: {joined_paths}"
            raise FileExistsError(msg)

    documents_path.write_text(json.dumps(SAMPLE_DOCUMENT) + "\n", encoding="utf-8")
    questions_path.write_text(json.dumps(SAMPLE_QUESTION) + "\n", encoding="utf-8")

    return documents_path, questions_path
