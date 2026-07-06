from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from rageval.core.schema import Document, EvaluationExample


def _read_jsonl(path: str | Path) -> Iterable[dict]:
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc


def load_documents(path: str | Path) -> list[Document]:
    return [Document.model_validate(row) for row in _read_jsonl(path)]


def load_examples(path: str | Path) -> list[EvaluationExample]:
    return [EvaluationExample.model_validate(row) for row in _read_jsonl(path)]
