from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rageval.core.schema import Document, EvaluationExample


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def load_documents(path: str | Path) -> list[Document]:
    return [Document(**record) for record in _read_jsonl(path)]


def load_examples(path: str | Path) -> list[EvaluationExample]:
    return [EvaluationExample(**record) for record in _read_jsonl(path)]
