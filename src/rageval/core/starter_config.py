from __future__ import annotations

from pathlib import Path


def create_starter_config(
    output_path: str | Path,
    dataset_dir: str | Path,
    overwrite: bool = False,
) -> Path:
    output = Path(output_path)
    dataset = Path(dataset_dir)

    if output.exists() and not overwrite:
        msg = f"Refusing to overwrite existing config file: {output}"
        raise FileExistsError(msg)

    output.parent.mkdir(parents=True, exist_ok=True)

    documents_path = dataset / "documents.jsonl"
    questions_path = dataset / "questions.jsonl"

    content = f"""run_name: starter_eval

dataset:
  documents_path: {documents_path.as_posix()}
  questions_path: {questions_path.as_posix()}

chunking:
  chunk_size: 500
  chunk_overlap: 50

embedding:
  provider: hash
  model_name: hash
  dimensions: 128

retriever:
  type: dense
  top_k: 3

report:
  output_dir: outputs
"""

    output.write_text(content, encoding="utf-8")
    return output
