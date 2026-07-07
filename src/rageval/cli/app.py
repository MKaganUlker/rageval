from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from rageval.chunking.fixed import FixedSizeChunker
from rageval.core.config import RagevalConfig
from rageval.datasets.jsonl import load_documents, load_examples
from rageval.embeddings.hash import HashEmbeddingModel
from rageval.evaluators.runner import EvaluationRunner
from rageval.reports.markdown import write_markdown_report
from rageval.retrievers.dense import DenseRetriever
from rageval.vectorstores.memory import InMemoryVectorStore

app = typer.Typer(help="Evaluate Retrieval-Augmented Generation systems.")
console = Console()


@app.command()
def run(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to a rageval YAML config."),
    ],
) -> None:
    cfg = RagevalConfig.from_yaml(config)

    documents = load_documents(cfg.dataset.documents_path)
    examples = load_examples(cfg.dataset.questions_path)

    chunker = FixedSizeChunker(
        chunk_size=cfg.chunking.chunk_size,
        chunk_overlap=cfg.chunking.chunk_overlap,
    )
    chunks = chunker.split(documents)

    embedding_model = HashEmbeddingModel(dimensions=cfg.embedding.dimensions)
    vectors = embedding_model.embed_texts([chunk.text for chunk in chunks])

    vector_store = InMemoryVectorStore()
    vector_store.add(chunks=chunks, vectors=vectors)

    retriever = DenseRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        top_k=cfg.retriever.top_k,
    )
    evaluation_run = EvaluationRunner(retriever=retriever, run_name=cfg.run_name).run(examples)

    output_dir = cfg.report.output_dir
    json_path = output_dir / f"{evaluation_run.name}.json"
    markdown_path = output_dir / f"{evaluation_run.name}.md"
    evaluation_run.save_json(json_path)
    write_markdown_report(evaluation_run, markdown_path)

    table = Table(title=f"RAG Evaluation: {evaluation_run.name}")
    table.add_column("Output", style="bold")
    table.add_column("Path")
    table.add_row("JSON", str(json_path))
    table.add_row("Markdown", str(markdown_path))
    console.print(table)


@app.command()
def version() -> None:
    from rageval import __version__

    console.print(__version__)


if __name__ == "__main__":
    app()
