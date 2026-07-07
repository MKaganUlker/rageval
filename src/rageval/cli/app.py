from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from rageval.chunking.fixed import FixedSizeChunker
from rageval.core.config import RagevalConfig
from rageval.core.starter_config import create_starter_config
from rageval.datasets.jsonl import load_documents, load_examples
from rageval.datasets.starter import create_starter_dataset
from rageval.embeddings.hash import HashEmbeddingModel
from rageval.evaluators.runner import EvaluationRunner
from rageval.reports.markdown import write_markdown_report
from rageval.retrievers.dense import DenseRetriever
from rageval.validation.dataset import validate_dataset
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

    validation_report = validate_dataset(documents=documents, examples=examples)
    if not validation_report.is_valid:
        console.print("[bold red]Dataset validation failed.[/bold red]")
        for issue in validation_report.issues:
            console.print(f"[{issue.level}] {issue.message}")
        raise typer.Exit(code=1)

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
def validate(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to a rageval YAML config."),
    ],
) -> None:
    cfg = RagevalConfig.from_yaml(config)
    documents = load_documents(cfg.dataset.documents_path)
    examples = load_examples(cfg.dataset.questions_path)

    validation_report = validate_dataset(documents=documents, examples=examples)

    table = Table(title="Dataset Validation")
    table.add_column("Level", style="bold")
    table.add_column("Message")

    if not validation_report.issues:
        table.add_row("success", "Dataset is valid.")
    else:
        for issue in validation_report.issues:
            table.add_row(issue.level, issue.message)

    console.print(table)

    if not validation_report.is_valid:
        raise typer.Exit(code=1)


@app.command("init-dataset")
def init_dataset(
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Directory where starter dataset files are created."),
    ],
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Overwrite existing starter dataset files."),
    ] = False,
) -> None:
    try:
        documents_path, questions_path = create_starter_dataset(
            output_dir=output,
            overwrite=overwrite,
        )
    except FileExistsError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    table = Table(title="Starter Dataset Created")
    table.add_column("File", style="bold")
    table.add_column("Path")
    table.add_row("Documents", str(documents_path))
    table.add_row("Questions", str(questions_path))
    console.print(table)

@app.command("init-config")
def init_config(
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Path where the YAML config is created."),
    ],
    dataset_dir: Annotated[
        Path,
        typer.Option(
            "--dataset-dir",
            "-d",
            help="Directory containing documents.jsonl and questions.jsonl.",
        ),
    ],
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Overwrite existing config file."),
    ] = False,
) -> None:
    try:
        config_path = create_starter_config(
            output_path=output,
            dataset_dir=dataset_dir,
            overwrite=overwrite,
        )
    except FileExistsError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    table = Table(title="Starter Config Created")
    table.add_column("File", style="bold")
    table.add_column("Path")
    table.add_row("Config", str(config_path))
    console.print(table)

@app.command()
def version() -> None:
    from rageval import __version__

    console.print(__version__)


if __name__ == "__main__":
    app()
