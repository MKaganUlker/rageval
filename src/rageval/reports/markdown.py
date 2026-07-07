from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean

from rageval.core.schema import EvaluationRun


def write_markdown_report(run: EvaluationRun, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    metric_values: dict[str, list[float]] = defaultdict(list)
    for result in run.results:
        for metric in result.metrics:
            metric_values[metric.name].append(metric.value)

    lines = [
        f"# RAG Evaluation Report: {run.name}",
        "",
        f"Run ID: `{run.id}`",
        f"Created at: `{run.created_at.isoformat()}`",
        "",
        "## Summary",
        "",
        "| Metric | Average |",
        "|---|---:|",
    ]

    for metric_name, values in sorted(metric_values.items()):
        lines.append(f"| {metric_name} | {mean(values):.4f} |")

    lines.extend(["", "## Questions", ""])
    for result in run.results:
        lines.extend(
            [
                f"### {result.example_id}: {result.question}",
                "",
                f"Latency: `{result.latency_ms:.2f} ms`",
                "",
                "| Metric | Value |",
                "|---|---:|",
            ]
        )
        for metric in result.metrics:
            lines.append(f"| {metric.name} | {metric.value:.4f} |")
        lines.extend(["", "Top retrieved contexts:", ""])
        for context in result.retrieved_contexts[:3]:
            preview = context.text.replace("\n", " ")[:220]
            lines.append(
                f"- Rank {context.rank}: `{context.document_id}` "
                f"score={context.score:.4f} — {preview}..."
            )
        lines.append("")

    output.write_text("\n".join(lines), encoding="utf-8")
