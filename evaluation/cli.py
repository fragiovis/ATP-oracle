"""CLI entry point for model evaluation."""
from __future__ import annotations
import logging
import typer
from rich.logging import RichHandler
from evaluation.runner import EvaluationRunner
from storage.writer import StorageWriter

logging.basicConfig(level=logging.INFO, handlers=[RichHandler(rich_tracebacks=True)])
app = typer.Typer(help="ATP Oracle – evaluation CLI")


@app.command()
def run(run_id: str = typer.Option(None, "--run-id", help="Evaluation run identifier")):
    """Evaluate all trained models and write metrics + plots to HDFS."""
    writer = StorageWriter()
    features = writer.read_gold_features()
    runner = EvaluationRunner()
    results = runner.run_all(features, run_id=run_id)
    for model_id, metrics in results.items():
        overall = metrics.get("overall", {})
        typer.echo(f"  {model_id}: {overall}")


if __name__ == "__main__":
    app()
