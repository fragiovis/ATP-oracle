"""CLI entry point for model training."""
from __future__ import annotations
import logging
import typer
from rich.logging import RichHandler
from models.trainer import ModelTrainer
from storage.writer import StorageWriter

logging.basicConfig(level=logging.INFO, handlers=[RichHandler(rich_tracebacks=True)])
logger = logging.getLogger(__name__)
app = typer.Typer(help="ATP Oracle – model training CLI")


@app.command()
def run():
    """Train all target models from the gold features table."""
    writer = StorageWriter()
    logger.info("Loading gold features from HDFS…")
    features = writer.read_gold_features()
    trainer = ModelTrainer()
    results = trainer.train_all(features)
    for model_id, meta in results.items():
        status = meta.get("status", "?")
        typer.echo(f"  {model_id}: {status}")


if __name__ == "__main__":
    app()
