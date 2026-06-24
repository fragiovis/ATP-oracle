"""CLI entry point for the serving/inference layer."""

from __future__ import annotations

import json
import logging

import typer
from rich import print as rprint
from rich.logging import RichHandler

from serving.predictor import MatchPredictor, MatchRequest

logging.basicConfig(level=logging.INFO, handlers=[RichHandler(rich_tracebacks=True)])
app = typer.Typer(help="ATP Oracle – inference CLI")


@app.command()
def predict(
    player_a: str = typer.Option(..., "--player-a", "-a", help="Player A name"),
    player_b: str = typer.Option(..., "--player-b", "-b", help="Player B name"),
    surface: str = typer.Option("hard", "--surface", "-s"),
    tournament_category: str = typer.Option("ATP 250", "--category", "-c"),
    round_: str = typer.Option("R32", "--round", "-r"),
    best_of: int = typer.Option(3, "--best-of"),
):
    """Run pre-match predictions for player_a vs player_b."""
    req = MatchRequest(
        player_a=player_a,
        player_b=player_b,
        surface=surface,
        tournament_category=tournament_category,
        round=round_,
        best_of=best_of,
    )
    predictor = MatchPredictor()
    result = predictor.predict(req)
    rprint(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    app()
