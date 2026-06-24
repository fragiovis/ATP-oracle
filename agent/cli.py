"""CLI entry point for the primary agent (interactive REPL or one-shot)."""

from __future__ import annotations

import logging

import typer
from rich import print as rprint
from rich.logging import RichHandler

from agent.agent import PrimaryAgent

logging.basicConfig(level=logging.WARNING, handlers=[RichHandler(rich_tracebacks=True)])
app = typer.Typer(help="ATP Oracle – natural-language agent CLI")
_agent: PrimaryAgent | None = None


def _get_agent() -> PrimaryAgent:
    global _agent
    if _agent is None:
        _agent = PrimaryAgent()
    return _agent


@app.command()
def ask(
    query: str = typer.Argument(None, help="Natural-language prediction query"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Start REPL mode"),
):
    """Ask a pre-match prediction question in natural language."""
    agent = _get_agent()

    if interactive or not query:
        typer.echo("ATP Oracle – type your prediction query. Ctrl-C to exit.\n")
        while True:
            try:
                q = typer.prompt("You")
                _run_and_print(agent, q)
            except (KeyboardInterrupt, EOFError):
                typer.echo("\nBye!")
                break
    else:
        _run_and_print(agent, query)


def _run_and_print(agent: PrimaryAgent, query: str) -> None:
    result = agent.run(query)
    rprint(f"\n[bold cyan]Agent:[/bold cyan] {result.answer}\n")
    if result.prediction:
        import json
        rprint("[dim]" + json.dumps(result.prediction, indent=2) + "[/dim]")


if __name__ == "__main__":
    app()
