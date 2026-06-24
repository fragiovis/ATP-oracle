"""CLI entry point for the scraping pipeline."""

from __future__ import annotations

import json
import logging
import os

import typer
from rich.logging import RichHandler

from scraper.pipeline import ScrapingPipeline
from storage.writer import StorageWriter

level_name = os.environ.get("ATP_LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, level_name, logging.INFO)
logging.basicConfig(
    level=log_level,
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)
app = typer.Typer(help="ATP Oracle – scraping CLI")


@app.command()
def run(
    output: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Write raw JSON output to this local file (optional, mainly for debugging)",
    ),
    skip_stats: bool = typer.Option(
        False,
        "--skip-stats",
        help="Skip per-match stats scraping (faster dry run)",
    ),
    no_bios: bool = typer.Option(
        False,
        "--no-bios",
        help="Skip player bio scraping (much faster; bios filled from Sackmann data instead)",
    ),
):
    """Run the full ATP scraping pipeline (downloads all historical data)."""
    pipeline = ScrapingPipeline()
    data = pipeline.run(skip_bios=no_bios)

    if output:
        with open(output, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Raw data written to %s", output)

    if data:
        logger.info("Writing data to S3 (raw JSON + silver Parquet)…")
        writer = StorageWriter()
        writer.write_all(data)
        logger.info("S3 write complete.")

    typer.echo(
        f"Done – rankings={len(data.get('rankings', []))}, "
        f"players={len(data.get('players', []))}, "
        f"matches={len(data.get('matches', []))}, "
        f"match_stats={len(data.get('match_stats', []))}"
    )


@app.command()
def update(
    rebuild_silver: bool = typer.Option(
        True,
        "--rebuild-silver/--no-silver",
        help="Rebuild the silver layer after adding new matches",
    ),
    rebuild_gold: bool = typer.Option(
        False,
        "--rebuild-gold/--no-gold",
        help="Also rebuild gold features and player profiles (slower)",
    ),
):
    """Incremental update: download new matches from tennis-data.co.uk and merge with S3.

    Examples:
        atp-scrape update                  # download + rebuild silver
        atp-scrape update --rebuild-gold   # also rebuild features/profiles
        atp-scrape update --no-silver      # only update raw matches
    """
    from scraper.updater import MatchUpdater

    updater = MatchUpdater()
    stats = updater.update(
        rebuild_silver=rebuild_silver,
        rebuild_gold=rebuild_gold,
    )

    typer.echo(
        f"Update complete – "
        f"existing={stats['existing']}, "
        f"fresh={stats['fresh_matches']}, "
        f"new={stats['new']}"
    )


@app.command()
def scrape_stats(
    year: int = typer.Option(
        2025,
        "--year",
        "-y",
        help="Year to scrape matches + stats for",
    ),
    max_tournaments: int = typer.Option(
        None,
        "--max-tournaments",
        "-n",
        help="Limit number of tournaments (for testing)",
    ),
    output: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Write scraped data to this local JSON file",
    ),
    upload: bool = typer.Option(
        True,
        "--upload/--no-upload",
        help="Upload scraped data to S3 raw layer",
    ),
):
    """Scrape matches + detailed stats from atptour.com using Playwright.

    Uses a headless browser to bypass Cloudflare and intercept the ATP
    Hawkeye API. Produces both match records AND per-player statistics
    (aces, serve %, break points, return stats, etc.).

    Supports checkpointing — safe to interrupt and resume.

    Examples:
        atp-scrape scrape-stats --year 2025                   # full 2025
        atp-scrape scrape-stats --year 2025 -n 3              # first 3 tournaments
        atp-scrape scrape-stats --year 2025 -o data.json      # save locally
        atp-scrape scrape-stats --year 2025 --no-upload       # local only
    """
    from scraper.atp_stats import ATPStatsScraper
    from datetime import date

    scraper = ATPStatsScraper()
    result = scraper.scrape_year(year, max_tournaments=max_tournaments)
    matches = result.get("matches", [])
    match_stats = result.get("match_stats", [])

    if output:
        import json as json_mod
        with open(output, "w") as f:
            json_mod.dump(result, f, indent=2, default=str)
        logger.info("Data written to %s", output)

    if upload and (matches or match_stats):
        writer = StorageWriter()
        today = date.today().isoformat()

        if matches:
            matches_path = f"{writer.cfg.raw_matches_path}/atp_hawkeye_{year}_{today}.json"
            logger.info("Uploading %d matches to %s", len(matches), matches_path)
            writer.client.write_jsonlines(matches, matches_path)

        if match_stats:
            stats_path = f"{writer.cfg.raw_match_stats_path}/atp_hawkeye_{year}_{today}.json"
            logger.info("Uploading %d stat records to %s", len(match_stats), stats_path)
            writer.client.write_jsonlines(match_stats, stats_path)

    typer.echo(
        f"Done – {len(matches)} matches, {len(match_stats)} stat records scraped for {year}."
    )


if __name__ == "__main__":
    app()
