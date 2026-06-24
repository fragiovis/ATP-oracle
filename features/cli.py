"""CLI entry point for the feature engineering pipeline."""
from __future__ import annotations
import logging
from pathlib import Path

import typer
from rich.logging import RichHandler

from features.pipeline import FeaturePipeline
from features.player_profiler import PlayerProfiler
from storage.writer import StorageWriter

logging.basicConfig(level=logging.INFO, handlers=[RichHandler(rich_tracebacks=True)])
logger = logging.getLogger(__name__)
app = typer.Typer(help="ATP Oracle – feature engineering CLI")

LOCAL_BACKUP = Path(".cache/pre_match_features.parquet")
LOCAL_PLAYERS_BACKUP = Path(".cache/player_profiles.parquet")


def _build_name_to_id_map(matches) -> dict[str, str]:
    """Build name → player_id from the unified matches table.

    Sackmann numeric IDs take priority; td_* IDs fill in players that only
    appear in tennis-data (no Sackmann history).
    """
    name_map: dict[str, str] = {}
    td_map: dict[str, str] = {}  # lower-priority td_* entries

    for col_id, col_name in [("player_a_id", "player_a_name"), ("player_b_id", "player_b_name")]:
        if col_id not in matches.columns or col_name not in matches.columns:
            continue
        for pid, name in zip(matches[col_id].astype(str), matches[col_name].fillna("")):
            if not name.strip():
                continue
            target = td_map if pid.startswith("td_") else name_map
            full = name.strip().lower()
            target[full] = pid
            parts = name.strip().split()
            if len(parts) > 1:
                surname = " ".join(parts[1:]).lower()
                target[surname] = pid

    # Merge: Sackmann entries override td_* for the same key
    merged = {**td_map, **name_map}
    return merged


def _resolve_name(ranking_name: str, name_map: dict[str, str]) -> str | None:
    """Resolve a ranking name like 'J. Sinner' or 'Felix Auger-Aliassime' to a Sackmann ID."""
    rn = ranking_name.strip()

    # Try full lowercase exact match
    if rn.lower() in name_map:
        return name_map[rn.lower()]

    parts = rn.split()

    # Remove leading initials (e.g. "J.", "F.A.") to get surname-only parts
    surname_parts = [p for p in parts if not (p.endswith(".") and len(p) <= 5)]

    # Try full surname (handles "Auger-Aliassime", "De Minaur", etc.)
    if surname_parts:
        surname = " ".join(surname_parts).lower()
        if surname in name_map:
            return name_map[surname]

    # Try last token only
    if parts:
        last = parts[-1].lower()
        if last in name_map:
            return name_map[last]

    # Try hyphenated surname parts: "Auger-Aliassime" → try "aliassime", "auger-aliassime"
    if surname_parts:
        last_part = surname_parts[-1].lower()
        if "-" in last_part:
            for chunk in last_part.split("-"):
                if chunk in name_map:
                    return name_map[chunk]

    # Partial match: find any key containing the last surname part
    if surname_parts:
        last_lower = surname_parts[-1].lower().replace("-", " ")
        for key, pid in name_map.items():
            if last_lower in key:
                return pid

    return None


@app.command("features")
def run_features(
    from_local: bool = typer.Option(
        False, "--from-local",
        help="Skip recomputation – upload the local backup file to S3 directly.",
    )
):
    """Build the gold-layer pre_match_features table and write to S3."""
    writer = StorageWriter()

    if from_local:
        if not LOCAL_BACKUP.exists():
            typer.echo(f"No local backup found at {LOCAL_BACKUP}. Run without --from-local first.")
            raise typer.Exit(1)
        logger.info("Loading local backup from %s …", LOCAL_BACKUP)
        import pandas as pd
        features_df = pd.read_parquet(LOCAL_BACKUP)
        logger.info("Loaded %d rows from local backup.", len(features_df))
    else:
        logger.info("Reading silver tables from S3…")
        matches     = writer.read_silver_table("matches")
        match_stats = writer.read_silver_table("match_player_stats")
        players     = writer.read_silver_table("players")
        h2h         = writer.read_silver_table("head_to_head_history")

        pipeline = FeaturePipeline()
        features, tournament_profiles = pipeline.build(matches, match_stats, players, h2h)
        features_df = features.reset_index()

        # Always save local backup BEFORE attempting S3 upload
        LOCAL_BACKUP.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_parquet(LOCAL_BACKUP, index=False)
        logger.info("Local backup saved to %s (%d rows)", LOCAL_BACKUP, len(features_df))

        # Write tournament profiles (best-effort)
        if not tournament_profiles.empty:
            try:
                writer.write_tournament_profiles(tournament_profiles.reset_index())
                logger.info("Tournament profiles written (%d tournaments)", len(tournament_profiles))
            except Exception as e:
                logger.warning("Could not write tournament profiles to S3: %s", e)

    try:
        writer.write_gold_features(features_df)
        typer.echo(f"Done – {len(features_df)} feature rows written to S3.")
    except Exception as e:
        typer.echo(f"[ERROR] S3 upload failed: {e}", err=True)
        typer.echo(f"Local backup is at {LOCAL_BACKUP.resolve()}")
        typer.echo("When credentials are refreshed, run:  atp-features features --from-local")
        raise typer.Exit(1)


@app.command("profiles")
def run_profiles(
    from_local: bool = typer.Option(
        False, "--from-local",
        help="Skip recomputation – upload the local backup file to S3 directly.",
    ),
    from_gold: bool = typer.Option(
        False, "--from-gold",
        help="Extract profiles from the local gold feature table (no S3 needed).",
    ),
    top_n: int = typer.Option(
        200, "--top-n",
        help="Number of top-ranked players to profile.",
    ),
):
    """Build player profiles for the top N ATP players and write to S3."""
    import pandas as pd

    writer = StorageWriter()

    if from_local:
        if not LOCAL_PLAYERS_BACKUP.exists():
            typer.echo(f"No local backup found at {LOCAL_PLAYERS_BACKUP}.")
            raise typer.Exit(1)
        profiles_df = pd.read_parquet(LOCAL_PLAYERS_BACKUP)
        logger.info("Loaded %d profiles from local backup.", len(profiles_df))

    elif from_gold:
        # Extract profiles from the cached gold feature table — works offline
        if not LOCAL_BACKUP.exists():
            typer.echo(f"No gold feature cache at {LOCAL_BACKUP}. Run 'atp-features features' first.")
            raise typer.Exit(1)
        logger.info("Extracting player profiles from gold feature table…")
        from features.player_profiler import extract_profiles_from_gold
        profiles = extract_profiles_from_gold(
            pd.read_parquet(LOCAL_BACKUP), top_n=top_n
        )
        profiles_df = profiles.reset_index()

        LOCAL_PLAYERS_BACKUP.parent.mkdir(parents=True, exist_ok=True)
        profiles_df.to_parquet(LOCAL_PLAYERS_BACKUP, index=False)
        logger.info("Local backup saved to %s (%d players)", LOCAL_PLAYERS_BACKUP, len(profiles_df))

    else:
        logger.info("Reading silver tables from S3…")
        matches     = writer.read_silver_table("matches")
        match_stats = writer.read_silver_table("match_player_stats")
        players     = writer.read_silver_table("players")
        rankings    = writer.read_silver_table("rankings_snapshot_current_top200")

        # Unify IDs first (same as feature pipeline)
        from features.pipeline import FeaturePipeline
        matches = FeaturePipeline._unify_player_ids(matches)

        # Build name → Sackmann ID mapping from unified matches
        name_to_id = _build_name_to_id_map(matches)

        # 1) Try live ATP rankings (current top-N from atptour.com)
        # 2) Fall back to Sackmann snapshot if live fetch fails
        # 3) Last resort: most active players in last 12 months
        from scraper.atp_rankings_live import fetch_atp_rankings
        live_rankings = fetch_atp_rankings(top_n)

        if live_rankings:
            ranked_names = [r["player_name"] for r in live_rankings]
            source = "live ATP rankings"
        elif not rankings.empty and "player_name" in rankings.columns:
            if "rank" in rankings.columns:
                rankings = rankings.sort_values("rank")
            ranked_names = rankings["player_name"].head(top_n).tolist()
            source = "Sackmann snapshot"
        else:
            ranked_names = []
            source = None

        if ranked_names:
            player_ids = []
            for name in ranked_names:
                pid = _resolve_name(name, name_to_id)
                if pid:
                    player_ids.append(pid)
                else:
                    logger.debug("Could not resolve ranking player: %s", name)
            logger.info(
                "Resolved %d / %d ranked players to match IDs (source: %s).",
                len(player_ids), len(ranked_names), source,
            )
        else:
            # Last resort: most active players in last 12 months
            logger.warning("No rankings available – using most active players in last 12 months.")
            matches["match_date"] = pd.to_datetime(matches["match_date"], errors="coerce")
            cutoff = pd.Timestamp.today() - pd.DateOffset(months=12)
            recent = matches[matches["match_date"] >= cutoff]
            if recent.empty:
                recent = matches
            all_ids = pd.concat([
                recent["player_a_id"].astype(str),
                recent["player_b_id"].astype(str),
            ])
            all_ids = all_ids[~all_ids.str.startswith("td_")]
            player_ids = all_ids.value_counts().head(top_n).index.tolist()
            logger.info("Selected %d players from recent match activity.", len(player_ids))

        profiler = PlayerProfiler()
        profiles = profiler.build_profiles(matches, match_stats, player_ids, players)
        profiles_df = profiles.reset_index()

        # Save local backup
        LOCAL_PLAYERS_BACKUP.parent.mkdir(parents=True, exist_ok=True)
        profiles_df.to_parquet(LOCAL_PLAYERS_BACKUP, index=False)
        logger.info("Local backup saved to %s (%d players)", LOCAL_PLAYERS_BACKUP, len(profiles_df))

    try:
        writer.write_player_profiles(profiles_df)
        typer.echo(f"Done – {len(profiles_df)} player profiles written to S3.")
    except Exception as e:
        typer.echo(f"[ERROR] S3 upload failed: {e}", err=True)
        typer.echo(f"Local backup is at {LOCAL_PLAYERS_BACKUP.resolve()}")
        raise typer.Exit(1)


LOCAL_TOURNAMENT_HISTORY_BACKUP = Path(".cache/player_tournament_history.parquet")


@app.command("tournament-history")
def run_tournament_history():
    """Build player×tournament history table and write to S3."""
    import pandas as pd
    from features.player_profiler import build_player_tournament_history

    writer = StorageWriter()

    logger.info("Reading silver matches from S3…")
    matches = writer.read_silver_table("matches")

    from features.pipeline import FeaturePipeline
    matches = FeaturePipeline._unify_player_ids(matches)

    history_df = build_player_tournament_history(matches)
    if history_df.empty:
        typer.echo("[ERROR] No tournament history computed.", err=True)
        raise typer.Exit(1)

    LOCAL_TOURNAMENT_HISTORY_BACKUP.parent.mkdir(parents=True, exist_ok=True)
    history_df.to_parquet(LOCAL_TOURNAMENT_HISTORY_BACKUP, index=False)
    logger.info("Local backup saved to %s", LOCAL_TOURNAMENT_HISTORY_BACKUP)

    try:
        writer.write_player_tournament_history(history_df)
        typer.echo(f"Done – {len(history_df)} player×tournament rows written to S3.")
    except Exception as e:
        typer.echo(f"[ERROR] S3 upload failed: {e}", err=True)
        typer.echo(f"Local backup is at {LOCAL_TOURNAMENT_HISTORY_BACKUP.resolve()}")
        raise typer.Exit(1)


@app.command("all")
def run_all(
    top_n: int = typer.Option(200, "--top-n"),
):
    """Build feature table, player profiles, and tournament history."""
    run_features(from_local=False)
    run_profiles(from_local=False, top_n=top_n)
    run_tournament_history()


# Keep backwards compat: bare `atp-features` still works
@app.callback(invoke_without_command=True)
def default(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        run_features(from_local=False)


if __name__ == "__main__":
    app()
