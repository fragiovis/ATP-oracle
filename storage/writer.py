"""Write raw and normalised data to S3 (with local cache fallback for reads).

Ties together the normaliser and the S3 client.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import settings
from storage.s3_client import S3Client
from storage.normaliser import DataNormaliser

LOCAL_FEATURES_CACHE = Path(".cache/pre_match_features.parquet")
LOCAL_PLAYERS_CACHE = Path(".cache/player_profiles.parquet")
LOCAL_TOURNAMENTS_CACHE = Path(".cache/tournament_profiles.parquet")
LOCAL_PLAYER_TOURNAMENT_CACHE = Path(".cache/player_tournament_history.parquet")
LOCAL_RANKINGS_CACHE = Path(".cache/rankings_snapshot.parquet")

logger = logging.getLogger(__name__)


class StorageWriter:
    """Persist raw + normalised ATP data to S3."""

    def __init__(self):
        self.client = S3Client()
        self.normaliser = DataNormaliser()
        self.cfg = settings.s3

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_raw(self, raw: dict[str, Any]) -> None:
        """Write each raw section as JSON on S3 under /data/raw/."""
        today = date.today().isoformat()

        sections = {
            "rankings":    self.cfg.raw_rankings_path,
            "players":     self.cfg.raw_players_path,
            "matches":     self.cfg.raw_matches_path,
            "match_stats": self.cfg.raw_match_stats_path,
        }

        for key, dir_path in sections.items():
            records = raw.get(key, [])
            if not records:
                logger.info("No records for %s – skipping.", key)
                continue
            s3_path = f"{dir_path}/{today}.json"
            logger.info("Writing raw %s (%d records) → %s", key, len(records), s3_path)
            self.client.write_jsonlines(records, s3_path)

    def write_silver(self, raw: dict[str, Any]) -> None:
        """Normalise raw data and write Parquet tables under /data/silver/."""
        tables = self.normaliser.normalise(raw)

        for table_name, df in tables.items():
            if df.empty:
                logger.warning("Empty table: %s – skipping.", table_name)
                continue
            s3_path = f"{self.cfg.silver_path}/{table_name}.parquet"
            logger.info("Writing silver table %s (%d rows) → %s", table_name, len(df), s3_path)
            self.client.write_parquet(df, s3_path)

    def write_all(self, raw: dict[str, Any]) -> None:
        """Convenience: write raw JSON *and* silver Parquet."""
        self.write_raw(raw)
        self.write_silver(raw)

    # ------------------------------------------------------------------
    # Raw matches (used by incremental updater)
    # ------------------------------------------------------------------

    def write_raw_matches(self, records: list[dict]) -> None:
        """Write the full raw matches list as a single JSONL file on S3."""
        s3_path = f"{self.cfg.raw_matches_path}/all_matches.jsonl"
        logger.info("Writing raw matches (%d records) → %s", len(records), s3_path)
        self.client.write_jsonlines(records, s3_path)

    def read_raw_matches(self) -> list[dict]:
        """Read the raw matches JSONL file from S3."""
        import json
        s3_path = f"{self.cfg.raw_matches_path}/all_matches.jsonl"
        try:
            key = self.client._key(s3_path)
            resp = self.client._s3.get_object(Bucket=self.client._bucket, Key=key)
            text = resp["Body"].read().decode("utf-8")
            return [json.loads(line) for line in text.strip().split("\n") if line.strip()]
        except Exception as e:
            if "NoSuchKey" in str(e) or "404" in str(e):
                logger.info("No existing raw matches file found on S3.")
                return []
            raise

    # ------------------------------------------------------------------
    # Read helpers (used by feature pipeline)
    # ------------------------------------------------------------------

    def read_silver_table(self, table_name: str):
        """Read a silver Parquet table from S3 into a pandas DataFrame."""
        s3_path = f"{self.cfg.silver_path}/{table_name}.parquet"
        logger.info("Reading %s", s3_path)
        return self.client.read_parquet(s3_path)

    def write_gold_features(self, df, filename: str = "pre_match_features.parquet") -> None:
        s3_path = f"{self.cfg.gold_features_path}/{filename}"
        logger.info("Writing gold features (%d rows) → %s", len(df), s3_path)
        self.client.write_parquet(df, s3_path)

    def read_gold_features(self, filename: str = "pre_match_features.parquet"):
        s3_path = f"{self.cfg.gold_features_path}/{filename}"
        try:
            return self.client.read_parquet(s3_path)
        except Exception as e:
            if "AccessDenied" in str(e) or "ExpiredToken" in str(e) or "credentials" in str(e).lower():
                local = LOCAL_FEATURES_CACHE if filename == "pre_match_features.parquet" \
                    else Path(".cache") / filename
                if local.exists():
                    logger.warning("S3 access denied – loading gold features from local cache %s", local)
                    return pd.read_parquet(local)
            raise

    # ------------------------------------------------------------------
    # Player profiles (one row per player in top 200)
    # ------------------------------------------------------------------

    def write_player_profiles(self, df: pd.DataFrame, filename: str = "player_profiles.parquet") -> None:
        s3_path = f"{self.cfg.players_profiles_path}/{filename}"
        logger.info("Writing player profiles (%d players) → %s", len(df), s3_path)
        self.client.write_parquet(df, s3_path)

    def read_player_profiles(self, filename: str = "player_profiles.parquet") -> pd.DataFrame:
        s3_path = f"{self.cfg.players_profiles_path}/{filename}"
        try:
            return self.client.read_parquet(s3_path)
        except Exception as e:
            if "AccessDenied" in str(e) or "ExpiredToken" in str(e) or "credentials" in str(e).lower():
                local = LOCAL_PLAYERS_CACHE if filename == "player_profiles.parquet" \
                    else Path(".cache") / filename
                if local.exists():
                    logger.warning("S3 access denied – loading player profiles from local cache %s", local)
                    return pd.read_parquet(local)
            raise

    # ------------------------------------------------------------------
    # Tournament profiles
    # ------------------------------------------------------------------

    def write_tournament_profiles(self, df: pd.DataFrame, filename: str = "tournament_profiles.parquet") -> None:
        s3_path = f"{self.cfg.tournaments_path}/{filename}"
        logger.info("Writing tournament profiles (%d tournaments) → %s", len(df), s3_path)
        self.client.write_parquet(df, s3_path)
        LOCAL_TOURNAMENTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(LOCAL_TOURNAMENTS_CACHE, index=True)

    def read_tournament_profiles(self, filename: str = "tournament_profiles.parquet") -> pd.DataFrame:
        s3_path = f"{self.cfg.tournaments_path}/{filename}"
        try:
            return self.client.read_parquet(s3_path)
        except Exception as e:
            if "AccessDenied" in str(e) or "ExpiredToken" in str(e) or "credentials" in str(e).lower():
                if LOCAL_TOURNAMENTS_CACHE.exists():
                    logger.warning("S3 access denied – loading tournament profiles from local cache")
                    return pd.read_parquet(LOCAL_TOURNAMENTS_CACHE)
            raise

    # ------------------------------------------------------------------
    # Player tournament history  (player_id × tournament_name → win_rate, matches)
    # ------------------------------------------------------------------

    def write_player_tournament_history(self, df: pd.DataFrame, filename: str = "player_tournament_history.parquet") -> None:
        s3_path = f"{self.cfg.players_profiles_path}/{filename}"
        logger.info("Writing player tournament history (%d rows) → %s", len(df), s3_path)
        self.client.write_parquet(df, s3_path)
        LOCAL_PLAYER_TOURNAMENT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(LOCAL_PLAYER_TOURNAMENT_CACHE, index=False)

    def read_player_tournament_history(self, filename: str = "player_tournament_history.parquet") -> pd.DataFrame:
        s3_path = f"{self.cfg.players_profiles_path}/{filename}"
        try:
            return self.client.read_parquet(s3_path)
        except Exception as e:
            if "AccessDenied" in str(e) or "ExpiredToken" in str(e) or "credentials" in str(e).lower():
                if LOCAL_PLAYER_TOURNAMENT_CACHE.exists():
                    logger.warning("S3 access denied – loading player tournament history from local cache")
                    return pd.read_parquet(LOCAL_PLAYER_TOURNAMENT_CACHE)
            raise

    def read_rankings_snapshot(self, filename: str = "rankings_snapshot_current_top200.parquet") -> pd.DataFrame:
        """Read the latest ATP rankings snapshot. Falls back to local cache or live scrape."""
        s3_path = f"{self.cfg.silver_path}/{filename}"
        try:
            return self.client.read_parquet(s3_path)
        except Exception as e:
            if "AccessDenied" in str(e) or "ExpiredToken" in str(e) or "credentials" in str(e).lower():
                if LOCAL_RANKINGS_CACHE.exists():
                    logger.warning("S3 access denied – loading rankings from local cache")
                    return pd.read_parquet(LOCAL_RANKINGS_CACHE)
            logger.warning("Could not read rankings snapshot from S3: %s", e)
            return pd.DataFrame()

    def write_match_odds(self, df: pd.DataFrame, filename: str = "match_odds.parquet") -> None:
        s3_path = f"{self.cfg.silver_path}/{filename}"
        logger.info("Writing match odds (%d rows) → %s", len(df), s3_path)
        self.client.write_parquet(df, s3_path)

    def read_match_odds(self, filename: str = "match_odds.parquet") -> pd.DataFrame:
        s3_path = f"{self.cfg.silver_path}/{filename}"
        return self.client.read_parquet(s3_path)

    # ------------------------------------------------------------------
    # Model artifact helpers (used by ModelTrainer)
    # ------------------------------------------------------------------

    def write_model(self, data: bytes, run_id: str, model_id: str) -> None:
        s3_path = f"{self.cfg.models_path}/{run_id}/{model_id}/model.pkl"
        logger.info("Writing model %s → %s", model_id, s3_path)
        self.client.write_bytes(data, s3_path)

    def read_model(self, run_id: str, model_id: str) -> bytes:
        s3_path = f"{self.cfg.models_path}/{run_id}/{model_id}/model.pkl"
        return self.client.read_bytes(s3_path)

    def latest_model_run(self) -> str | None:
        return self.client.latest_subdir(self.cfg.models_path)
