"""Incremental match updater — tennis-data.co.uk only.

Downloads the latest match data from tennis-data.co.uk, merges with
existing raw matches on S3, rebuilds silver layer.
"""

from __future__ import annotations

import logging
from typing import Any

from storage.writer import StorageWriter

logger = logging.getLogger(__name__)


class MatchUpdater:
    """Fetch recent matches from tennis-data.co.uk and merge into S3."""

    def __init__(self):
        self._writer = StorageWriter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        rebuild_silver: bool = True,
        rebuild_gold: bool = False,
    ) -> dict[str, int]:
        """
        Run the incremental update pipeline.

        1. Load existing raw matches from S3.
        2. Download fresh matches from tennis-data.co.uk.
        3. Deduplicate and find new matches.
        4. Merge and write updated raw matches to S3.
        5. If rebuild_silver: re-normalise the full matches table.
        6. If rebuild_gold: rebuild features + player profiles.

        Returns summary stats.
        """
        stats = {"existing": 0, "fresh_matches": 0, "new": 0}

        # Step 1 – Load existing match IDs from silver parquet (fast, columnar)
        # We only need match_id + a fingerprint for deduplication — no need to
        # download the 63 MB raw JSONL which contains all Sackmann history.
        logger.info("=== Step 1: Loading existing match IDs from silver table ===")
        existing_ids: set[str] = set()
        existing_fps: set[str] = set()

        def _fp(m: dict) -> str:
            a = (m.get("player_a_name") or "").lower().strip()
            b = (m.get("player_b_name") or "").lower().strip()
            if a > b:
                a, b = b, a
            return f"{a}|{b}|{str(m.get('match_date',''))[:10]}|{m.get('round','')}"

        try:
            import pandas as pd
            silver_df = self._writer.read_silver_table("matches")
            existing_ids = set(silver_df["match_id"].astype(str).tolist())
            # Build fingerprints only from td_* matches (tennis-data rows that may
            # have non-unique IDs across runs)
            td_mask = silver_df["match_id"].astype(str).str.startswith("td_")
            td_rows = silver_df[td_mask]
            for row in td_rows.itertuples(index=False):
                a = str(getattr(row, "player_a_name", "") or "").lower().strip()
                b = str(getattr(row, "player_b_name", "") or "").lower().strip()
                if a > b:
                    a, b = b, a
                d = str(getattr(row, "match_date", "") or "")[:10]
                r = str(getattr(row, "round", "") or "")
                existing_fps.add(f"{a}|{b}|{d}|{r}")
            logger.info(
                "Existing matches: %d (td_* fingerprints: %d)",
                len(existing_ids), len(existing_fps),
            )
            stats["existing"] = len(existing_ids)
        except Exception as e:
            logger.warning("Could not load silver matches: %s. Dedup will be empty.", e)

        # Step 2 – Download from tennis-data.co.uk
        logger.info("=== Step 2: Downloading fresh data from tennis-data.co.uk ===")
        from scraper.tennisdata import TennisDataLoader
        td = TennisDataLoader()
        td_data = td.load_all()
        fresh_matches = td_data.get("matches", [])
        logger.info("tennis-data matches downloaded: %d", len(fresh_matches))
        stats["fresh_matches"] = len(fresh_matches)

        # Step 3 – Deduplicate
        logger.info("=== Step 3: Deduplicating ===")
        new_matches = [
            m for m in fresh_matches
            if m.get("match_id", "") not in existing_ids and _fp(m) not in existing_fps
        ]
        stats["new"] = len(new_matches)
        logger.info("New matches to add: %d", len(new_matches))

        if not new_matches:
            logger.info("No new matches found. Dataset is up to date.")
            return stats

        # Step 4 – Write fresh tennis-data matches to S3 as a dated snapshot.
        # We no longer maintain a merged JSONL of all history — the silver parquet
        # is the source of truth. We just write the full fresh tennis-data dump so
        # it can be re-loaded on the next run if needed.
        logger.info("=== Step 4: Writing fresh tennis-data snapshot to S3 ===")
        from datetime import date as _date
        snapshot_path = f"{self._writer.cfg.raw_matches_path}/tennisdata_{_date.today()}.jsonl"
        self._writer.client.write_jsonlines(fresh_matches, snapshot_path)
        logger.info("Snapshot written: %d matches → %s", len(fresh_matches), snapshot_path)

        # Step 4b – Extract and persist bookmaker odds to silver
        logger.info("=== Step 4b: Persisting match odds to silver ===")
        self._rebuild_odds(fresh_matches)

        # Step 5 – Rebuild silver layer from full fresh tennis-data dump
        # (not just new_matches — _refresh_td_years replaces stale td_* rows)
        if rebuild_silver:
            logger.info("=== Step 5: Rebuilding silver layer ===")
            self._rebuild_silver(fresh_matches)

        # Step 6 – Rebuild gold
        if rebuild_gold:
            logger.info("=== Step 6: Rebuilding gold layer ===")
            self._rebuild_gold()

        logger.info("=== Update complete ===")
        return stats

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _rebuild_odds(self, fresh_matches: list[dict]) -> None:
        """Extract bookmaker odds from fresh matches and merge into silver match_odds table.

        Saves player names and date alongside odds so the backtest can join
        on a name+date fingerprint against the gold feature table (which uses
        Sackmann IDs incompatible with tennis-data td_* IDs).
        """
        import pandas as pd

        keep_cols = [
            "match_id", "match_date", "year",
            "player_a_name", "player_b_name",
            "b365_a", "b365_b", "ps_a", "ps_b", "max_a", "max_b",
        ]
        rows = [
            {c: m.get(c) for c in keep_cols}
            for m in fresh_matches
            if any(m.get(c) is not None for c in ("b365_a", "b365_b", "ps_a", "ps_b"))
        ]
        if not rows:
            logger.info("No odds data found in fresh matches — skipping.")
            return

        new_df = pd.DataFrame(rows).dropna(subset=["match_id"])

        # Build a join fingerprint: lastname_a|lastname_b|date (order-independent)
        def _lastname(name: str) -> str:
            # tennis-data format: "Djokovic N." → "djokovic"
            return str(name or "").split()[0].lower().rstrip(".")

        def _fp(row) -> str:
            a = _lastname(row.get("player_a_name", ""))
            b = _lastname(row.get("player_b_name", ""))
            if a > b:
                a, b = b, a
            d = str(row.get("match_date", "") or "")[:10]
            return f"{a}|{b}|{d}"

        new_df["odds_fp"] = [_fp(m) for m in rows if m.get("match_id")]
        logger.info("New odds rows: %d", len(new_df))

        # Merge with existing odds (replace rows for same match_ids)
        try:
            existing = self._writer.read_match_odds()
            keep = existing[~existing["match_id"].isin(new_df["match_id"])]
            merged = pd.concat([keep, new_df], ignore_index=True)
            logger.info("Odds table: kept %d existing + %d new = %d total", len(keep), len(new_df), len(merged))
        except Exception:
            merged = new_df

        self._writer.write_match_odds(merged)

    @staticmethod
    def _refresh_td_years(merged_raw: list[dict], fresh_td: list[dict]) -> list[dict]:
        """Replace td_* entries for scraped years with fresh versions.

        Corrects stale rows (NaT dates, changed scores) written in previous runs.
        """
        if not fresh_td:
            return merged_raw

        years_refreshed = {
            str(m.get("year", ""))
            for m in fresh_td
            if str(m.get("match_id", "")).startswith("td_")
        }
        if not years_refreshed:
            return merged_raw

        cleaned = [
            m for m in merged_raw
            if not (
                str(m.get("match_id", "")).startswith("td_")
                and str(m.get("year", "")) in years_refreshed
            )
        ]
        result = cleaned + fresh_td
        logger.info(
            "Refreshed td_* entries for years %s: %d → %d records",
            sorted(years_refreshed), len(merged_raw), len(result),
        )
        return result

    def _rebuild_silver(self, fresh_td_matches: list[dict]) -> None:
        """Merge fresh tennis-data matches into silver and rewrite to S3.

        Strategy:
          1. Load existing silver matches (parquet — fast).
          2. Normalise the fresh tennis-data dump into a DataFrame.
          3. Drop old td_* rows for the same years as the fresh dump.
          4. Append the fresh rows and write back to S3.
        """
        import pandas as pd
        from storage.normaliser import DataNormaliser

        normaliser = DataNormaliser()

        # Normalise fresh tennis-data matches
        players: list[dict] = []
        rankings: list[dict] = []
        try:
            players_df = self._writer.read_silver_table("players")
            players = players_df.to_dict("records")
        except Exception:
            pass
        try:
            rankings_df = self._writer.read_silver_table("rankings_snapshot_current_top200")
            rankings = rankings_df.to_dict("records")
        except Exception:
            pass

        raw_data = {
            "matches": fresh_td_matches,
            "match_stats": [],
            "players": players,
            "rankings": rankings,
        }
        new_tables = normaliser.normalise(raw_data)
        new_matches_df = new_tables.get("matches", pd.DataFrame())

        if new_matches_df.empty:
            logger.warning("No matches after normalisation — skipping silver rebuild.")
            return

        # Load existing silver and replace td_* rows for the refreshed years
        try:
            existing_df = self._writer.read_silver_table("matches")
            td_years = set(
                new_matches_df.loc[
                    new_matches_df["match_id"].astype(str).str.startswith("td_"), "match_date"
                ].astype(str).str[:4].unique()
            )
            keep_mask = ~(
                existing_df["match_id"].astype(str).str.startswith("td_") &
                existing_df["match_date"].astype(str).str[:4].isin(td_years)
            )
            merged_df = pd.concat([existing_df[keep_mask], new_matches_df], ignore_index=True)
            logger.info(
                "Silver matches: kept %d existing + %d new = %d total",
                keep_mask.sum(), len(new_matches_df), len(merged_df),
            )
        except Exception as e:
            logger.warning("Could not load existing silver matches (%s) — using only fresh data.", e)
            merged_df = new_matches_df

        s3_path = f"{self._writer.cfg.silver_path}/matches.parquet"
        logger.info("Writing silver matches (%d rows) → %s", len(merged_df), s3_path)
        self._writer.client.write_parquet(merged_df, s3_path)

        # Write other normalised tables (players, rankings, stats) if present
        for table_name, df in new_tables.items():
            if table_name == "matches" or df.empty:
                continue
            s3_path = f"{self._writer.cfg.silver_path}/{table_name}.parquet"
            logger.info("Writing silver table %s (%d rows)", table_name, len(df))
            self._writer.client.write_parquet(df, s3_path)

    def _rebuild_gold(self) -> None:
        """Rebuild features and player profiles from updated silver data."""
        from features.pipeline import FeaturePipeline
        from features.player_profiler import PlayerProfiler, build_player_tournament_history
        from features.cli import _resolve_name, _build_name_to_id_map
        from scraper.atp_rankings_live import fetch_atp_rankings

        writer = self._writer

        matches = writer.read_silver_table("matches")
        match_stats = writer.read_silver_table("match_player_stats")
        players = writer.read_silver_table("players")
        h2h = writer.read_silver_table("head_to_head_history")

        logger.info("Building pre-match features...")
        pipeline = FeaturePipeline()
        features_df, tournament_profiles = pipeline.build(matches, match_stats, players, h2h)
        if not features_df.empty:
            writer.write_gold_features(features_df.reset_index())
            logger.info("Gold features written: %d rows", len(features_df))
        if not tournament_profiles.empty:
            try:
                writer.write_tournament_profiles(tournament_profiles.reset_index())
            except Exception as e:
                logger.warning("Could not write tournament profiles: %s", e)

        logger.info("Building player profiles...")
        matches_unified = FeaturePipeline._unify_player_ids(matches)
        name_to_id = _build_name_to_id_map(matches_unified)
        live_rankings = fetch_atp_rankings(200)
        if live_rankings:
            ranked_names = [r["player_name"] for r in live_rankings]
        else:
            ranked_names = []

        player_ids = [pid for name in ranked_names if (pid := _resolve_name(name, name_to_id))]
        logger.info("Resolved %d / %d ranked players.", len(player_ids), len(ranked_names))

        profiler = PlayerProfiler()
        profiles_df = profiler.build_profiles(matches_unified, match_stats, player_ids, players_meta=players)
        if not profiles_df.empty:
            writer.write_player_profiles(profiles_df.reset_index())
            logger.info("Player profiles written: %d players", len(profiles_df))

        logger.info("Building player tournament history...")
        history_df = build_player_tournament_history(matches_unified)
        if not history_df.empty:
            writer.write_player_tournament_history(history_df)
            logger.info("Tournament history written: %d rows", len(history_df))
