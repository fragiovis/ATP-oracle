"""Data acquisition via Jeff Sackmann's tennis_atp public dataset.

Repository: https://github.com/JeffSackmann/tennis_atp
License:    Creative Commons Attribution-NonCommercial-ShareAlike 4.0

This module replaces ATP website scraping for match history.
It downloads CSV files directly from GitHub (no JavaScript, no anti-scraping).

CSV columns (match files):
  tourney_id, tourney_name, surface, draw_size, tourney_level, tourney_date,
  match_num, winner_id, winner_name, winner_hand, winner_ht, winner_ioc,
  loser_id,  loser_name,  loser_hand,  loser_ht,  loser_ioc,
  score, best_of, round, minutes,
  w_ace, w_df, w_svpt, w_1stIn, w_1stWon, w_2ndWon, w_SvGms, w_bpSaved, w_bpFaced,
  l_ace, l_df, l_svpt, l_1stIn, l_1stWon, l_2ndWon, l_SvGms, l_bpSaved, l_bpFaced,
  winner_rank, winner_rank_points, loser_rank, loser_rank_points
"""

from __future__ import annotations

import io
import logging
from datetime import date
from typing import Any

import pandas as pd
import requests

from config.settings import settings

logger = logging.getLogger(__name__)

BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"

# tourney_level → tournament_category
LEVEL_MAP = {
    "G": "Grand Slam",
    "M": "ATP 1000",
    "A": "ATP 500",   # covers both 500 and 250; refined below
    "F": "ATP Finals",
    "D": "Davis Cup",
    "C": "Challenger",
}

# Surfaces are always one of clay/hard/grass — carpet collapses to "hard" and
# the indoor/outdoor distinction is carried by the `indoor` flag instead.
SURFACE_MAP = {
    "Clay":  "clay",
    "Hard":  "hard",
    "Grass": "grass",
    "Carpet": "hard",
}


class SackmannLoader:
    """Download and normalise Jeff Sackmann's tennis_atp CSV dataset."""

    def __init__(self):
        cfg = settings.scraper
        current_year = date.today().year
        self.years = list(range(current_year - cfg.years_back, current_year + 1))
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "atp-oracle/0.1 (educational project)"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_all(
        self,
        player_ids: set[str] | None = None,
        years: list[int] | None = None,
    ) -> dict[str, Any]:
        """
        Download match + player data.

        Parameters
        ----------
        player_ids : if provided, filter matches to only those involving
                     these Sackmann player IDs. Pass None to keep all.
        years      : override the year range to load. Defaults to self.years.
        """
        players_df   = self._load_players()
        rankings_df  = self._load_current_rankings()
        matches_df   = self._load_matches(player_ids, years=years)

        # Split matches into base match info + per-player stats
        match_rows, stat_rows = self._split_matches(matches_df)

        return {
            "players":     players_df.to_dict("records"),
            "rankings":    rankings_df.to_dict("records"),
            "matches":     match_rows,
            "match_stats": stat_rows,
        }

    # ------------------------------------------------------------------
    # Players
    # ------------------------------------------------------------------

    def _load_players(self) -> pd.DataFrame:
        url = f"{BASE_URL}/atp_players.csv"
        logger.info("Downloading players list: %s", url)
        df = self._fetch_csv(url)
        if df.empty:
            return df
        df = df.rename(columns={
            "player_id": "player_id",
            "name_first": "first_name",
            "name_last":  "last_name",
            "hand":       "hand",
            "dob":        "birth_date",
            "ioc":        "country",
        })
        df["player_name"] = df["first_name"].fillna("") + " " + df["last_name"].fillna("")
        df["player_name"] = df["player_name"].str.strip()
        df["player_id"]   = df["player_id"].astype(str)
        return df[["player_id", "player_name", "country", "hand", "birth_date"]]

    # ------------------------------------------------------------------
    # Rankings
    # ------------------------------------------------------------------

    def _load_current_rankings(self) -> pd.DataFrame:
        url = f"{BASE_URL}/atp_rankings_current.csv"
        logger.info("Downloading current rankings: %s", url)
        df = self._fetch_csv(url)
        if df.empty:
            return df

        df.columns = [c.strip() for c in df.columns]
        # Expected columns: ranking_date, rank, player, points
        col_map = {}
        for c in df.columns:
            cl = c.lower()
            if "date" in cl:    col_map[c] = "snapshot_date"
            elif "rank" in cl and "point" not in cl: col_map[c] = "rank"
            elif "player" in cl or "id" in cl:       col_map[c] = "player_id"
            elif "point" in cl:                      col_map[c] = "ranking_points"
        df = df.rename(columns=col_map)

        if "snapshot_date" in df.columns:
            df["snapshot_date"] = pd.to_datetime(
                df["snapshot_date"].astype(str), format="%Y%m%d", errors="coerce"
            ).dt.strftime("%Y-%m-%d")
            # Keep only the most recent snapshot
            df = df[df["snapshot_date"] == df["snapshot_date"].max()]

        df["player_id"] = df["player_id"].astype(str)
        n = settings.scraper.player_universe_size
        df = df.sort_values("rank").head(n).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Matches
    # ------------------------------------------------------------------

    def _load_matches(self, player_ids: set[str] | None, years: list[int] | None = None) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for year in (years if years is not None else self.years):
            df = self._load_year(year)
            if df is not None and not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame()

        all_matches = pd.concat(frames, ignore_index=True)

        if player_ids:
            mask = (
                all_matches["winner_id"].astype(str).isin(player_ids)
                | all_matches["loser_id"].astype(str).isin(player_ids)
            )
            all_matches = all_matches[mask]

        logger.info("Total matches loaded: %d", len(all_matches))
        return all_matches

    def _load_year(self, year: int) -> pd.DataFrame | None:
        url = f"{BASE_URL}/atp_matches_{year}.csv"
        logger.info("Downloading matches %d …", year)
        df = self._fetch_csv(url)
        if df is None or df.empty:
            logger.warning("No data for year %d", year)
            return None
        df["year"] = year
        return df

    # ------------------------------------------------------------------
    # Split into match rows + stat rows
    # ------------------------------------------------------------------

    def _split_matches(
        self, df: pd.DataFrame
    ) -> tuple[list[dict], list[dict]]:
        if df.empty:
            return [], []

        match_rows: list[dict] = []
        stat_rows:  list[dict] = []

        for _, row in df.iterrows():
            wid = str(row.get("winner_id", ""))
            lid = str(row.get("loser_id",  ""))
            mid = f"{row.get('tourney_id','')}_{row.get('match_num','')}_{wid}_{lid}"

            # Normalise date
            raw_date = str(row.get("tourney_date", ""))
            match_date = ""
            if len(raw_date) == 8:
                match_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"

            surf_raw = str(row.get("surface", ""))
            surface = SURFACE_MAP.get(surf_raw, "hard")
            level   = str(row.get("tourney_level", "A"))
            cat     = LEVEL_MAP.get(level, "ATP 250")
            best_of = int(row["best_of"]) if pd.notna(row.get("best_of")) else 3
            indoor  = surf_raw == "Carpet"

            match_rows.append({
                "match_id":            mid,
                "match_date":          match_date,
                "year":                int(row.get("year", 0)),
                "tournament_id":       str(row.get("tourney_id", "")),
                "tournament_name":     str(row.get("tourney_name", "")),
                "tournament_category": cat,
                "round":               str(row.get("round", "")),
                "surface":             surface,
                "indoor":              indoor,
                "best_of":             best_of,
                "player_a_id":         wid,
                "player_b_id":         lid,
                "player_a_name":       str(row.get("winner_name", "")),
                "player_b_name":       str(row.get("loser_name",  "")),
                "winner_id":           wid,
                "scoreline":           str(row.get("score", "")),
            })

            # Winner stats (return stats derived from loser's serve data)
            ws = self._extract_stats(row, prefix="w_", player_id=wid, match_id=mid,
                                     opp_prefix="l_")
            if ws:
                stat_rows.append(ws)

            # Loser stats (return stats derived from winner's serve data)
            ls = self._extract_stats(row, prefix="l_", player_id=lid, match_id=mid,
                                     opp_prefix="w_")
            if ls:
                stat_rows.append(ls)

        return match_rows, stat_rows

    @staticmethod
    def _extract_stats(
        row, prefix: str, player_id: str, match_id: str, opp_prefix: str = ""
    ) -> dict | None:
        def safe(col, pfx=None):
            v = row.get((pfx or prefix) + col)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return None
            try:
                return float(v)
            except (ValueError, TypeError):
                return None

        svpt = safe("svpt")
        in1  = safe("1stIn")
        won1 = safe("1stWon")
        won2 = safe("2ndWon")
        bps  = safe("bpSaved")
        bpf  = safe("bpFaced")
        sgs  = safe("SvGms")

        stats = {
            "match_id":              match_id,
            "player_id":             player_id,
            "aces":                  safe("ace"),
            "double_faults":         safe("df"),
            "service_pts_total":     svpt,
            "first_serve_in":        in1,
            "first_serve_pts_won":   won1,
            "second_serve_pts_won":  won2,
            "service_games":         sgs,
            "bp_saved":              bps,
            "bp_faced":              bpf,
        }

        # Derived serve percentages
        if svpt and svpt > 0:
            if in1 is not None:
                stats["first_serve_pct"] = in1 / svpt
            if won1 is not None and in1 and in1 > 0:
                stats["first_serve_pts_won_pct"] = won1 / in1
            if won2 is not None:
                sec_svpt = svpt - (in1 or 0)
                if sec_svpt > 0:
                    stats["second_serve_pts_won_pct"] = won2 / sec_svpt
            total_won = (won1 or 0) + (won2 or 0)
            stats["service_pts_won_pct"] = total_won / svpt

        if bpf and bpf > 0 and bps is not None:
            stats["bp_saved_pct"] = bps / bpf

        # Return stats — derived from opponent's serve data
        if opp_prefix:
            opp_svpt = safe("svpt", opp_prefix)
            opp_won1 = safe("1stWon", opp_prefix)
            opp_won2 = safe("2ndWon", opp_prefix)
            opp_bpf  = safe("bpFaced", opp_prefix)
            opp_bps  = safe("bpSaved", opp_prefix)

            if opp_svpt and opp_svpt > 0 and opp_won1 is not None and opp_won2 is not None:
                opp_total_won = (opp_won1 or 0) + (opp_won2 or 0)
                stats["return_pts_won_pct"] = 1.0 - (opp_total_won / opp_svpt)

            # bp_converted = break points won / break points faced by opponent
            if opp_bpf and opp_bpf > 0 and opp_bps is not None:
                stats["bp_converted_pct"] = (opp_bpf - opp_bps) / opp_bpf

        # Skip if all stat values are None
        if all(v is None for k, v in stats.items() if k not in ("match_id", "player_id")):
            return None

        return stats

    # ------------------------------------------------------------------
    # HTTP helper
    # ------------------------------------------------------------------

    def _fetch_csv(self, url: str) -> pd.DataFrame:
        try:
            resp = self._session.get(url, timeout=30)
            resp.raise_for_status()
            return pd.read_csv(io.StringIO(resp.text), low_memory=False)
        except Exception as exc:
            logger.warning("Could not fetch %s: %s", url, exc)
            return pd.DataFrame()
