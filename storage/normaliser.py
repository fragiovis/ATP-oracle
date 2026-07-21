"""Normalise raw scraped data into clean relational tables (silver layer).

Produces pandas DataFrames for:
  - players
  - rankings_snapshot_current_top200
  - matches
  - match_player_stats
  - head_to_head_history
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import date
from typing import Any

import pandas as pd

from storage.canonical import canonical_surface, canonical_tournament

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scoreline parsing utilities
# ---------------------------------------------------------------------------

def parse_scoreline(scoreline: str) -> dict[str, Any]:
    """
    Parse a raw scoreline string into structured fields.

    Returns:
      sets_a, sets_b                       – number of sets won
      games_a, games_b                     – total games won per player
      total_games                          – sum of all games
      has_tiebreak                         – at least one tiebreak occurred
      deciding_set                         – match went to a deciding set
      set_scores                           – list of (ga, gb) tuples per set
    """
    result: dict[str, Any] = {
        "sets_a": 0, "sets_b": 0,
        "games_a": 0, "games_b": 0,
        "total_games": 0,
        "has_tiebreak": False,
        "deciding_set": False,
        "set_scores": [],
    }

    if not scoreline or not isinstance(scoreline, str):
        return result

    # Normalise: remove RET/W/O annotations, strip extra whitespace
    clean = re.sub(r"\b(ret|w/o|retired|def\.?)\b.*", "", scoreline, flags=re.I).strip()

    # Split into individual set scores: "6-4", "7-6(5)", "3-6"
    set_pattern = re.compile(r"(\d+)-(\d+)(?:\((\d+)\))?")
    sets = set_pattern.findall(clean)

    for ga_s, gb_s, tb_s in sets:
        ga, gb = int(ga_s), int(gb_s)
        result["set_scores"].append((ga, gb))
        result["games_a"] += ga
        result["games_b"] += gb
        result["total_games"] += ga + gb
        if ga > gb:
            result["sets_a"] += 1
        else:
            result["sets_b"] += 1
        # A set ending 7-6 (or 6-7) is by definition decided via tiebreak —
        # derive the flag structurally instead of relying on the "(n)" marker,
        # which is inconsistently present across scraped sources.
        if (ga == 7 and gb == 6) or (ga == 6 and gb == 7):
            result["has_tiebreak"] = True
        elif tb_s:
            result["has_tiebreak"] = True

    total_sets = result["sets_a"] + result["sets_b"]
    if total_sets == 3:
        result["deciding_set"] = True  # Bo3
    elif total_sets == 5:
        result["deciding_set"] = True  # Bo5

    return result


def generate_match_id(player_a_id: str, player_b_id: str, match_date: str, round_: str) -> str:
    raw = f"{min(player_a_id, player_b_id)}_{max(player_a_id, player_b_id)}_{match_date}_{round_}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Normaliser
# ---------------------------------------------------------------------------

class DataNormaliser:
    """Convert raw scraped dicts into clean pandas DataFrames."""

    def normalise(self, raw: dict[str, Any]) -> dict[str, pd.DataFrame]:
        """
        Parameters
        ----------
        raw : dict returned by ScrapingPipeline.run()

        Returns
        -------
        dict mapping table name → DataFrame
        """
        tables: dict[str, pd.DataFrame] = {}
        tables["players"] = self._normalise_players(raw.get("players", []))
        tables["rankings_snapshot_current_top200"] = self._normalise_rankings(
            raw.get("rankings", [])
        )
        matches_df, stats_df, h2h_df = self._normalise_matches(
            raw.get("matches", []), raw.get("match_stats", [])
        )
        tables["matches"] = matches_df
        tables["match_player_stats"] = stats_df
        tables["head_to_head_history"] = h2h_df
        return tables

    # ------------------------------------------------------------------
    # Players
    # ------------------------------------------------------------------

    def _normalise_players(self, raw_players: list[dict]) -> pd.DataFrame:
        if not raw_players:
            return pd.DataFrame()
        df = pd.DataFrame(raw_players)
        keep = [c for c in [
            "player_id", "player_name", "country", "birth_date",
            "height_cm", "hand", "turned_pro",
        ] if c in df.columns]
        df = df[keep].drop_duplicates(subset=["player_id"])
        return df

    # ------------------------------------------------------------------
    # Rankings
    # ------------------------------------------------------------------

    def _normalise_rankings(self, raw_rankings: list[dict]) -> pd.DataFrame:
        if not raw_rankings:
            return pd.DataFrame()
        df = pd.DataFrame(raw_rankings)
        keep = [c for c in ["snapshot_date", "rank", "player_id", "player_name", "country"]
                if c in df.columns]
        return df[keep]

    # ------------------------------------------------------------------
    # Matches + stats + H2H
    # ------------------------------------------------------------------

    def _normalise_matches(
        self,
        raw_matches: list[dict],
        raw_stats: list[dict],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if not raw_matches:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        match_rows: list[dict] = []
        stat_rows: list[dict] = []

        # Build a lookup from match_id → list of stat rows (Sackmann schema)
        stats_by_match: dict[str, list[dict]] = {}
        for s in raw_stats:
            mid_key = s.get("match_id", "")
            if mid_key:
                stats_by_match.setdefault(mid_key, []).append(s)

        # Deduplicate matches: use the match_id that comes from the source, or
        # generate one from (pa, pb, date, round) for deduplication.
        seen_ids: set[str] = set()

        for m in raw_matches:
            # Support both Sackmann/TennisData schema (player_a_id / player_b_id)
            # and the legacy ATP-scraper schema (player_id / opponent_id).
            player_a_id = m.get("player_a_id") or m.get("player_id", "")
            player_b_id = m.get("player_b_id") or m.get("opponent_id", "")
            match_date  = m.get("match_date", "")
            round_      = m.get("round", "")

            if not (player_a_id and player_b_id):
                continue

            # Prefer source-supplied match_id; fall back to generated one
            src_mid = m.get("match_id", "")
            mid = src_mid if src_mid else generate_match_id(
                player_a_id, player_b_id, match_date, round_
            )
            if mid in seen_ids:
                continue
            seen_ids.add(mid)

            scoreline = m.get("scoreline", "")
            parsed = parse_scoreline(scoreline)

            # Determine winner_id: use source field if present, else derive from player_won
            winner_id = m.get("winner_id", "")
            if not winner_id:
                player_a_won = m.get("player_won", False)
                winner_id = player_a_id if player_a_won else player_b_id

            # Surface normalisation
            surface = self._normalise_surface(m.get("surface", ""))

            match_row = {
                "match_id": mid,
                "match_date": match_date,
                "tournament_id": m.get("tournament_id", ""),
                "tournament_name": canonical_tournament(m.get("tournament_name", "")),
                "tournament_category": m.get("tournament_category", ""),
                "round": round_,
                "surface": surface,
                "indoor": m.get("indoor", None),
                "best_of": m.get("best_of", None),
                "player_a_id": player_a_id,
                "player_b_id": player_b_id,
                "player_a_name": m.get("player_a_name", ""),
                "player_b_name": m.get("player_b_name", ""),
                "winner_id": winner_id,
                "scoreline": scoreline,
                "sets_a": parsed["sets_a"],
                "sets_b": parsed["sets_b"],
                "total_games": parsed["total_games"],
                "has_tiebreak": parsed["has_tiebreak"],
                "deciding_set": parsed["deciding_set"],
            }
            match_rows.append(match_row)

            # Attach stats: look up by the source match_id (Sackmann schema)
            for stat in stats_by_match.get(src_mid, []):
                pid = stat.get("player_id", "")
                if pid:
                    # Re-key the stat with the (possibly regenerated) match_id
                    stat_row = {k: v for k, v in stat.items() if k != "match_id"}
                    stat_rows.append({"match_id": mid, **stat_row})

        matches_df = pd.DataFrame(match_rows)
        stats_df = pd.DataFrame(stat_rows) if stat_rows else pd.DataFrame()

        # H2H
        h2h_df = self._build_h2h(matches_df) if not matches_df.empty else pd.DataFrame()

        # Parse dates
        if not matches_df.empty and "match_date" in matches_df.columns:
            matches_df["match_date"] = pd.to_datetime(
                matches_df["match_date"], errors="coerce"
            )

        return matches_df, stats_df, h2h_df

    @staticmethod
    def _build_h2h(matches: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, m in matches.iterrows():
            rows.append({
                "player_a_id": m["player_a_id"],
                "player_b_id": m["player_b_id"],
                "match_id": m["match_id"],
                "winner_id": m["winner_id"],
                "surface": m.get("surface", ""),
                "match_date": m.get("match_date"),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _normalise_surface(raw: str) -> str:
        """Map to one of clay/hard/grass. Indoor hard and carpet become "hard";
        the indoor/outdoor distinction lives in the separate `indoor` column."""
        return canonical_surface(raw)
