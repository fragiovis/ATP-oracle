"""Loader for tennis-data.co.uk Excel files (2025 / 2026 and beyond).

URL pattern: http://www.tennis-data.co.uk/{year}/{year}.xlsx

Columns present:
  ATP, Location, Tournament, Date, Series, Court, Surface, Round, Best of,
  Winner, Loser, WRank, LRank, WPts, LPts,
  W1..W5, L1..L5,   ← games per set (winner / loser)
  Wsets, Lsets, Comment

No per-match serve stats (aces, df, etc.) — only result + ranking data.
Scoreline is reconstructed from the set-score columns.
"""

from __future__ import annotations

import hashlib
import io
import logging
from datetime import date
from typing import Any

import pandas as pd
import requests

from config.settings import settings

logger = logging.getLogger(__name__)

BASE_URL = "http://www.tennis-data.co.uk/{year}/{year}.xlsx"

SERIES_MAP = {
    "grand slam":   "Grand Slam",
    "masters":      "ATP 1000",
    "masters 1000": "ATP 1000",
    "atp500":       "ATP 500",
    "atp 500":      "ATP 500",
    "atp250":       "ATP 250",
    "atp 250":      "ATP 250",
    "international": "ATP 250",
    "international gold": "ATP 500",
}

# Surfaces are always one of clay/hard/grass — carpet and indoor hard collapse
# to "hard" and the indoor/outdoor distinction is carried by `indoor` instead.
SURFACE_MAP = {
    "clay":   "clay",
    "hard":   "hard",
    "grass":  "grass",
    "carpet": "hard",
    "indoor hard": "hard",
}

# Raw surface values that imply an indoor court
_INDOOR_SURFACES = {"carpet", "indoor hard"}


class TennisDataLoader:
    """Load recent ATP match data from tennis-data.co.uk for years not yet
    covered by the Sackmann dataset."""

    def __init__(self, years: list[int] | None = None):
        if years is None:
            current = date.today().year
            # Use for years AFTER Sackmann's last available year (2024)
            self.years = [y for y in range(current - settings.scraper.years_back, current + 1)
                          if y >= 2025]
        else:
            self.years = years
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "atp-oracle/0.1 (educational)"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_all(self) -> dict[str, list[dict]]:
        """Return {'matches': [...], 'match_stats': []} for all configured years."""
        all_matches: list[dict] = []

        for year in self.years:
            rows = self._load_year(year)
            all_matches.extend(rows)
            logger.info("tennis-data %d: %d matches loaded", year, len(rows))

        return {"matches": all_matches, "match_stats": []}

    # ------------------------------------------------------------------

    def _load_year(self, year: int) -> list[dict]:
        url = BASE_URL.format(year=year)
        logger.info("Downloading tennis-data %d: %s", year, url)
        try:
            resp = self._session.get(url, timeout=30)
            resp.raise_for_status()
            df = pd.read_excel(io.BytesIO(resp.content), engine="openpyxl")
        except Exception as exc:
            logger.warning("Could not load tennis-data %d: %s", year, exc)
            return []

        return self._parse(df, year)

    def _parse(self, df: pd.DataFrame, year: int) -> list[dict]:
        rows: list[dict] = []

        for _, row in df.iterrows():
            winner = str(row.get("Winner", "") or "").strip()
            loser  = str(row.get("Loser",  "") or "").strip()
            if not winner or not loser or winner == "nan":
                continue

            # Synthetic stable IDs from names (no numeric Sackmann IDs here)
            wid = "td_" + hashlib.md5(winner.lower().encode()).hexdigest()[:8]
            lid = "td_" + hashlib.md5(loser.lower().encode()).hexdigest()[:8]
            mid = f"td_{year}_{wid}_{lid}_{abs(hash(str(row.get('Date',''))+str(row.get('Round',''))))%100000}"

            # Date
            raw_date = row.get("Date")
            if pd.notna(raw_date):
                try:
                    match_date = pd.to_datetime(raw_date).strftime("%Y-%m-%d")
                except Exception:
                    match_date = None
            else:
                match_date = None

            # Surface
            surf_raw = str(row.get("Surface", "") or "").lower()
            surface  = SURFACE_MAP.get(surf_raw, "hard")
            indoor   = surf_raw in _INDOOR_SURFACES or str(row.get("Court", "")).lower() == "indoor"

            # Tournament category
            series_raw = str(row.get("Series", "") or "").lower().strip()
            category   = SERIES_MAP.get(series_raw, "ATP 250")

            # Best of
            bo_raw = row.get("Best of")
            best_of = int(bo_raw) if pd.notna(bo_raw) else (5 if category == "Grand Slam" else 3)

            # Scoreline reconstruction from W1/L1 … W5/L5
            scoreline = self._build_scoreline(row)

            # Rankings
            wrank = self._safe_int(row.get("WRank"))
            lrank = self._safe_int(row.get("LRank"))

            # Bookmaker odds (winner = player_a, loser = player_b)
            b365w = self._safe_float(row.get("B365W"))
            b365l = self._safe_float(row.get("B365L"))
            psw   = self._safe_float(row.get("PSW"))
            psl   = self._safe_float(row.get("PSL"))
            maxw  = self._safe_float(row.get("MaxW"))
            maxl  = self._safe_float(row.get("MaxL"))

            rows.append({
                "match_id":            mid,
                "match_date":          match_date,
                "year":                year,
                "tournament_id":       str(row.get("ATP", "")).strip(),
                "tournament_name":     str(row.get("Tournament", "") or "").strip(),
                "tournament_category": category,
                "round":               self._normalise_round(str(row.get("Round", "") or "")),
                "surface":             surface,
                "indoor":              indoor,
                "best_of":             best_of,
                "player_a_id":         wid,
                "player_b_id":         lid,
                "player_a_name":       winner,
                "player_b_name":       loser,
                "winner_id":           wid,
                "scoreline":           scoreline,
                "winner_rank":         wrank,
                "loser_rank":          lrank,
                # Bookmaker odds: player_a = winner, player_b = loser
                "b365_a":              b365w,
                "b365_b":              b365l,
                "ps_a":                psw,
                "ps_b":                psl,
                "max_a":               maxw,
                "max_b":               maxl,
            })

        return rows

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_scoreline(row) -> str:
        """Reconstruct '6-4 7-5 3-6' from W1/L1..W5/L5 columns."""
        sets = []
        for i in range(1, 6):
            w = row.get(f"W{i}")
            l = row.get(f"L{i}")
            if pd.notna(w) and pd.notna(l) and (int(w) > 0 or int(l) > 0):
                sets.append(f"{int(w)}-{int(l)}")
        return " ".join(sets)

    @staticmethod
    def _normalise_round(raw: str) -> str:
        mapping = {
            "the final": "F", "final": "F",
            "semifinals": "SF", "semi-finals": "SF", "semi finals": "SF",
            "quarterfinals": "QF", "quarter-finals": "QF",
            "round of 16": "R16", "4th round": "R16",
            "round of 32": "R32", "3rd round": "R32",
            "round of 64": "R64", "2nd round": "R64",
            "round of 128": "R128", "1st round": "R128",
            "robin round": "RR",
        }
        return mapping.get(raw.lower().strip(), raw)

    @staticmethod
    def _safe_int(v) -> int | None:
        try:
            return int(v) if pd.notna(v) else None
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_float(v) -> float | None:
        try:
            f = float(v)
            return f if f > 0 else None
        except (ValueError, TypeError):
            return None
