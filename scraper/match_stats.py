"""Scrape per-match player statistics from the ATP match-stats page.

Scraped fields (per player per match):
  aces, double_faults,
  first_serve_pct,
  first_serve_pts_won_pct, second_serve_pts_won_pct,
  service_pts_won, service_pts_total,
  return_pts_won, return_pts_total,
  total_pts_won,
  bp_saved, bp_faced,
  bp_converted, bp_created
"""

from __future__ import annotations

import logging
import re
from typing import Any

from bs4 import BeautifulSoup

from scraper.base import BaseScraper, ScraperError

logger = logging.getLogger(__name__)

ATP_BASE_URL = "https://www.atptour.com"


class MatchStatsScraper(BaseScraper):
    """Scrape ATP match-level player statistics for a given match stats URL."""

    def __init__(self):
        super().__init__(checkpoint_name="match_stats")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrape_match(self, stats_url: str, match_id: str) -> dict[str, Any]:
        """
        Return a dict with keys 'player_a' and 'player_b', each containing
        their stat dicts.  Returns an empty dict on failure (graceful degradation).
        """
        if not stats_url:
            return {}

        if self.checkpoint.is_done(match_id):
            logger.debug("Skip (checkpoint): match_id=%s", match_id)
            return {}

        try:
            html = self.get_html(stats_url)
        except ScraperError:
            logger.warning("Could not fetch match stats: %s", stats_url)
            return {}

        result = self._parse(html, match_id)
        if result:
            self.checkpoint.mark_done(match_id)
        return result

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse(self, html: str, match_id: str) -> dict[str, Any]:
        soup = BeautifulSoup(html, "lxml")
        stats: dict[str, Any] = {"match_id": match_id, "player_a": {}, "player_b": {}}

        # Look for the stats table; ATP uses various selectors over time
        stat_rows = (
            soup.select("table.match-stats-table tr")
            or soup.select("div.match-stats-row")
            or soup.select("tr[class*='stat']")
        )

        for row in stat_rows:
            self._parse_stat_row(row, stats)

        return stats if (stats["player_a"] or stats["player_b"]) else {}

    def _parse_stat_row(self, row, stats: dict[str, Any]) -> None:
        """Extract a single stat row (label + two player values)."""
        try:
            cells = row.find_all("td") or row.find_all("div", class_=re.compile("value|stat"))
            if len(cells) < 3:
                return

            label = cells[1].get_text(strip=True).lower() if len(cells) > 1 else ""
            val_a_raw = cells[0].get_text(strip=True)
            val_b_raw = cells[2].get_text(strip=True)

            field = self._label_to_field(label)
            if not field:
                return

            stats["player_a"][field] = self._parse_value(val_a_raw, field)
            stats["player_b"][field] = self._parse_value(val_b_raw, field)
        except Exception as exc:
            logger.debug("Stat row parse error: %s", exc)

    # ------------------------------------------------------------------
    # Field mapping & value parsing
    # ------------------------------------------------------------------

    _LABEL_MAP: dict[str, str] = {
        "aces": "aces",
        "double faults": "double_faults",
        "double fault": "double_faults",
        "1st serve": "first_serve_pct",
        "first serve": "first_serve_pct",
        "1st serve points won": "first_serve_pts_won_pct",
        "2nd serve points won": "second_serve_pts_won_pct",
        "service points won": "service_pts_won_pct",
        "total service points won": "service_pts_won_pct",
        "break points saved": "bp_saved_pct",
        "break point saved": "bp_saved_pct",
        "break points converted": "bp_converted_pct",
        "return points won": "return_pts_won_pct",
        "total points won": "total_pts_won",
        "winners": "winners",
        "unforced errors": "unforced_errors",
        "net points won": "net_pts_won_pct",
        "receiving points won": "return_pts_won_pct",
    }

    @classmethod
    def _label_to_field(cls, label: str) -> str:
        return cls._LABEL_MAP.get(label.strip().lower(), "")

    @staticmethod
    def _parse_value(raw: str, field: str) -> float | int | None:
        """
        Parse a raw stat string.
        - Percentage: "73% (22/30)" → 0.73 and store numerator/denominator separately
        - Integer: "12" → 12
        - Fraction: "4/6" → keep the fraction as float
        """
        raw = raw.strip()
        if not raw or raw in ("-", "N/A", ""):
            return None

        # Pattern: "73% (22/30)" or "73%"
        pct_match = re.match(r"(\d+(?:\.\d+)?)\s*%", raw)
        if pct_match:
            return float(pct_match.group(1)) / 100.0

        # Pattern: "22/30"
        frac_match = re.match(r"(\d+)\s*/\s*(\d+)", raw)
        if frac_match:
            num, den = int(frac_match.group(1)), int(frac_match.group(2))
            return num / den if den else None

        # Plain integer / float
        try:
            return int(raw)
        except ValueError:
            pass
        try:
            return float(raw)
        except ValueError:
            return None
