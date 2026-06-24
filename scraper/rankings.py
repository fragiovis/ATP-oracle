"""Scrape the current ATP Top-N singles rankings page.

Returns a list of dicts with: rank, player_id, player_name, country,
and the player profile URL.

ATP rankings URL (as of 2024):
  https://www.atptour.com/en/rankings/singles
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any

from bs4 import BeautifulSoup

from config.settings import settings
from scraper.base import BaseScraper, ScraperError

logger = logging.getLogger(__name__)

ATP_RANKINGS_URL = "https://www.atptour.com/en/rankings/singles?rankRange=1-{n}"
ATP_BASE_URL = "https://www.atptour.com"


class RankingsScraper(BaseScraper):
    """Scrape ATP singles rankings page and extract the Top-N player list."""

    def __init__(self):
        super().__init__(checkpoint_name="rankings")
        self.n = settings.scraper.player_universe_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrape(self) -> list[dict[str, Any]]:
        """Return a list of player dicts for the current top-N ranked players."""
        url = ATP_RANKINGS_URL.format(n=self.n)
        logger.info("Scraping ATP rankings (top %d): %s", self.n, url)

        try:
            html = self.get_html(url)
        except ScraperError:
            logger.error("Could not retrieve rankings page – aborting.")
            return []

        players = self._parse(html)
        logger.info("Found %d players in rankings.", len(players))
        return players

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse(self, html: str) -> list[dict[str, Any]]:
        soup = BeautifulSoup(html, "lxml")
        players: list[dict[str, Any]] = []

        # The ATP rankings table has <tr> rows – try multiple selectors
        rows = soup.select("table.rankings-table tbody tr")
        if not rows:
            # Fallback: any table row with a rank cell
            rows = soup.select("tr[class*='rankings-table']") or soup.select("tr")

        for row in rows:
            player = self._parse_row(row)
            if player and len(players) < self.n:
                players.append(player)

        return players

    def _parse_row(self, row) -> dict[str, Any] | None:
        """Extract fields from a single rankings table row."""
        try:
            cells = row.find_all("td")
            if len(cells) < 4:
                return None

            # Rank
            rank_text = cells[0].get_text(strip=True)
            rank = int(re.sub(r"[^\d]", "", rank_text))
            if rank < 1 or rank > self.n:
                return None

            # Player name + profile link
            name_cell = row.find("a", href=re.compile(r"/en/players/"))
            if name_cell is None:
                return None
            player_name = name_cell.get_text(strip=True)
            profile_href = name_cell["href"]
            profile_url = ATP_BASE_URL + profile_href if profile_href.startswith("/") else profile_href

            # Player ID (extracted from URL slug, e.g. /en/players/jannik-sinner/s0ag/overview)
            player_id = self._extract_player_id(profile_href)

            # Country
            country_img = row.find("img", alt=True)
            country = country_img["alt"] if country_img else ""

            return {
                "snapshot_date": date.today().isoformat(),
                "rank": rank,
                "player_id": player_id,
                "player_name": player_name,
                "country": country,
                "profile_url": profile_url,
            }
        except Exception as exc:
            logger.debug("Could not parse ranking row: %s", exc)
            return None

    @staticmethod
    def _extract_player_id(href: str) -> str:
        """
        Extract the stable 4-char ATP player ID from a profile URL.
        Example: /en/players/jannik-sinner/s0ag/overview → 's0ag'
        """
        parts = href.strip("/").split("/")
        # Typical structure: en / players / <slug> / <id> / overview
        if len(parts) >= 4:
            return parts[3]
        return parts[-1]
