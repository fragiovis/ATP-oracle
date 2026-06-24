"""Scrape ATP player profile pages.

Extracts: player bio data (country, birth date, height, hand, turned pro)
and optionally season/career surface stats aggregates.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from bs4 import BeautifulSoup

from scraper.base import BaseScraper, ScraperError

logger = logging.getLogger(__name__)

ATP_BASE_URL = "https://www.atptour.com"
PROFILE_URL = "https://www.atptour.com/en/players/{slug}/{player_id}/overview"
STATS_URL = "https://www.atptour.com/en/players/{slug}/{player_id}/player-stats"


class PlayerScraper(BaseScraper):
    """Scrape profile and career-stats for a single ATP player."""

    def __init__(self):
        super().__init__(checkpoint_name="players")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrape_player(self, player_id: str, player_slug: str) -> dict[str, Any]:
        """Return a merged dict with bio + aggregate stats for the player."""
        if self.checkpoint.is_done(player_id):
            logger.debug("Skip (checkpoint): player=%s", player_id)
            return {}

        bio = self._scrape_bio(player_id, player_slug)
        stats = self._scrape_stats(player_id, player_slug)
        result = {**bio, **stats, "player_id": player_id}

        self.checkpoint.mark_done(player_id)
        return result

    # ------------------------------------------------------------------
    # Bio
    # ------------------------------------------------------------------

    def _scrape_bio(self, player_id: str, slug: str) -> dict[str, Any]:
        url = PROFILE_URL.format(slug=slug, player_id=player_id)
        logger.info("Scraping bio: %s", url)
        try:
            html = self.get_html(url)
        except ScraperError:
            return {}
        return self._parse_bio(html)

    def _parse_bio(self, html: str) -> dict[str, Any]:
        soup = BeautifulSoup(html, "lxml")
        bio: dict[str, Any] = {}

        # Name
        name_el = soup.select_one("h1.player-name, .player-profile-hero-name h3")
        if name_el:
            bio["player_name"] = name_el.get_text(strip=True)

        # Country
        country_el = soup.select_one(
            "img.player-flag, .player-profile-hero-flag img, .country-item img"
        )
        if country_el:
            bio["country"] = country_el.get("alt", "")

        # Bio table (birthdate, height, plays, turned pro…)
        for li in soup.select("ul.player-profile-hero-list li, .player-stats-table li"):
            label_el = li.select_one("span.label, .title")
            value_el = li.select_one("span.value, .stat-value")
            if label_el and value_el:
                label = label_el.get_text(strip=True).lower()
                value = value_el.get_text(strip=True)
                if "age" in label or "born" in label or "birth" in label:
                    bio["birth_date"] = value
                elif "height" in label:
                    bio["height_cm"] = self._parse_height(value)
                elif "plays" in label or "hand" in label:
                    bio["hand"] = value
                elif "turned pro" in label:
                    bio["turned_pro"] = value

        return bio

    # ------------------------------------------------------------------
    # Career/surface stats
    # ------------------------------------------------------------------

    def _scrape_stats(self, player_id: str, slug: str) -> dict[str, Any]:
        url = STATS_URL.format(slug=slug, player_id=player_id)
        logger.info("Scraping career stats: %s", url)
        try:
            html = self.get_html(url)
        except ScraperError:
            return {}
        return self._parse_stats(html)

    def _parse_stats(self, html: str) -> dict[str, Any]:
        """Parse surface/career aggregate stats from the player stats page."""
        soup = BeautifulSoup(html, "lxml")
        stats: dict[str, Any] = {}

        # Surface tabs: overall, clay, hard, grass
        for surface in ("overall", "clay", "hard", "grass"):
            section = soup.find(id=re.compile(surface, re.I)) or soup.find(
                class_=re.compile(surface, re.I)
            )
            if not section:
                continue
            surface_stats = self._parse_surface_section(section, surface)
            stats.update(surface_stats)

        return stats

    def _parse_surface_section(self, section, surface: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        prefix = f"career_{surface}_"

        rows = section.select("tr") or section.select(".stat-row")
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            label = cells[0].get_text(strip=True).lower().replace(" ", "_")
            value_raw = cells[1].get_text(strip=True)
            try:
                result[prefix + label] = float(value_raw.replace("%", "").strip()) / (
                    100 if "%" in value_raw else 1
                )
            except ValueError:
                result[prefix + label] = value_raw

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_height(raw: str) -> int | None:
        """Parse height string like '188 cm' or '6'2"' into cm."""
        m = re.search(r"(\d{2,3})\s*cm", raw)
        if m:
            return int(m.group(1))
        # feet/inches fallback
        m = re.match(r"(\d)'(\d+)", raw)
        if m:
            return int(m.group(1)) * 30 + int(m.group(2)) * 3
        return None
