"""Scrape historical match results for a given ATP player.

The ATP player activity page lists matches year-by-year.  This scraper
iterates over the configured year range, parsing each result row.

Scraped fields per match:
  tournament_name, tournament_id, match_date, round, surface,
  best_of (if determinable), opponent_name, opponent_id,
  winner (player / opponent), scoreline, match_url
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

ATP_BASE_URL = "https://www.atptour.com"
# Activity page URL – one page per player per year
ACTIVITY_URL = (
    "https://www.atptour.com/en/players/{slug}/{player_id}/player-activity"
    "?year={year}"
)


class MatchScraper(BaseScraper):
    """Scrape match-result history for a single player."""

    def __init__(self):
        super().__init__(checkpoint_name="matches")
        cfg = settings.scraper
        current_year = date.today().year
        self.years = list(range(current_year - cfg.years_back, current_year + 1))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrape_player(
        self, player_id: str, player_slug: str, player_name: str
    ) -> list[dict[str, Any]]:
        """
        Return all scraped matches for *player_id*.
        Already-completed (year, player_id) combos are skipped.
        """
        all_matches: list[dict[str, Any]] = []

        for year in self.years:
            ck_key = f"{player_id}_{year}"
            if self.checkpoint.is_done(ck_key):
                logger.debug("Skip (checkpoint): player=%s year=%d", player_id, year)
                continue

            url = ACTIVITY_URL.format(
                slug=player_slug, player_id=player_id, year=year
            )
            logger.info("Scraping matches: player=%s year=%d", player_name, year)

            try:
                html = self.get_html(url)
            except ScraperError:
                logger.warning("Could not fetch activity page for %s %d", player_name, year)
                continue

            matches = self._parse_activity(html, player_id, player_name, year)
            logger.info(
                "  Found %d matches for %s in %d", len(matches), player_name, year
            )
            all_matches.extend(matches)
            self.checkpoint.mark_done(ck_key)

        return all_matches

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_activity(
        self, html: str, player_id: str, player_name: str, year: int
    ) -> list[dict[str, Any]]:
        soup = BeautifulSoup(html, "lxml")
        matches: list[dict[str, Any]] = []

        # ATP activity pages group matches by tournament in sections
        tournament_sections = soup.select("div.activity-tournament-table") or []

        for section in tournament_sections:
            tournament_meta = self._parse_tournament_meta(section)
            match_rows = section.select("tr.activity-table-content")

            for row in match_rows:
                match = self._parse_match_row(
                    row, player_id, player_name, tournament_meta, year
                )
                if match:
                    matches.append(match)

        # Fallback: generic table rows if the structured layout isn't found
        if not tournament_sections:
            rows = soup.select("tr") or []
            for row in rows:
                match = self._parse_match_row(
                    row, player_id, player_name, {}, year
                )
                if match:
                    matches.append(match)

        return matches

    # ------------------------------------------------------------------

    def _parse_tournament_meta(self, section) -> dict[str, Any]:
        meta: dict[str, Any] = {}
        try:
            name_el = section.select_one(".activity-tournament-title a, .event-name a")
            if name_el:
                meta["tournament_name"] = name_el.get_text(strip=True)
                href = name_el.get("href", "")
                # extract tournament id from URL if present
                tid_match = re.search(r"/tournaments/[^/]+/([^/]+)/", href)
                meta["tournament_id"] = tid_match.group(1) if tid_match else ""

            surface_el = section.select_one(".activity-tournament-surface, .surface-type")
            if surface_el:
                meta["surface"] = surface_el.get_text(strip=True).lower()

            category_el = section.select_one(".activity-tournament-category, .event-category")
            if category_el:
                meta["tournament_category"] = category_el.get_text(strip=True)
        except Exception:
            pass
        return meta

    def _parse_match_row(
        self,
        row,
        player_id: str,
        player_name: str,
        tournament_meta: dict[str, Any],
        year: int,
    ) -> dict[str, Any] | None:
        try:
            cells = row.find_all("td")
            if len(cells) < 3:
                return None

            text_cells = [c.get_text(strip=True) for c in cells]

            # Round
            round_text = self._find_round(text_cells)

            # Opponent
            opponent_link = row.find("a", href=re.compile(r"/en/players/"))
            if not opponent_link:
                return None
            opponent_name = opponent_link.get_text(strip=True)
            opponent_href = opponent_link.get("href", "")
            opponent_id = self._extract_player_id(opponent_href)

            # Scoreline – looks like "6-4 7-5" or "W/O" or "RET"
            scoreline = self._find_scoreline(text_cells)

            # Win/loss
            result_el = row.find(class_=re.compile(r"W|L|win|loss", re.I))
            is_winner = self._determine_winner(row, scoreline)

            # Match date
            match_date = self._find_date(text_cells, year)

            # Match stats URL (may not exist)
            stats_url = self._find_stats_url(row)

            return {
                "player_id": player_id,
                "player_name": player_name,
                "opponent_id": opponent_id,
                "opponent_name": opponent_name,
                "match_date": match_date,
                "round": round_text,
                "scoreline": scoreline,
                "player_won": is_winner,
                "stats_url": stats_url,
                **tournament_meta,
            }
        except Exception as exc:
            logger.debug("Could not parse match row: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Field extractors
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_player_id(href: str) -> str:
        parts = href.strip("/").split("/")
        return parts[3] if len(parts) >= 4 else parts[-1]

    @staticmethod
    def _find_round(cells: list[str]) -> str:
        round_patterns = re.compile(
            r"\b(R128|R64|R32|R16|QF|SF|F|RR|1R|2R|3R|4R|Finals?)\b", re.I
        )
        for c in cells:
            m = round_patterns.search(c)
            if m:
                return m.group(0).upper()
        return ""

    @staticmethod
    def _find_scoreline(cells: list[str]) -> str:
        score_pattern = re.compile(r"\d[-–]\d")
        for c in cells:
            if score_pattern.search(c):
                return c
        return ""

    @staticmethod
    def _find_date(cells: list[str], year: int) -> str:
        date_pattern = re.compile(r"\d{2}\.\d{2}|\d{4}-\d{2}-\d{2}|\w+ \d{1,2}")
        for c in cells:
            m = date_pattern.search(c)
            if m:
                raw = m.group(0)
                # Try to normalise to YYYY-MM-DD
                for fmt in ("%d.%m", "%Y-%m-%d"):
                    try:
                        import datetime
                        if fmt == "%d.%m":
                            d = datetime.datetime.strptime(f"{raw}.{year}", "%d.%m.%Y")
                        else:
                            d = datetime.datetime.strptime(raw, fmt)
                        return d.strftime("%Y-%m-%d")
                    except ValueError:
                        pass
        return str(year)

    @staticmethod
    def _determine_winner(row, scoreline: str) -> bool:
        """Heuristic: look for a W/L indicator or 'win' class in the row."""
        text = row.get_text()
        classes = " ".join(row.get("class", []))
        if re.search(r"\bW\b", classes) or "win" in classes.lower():
            return True
        if re.search(r"\bL\b", classes) or "loss" in classes.lower():
            return False
        # Fallback: if scoreline is present and no 'RET'/'W/O', assume scraped from winner
        return bool(scoreline and "ret" not in scoreline.lower())

    @staticmethod
    def _find_stats_url(row) -> str:
        link = row.find("a", href=re.compile(r"match-stats|stats"))
        if link:
            href = link.get("href", "")
            return (ATP_BASE_URL + href) if href.startswith("/") else href
        return ""
