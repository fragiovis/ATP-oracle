"""Orchestrate the full data acquisition pipeline.

Sources:
  - ATP Tour website  → current Top-200 rankings  (scraper/rankings.py)
  - Jeff Sackmann CSV → match history + stats      (scraper/sackmann.py)
  - ATP Tour Hawkeye  → 2025+ matches with stats   (scraper/atp_stats.py)
  - tennis-data.co.uk → 2025+ fallback (no stats)  (scraper/tennisdata.py)
  - ATP Tour website  → player bio (optional)

Match history: ALL ATP matches for the configured years_back window.
No filtering by current rankings – the full historical dataset is used for
training. The current Top-200 rankings are stored separately and used only
for the inference layer (predictor.py).
"""

from __future__ import annotations

import logging
from typing import Any

from scraper.rankings import RankingsScraper
from scraper.player import PlayerScraper
from scraper.sackmann import SackmannLoader
from scraper.tennisdata import TennisDataLoader

logger = logging.getLogger(__name__)


class ScrapingPipeline:
    def __init__(self):
        self.rankings_scraper = RankingsScraper()
        self.player_scraper   = PlayerScraper()
        self.sackmann         = SackmannLoader()
        self.tennisdata       = TennisDataLoader()

    def run(self, skip_bios: bool = False) -> dict[str, Any]:
        """
        Execute the full pipeline.

        Returns
        -------
        dict with keys:
          rankings    – current ATP Top-200 (for inference only)
          players     – current Top-200 player profiles (for inference only)
          matches     – ALL historical ATP matches (for training)
          match_stats – per-player stats for all historical matches
        """

        # --- Step 1: Current ATP rankings ---
        logger.info("=== Step 1: Scraping current ATP Top-200 rankings ===")
        rankings = self.rankings_scraper.scrape()
        if not rankings:
            logger.error("No rankings found – aborting.")
            return {}
        logger.info("Current top-%d rankings fetched.", len(rankings))

        # --- Step 2: Player bios for current Top-200 (optional) ---
        if skip_bios:
            logger.info("=== Step 2: Skipping player bios (--no-bios) ===")
            players: list[dict[str, Any]] = list(rankings)
        else:
            logger.info("=== Step 2: Scraping player bios for current Top-200 ===")
            players = []
            for r in rankings:
                try:
                    profile = self.player_scraper.scrape_player(
                        player_id=r["player_id"],
                        player_slug=self._to_slug(r["player_name"]),
                    )
                    players.append({**r, **profile})
                except Exception as exc:
                    logger.debug("Bio fetch failed for %s: %s", r["player_name"], exc)
                    players.append(r)

        # --- Step 3: ALL historical matches from Sackmann (no player filter) ---
        logger.info("=== Step 3: Loading ALL historical matches from Sackmann ===")
        sackmann_data = self.sackmann.load_all(player_ids=None)  # None = no filter

        all_matches = sackmann_data.get("matches", [])
        all_stats   = sackmann_data.get("match_stats", [])
        logger.info("Sackmann matches: %d  stats: %d", len(all_matches), len(all_stats))

        # --- Step 4: 2025 / 2026 data from tennis-data.co.uk ---
        logger.info("=== Step 4: Loading 2025/2026 data from tennis-data.co.uk ===")
        recent = self.tennisdata.load_all()
        recent_matches = recent.get("matches", [])
        logger.info("Recent matches (2025/2026): %d", len(recent_matches))
        all_matches = all_matches + recent_matches
        logger.info("Total matches combined: %d", len(all_matches))

        # --- Players table: Sackmann full roster merged with Top-200 bios ---
        # Use the full Sackmann player list as the players table for training
        # (needed so feature engineering can look up any historical player).
        sack_players = sackmann_data.get("players", [])
        logger.info("Sackmann player roster: %d players", len(sack_players))

        return {
            "rankings":    rankings,       # current Top-200 (inference)
            "players":     sack_players,   # full historical roster (training + inference)
            "matches":     all_matches,    # all historical + recent (training)
            "match_stats": all_stats,      # stats for all matches (training)
        }

    @staticmethod
    def _to_slug(name: str) -> str:
        return name.lower().replace(" ", "-")
