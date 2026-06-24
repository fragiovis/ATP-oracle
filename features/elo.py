"""Compute Elo ratings per player over time (overall and per surface).

Elo is computed chronologically from the full match history so that each
match's pre-match Elo is a true pre-match feature (no leakage).

Usage:
    from features.elo import EloComputer
    elo_df = EloComputer().compute(matches_df)

Returns a DataFrame indexed by match_id with columns:
    player_a_elo_before, player_b_elo_before,
    player_a_elo_clay_before, player_b_elo_clay_before, ... (per surface)
    elo_diff  (player_a - player_b)
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict

import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)


class EloComputer:
    def __init__(self):
        cfg = settings.features
        self.k = cfg.elo_k_factor
        self.k_surface = cfg.elo_surface_k_factor
        self.initial = cfg.elo_initial_rating
        self.surfaces = cfg.surfaces
        self.halflife_days = cfg.elo_time_decay_halflife_days  # 0 = no decay

    def _decay_weight(self, days_since: float) -> float:
        """Exponential decay weight: 1.0 at 0 days, 0.5 at halflife_days."""
        if self.halflife_days <= 0 or days_since <= 0:
            return 1.0
        return math.exp(-math.log(2) * days_since / self.halflife_days)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, matches: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Elo ratings and return a DataFrame with pre-match Elo columns.
        *matches* must be sorted by match_date ascending.
        """
        if matches.empty:
            return pd.DataFrame()

        matches = matches.copy().sort_values("match_date", na_position="last")

        # Overall Elo
        elo: dict[str, float] = defaultdict(lambda: self.initial)
        # Per-surface Elo
        elo_surface: dict[str, dict[str, float]] = {
            s: defaultdict(lambda: self.initial) for s in self.surfaces
        }

        rows: list[dict] = []
        # Track the last match date per player for time-decay K weighting
        last_match_date: dict[str, pd.Timestamp] = {}

        for _, row in matches.iterrows():
            pa = str(row["player_a_id"])
            pb = str(row["player_b_id"])
            winner = str(row["winner_id"])
            surface = str(row.get("surface", "")).lower()
            match_date = row.get("match_date")

            # Time-decay weights based on days since each player's last match
            if pd.notnull(match_date):
                match_ts = pd.Timestamp(match_date)
                days_a = (match_ts - last_match_date[pa]).days if pa in last_match_date else 0
                days_b = (match_ts - last_match_date[pb]).days if pb in last_match_date else 0
                w_a = self._decay_weight(days_a)
                w_b = self._decay_weight(days_b)
            else:
                w_a = w_b = 1.0

            # Snapshot pre-match Elo
            record: dict = {
                "match_id": row["match_id"],
                "player_a_elo_before": elo[pa],
                "player_b_elo_before": elo[pb],
            }
            for s in self.surfaces:
                record[f"player_a_elo_{s}_before"] = elo_surface[s][pa]
                record[f"player_b_elo_{s}_before"] = elo_surface[s][pb]

            record["elo_diff"] = record["player_a_elo_before"] - record["player_b_elo_before"]

            rows.append(record)

            # Update overall Elo with decay-weighted K
            score_a = 1.0 if winner == pa else 0.0
            k_a = self.k * w_a
            k_b = self.k * w_b
            elo[pa], elo[pb] = self._update_asymmetric(elo[pa], elo[pb], score_a, k_a, k_b)

            # Update surface Elo with decay-weighted K
            for s in self.surfaces:
                if s in surface:
                    ks_a = self.k_surface * w_a
                    ks_b = self.k_surface * w_b
                    sa, sb = self._update_asymmetric(
                        elo_surface[s][pa], elo_surface[s][pb], score_a, ks_a, ks_b
                    )
                    elo_surface[s][pa] = sa
                    elo_surface[s][pb] = sb

            # Update last-match-date after processing
            if pd.notnull(match_date):
                ts = pd.Timestamp(match_date)
                last_match_date[pa] = ts
                last_match_date[pb] = ts

        return pd.DataFrame(rows).set_index("match_id")

    def current_ratings(self, matches: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full Elo computation and return the final (current) rating
        for every player seen in the history.

        Returns DataFrame indexed by player_id with columns:
            elo, elo_clay, elo_hard, elo_grass
        """
        if matches.empty:
            return pd.DataFrame()

        matches = matches.copy().sort_values("match_date", na_position="last")

        elo: dict[str, float] = defaultdict(lambda: self.initial)
        elo_surface: dict[str, dict[str, float]] = {
            s: defaultdict(lambda: self.initial) for s in self.surfaces
        }
        last_match_date: dict[str, pd.Timestamp] = {}

        for _, row in matches.iterrows():
            pa = str(row["player_a_id"])
            pb = str(row["player_b_id"])
            winner = str(row["winner_id"])
            surface = str(row.get("surface", "")).lower()
            match_date = row.get("match_date")

            if pd.notnull(match_date):
                match_ts = pd.Timestamp(match_date)
                days_a = (match_ts - last_match_date[pa]).days if pa in last_match_date else 0
                days_b = (match_ts - last_match_date[pb]).days if pb in last_match_date else 0
                w_a = self._decay_weight(days_a)
                w_b = self._decay_weight(days_b)
            else:
                w_a = w_b = 1.0

            score_a = 1.0 if winner == pa else 0.0
            k_a = self.k * w_a
            k_b = self.k * w_b
            elo[pa], elo[pb] = self._update_asymmetric(elo[pa], elo[pb], score_a, k_a, k_b)

            for s in self.surfaces:
                if s in surface:
                    ks_a = self.k_surface * w_a
                    ks_b = self.k_surface * w_b
                    sa, sb = self._update_asymmetric(
                        elo_surface[s][pa], elo_surface[s][pb], score_a, ks_a, ks_b
                    )
                    elo_surface[s][pa] = sa
                    elo_surface[s][pb] = sb

            if pd.notnull(match_date):
                ts = pd.Timestamp(match_date)
                last_match_date[pa] = ts
                last_match_date[pb] = ts

        rows = []
        for pid in elo:
            row_dict = {"player_id": pid, "elo": elo[pid]}
            for s in self.surfaces:
                row_dict["elo_" + s] = elo_surface[s][pid]
            rows.append(row_dict)

        return pd.DataFrame(rows).set_index("player_id")

    # ------------------------------------------------------------------

    @staticmethod
    def _expected(ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400))

    def _update(
        self, ra: float, rb: float, score_a: float, k: float
    ) -> tuple[float, float]:
        return self._update_asymmetric(ra, rb, score_a, k, k)

    def _update_asymmetric(
        self, ra: float, rb: float, score_a: float, k_a: float, k_b: float
    ) -> tuple[float, float]:
        """Update Elo with separate K-factors for each player (used for time decay)."""
        ea = self._expected(ra, rb)
        new_ra = ra + k_a * (score_a - ea)
        new_rb = rb + k_b * ((1 - score_a) - (1 - ea))
        return new_ra, new_rb
