"""Compute head-to-head features for a pair of players before each match."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class H2HFeatureComputer:
    """Derive head-to-head features from the h2h_history table."""

    def compute(
        self,
        matches: pd.DataFrame,
        h2h: pd.DataFrame,
        match_stats: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        For each match, compute H2H features using only H2H records
        strictly before the match date.

        Returns DataFrame indexed by match_id.
        """
        if matches.empty or h2h.empty:
            return pd.DataFrame()

        matches = matches.copy().sort_values("match_date", na_position="last")
        if "match_date" in h2h.columns:
            h2h = h2h.copy()
            h2h["match_date"] = pd.to_datetime(h2h["match_date"], errors="coerce")

        # Pre-build match-level lookups for new H2H features
        # match_id → {has_tiebreak, total_games, sets_played}
        match_lookup: dict[str, dict] = {}
        for _, m in matches.iterrows():
            mid = m["match_id"]
            has_tb = bool(m.get("has_tiebreak", False))
            total_g = m.get("total_games")
            sets_a = m.get("sets_a")
            sets_b = m.get("sets_b")
            sets_pl = (sets_a + sets_b) if (pd.notnull(sets_a) and pd.notnull(sets_b)) else None
            match_lookup[mid] = {
                "has_tiebreak": has_tb,
                "total_games": float(total_g) if pd.notnull(total_g) else None,
                "sets_played": float(sets_pl) if sets_pl is not None else None,
            }

        # Pre-build aces lookup: (match_id, player_id) → aces
        aces_lookup: dict[tuple, float] = {}
        if match_stats is not None and not match_stats.empty and "aces" in match_stats.columns:
            for _, s in match_stats.iterrows():
                mid = s.get("match_id")
                pid = s.get("player_id")
                ace = s.get("aces")
                if mid and pid and pd.notnull(ace):
                    aces_lookup[(str(mid), str(pid))] = float(ace)

        rows: list[dict] = []

        for _, m in matches.iterrows():
            mid = m["match_id"]
            pa = m["player_a_id"]
            pb = m["player_b_id"]
            match_date = m.get("match_date")
            surface = str(m.get("surface", "")).lower()

            # All prior H2H meetings between these two players
            prior = h2h[
                (
                    ((h2h["player_a_id"] == pa) & (h2h["player_b_id"] == pb))
                    | ((h2h["player_a_id"] == pb) & (h2h["player_b_id"] == pa))
                )
                & (h2h["match_id"] != mid)
            ]
            if pd.notnull(match_date) and "match_date" in prior.columns:
                prior = prior[prior["match_date"] < match_date]

            feats = self._compute_h2h(
                pa, pb, prior, surface, match_date, match_lookup, aces_lookup
            )
            feats["match_id"] = mid
            rows.append(feats)

        return pd.DataFrame(rows).set_index("match_id")

    # ------------------------------------------------------------------

    def _compute_h2h(
        self,
        pa: str,
        pb: str,
        prior: pd.DataFrame,
        surface: str,
        match_date,
        match_lookup: dict,
        aces_lookup: dict,
    ) -> dict[str, Any]:
        feats: dict[str, Any] = {}

        n = len(prior)
        feats["h2h_total_matches"] = n

        if n == 0:
            feats["h2h_wins_a"] = np.nan
            feats["h2h_wins_b"] = np.nan
            feats["h2h_win_rate_a"] = np.nan
            feats["h2h_win_rate_a_surface"] = np.nan
            feats["h2h_days_since_last"] = np.nan
            feats["h2h_last5_win_rate_a"] = np.nan
            feats["h2h_tiebreak_pct"] = np.nan
            feats["h2h_avg_aces_a"] = np.nan
            feats["h2h_avg_total_aces"] = np.nan
            feats["h2h_avg_total_games"] = np.nan
            feats["h2h_avg_sets_played"] = np.nan
            return feats

        wins_a = (prior["winner_id"] == pa).sum()
        wins_b = (prior["winner_id"] == pb).sum()
        feats["h2h_wins_a"] = int(wins_a)
        feats["h2h_wins_b"] = int(wins_b)
        feats["h2h_win_rate_a"] = wins_a / n

        # Surface-specific H2H
        surface_prior = prior[prior["surface"].str.contains(surface, na=False, case=False)]
        ns = len(surface_prior)
        if ns > 0:
            feats["h2h_win_rate_a_surface"] = (surface_prior["winner_id"] == pa).sum() / ns
        else:
            feats["h2h_win_rate_a_surface"] = np.nan

        # Time since last meeting
        if "match_date" in prior.columns and pd.notnull(match_date):
            last_date = prior["match_date"].dropna().max()
            if pd.notnull(last_date):
                delta = pd.Timestamp(match_date) - pd.Timestamp(last_date)
                feats["h2h_days_since_last"] = delta.days
            else:
                feats["h2h_days_since_last"] = np.nan
        else:
            feats["h2h_days_since_last"] = np.nan

        # Last 5 H2H win rate
        recent = prior.sort_values("match_date", ascending=False).head(5)
        feats["h2h_last5_win_rate_a"] = (recent["winner_id"] == pa).sum() / len(recent)

        # ── New H2H features ────────────────────────────────────────────

        prior_mids = prior["match_id"].tolist()

        # h2h_tiebreak_pct: % of H2H matches with a tiebreak
        tb_vals = [
            match_lookup[mid]["has_tiebreak"]
            for mid in prior_mids
            if mid in match_lookup
        ]
        feats["h2h_tiebreak_pct"] = float(np.mean(tb_vals)) if tb_vals else np.nan

        # h2h_avg_total_games: avg total games in H2H
        tg_vals = [
            match_lookup[mid]["total_games"]
            for mid in prior_mids
            if mid in match_lookup and match_lookup[mid]["total_games"] is not None
        ]
        feats["h2h_avg_total_games"] = float(np.mean(tg_vals)) if tg_vals else np.nan

        # h2h_avg_sets_played: avg sets played in H2H
        sp_vals = [
            match_lookup[mid]["sets_played"]
            for mid in prior_mids
            if mid in match_lookup and match_lookup[mid]["sets_played"] is not None
        ]
        feats["h2h_avg_sets_played"] = float(np.mean(sp_vals)) if sp_vals else np.nan

        # h2h_avg_aces_a: avg aces by player_a in H2H
        aces_a_vals: list[float] = []
        aces_total_vals: list[float] = []
        pa_str, pb_str = str(pa), str(pb)
        for mid in prior_mids:
            mid_str = str(mid)
            # Determine which side player_a was in this prior match
            prior_row = prior[prior["match_id"] == mid]
            if prior_row.empty:
                continue
            pr = prior_row.iloc[0]
            # In this prior meeting, which player_id corresponds to pa?
            if str(pr.get("player_a_id", "")) == pa_str:
                pa_role_id = pa_str
            else:
                pa_role_id = pa_str  # pa is always pa regardless of H2H orientation

            ace_a = aces_lookup.get((mid_str, pa_role_id))
            ace_b = aces_lookup.get((mid_str, pb_str))
            if ace_a is not None:
                aces_a_vals.append(ace_a)
            if ace_a is not None and ace_b is not None:
                aces_total_vals.append(ace_a + ace_b)
            elif ace_a is None and ace_b is None:
                # try total from match aces sum
                total = sum(
                    v for k, v in aces_lookup.items()
                    if k[0] == mid_str
                )
                if total > 0:
                    aces_total_vals.append(float(total))

        feats["h2h_avg_aces_a"] = float(np.mean(aces_a_vals)) if aces_a_vals else np.nan
        feats["h2h_avg_total_aces"] = float(np.mean(aces_total_vals)) if aces_total_vals else np.nan

        return feats
