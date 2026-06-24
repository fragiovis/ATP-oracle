"""Compute rolling player-level features strictly before each match (no leakage).

Kept features (per player, prefix pa_/pb_):
  Career overall:
    - career_aces_per_match
    - career_first_serve_pct        (% first serves in)
    - career_first_serve_pts_won_pct
    - career_service_pts_won_pct
    - career_return_pts_won_pct
    - career_win_rate
  Per active surface (clay/hard/grass):
    - {surface}_career_aces_per_match
    - {surface}_career_first_serve_pts_won_pct
    - {surface}_career_service_pts_won_pct
    - {surface}_career_return_pts_won_pct
    - {surface}_career_win_rate
  Recent form (last 10 matches):
    - last10_win_rate
    - last10_aces_per_match
    - last10_first_serve_pts_won_pct
    - last10_service_pts_won_pct
    - last10_return_pts_won_pct
  Momentum:
    - form_last5  (win rate last 5)
  Surface form (last 20 matches on the active surface):
    - surface_win_rate_last20
  Opponent quality (last 10 wins):
    - avg_opp_elo_won_last10  (mean Elo of opponents defeated in last 10 won matches)
  Tournament history (strictly before this match in this tournament):
    - tournament_win_rate  (win rate in this tournament; 0 if no history)
    - tournament_matches   (number of matches played in this tournament)

Removed vs previous version:
  - second_serve_pts_won_pct  (collinear with service_pts_won_pct)
  - bp_saved_pct / bp_converted_pct  (noisy, sparse)
  - bo3/bo5 stratified stats  (too sparse for most players)
  - win_streak  (captures momentum already in form_last5)
  - n_matches  (not a predictive feature)
  - {surface}_n_matches  (not predictive)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)

SURFACES = ["clay", "hard", "grass"]
MIN_STAT = 3   # minimum matches with stat data required to emit a value


class PlayerFeatureComputer:
    def __init__(self):
        self.min_matches = settings.features.min_matches_for_feature

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, matches: pd.DataFrame, match_stats: pd.DataFrame) -> pd.DataFrame:
        if matches.empty:
            return pd.DataFrame()

        matches = matches.copy()
        matches["match_date"] = pd.to_datetime(matches["match_date"], errors="coerce")
        matches = matches.sort_values("match_date", na_position="last").reset_index(drop=True)

        logger.info("Building per-player history index…")
        player_hist = self._build_player_history(matches, match_stats)

        logger.info("Computing features for %d matches…", len(matches))
        result_rows: list[dict] = []

        for _, m in matches.iterrows():
            mid        = m["match_id"]
            pa         = str(m["player_a_id"])
            pb         = str(m["player_b_id"])
            match_date = m["match_date"]
            surface    = str(m.get("surface", "")).lower()
            tournament = str(m.get("tournament_name", "")).strip().lower()

            feats: dict[str, Any] = {"match_id": mid}

            for prefix, pid in [("pa", pa), ("pb", pb)]:
                hist = player_hist.get(pid, pd.DataFrame())
                if not hist.empty and pd.notnull(match_date):
                    hist = hist[hist["match_date"] < match_date]
                feats.update(self._player_feats(prefix, hist, surface=surface, tournament=tournament))

            result_rows.append(feats)

        return pd.DataFrame(result_rows).set_index("match_id")

    # ------------------------------------------------------------------
    # Build per-player history dict
    # ------------------------------------------------------------------

    def _build_player_history(
        self, matches: pd.DataFrame, stats: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
        STAT_COLS = [
            "aces", "first_serve_pct", "first_serve_pts_won_pct",
            "service_pts_won_pct", "return_pts_won_pct",
        ]
        stats_lookup: dict[tuple, dict] = {}
        if not stats.empty:
            for row in stats.itertuples(index=False):
                key = (getattr(row, "match_id", None), str(getattr(row, "player_id", "")))
                stats_lookup[key] = {
                    c: getattr(row, c, np.nan) for c in STAT_COLS if hasattr(row, c)
                }

        # Build opponent Elo lookup: match_id → {player_a_id: elo_b_before, player_b_id: elo_a_before}
        # (the opponent's Elo before the match, keyed by this player's id)
        has_elo_a = "player_a_elo_before" in matches.columns
        has_elo_b = "player_b_elo_before" in matches.columns

        rows: list[dict] = []
        for m in matches.itertuples(index=False):
            mid        = m.match_id
            date       = m.match_date
            surf       = str(getattr(m, "surface", "")).lower()
            pa         = str(m.player_a_id)
            pb         = str(m.player_b_id)
            won_a      = (str(getattr(m, "winner_id", "")) == pa)
            best_of    = int(getattr(m, "best_of", 3) or 3)
            tournament = str(getattr(m, "tournament_name", "") or "").strip().lower()

            elo_a = float(getattr(m, "player_a_elo_before", np.nan)) if has_elo_a else np.nan
            elo_b = float(getattr(m, "player_b_elo_before", np.nan)) if has_elo_b else np.nan

            for pid, won, opp_elo in [(pa, won_a, elo_b), (pb, not won_a, elo_a)]:
                row: dict[str, Any] = {
                    "player_id":   pid,
                    "match_date":  date,
                    "surface":     surf,
                    "best_of":     best_of,
                    "won":         int(won),
                    "opponent_elo": opp_elo,
                    "tournament":  tournament,
                }
                s = stats_lookup.get((mid, pid), {})
                for c in STAT_COLS:
                    row[c] = s.get(c, np.nan)
                rows.append(row)

        df = pd.DataFrame(rows)
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
        df = df.sort_values("match_date", na_position="last")

        return {pid: grp.reset_index(drop=True) for pid, grp in df.groupby("player_id")}

    # ------------------------------------------------------------------
    # Feature computation for one player
    # ------------------------------------------------------------------

    def _player_feats(
        self,
        prefix: str,
        hist: pd.DataFrame,
        surface: str = "",
        tournament: str = "",
    ) -> dict[str, Any]:
        if hist.empty:
            return {}

        hist = hist.sort_values("match_date", ascending=False)
        f: dict[str, Any] = {}

        # ── Overall career ──────────────────────────────────────────────
        f[f"{prefix}_career_win_rate"] = hist["won"].mean()
        f[f"{prefix}_form_last5"]      = hist.head(5)["won"].mean()

        # Last-10 win rate (form indicator)
        f[f"{prefix}_last10_win_rate"] = hist.head(10)["won"].mean()

        # Career serve/return stats (window: last 30 matches with real data)
        CAREER_STATS = {
            "aces":                    "career_aces_per_match",
            "first_serve_pct":         "career_first_serve_pct",
            "first_serve_pts_won_pct": "career_first_serve_pts_won_pct",
            "service_pts_won_pct":     "career_service_pts_won_pct",
            "return_pts_won_pct":      "career_return_pts_won_pct",
        }
        for col, fname in CAREER_STATS.items():
            if col in hist.columns:
                wd = hist[hist[col].notna()].head(30)
                if len(wd) >= MIN_STAT:
                    f[f"{prefix}_{fname}"] = wd[col].mean()

        # Last-10 serve/return stats
        LAST10_STATS = {
            "aces":                    "last10_aces_per_match",
            "first_serve_pts_won_pct": "last10_first_serve_pts_won_pct",
            "service_pts_won_pct":     "last10_service_pts_won_pct",
            "return_pts_won_pct":      "last10_return_pts_won_pct",
        }
        for col, fname in LAST10_STATS.items():
            if col in hist.columns:
                wd = hist[hist[col].notna()].head(10)
                if len(wd) >= MIN_STAT:
                    f[f"{prefix}_{fname}"] = wd[col].mean()

        # ── BO3 / BO5 ace rate (only aces, not all stats — sparse otherwise) ──
        if "best_of" in hist.columns and "aces" in hist.columns:
            for bo_val, bo_label in [(5, "bo5"), (3, "bo3")]:
                bo_h = hist[hist["best_of"] == bo_val]
                wd = bo_h[bo_h["aces"].notna()].head(30)
                if len(wd) >= MIN_STAT:
                    f[f"{prefix}_{bo_label}_career_aces_per_match"] = wd["aces"].mean()

        # ── Per-surface stats (all 3 surfaces always computed) ──────────
        # The predictor and training pipeline NaN-out inactive surface
        # columns so models only see the relevant surface at inference/train.
        SURFACE_STATS = {
            "aces":                    "career_aces_per_match",
            "first_serve_pts_won_pct": "career_first_serve_pts_won_pct",
            "service_pts_won_pct":     "career_service_pts_won_pct",
            "return_pts_won_pct":      "career_return_pts_won_pct",
        }
        for surf in SURFACES:
            sh = hist[hist["surface"].str.contains(surf, na=False, case=False)]
            if len(sh) >= 1:
                f[f"{prefix}_{surf}_career_win_rate"] = sh["won"].mean()
            for stat_col, stat_name in SURFACE_STATS.items():
                if stat_col in sh.columns:
                    wd = sh[sh[stat_col].notna()].head(20)
                    if len(wd) >= MIN_STAT:
                        f[f"{prefix}_{surf}_{stat_name}"] = wd[stat_col].mean()

        # ── Surface form: win rate on active surface (last 20) ──────────
        # Only computed when surface is known; surface masking ensures only
        # the active surface column survives at training/inference time.
        if surface and "surface" in hist.columns:
            sh_active = hist[hist["surface"].str.contains(surface, na=False, case=False)]
            if len(sh_active) >= 1:
                f[f"{prefix}_surface_win_rate_last20"] = sh_active.head(20)["won"].mean()
            else:
                f[f"{prefix}_surface_win_rate_last20"] = np.nan
        else:
            # Fallback: use all surfaces
            f[f"{prefix}_surface_win_rate_last20"] = hist.head(20)["won"].mean()

        # ── Opponent quality: mean Elo of opponents in last 10 WON matches ──
        if "opponent_elo" in hist.columns:
            won_hist = hist[hist["won"] == 1].head(10)
            opp_elos = won_hist["opponent_elo"].dropna()
            if len(opp_elos) >= 1:
                f[f"{prefix}_avg_opp_elo_won_last10"] = opp_elos.mean()

        # ── Tournament history ───────────────────────────────────────────
        if tournament and "tournament" in hist.columns:
            th = hist[hist["tournament"] == tournament]
            n_played = len(th)
            f[f"{prefix}_tournament_matches"] = n_played
            f[f"{prefix}_tournament_win_rate"] = th["won"].mean() if n_played > 0 else 0.0
        else:
            f[f"{prefix}_tournament_matches"] = 0
            f[f"{prefix}_tournament_win_rate"] = 0.0

        return f
