"""Matchup / interaction features combining both players' profiles.

Only signed differences are produced — no absolute values to avoid
accidentally encoding outcome direction.

Removed vs previous version:
  - bo3/bo5 ace/win-rate diffs (bo stats removed from player_features)
  - ia_bp_edge / ib_bp_edge (bp stats removed)
  - ia/ib_serve1_vs_return_l10 replaced by direct last10 serve/return diffs
"""

from __future__ import annotations

import pandas as pd


def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)

    def diff(a: str, b: str, out: str) -> None:
        if a in df.columns and b in df.columns:
            result[out] = df[a] - df[b]

    # Elo surface gaps (overall elo_diff already computed by elo.py)
    for s in ("clay", "hard", "grass"):
        diff(f"player_a_elo_{s}_before", f"player_b_elo_{s}_before", f"elo_{s}_diff")

    # Win-rate gaps
    diff("pa_career_win_rate",   "pb_career_win_rate",   "win_rate_diff_career")
    diff("pa_last10_win_rate",   "pb_last10_win_rate",   "win_rate_diff_last10")
    diff("pa_form_last5",        "pb_form_last5",         "form_diff_last5")

    # Surface-specific win-rate gaps
    for s in ("clay", "hard", "grass"):
        diff(f"pa_{s}_career_win_rate", f"pb_{s}_career_win_rate", f"win_rate_{s}_diff")

    # Serve vs return (career): A's serve dominance over B's return
    diff("pa_career_first_serve_pts_won_pct", "pb_career_return_pts_won_pct",
         "ia_serve1_vs_return")
    diff("pb_career_first_serve_pts_won_pct", "pa_career_return_pts_won_pct",
         "ib_serve1_vs_return")

    # Serve vs return (last10 — recent form)
    diff("pa_last10_first_serve_pts_won_pct", "pb_last10_return_pts_won_pct",
         "ia_serve1_vs_return_l10")
    diff("pb_last10_first_serve_pts_won_pct", "pa_last10_return_pts_won_pct",
         "ib_serve1_vs_return_l10")

    # Ace rate gap (career and last10)
    diff("pa_career_aces_per_match", "pb_career_aces_per_match", "ace_rate_diff")
    diff("pa_last10_aces_per_match", "pb_last10_aces_per_match", "ace_rate_diff_l10")

    # Surface-specific serve dominance gap
    for s in ("clay", "hard", "grass"):
        diff(f"pa_{s}_career_service_pts_won_pct", f"pb_{s}_career_return_pts_won_pct",
             f"ia_{s}_serve_vs_return")
        diff(f"pb_{s}_career_service_pts_won_pct", f"pa_{s}_career_return_pts_won_pct",
             f"ib_{s}_serve_vs_return")

    # Surface form gap (win rate on active surface, last 20)
    diff("pa_surface_win_rate_last20", "pb_surface_win_rate_last20",
         "win_rate_surface_last20_diff")

    # Opponent quality gap (mean Elo of defeated opponents in last 10 wins)
    diff("pa_avg_opp_elo_won_last10", "pb_avg_opp_elo_won_last10",
         "avg_opp_elo_won_diff")

    # Tournament win-rate gap
    diff("pa_tournament_win_rate", "pb_tournament_win_rate",
         "tournament_win_rate_diff")

    return result
