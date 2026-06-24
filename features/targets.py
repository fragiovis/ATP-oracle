"""Derive training labels from the matches table.

All targets must be computable from the scoreline and match metadata –
they do NOT use in-match stats of the target match (those are the labels,
not the features).

Targets produced:
  - winner_label          (1 if player_a wins)
  - deciding_set_label    (1 if match went to deciding set)
  - tiebreak_present_label(1 if at least one tiebreak occurred)
  - first_set_winner_label(1 if player_a won the first set)
  - player_aces_target    (aces by player_a)
  - total_aces_target     (total aces in the match)
  - total_games_target    (total games in the match)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from storage.normaliser import parse_scoreline


def build_targets(matches: pd.DataFrame, match_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    matches     : silver matches table (must contain set_scores, has_tiebreak, deciding_set)
    match_stats : silver match_player_stats table

    Returns
    -------
    DataFrame indexed by match_id with all target columns.
    """
    df = matches.set_index("match_id").copy()

    # Classification
    df["winner_label"] = (df["winner_id"] == df["player_a_id"]).astype(int)

    # Defensive: re-derive has_tiebreak from scoreline when available.
    # The silver `has_tiebreak` flag has been corrupted in the past for
    # sources that don't include the "(n)" tiebreak marker, even though the
    # underlying scoreline (e.g. "7-6 6-4") is unambiguous. Re-parse when we
    # can, so a stale silver layer can still produce correct labels.
    if "scoreline" in df.columns:
        tb_from_score = df["scoreline"].map(
            lambda s: parse_scoreline(s)["has_tiebreak"] if isinstance(s, str) else None
        )
        # Combine with the stored flag (OR): either source is sufficient.
        df["tiebreak_present_label"] = (
            df["has_tiebreak"].fillna(False).astype(bool) | tb_from_score.fillna(False).astype(bool)
        ).astype(int)
    else:
        df["tiebreak_present_label"] = df["has_tiebreak"].astype(int)

    # First set winner
    df["first_set_winner_label"] = df.apply(_first_set_winner, axis=1)

    # Regression
    df["total_games_target"] = df["total_games"].astype(float)

    # Aces – from match_stats
    aces_df = _compute_aces(df, match_stats)
    df = df.join(aces_df, how="left")

    # Exact score: number of sets played (2 or 3 for bo3; 3,4,5 for bo5)
    # Derived from winner sets + loser sets, filtered to valid completed matches
    if "sets_a" in df.columns and "sets_b" in df.columns:
        sets_played = df["sets_a"] + df["sets_b"]
        # Only keep rows where the result makes sense (no retirements/walkovers)
        valid = sets_played.between(2, 5)
        df["sets_played_target"] = sets_played.where(valid)

    target_cols = [
        "winner_label",
        "tiebreak_present_label",
        "player_aces_target",
        "total_aces_target",
        "total_games_target",
        "sets_played_target",
    ]
    return df[[c for c in target_cols if c in df.columns]]


def _first_set_winner(row) -> int | float:
    """Return 1 if player_a won the first set, 0 if player_b, NaN if unknown."""
    set_scores = row.get("set_scores")
    if not isinstance(set_scores, list) or not set_scores:
        return np.nan
    ga, gb = set_scores[0]
    if ga > gb:
        return 1
    if gb > ga:
        return 0
    return np.nan  # tie-break set with equal score shouldn't happen


def _compute_aces(matches_idx: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Compute player_aces_target and total_aces_target from the stats table."""
    if stats.empty or "aces" not in stats.columns:
        empty = pd.DataFrame(
            index=matches_idx.index,
            data={"player_aces_target": np.nan, "total_aces_target": np.nan},
        )
        return empty

    # Aces per match: sum aces for both players
    aces_by_match = (
        stats.groupby("match_id")["aces"].sum().rename("total_aces_target")
    )

    # Player A aces: filter to player_a_id per match
    def player_a_aces(row):
        pa = row["player_a_id"]
        mid = row.name  # match_id is the index
        sub = stats[(stats["match_id"] == mid) & (stats["player_id"] == pa)]
        if sub.empty:
            return np.nan
        return sub["aces"].iloc[0]

    player_a_aces_s = matches_idx.apply(player_a_aces, axis=1).rename("player_aces_target")

    return pd.concat([player_a_aces_s, aces_by_match], axis=1)
