"""Unit tests for feature engineering and scoreline parsing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.targets import build_targets, parse_scoreline
from features.elo import EloComputer
from features.h2h_features import H2HFeatureComputer


# ---------------------------------------------------------------------------
# Scoreline parsing
# ---------------------------------------------------------------------------

class TestScorelineParsing:
    def test_straightset_win(self):
        r = parse_scoreline("6-4 6-3")
        assert r["sets_a"] == 2
        assert r["sets_b"] == 0
        assert r["total_games"] == 19
        assert not r["has_tiebreak"]
        assert not r["deciding_set"]

    def test_three_set_match(self):
        r = parse_scoreline("6-4 3-6 7-5")
        assert r["sets_a"] == 2
        assert r["sets_b"] == 1
        assert r["deciding_set"]

    def test_tiebreak_detected(self):
        r = parse_scoreline("7-6(5) 6-3")
        assert r["has_tiebreak"]

    def test_empty_scoreline(self):
        r = parse_scoreline("")
        assert r["total_games"] == 0
        assert r["sets_a"] == 0

    def test_retired_stripped(self):
        r = parse_scoreline("6-1 3-0 RET")
        # Only completed sets counted
        assert r["total_games"] == 10

    def test_bo5_deciding(self):
        r = parse_scoreline("6-4 3-6 6-3 4-6 7-5")
        assert r["deciding_set"]
        assert r["sets_a"] == 3
        assert r["sets_b"] == 2


# ---------------------------------------------------------------------------
# Elo computer
# ---------------------------------------------------------------------------

def _make_matches(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df["match_date"] = pd.to_datetime(df["match_date"])
    return df


class TestEloComputer:
    def test_initial_elo(self):
        matches = _make_matches([
            {
                "match_id": "m1",
                "player_a_id": "p1",
                "player_b_id": "p2",
                "winner_id": "p1",
                "match_date": "2020-01-01",
                "surface": "hard",
            }
        ])
        elo_df = EloComputer().compute(matches)
        assert "player_a_elo_before" in elo_df.columns
        # First match ever: both start at 1500
        assert elo_df.loc["m1", "player_a_elo_before"] == pytest.approx(1500.0)
        assert elo_df.loc["m1", "player_b_elo_before"] == pytest.approx(1500.0)

    def test_elo_updates_after_win(self):
        matches = _make_matches([
            {
                "match_id": "m1",
                "player_a_id": "p1",
                "player_b_id": "p2",
                "winner_id": "p1",
                "match_date": "2020-01-01",
                "surface": "hard",
            },
            {
                "match_id": "m2",
                "player_a_id": "p1",
                "player_b_id": "p2",
                "winner_id": "p2",
                "match_date": "2020-02-01",
                "surface": "hard",
            },
        ])
        elo_df = EloComputer().compute(matches)
        # After m1, p1 should have more than 1500 Elo before m2
        assert elo_df.loc["m2", "player_a_elo_before"] > 1500.0
        assert elo_df.loc["m2", "player_b_elo_before"] < 1500.0


# ---------------------------------------------------------------------------
# H2H features
# ---------------------------------------------------------------------------

class TestH2HFeatures:
    def _make_data(self):
        matches = pd.DataFrame([
            {
                "match_id": "m3",
                "player_a_id": "p1",
                "player_b_id": "p2",
                "winner_id": "p1",
                "match_date": pd.Timestamp("2024-06-01"),
                "surface": "clay",
                "scoreline": "6-4 6-3",
                "total_games": 19,
                "has_tiebreak": False,
                "deciding_set": False,
            }
        ])

        h2h = pd.DataFrame([
            {
                "match_id": "m1",
                "player_a_id": "p1",
                "player_b_id": "p2",
                "winner_id": "p1",
                "surface": "clay",
                "match_date": pd.Timestamp("2022-01-01"),
            },
            {
                "match_id": "m2",
                "player_a_id": "p2",
                "player_b_id": "p1",
                "winner_id": "p2",
                "surface": "hard",
                "match_date": pd.Timestamp("2023-06-01"),
            },
        ])
        return matches, h2h

    def test_h2h_counts(self):
        matches, h2h = self._make_data()
        result = H2HFeatureComputer().compute(matches, h2h)
        assert "h2h_total_matches" in result.columns
        assert result.loc["m3", "h2h_total_matches"] == 2

    def test_h2h_win_rate(self):
        matches, h2h = self._make_data()
        result = H2HFeatureComputer().compute(matches, h2h)
        # p1 won 1 out of 2 prior meetings
        assert result.loc["m3", "h2h_wins_a"] == 1
        assert result.loc["m3", "h2h_win_rate_a"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------

class TestTargetConstruction:
    def _sample_matches(self):
        return pd.DataFrame([
            {
                "match_id": "t1",
                "player_a_id": "p1",
                "player_b_id": "p2",
                "winner_id": "p1",
                "scoreline": "6-4 7-6(3)",
                "surface": "hard",
                "match_date": "2023-01-01",
                "total_games": 23,
                "has_tiebreak": True,
                "deciding_set": False,
                "set_scores": [(6, 4), (7, 6)],
            }
        ])

    def test_winner_label(self):
        matches = self._sample_matches()
        targets = build_targets(matches, pd.DataFrame())
        assert targets.loc["t1", "winner_label"] == 1

    def test_tiebreak_label(self):
        matches = self._sample_matches()
        targets = build_targets(matches, pd.DataFrame())
        assert targets.loc["t1", "tiebreak_present_label"] == 1

    def test_deciding_set_label(self):
        matches = self._sample_matches()
        targets = build_targets(matches, pd.DataFrame())
        assert targets.loc["t1", "deciding_set_label"] == 0

    def test_first_set_label(self):
        matches = self._sample_matches()
        targets = build_targets(matches, pd.DataFrame())
        assert targets.loc["t1", "first_set_winner_label"] == 1  # p1 won 6-4 first set
