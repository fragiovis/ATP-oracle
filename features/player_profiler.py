"""Compute player profiles for the top N ATP players.

Reads silver matches + match_stats tables and produces one row per player
with all career/surface/format statistics.  The output lives in /data/players/
and is the single source of truth for inference-time player stats.

The feature columns produced here are the same pa_* columns used in the
training feature table, so inference rows are consistent with training rows.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from config.settings import settings
from features.elo import EloComputer
from features.player_features import PlayerFeatureComputer

logger = logging.getLogger(__name__)

# Columns with >30% NaN across the player population are dropped
MAX_NAN_RATE = 0.30


class PlayerProfiler:
    """Build a profile table: one row per player, all pa_* statistics."""

    def __init__(self):
        self._feat_computer = PlayerFeatureComputer()
        self._elo_computer = EloComputer()

    def build_profiles(
        self,
        matches: pd.DataFrame,
        match_stats: pd.DataFrame,
        player_ids: list[str],
        players_meta: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Compute profiles for the given player IDs using their full match history.

        Parameters
        ----------
        matches      : silver matches table (all historical matches)
        match_stats  : silver match_player_stats table
        player_ids   : list of player IDs to profile (e.g. top 200)
        players_meta : silver players table (optional, for name/country)

        Returns
        -------
        DataFrame with one row per player, indexed by player_id.
        Columns are pa_* statistics (same names as training features).
        """
        matches = matches.copy()
        matches["match_date"] = pd.to_datetime(matches["match_date"], errors="coerce")
        matches = matches.sort_values("match_date", na_position="last").reset_index(drop=True)

        # Inject per-match Elo into matches so opponent quality can be computed.
        elo_ratings = self._elo_computer.compute(matches)
        matches_idx = matches.set_index("match_id")
        for col in ["player_a_elo_before", "player_b_elo_before"]:
            if col in elo_ratings.columns:
                matches_idx[col] = elo_ratings[col]
        matches = matches_idx.reset_index()

        # Build the per-player history dict once (same as training pipeline)
        logger.info("Building per-player history index for profiling…")
        player_hist = self._feat_computer._build_player_history(matches, match_stats)

        logger.info("Computing profiles for %d players…", len(player_ids))
        rows: list[dict[str, Any]] = []

        for pid in player_ids:
            pid_str = str(pid)
            hist = player_hist.get(pid_str, pd.DataFrame())

            if hist.empty:
                logger.debug("No history for player %s – skipping.", pid_str)
                continue

            # Use ALL history (no date cutoff – this is the current profile)
            feats = self._feat_computer._player_feats("pa", hist)
            feats["player_id"] = pid_str
            rows.append(feats)

        if not rows:
            logger.warning("No player profiles computed.")
            return pd.DataFrame()

        profiles = pd.DataFrame(rows).set_index("player_id")

        # Attach current Elo ratings (reuse the already-computed per-match Elo to get current ratings)
        logger.info("Computing current Elo ratings…")
        elo_ratings = self._elo_computer.current_ratings(matches.drop(
            columns=[c for c in ["player_a_elo_before", "player_b_elo_before"] if c in matches.columns],
            errors="ignore",
        ))
        if not elo_ratings.empty:
            profiles = profiles.join(elo_ratings, how="left")
            # Fill NaN Elo with initial rating so these columns survive the NaN filter.
            # Players with no match history in the dataset default to the initial rating.
            initial = self._elo_computer.initial
            for col in ["elo", "elo_clay", "elo_hard", "elo_grass"]:
                if col in profiles.columns:
                    profiles[col] = profiles[col].fillna(initial)
            logger.info("Elo attached for %d players.", profiles["elo"].notna().sum())

        # Attach metadata (name, country) if available
        if players_meta is not None and not players_meta.empty:
            meta_cols = []
            if "player_name" in players_meta.columns:
                meta_cols.append("player_name")
            if "country" in players_meta.columns:
                meta_cols.append("country")
            if meta_cols:
                meta = players_meta.set_index("player_id")[meta_cols]
                meta.index = meta.index.astype(str)
                profiles = profiles.join(meta, how="left")

        # Filter out columns with >30% NaN — but never drop surface-specific or Elo columns.
        # Surface stats are expected to be NaN for players who haven't played on that surface.
        nan_rates = profiles.select_dtypes(include=[np.number]).isna().mean()
        high_nan_cols = [
            c for c in nan_rates[nan_rates > MAX_NAN_RATE].index
            if not c.startswith("elo")
            and not any(f"_{s}_" in c or c.endswith(f"_{s}") for s in ("clay", "hard", "grass"))
        ]
        if high_nan_cols:
            logger.info(
                "Dropping %d columns with >%.0f%% NaN: %s",
                len(high_nan_cols), MAX_NAN_RATE * 100, high_nan_cols,
            )
            profiles = profiles.drop(columns=high_nan_cols)

        logger.info("Player profiles: %d players × %d features", *profiles.shape)
        return profiles


# ---------------------------------------------------------------------------
# Offline fallback: extract profiles from the gold feature table
# ---------------------------------------------------------------------------

def extract_profiles_from_gold(
    gold_df: pd.DataFrame, top_n: int = 200
) -> pd.DataFrame:
    """
    Extract one profile per player from the gold feature table.

    For each player we merge:
      - Their most recent overall pa_* row (career + last10 stats).
      - Their most recent row on each surface (to recover surface-specific
        stats that were NaN'd by masking on non-matching rows).
      - Their Elo columns (player_a_elo_before / player_a_elo_*_before)
        from the most recent overall row.
    """
    if gold_df.empty:
        return pd.DataFrame()

    SURFACES = ["clay", "hard", "grass"]

    if "year" in gold_df.columns:
        gold_df = gold_df.sort_values("year", ascending=False)

    pa_cols      = [c for c in gold_df.columns if c.startswith("pa_")]
    elo_a_cols   = [c for c in gold_df.columns if c.startswith("player_a_elo")]

    # Group rows by player_a_id for fast surface slicing
    has_id = "player_a_id" in gold_df.columns
    has_name = "player_a_name" in gold_df.columns

    # Build player list ordered by most-recent appearance
    seen_ids: set[str] = set()
    player_order: list[tuple[str, str]] = []  # (player_id, player_name)
    for _, row in gold_df.iterrows():
        pid  = str(row.get("player_a_id", "")) if has_id else ""
        name = str(row.get("player_a_name", "")).strip() if has_name else ""
        key  = pid if pid else name.lower()
        if not key or key in seen_ids:
            continue
        seen_ids.add(key)
        player_order.append((pid, name))
        if len(player_order) >= top_n:
            break

    rows: list[dict[str, Any]] = []
    for pid, name in player_order:
        # Subset of rows where this player is player_a
        if pid:
            sub = gold_df[gold_df["player_a_id"].astype(str) == pid]
        else:
            sub = gold_df[gold_df["player_a_name"].fillna("").str.strip() == name]

        if sub.empty:
            continue

        # Most recent row → career + last10 + Elo
        latest = sub.iloc[0]
        profile: dict[str, Any] = {}
        if name:
            profile["player_name"] = name
        if pid:
            profile["player_id"] = pid

        for c in pa_cols:
            val = latest.get(c)
            if isinstance(val, (int, float, np.integer, np.floating)) and val == val:
                profile[c] = float(val)

        # Elo: rename player_a_elo_before → elo, player_a_elo_clay_before → elo_clay …
        for c in elo_a_cols:
            val = latest.get(c)
            if not isinstance(val, (int, float, np.integer, np.floating)) or val != val:
                continue
            if c == "player_a_elo_before":
                profile["elo"] = float(val)
            else:
                # player_a_elo_clay_before → elo_clay
                short = c.replace("player_a_elo_", "").replace("_before", "")
                profile[f"elo_{short}"] = float(val)

        # Surface-specific pa_* — take the most recent row on each surface
        # where those columns are not NaN (masking left them populated only
        # for the matching surface).
        for surf in SURFACES:
            surf_col = f"surface_{surf}"
            if surf_col not in sub.columns:
                continue
            surf_rows = sub[sub[surf_col] == 1]
            if surf_rows.empty:
                continue
            surf_latest = surf_rows.iloc[0]
            surf_pa_cols = [c for c in pa_cols if f"_{surf}_" in c or c.endswith(f"_{surf}")]
            for c in surf_pa_cols:
                if c in profile:
                    continue  # already set
                val = surf_latest.get(c)
                if isinstance(val, (int, float, np.integer, np.floating)) and val == val:
                    profile[c] = float(val)

        rows.append(profile)

    if not rows:
        return pd.DataFrame()

    profiles = pd.DataFrame(rows)
    if "player_id" in profiles.columns:
        profiles = profiles.set_index("player_id")

    # Filter out columns with >30% NaN — but never drop Elo or surface-specific columns
    nan_rates = profiles.select_dtypes(include=[np.number]).isna().mean()
    high_nan_cols = [
        c for c in nan_rates[nan_rates > MAX_NAN_RATE].index
        if not c.startswith("elo")
        and not any(f"_{s}_" in c or c.endswith(f"_{s}") for s in ("clay", "hard", "grass"))
    ]
    if high_nan_cols:
        logger.info(
            "Dropping %d columns with >%.0f%% NaN: %s",
            len(high_nan_cols), MAX_NAN_RATE * 100, high_nan_cols,
        )
        profiles = profiles.drop(columns=high_nan_cols)

    logger.info(
        "Extracted %d player profiles from gold table (%d features)",
        len(profiles), len([c for c in profiles.columns if c.startswith("pa_")]),
    )
    return profiles


# ---------------------------------------------------------------------------
# Player tournament history: player_id × tournament_name → win_rate, n_matches
# ---------------------------------------------------------------------------

def build_player_tournament_history(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Compute each player's overall win rate and match count per tournament,
    using the full match history (no date cutoff — this is the current profile).

    Returns a DataFrame with columns:
        player_id, tournament_name, tournament_win_rate, tournament_matches
    """
    if matches.empty:
        return pd.DataFrame()

    matches = matches.copy()
    matches["match_date"] = pd.to_datetime(matches["match_date"], errors="coerce")

    required = {"player_a_id", "player_b_id", "winner_id", "tournament_name"}
    if not required.issubset(matches.columns):
        logger.warning("build_player_tournament_history: missing columns %s", required - set(matches.columns))
        return pd.DataFrame()

    rows = []
    for m in matches.itertuples(index=False):
        pa = str(m.player_a_id)
        pb = str(m.player_b_id)
        winner = str(m.winner_id)
        tourn = str(getattr(m, "tournament_name", "") or "").strip().lower()
        if not tourn:
            continue
        rows.append({"player_id": pa, "tournament_name": tourn, "won": int(winner == pa)})
        rows.append({"player_id": pb, "tournament_name": tourn, "won": int(winner == pb)})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    agg = (
        df.groupby(["player_id", "tournament_name"])["won"]
        .agg(tournament_win_rate="mean", tournament_matches="count")
        .reset_index()
    )
    logger.info(
        "Player tournament history: %d (player, tournament) pairs across %d players.",
        len(agg), agg["player_id"].nunique(),
    )
    return agg
