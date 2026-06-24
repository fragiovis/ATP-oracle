"""Feature engineering pipeline.

Assembles the gold-layer pre_match_features table by:
  1. Reading silver tables from S3.
  2. Computing Elo ratings.
  3. Computing player rolling features.
  4. Computing H2H features.
  5. Building match context features.
  6. Building interaction (matchup) features.
  7. Computing tournament features (t_* columns).
  8. Applying surface masking: NaN out inactive-surface columns per row.
  9. Joining all features + targets into one wide table.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config.settings import settings
from features.elo import EloComputer
from features.player_features import PlayerFeatureComputer
from features.h2h_features import H2HFeatureComputer
from features.interaction_features import compute_interaction_features
from features.targets import build_targets
from features.tournament_features import TournamentFeatureComputer

logger = logging.getLogger(__name__)

SURFACES = ["clay", "hard", "grass"]


class FeaturePipeline:
    def __init__(self):
        self.elo_computer = EloComputer()
        self.player_feat_computer = PlayerFeatureComputer()
        self.h2h_computer = H2HFeatureComputer()
        self.tournament_computer = TournamentFeatureComputer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        matches: pd.DataFrame,
        match_stats: pd.DataFrame,
        players: pd.DataFrame,
        h2h: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build the full pre_match_features table and tournament profiles.

        Parameters
        ----------
        matches      : silver matches table
        match_stats  : silver match_player_stats table
        players      : silver players table
        h2h          : silver head_to_head_history table

        Returns
        -------
        features_df       : Wide DataFrame indexed by match_id (gold layer).
        tournament_profiles : DataFrame indexed by tournament_name (t_* cols).
        """
        logger.info("Unifying player IDs (Sackmann ↔ tennis-data)…")
        matches = self._unify_player_ids(matches)

        logger.info("Computing Elo ratings…")
        elo_df = self.elo_computer.compute(matches)

        # Inject per-match Elo into matches so player_features can compute
        # opponent quality (avg Elo of opponents defeated in last 10 wins).
        matches_with_elo = matches.copy()
        matches_with_elo = matches_with_elo.set_index("match_id")
        for col in ["player_a_elo_before", "player_b_elo_before"]:
            if col in elo_df.columns:
                matches_with_elo[col] = elo_df[col]
        matches_with_elo = matches_with_elo.reset_index()

        logger.info("Computing player rolling features…")
        player_df = self.player_feat_computer.compute(matches_with_elo, match_stats)

        logger.info("Computing H2H features…")
        h2h_df = self.h2h_computer.compute(matches, h2h, match_stats=match_stats)

        logger.info("Computing match context features…")
        context_df = self._context_features(matches)

        logger.info("Joining feature blocks…")
        features = (
            context_df
            .join(elo_df, how="left")
            .join(player_df, how="left")
            .join(h2h_df, how="left")
        )

        logger.info("Computing interaction features…")
        interaction_df = compute_interaction_features(features)
        features = features.join(interaction_df, how="left")

        logger.info("Building target labels…")
        targets_df = build_targets(matches, match_stats)
        features = features.join(targets_df, how="left")

        # Drop rows with no label information at all
        label_cols = [c for c in targets_df.columns if c in features.columns]
        features = features.dropna(subset=label_cols, how="all")

        logger.info("Computing tournament features…")
        tournament_profiles, tourn_match_df = self.tournament_computer.compute(
            matches, match_stats, targets=targets_df
        )
        features = features.join(tourn_match_df, how="left")

        # --- Surface masking ---
        # NaN out all surface-specific columns for surfaces that are NOT
        # the active surface of each match row. This ensures models trained
        # on this table only see relevant surface data for each row — same
        # contract enforced at inference by the predictor.
        logger.info("Applying per-row surface masking…")
        features = self._apply_surface_masking(features)

        # --- Critical: randomise player_a/player_b roles ---
        # In Sackmann data player_a is ALWAYS the winner (winner_label=1 for
        # every row). We randomly swap half the rows so the winner model has a
        # balanced target and can actually learn something.
        logger.info("Randomising player roles (50%% swap) to balance winner_label…")
        features = self._randomise_roles(features)

        logger.info("Feature table shape: %s", features.shape)
        return features, tournament_profiles

    # ------------------------------------------------------------------
    # Surface masking
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_surface_masking(df: pd.DataFrame) -> pd.DataFrame:
        """
        For each row, NaN out all surface-specific columns that belong to
        surfaces OTHER than the active surface of that match.

        Surface-specific columns are identified by the pattern _{surface}_
        anywhere in the column name (e.g. pa_clay_career_win_rate,
        elo_hard_diff, ia_grass_serve_vs_return, win_rate_clay_diff).

        This enforces the same contract as the predictor at inference, so
        models can never learn spurious cross-surface correlations.
        """
        df = df.copy()

        # Pre-compute column sets per surface for speed
        surf_cols: dict[str, list[str]] = {}
        for s in SURFACES:
            surf_cols[s] = [
                c for c in df.columns
                if f"_{s}_" in c or c.endswith(f"_{s}")
            ]

        for s in SURFACES:
            # Rows where this surface is NOT active
            if "surface_" + s in df.columns:
                inactive = df["surface_" + s] == 0
            else:
                inactive = pd.Series(False, index=df.index)
            if inactive.any():
                df.loc[inactive, surf_cols[s]] = np.nan

        return df

    # ------------------------------------------------------------------
    # Player ID unification (Sackmann numeric IDs ↔ tennis-data td_* IDs)
    # ------------------------------------------------------------------

    @staticmethod
    def _unify_player_ids(matches: pd.DataFrame) -> pd.DataFrame:
        """
        Replace tennis-data.co.uk player IDs (td_*) with the corresponding
        Sackmann numeric ID, resolved by surname matching.

        Applies to player_a_id, player_b_id AND winner_id so that
        won_a comparisons in player_features work correctly.
        """
        matches = matches.copy()

        def _surname(name: str) -> str:
            parts = str(name).strip().split()
            if not parts:
                return ""
            last = parts[-1]
            # tennis-data trailing initial: "R.", "F.", "L.C."
            if last.endswith(".") and all(c.isalpha() or c == "." for c in last) and len(last) <= 5:
                return " ".join(parts[:-1]).lower().replace("-", " ")
            # Sackmann: "Given Surname" or "Given Double Surname"
            return " ".join(parts[1:]).lower().replace("-", " ") if len(parts) > 1 else parts[0].lower()

        # Step 1 — build surname → Sackmann ID from non-td rows
        surname_to_id: dict[str, str] = {}
        surname_ambiguous: set[str] = set()
        fullname_to_id: dict[str, str] = {}   # full lowercased name → id (for disambiguation)

        for col_id, col_name in [("player_a_id", "player_a_name"), ("player_b_id", "player_b_name")]:
            if col_id not in matches.columns or col_name not in matches.columns:
                continue
            mask = ~matches[col_id].astype(str).str.startswith("td_")
            for pid, name in zip(matches.loc[mask, col_id], matches.loc[mask, col_name]):
                if not pd.notna(name) or not str(name).strip():
                    continue
                s = _surname(name)
                full = str(name).strip().lower()
                fullname_to_id[full] = str(pid)
                if s in surname_to_id and surname_to_id[s] != str(pid):
                    surname_ambiguous.add(s)
                else:
                    surname_to_id[s] = str(pid)

        # Step 2 — resolve a tennis-data name to a canonical Sackmann ID
        def _resolve(name: str) -> str | None:
            s = _surname(name)
            if s in surname_ambiguous:
                parts = name.split()
                initial = parts[-1].rstrip(".").lower() if len(parts) >= 2 else ""
                for full, pid in fullname_to_id.items():
                    f_parts = full.split()
                    if f_parts and f_parts[-1] == s and f_parts[0].startswith(initial):
                        return pid
                return None
            return surname_to_id.get(s)

        # Step 3 — build original td_id → canonical map BEFORE modifying the DataFrame.
        # winner_id stores the original td_* value, so we need this pre-change mapping.
        # For players not found in Sackmann we still record their td_* → td_* (identity)
        # so that winner_id can always be aligned with player_a_id / player_b_id.
        td_to_canonical: dict[str, str] = {}
        for col_id, col_name in [("player_a_id", "player_a_name"), ("player_b_id", "player_b_name")]:
            if col_id not in matches.columns or col_name not in matches.columns:
                continue
            td_mask = matches[col_id].astype(str).str.startswith("td_")
            for td_id, name in zip(matches.loc[td_mask, col_id], matches.loc[td_mask, col_name]):
                td_id = str(td_id)
                canonical = _resolve(str(name).strip())
                # If not resolvable, keep td_* as-is (identity mapping)
                td_to_canonical[td_id] = canonical if canonical else td_id

        # Step 4 — apply to player_a_id, player_b_id, winner_id
        replaced = 0
        for col in ("player_a_id", "player_b_id", "winner_id"):
            if col not in matches.columns:
                continue
            td_mask = matches[col].astype(str).str.startswith("td_")
            for idx in matches[td_mask].index:
                mapped = td_to_canonical.get(str(matches.at[idx, col]))
                if mapped:
                    matches.at[idx, col] = mapped
                    if not mapped.startswith("td_"):
                        replaced += 1

        logger.info(
            "ID unification: %d td_* values replaced across player_a_id/player_b_id/winner_id "
            "(name map: %d unique surnames).",
            replaced, len(surname_to_id),
        )
        return matches

    # ------------------------------------------------------------------
    # Role randomisation (fix for Sackmann winner=player_a bias)
    # ------------------------------------------------------------------

    @staticmethod
    def _randomise_roles(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
        """
        Randomly swap player_a / player_b roles for 50% of rows.

        Swapped columns:
          pa_* ↔ pb_*
          player_a_* ↔ player_b_*
          ia_* ↔ ib_*
          elo_diff sign flipped
          player_a_id ↔ player_b_id
          player_a_name ↔ player_b_name
          winner_label flipped (0→1, 1→0)
          first_set_winner_label flipped
          player_aces_target set NaN (can't cheaply determine player_b aces)
        """
        rng = np.random.default_rng(seed)
        swap = rng.random(len(df)) < 0.5
        if not swap.any():
            return df

        df = df.copy()

        def _swap_pair(col_a: str, col_b: str) -> None:
            if col_a in df.columns and col_b in df.columns:
                tmp = df.loc[swap, col_a].copy()
                df.loc[swap, col_a] = df.loc[swap, col_b]
                df.loc[swap, col_b] = tmp

        # pa_* ↔ pb_*
        pa_cols = [c for c in df.columns if c.startswith("pa_")]
        for c in pa_cols:
            _swap_pair(c, "pb_" + c[3:])

        # player_a_* ↔ player_b_*
        pA_cols = [c for c in df.columns if c.startswith("player_a_")]
        for c in pA_cols:
            _swap_pair(c, "player_b_" + c[9:])

        # ia_* ↔ ib_*
        ia_cols = [c for c in df.columns if c.startswith("ia_")]
        for c in ia_cols:
            _swap_pair(c, "ib_" + c[3:])

        # Negate signed diff/gap columns — their sign flips when roles swap.
        # ia_*/ib_* are already handled by the swap above (ia↔ib), so they
        # must NOT be negated here — swap alone is the correct transform.
        diff_cols = [
            c for c in df.columns
            if (
                "_diff" in c
                or c == "form_diff_last5"
            )
            and not c.startswith("ia_")
            and not c.startswith("ib_")
        ]
        for col in diff_cols:
            df.loc[swap, col] = -df.loc[swap, col]

        # Flip binary labels
        for col in ("winner_label", "first_set_winner_label"):
            if col in df.columns:
                df.loc[swap, col] = (1 - df.loc[swap, col]).where(
                    df.loc[swap, col].notna()
                )

        # H2H: swap wins_a ↔ wins_b and flip rate columns
        _swap_pair("h2h_wins_a", "h2h_wins_b")
        for col in ("h2h_win_rate_a", "h2h_last5_win_rate_a", "h2h_win_rate_a_surface"):
            if col in df.columns:
                # rate = wins_a / total → when roles swap becomes 1 - rate
                df.loc[swap, col] = (1 - df.loc[swap, col]).where(
                    df.loc[swap, col].notna()
                )

        # player_aces_target can't be cheaply swapped → set NaN for swapped rows
        if "player_aces_target" in df.columns:
            df.loc[swap, "player_aces_target"] = np.nan

        return df

    # ------------------------------------------------------------------
    # Match context features
    # ------------------------------------------------------------------

    @staticmethod
    def _context_features(matches: pd.DataFrame) -> pd.DataFrame:
        df = matches.set_index("match_id").copy()
        ctx = pd.DataFrame(index=df.index)

        # Player IDs and names (needed by the predictor for inference look-ups)
        for col in ["player_a_id", "player_b_id", "winner_id", "player_a_name", "player_b_name"]:
            if col in df.columns:
                ctx[col] = df[col]

        # Surface (one-hot)
        for surface in ["clay", "hard", "grass"]:
            ctx[f"surface_{surface}"] = (
                df["surface"].str.contains(surface, na=False, case=False).astype(int)
            )

        # Best-of
        if "best_of" in df.columns:
            ctx["best_of_5"] = (df["best_of"] == 5).astype(int)

        # Round (ordinal encoding)
        round_map = {
            "R128": 1, "R64": 2, "R32": 3, "R16": 4,
            "QF": 5, "SF": 6, "F": 7, "RR": 3,
            "1R": 1, "2R": 2, "3R": 3, "4R": 4,
        }
        if "round" in df.columns:
            ctx["round_ordinal"] = df["round"].map(round_map).fillna(3).astype(int)

        # Tournament category (ordinal)
        cat_map = {"250": 1, "500": 2, "1000": 3, "GS": 4, "ATP 250": 1, "ATP 500": 2, "ATP 1000": 3, "Grand Slam": 4}
        if "tournament_category" in df.columns:
            ctx["tournament_category_ordinal"] = (
                df["tournament_category"].map(cat_map).fillna(1).astype(int)
            )

        # Season / month / date
        if "match_date" in df.columns:
            dt = pd.to_datetime(df["match_date"], errors="coerce")
            ctx["match_date"] = dt.dt.strftime("%Y-%m-%d")  # kept as string for join
            ctx["month"] = dt.dt.month.fillna(0).astype(int)
            ctx["year"] = dt.dt.year.fillna(0).astype(int)

        # Grand Slam flags — each Slam has distinct rules (surface, tiebreak policy, best-of)
        # Names are normalised across Sackmann and tennis-data sources
        if "tournament_name" in df.columns:
            tn = df["tournament_name"].fillna("").str.strip().str.lower()
            ctx["is_australian_open"] = tn.str.contains("australian open").astype(int)
            ctx["is_roland_garros"]   = tn.isin(["french open", "roland garros"]).astype(int)
            ctx["is_wimbledon"]       = tn.str.contains("wimbledon").astype(int)
            ctx["is_us_open"]         = tn.str.contains("us open").astype(int)

        return ctx
