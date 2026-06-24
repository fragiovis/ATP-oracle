"""Inference layer: given a structured match request, return all predictions.

Workflow:
  1. Accept a MatchRequest (player_a, player_b, surface, context).
  2. Look up both players' pre-computed profiles from /data/players/.
  3. Assemble pa_*/pb_* into a single feature row + context + interaction features.
  4. Load all trained models (lazy, cached in memory).
  5. Run inference for all targets.
  6. Return a MatchPrediction dict.

Design note: this module is intentionally decoupled from the agent layer.
It only accepts structured input and calls models; it does no NLP.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from models.targets import ALL_MODELS, BaseModel
from models.trainer import ModelTrainer
from storage.writer import StorageWriter

try:
    import shap as _shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response types
# ---------------------------------------------------------------------------

@dataclass
class MatchRequest:
    player_a: str
    player_b: str
    surface: str = "hard"
    tournament_category: str = "ATP 250"
    tournament_name: str = ""          # e.g. "Wimbledon", "Roland Garros"
    round: str = "R32"
    best_of: int = 3                   # 3 = bo3 (default ATP), 5 = Grand Slam / Davis Cup
    match_date: str | None = None


@dataclass
class MatchPrediction:
    win_probability_player_a: float | None = None
    win_probability_player_b: float | None = None
    predicted_winner: str | None = None
    expected_aces_player_a: float | None = None
    expected_total_aces: float | None = None
    expected_total_games: float | None = None
    expected_sets_played: float | None = None
    prob_tiebreak: float | None = None
    prob_player_a_first_set: float | None = None
    explanation: list[dict] | None = None   # top SHAP features for winner model
    players_not_found: list[str] | None = None  # names of players with no profile

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if v is not None}
        # Always include players_not_found even if empty list
        if "players_not_found" not in d:
            d["players_not_found"] = []
        return d


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

def _align(model, df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has all columns the model expects.
    Missing columns are added as NaN — LightGBM/XGBoost handle NaN natively."""
    missing = [c for c in model.feature_cols if c not in df.columns]
    if missing:
        df = df.copy()
        for c in missing:
            df[c] = np.nan
    return df[model.feature_cols]


class MatchPredictor:
    """
    Load player profiles + trained models once and serve predictions.

    Player profiles are read from /data/players/ (one row per player,
    columns = pa_* statistics). Tournament profiles are read from
    /data/tournaments/ (one row per tournament, columns = t_* statistics).
    At inference time we look up both players, rename player_b's pa_* → pb_*,
    add context, interaction, and tournament features, apply surface masking,
    and pass the assembled row to all models.
    """

    def __init__(self):
        self._trainer = ModelTrainer()
        self._writer = StorageWriter()
        self._models: dict[str, BaseModel] = {}
        self._profiles: pd.DataFrame | None = None
        self._tournament_profiles: pd.DataFrame | None = None
        self._player_tournament_history: pd.DataFrame | None = None
        self._rankings: pd.DataFrame | None = None  # player_name → rank

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, request: MatchRequest) -> MatchPrediction:
        """Produce all pre-match predictions for the given match request."""
        self._ensure_loaded()

        self._last_players_not_found = []
        feature_row = self._build_feature_row(request)
        df = pd.DataFrame([feature_row])

        pred = MatchPrediction()
        pred.players_not_found = self._last_players_not_found or []

        # Winner probability
        # Run the model twice (A vs B and B vs A) and average to cancel out
        # the positional bias caused by asymmetric pa_*/pb_* feature importances.
        if "winner" in self._models:
            m = self._models["winner"]
            try:
                proba_ab = m.predict_proba(_align(m, df))
                p_ab = float(proba_ab[0][1]) if proba_ab.ndim == 2 else float(proba_ab[0])

                request_ba = MatchRequest(
                    player_a=request.player_b,
                    player_b=request.player_a,
                    surface=request.surface,
                    tournament_category=request.tournament_category,
                    tournament_name=request.tournament_name,
                    round=request.round,
                    best_of=request.best_of,
                    match_date=request.match_date,
                )
                feature_row_ba = self._build_feature_row(request_ba)
                df_ba = pd.DataFrame([feature_row_ba])
                proba_ba = m.predict_proba(_align(m, df_ba))
                p_ba = float(proba_ba[0][1]) if proba_ba.ndim == 2 else float(proba_ba[0])

                p_a = (p_ab + (1 - p_ba)) / 2
                pred.win_probability_player_a = round(p_a, 4)
                pred.win_probability_player_b = round(1 - p_a, 4)
                pred.predicted_winner = (
                    request.player_a if p_a >= 0.5 else request.player_b
                )
            except Exception as e:
                logger.warning("Winner model error: %s", e)

        # Tiebreak probability
        if "tiebreak" in self._models:
            m = self._models["tiebreak"]
            try:
                proba = m.predict_proba(_align(m, df))
                pred.prob_tiebreak = round(float(proba[0][1]), 4)
            except Exception as e:
                logger.warning("Tiebreak model error: %s", e)

        # First set winner probability
        if "first_set" in self._models:
            m = self._models["first_set"]
            try:
                proba = m.predict_proba(_align(m, df))
                pred.prob_player_a_first_set = round(float(proba[0][1]), 4)
            except Exception as e:
                logger.warning("First set model error: %s", e)

        # Player aces (regression)
        if "player_aces" in self._models:
            m = self._models["player_aces"]
            try:
                pred.expected_aces_player_a = round(float(m.predict(_align(m, df))[0]), 2)
            except Exception as e:
                logger.warning("Player aces model error: %s", e)

        # Total aces (regression)
        if "total_aces" in self._models:
            m = self._models["total_aces"]
            try:
                pred.expected_total_aces = round(float(m.predict(_align(m, df))[0]), 2)
            except Exception as e:
                logger.warning("Total aces model error: %s", e)

        # Total games (regression)
        if "total_games" in self._models:
            m = self._models["total_games"]
            try:
                pred.expected_total_games = round(float(m.predict(_align(m, df))[0]), 2)
            except Exception as e:
                logger.warning("Total games model error: %s", e)

        # Sets played (regression)
        if "sets_played" in self._models:
            m = self._models["sets_played"]
            try:
                pred.expected_sets_played = round(float(m.predict(_align(m, df))[0]), 2)
            except Exception as e:
                logger.warning("Sets played model error: %s", e)

        # SHAP explanation for the winner model
        if "winner" in self._models and _SHAP_AVAILABLE:
            try:
                pred.explanation = self._explain(self._models["winner"], df)
            except Exception as e:
                logger.debug("SHAP explanation failed: %s", e)

        return pred

    # ------------------------------------------------------------------
    # Feature construction for inference (NEW ARCHITECTURE)
    # ------------------------------------------------------------------

    def _build_feature_row(self, req: MatchRequest) -> dict[str, Any]:
        """
        Build a feature dict for an unseen match.

        Strategy:
          1. Look up player_a's profile from /data/players/ → pa_* columns.
          2. Look up player_b's profile → pa_* columns, rename to pb_*.
          3. Compute interaction features (diffs, serve-vs-return, etc.).
          4. Add context features (surface, round, tournament, best_of).
          5. Apply format-specific stat override (BO3/BO5).
        """
        profile_a = self._lookup_player(req.player_a)
        profile_b = self._lookup_player(req.player_b)

        self._last_players_not_found = []
        if profile_a is None:
            logger.warning("No profile found for player_a=%s", req.player_a)
            self._last_players_not_found.append(req.player_a)
            profile_a = {}
        if profile_b is None:
            logger.warning("No profile found for player_b=%s", req.player_b)
            self._last_players_not_found.append(req.player_b)
            profile_b = {}

        # Rename player_b profile: pa_* → pb_*, elo → player_b_elo_*
        pb_renamed: dict[str, Any] = {}
        for k, v in profile_b.items():
            if k.startswith("pa_"):
                pb_renamed["pb_" + k[3:]] = v
            elif k == "elo":
                pb_renamed["player_b_elo_before"] = v
            elif k.startswith("elo_"):
                pb_renamed["player_b_" + k + "_before"] = v
            else:
                pb_renamed[k] = v

        # Rename player_a Elo columns
        if "elo" in profile_a:
            profile_a["player_a_elo_before"] = profile_a.pop("elo")
        for s in ("clay", "hard", "grass"):
            k = "elo_" + s
            if k in profile_a:
                profile_a["player_a_" + k + "_before"] = profile_a.pop(k)

        row: dict[str, Any] = {**profile_a, **pb_renamed}

        # Compute elo_diff
        pa_elo = row.get("player_a_elo_before", 1500.0)
        pb_elo = row.get("player_b_elo_before", 1500.0)
        row["elo_diff"] = float(pa_elo) - float(pb_elo)

        # Per-surface elo diffs
        for s in ("clay", "hard", "grass"):
            ea = row.get("player_a_elo_{}_before".format(s), 1500.0)
            eb = row.get("player_b_elo_{}_before".format(s), 1500.0)
            row["elo_{}_diff".format(s)] = float(ea) - float(eb)

        # Absolute elo diff (for deciding_set / tiebreak)
        row["elo_abs_diff"] = abs(row["elo_diff"])

        # ── Interaction features ────────────────────────────────────────
        def diff_feat(a: str, b: str) -> float | None:
            av, bv = row.get(a), row.get(b)
            if av is not None and bv is not None:
                return float(av) - float(bv)
            return None

        for out, a, b in [
            ("ia_serve1_vs_return",        "pa_career_first_serve_pts_won_pct",   "pb_career_return_pts_won_pct"),
            ("ib_serve1_vs_return",        "pb_career_first_serve_pts_won_pct",   "pa_career_return_pts_won_pct"),
            ("ia_serve1_vs_return_l10",    "pa_last10_first_serve_pts_won_pct",   "pb_last10_return_pts_won_pct"),
            ("ib_serve1_vs_return_l10",    "pb_last10_first_serve_pts_won_pct",   "pa_last10_return_pts_won_pct"),
            ("ia_bp_edge",                 "pa_career_bp_saved_pct",              "pb_career_bp_converted_pct"),
            ("ib_bp_edge",                 "pb_career_bp_saved_pct",              "pa_career_bp_converted_pct"),
            ("ace_rate_diff",              "pa_career_aces_per_match",            "pb_career_aces_per_match"),
            ("ace_rate_diff_l10",          "pa_last10_aces_per_match",            "pb_last10_aces_per_match"),
            ("form_diff_last5",            "pa_form_last5",                       "pb_form_last5"),
            ("win_rate_diff_career",       "pa_career_win_rate",                  "pb_career_win_rate"),
            ("win_rate_diff_last10",       "pa_last10_win_rate",                  "pb_last10_win_rate"),
            ("ace_rate_diff_bo5",          "pa_bo5_career_aces_per_match",        "pb_bo5_career_aces_per_match"),
            ("ace_rate_diff_bo3",          "pa_bo3_career_aces_per_match",        "pb_bo3_career_aces_per_match"),
            ("win_rate_diff_bo5",          "pa_bo5_career_win_rate",              "pb_bo5_career_win_rate"),
            ("win_rate_diff_bo3",          "pa_bo3_career_win_rate",              "pb_bo3_career_win_rate"),
        ]:
            v = diff_feat(a, b)
            if v is not None:
                row[out] = v

        # Surface-specific win-rate diffs
        for s in ("clay", "hard", "grass"):
            v = diff_feat(f"pa_{s}_career_win_rate", f"pb_{s}_career_win_rate")
            if v is not None:
                row[f"win_rate_{s}_diff"] = v

        # Surface-specific serve vs return
        for s in ("clay", "hard", "grass"):
            v = diff_feat(f"pa_{s}_career_service_pts_won_pct", f"pb_{s}_career_return_pts_won_pct")
            if v is not None:
                row[f"ia_{s}_serve_vs_return"] = v
            v = diff_feat(f"pb_{s}_career_service_pts_won_pct", f"pa_{s}_career_return_pts_won_pct")
            if v is not None:
                row[f"ib_{s}_serve_vs_return"] = v

        # Surface form (last 20): map from surface-specific career win rate in the profile.
        # The profile stores pa_{surface}_career_win_rate computed from last 20 surface matches.
        # At inference we expose it as pa_surface_win_rate_last20 for the active surface only.
        active_surface = (req.surface or "hard").lower()
        for s in ("clay", "hard", "grass"):
            key = f"pa_{s}_career_win_rate"
            if key in row:
                row[f"pa_surface_win_rate_last20"] = row[key] if s in active_surface else np.nan
                break
        for s in ("clay", "hard", "grass"):
            key = f"pb_{s}_career_win_rate"
            if key in row:
                row[f"pb_surface_win_rate_last20"] = row[key] if s in active_surface else np.nan
                break
        # Overwrite with the correct active surface value
        row["pa_surface_win_rate_last20"] = row.get(f"pa_{active_surface}_career_win_rate", np.nan)
        row["pb_surface_win_rate_last20"] = row.get(f"pb_{active_surface}_career_win_rate", np.nan)
        v = diff_feat("pa_surface_win_rate_last20", "pb_surface_win_rate_last20")
        if v is not None:
            row["win_rate_surface_last20_diff"] = v

        # Opponent quality and tournament history: not available at inference from profile alone.
        # Default to NaN — LightGBM handles missing values natively and the model
        # learns these are often unknown at inference.
        if "pa_avg_opp_elo_won_last10" not in row:
            row["pa_avg_opp_elo_won_last10"] = np.nan
        if "pb_avg_opp_elo_won_last10" not in row:
            row["pb_avg_opp_elo_won_last10"] = np.nan
        row["avg_opp_elo_won_diff"] = np.nan

        # Tournament history: look up from pre-computed player×tournament table
        tn_lower = (req.tournament_name or "").strip().lower()
        pa_tid = self._lookup_player_id(req.player_a)
        pb_tid = self._lookup_player_id(req.player_b)
        th = self._get_player_tournament_history()

        def _tourn_feats(player_id: str | None) -> tuple[float, int]:
            if player_id is None or th.empty or not tn_lower:
                return 0.0, 0
            mask = (th["player_id"].astype(str) == str(player_id)) & (th["tournament_name"] == tn_lower)
            sub = th[mask]
            if sub.empty:
                return 0.0, 0
            r = sub.iloc[0]
            return float(r["tournament_win_rate"]), int(r["tournament_matches"])

        pa_twr, pa_tm = _tourn_feats(pa_tid)
        pb_twr, pb_tm = _tourn_feats(pb_tid)
        row["pa_tournament_win_rate"] = pa_twr
        row["pa_tournament_matches"]  = pa_tm
        row["pb_tournament_win_rate"] = pb_twr
        row["pb_tournament_matches"]  = pb_tm
        row["tournament_win_rate_diff"] = pa_twr - pb_twr

        # Absolute gaps (for deciding_set / tiebreak models)
        if "win_rate_diff_career" in row:
            row["win_rate_abs_diff_career"] = abs(row["win_rate_diff_career"])

        # ── ATP ranking (live snapshot) ─────────────────────────────────
        pa_rank = self._lookup_rank(req.player_a)
        pb_rank = self._lookup_rank(req.player_b)
        row["pa_atp_rank"] = pa_rank
        row["pb_atp_rank"] = pb_rank
        if not (pa_rank != pa_rank) and not (pb_rank != pb_rank):  # both non-NaN
            row["atp_rank_diff"] = pa_rank - pb_rank  # negative = A is higher ranked
        else:
            row["atp_rank_diff"] = float("nan")

        # ── Context features ────────────────────────────────────────────
        surface = (req.surface or "hard").lower()
        for s in ["clay", "hard", "grass"]:
            row[f"surface_{s}"] = int(s in surface)

        round_map = {"R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6, "F": 7}
        cat_map   = {"ATP 250": 1, "250": 1, "ATP 500": 2, "500": 2,
                     "ATP 1000": 3, "1000": 3, "Grand Slam": 4, "GS": 4}
        row["best_of_5"]                    = int(req.best_of == 5)
        row["round_ordinal"]                = round_map.get(req.round, 3)
        row["tournament_category_ordinal"]  = cat_map.get(req.tournament_category, 1)

        # Grand Slam tournament flags
        tn = (req.tournament_name or "").strip().lower()
        row["is_australian_open"] = int("australian" in tn)
        row["is_roland_garros"]   = int(tn in ("french open", "roland garros") or "roland" in tn or "french open" in tn)
        row["is_wimbledon"]       = int("wimbledon" in tn)
        row["is_us_open"]         = int("us open" in tn)

        # ── Tournament features ─────────────────────────────────────────
        t_profile = self._lookup_tournament(req.tournament_name)
        if t_profile:
            row.update(t_profile)

        # ── Surface masking ─────────────────────────────────────────────
        # NaN out all surface-specific columns for inactive surfaces.
        # This mirrors the masking applied during training (pipeline._apply_surface_masking)
        # so models see the same data distribution at inference.
        for s in ("clay", "hard", "grass"):
            if s not in surface:
                for col in list(row.keys()):
                    if f"_{s}_" in col or col.endswith(f"_{s}"):
                        row[col] = np.nan

        return row

    # ------------------------------------------------------------------
    # Player profile lookup
    # ------------------------------------------------------------------

    def _lookup_player(self, player_name: str) -> dict[str, Any] | None:
        """
        Find a player in the profiles table by name (fuzzy match).
        Returns their pa_* stats as a dict.
        """
        profiles = self._get_profiles()
        if profiles.empty:
            return None

        name_lower = player_name.lower().strip()

        # Try exact substring match on player_name column
        if "player_name" in profiles.columns:
            mask = profiles["player_name"].fillna("").str.lower().apply(
                lambda n: name_lower in n or any(
                    tok in name_lower for tok in n.split() if len(tok) > 2
                )
            )
            matched = profiles[mask]
            if not matched.empty:
                row = matched.iloc[0]
                return self._row_to_profile(row)

        # Fallback: try player_id index if it looks like a name was passed
        # (shouldn't normally happen since profiles have player_name)
        return None

    def _row_to_profile(self, row: pd.Series) -> dict[str, Any]:
        """Extract pa_* and elo columns from a profile row."""
        return {
            k: v for k, v in row.items()
            if isinstance(v, (int, float, np.integer, np.floating))
            and (str(k).startswith("pa_") or str(k).startswith("elo"))
        }

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if not self._models:
            self._load_models()
        if self._tournament_profiles is None:
            self._get_tournament_profiles()
        if self._player_tournament_history is None:
            self._get_player_tournament_history()
        if self._rankings is None:
            self._load_rankings()

    def _load_models(self) -> None:
        logger.info("Loading trained models…")
        for model_cls in ALL_MODELS:
            try:
                m = self._trainer.load_latest_model(model_cls)
                self._models[m.model_id] = m
                logger.info("  Loaded: %s", m.model_id)
            except FileNotFoundError:
                logger.warning("  Not found: %s", model_cls.model_id)

    def _get_profiles(self) -> pd.DataFrame:
        """Load player profiles from /data/players/ (cached)."""
        if self._profiles is None:
            try:
                self._profiles = self._writer.read_player_profiles()
                logger.info("Loaded %d player profiles.", len(self._profiles))
            except Exception as e:
                logger.warning("Could not load player profiles: %s", e)
                self._profiles = pd.DataFrame()
        return self._profiles

    def _lookup_player_id(self, player_name: str) -> str | None:
        """Return the player_id (index) for a player name, or None if not found."""
        profiles = self._get_profiles()
        if profiles.empty:
            return None
        name_lower = player_name.lower().strip()
        if "player_name" in profiles.columns:
            mask = profiles["player_name"].fillna("").str.lower().apply(
                lambda n: name_lower in n or any(tok in name_lower for tok in n.split() if len(tok) > 2)
            )
            matched = profiles[mask]
            if not matched.empty:
                return str(matched.index[0])
        return None

    def _get_player_tournament_history(self) -> pd.DataFrame:
        """Load player×tournament history table (cached)."""
        if self._player_tournament_history is None:
            try:
                self._player_tournament_history = self._writer.read_player_tournament_history()
                logger.info("Loaded player tournament history (%d rows).", len(self._player_tournament_history))
            except Exception as e:
                logger.warning("Could not load player tournament history: %s", e)
                self._player_tournament_history = pd.DataFrame()
        return self._player_tournament_history

    def _lookup_tournament(self, tournament_name: str | None) -> dict[str, Any] | None:
        """Find tournament profile by name (exact or fuzzy substring match)."""
        if not tournament_name:
            return None
        profiles = self._get_tournament_profiles()
        if profiles.empty:
            return None

        tn_lower = tournament_name.strip().lower()
        idx = profiles.index.str.lower()

        # Exact match first
        exact = profiles[idx == tn_lower]
        if not exact.empty:
            return self._row_to_tournament_feats(exact.iloc[0])

        # Substring match
        sub = profiles[idx.str.contains(tn_lower, regex=False, na=False)]
        if not sub.empty:
            return self._row_to_tournament_feats(sub.iloc[0])

        return None

    def _row_to_tournament_feats(self, row: pd.Series) -> dict[str, Any]:
        return {
            k: v for k, v in row.items()
            if str(k).startswith("t_") and pd.notnull(v)
        }

    def _load_rankings(self) -> None:
        """Load ATP rankings snapshot from S3 (player_name → rank mapping)."""
        try:
            df = self._writer.read_rankings_snapshot()
            if not df.empty and "player_name" in df.columns and "rank" in df.columns:
                # Keep the most recent snapshot if multiple dates are present
                if "snapshot_date" in df.columns:
                    latest = df["snapshot_date"].max()
                    df = df[df["snapshot_date"] == latest]
                self._rankings = df[["player_name", "rank"]].copy()
                self._rankings["player_name_lower"] = self._rankings["player_name"].str.lower().str.strip()
                logger.info("Loaded ATP rankings snapshot: %d players (date=%s)",
                            len(self._rankings),
                            df["snapshot_date"].iloc[0] if "snapshot_date" in df.columns else "unknown")
            else:
                logger.warning("Rankings snapshot empty or missing columns — rank features will be NaN")
                self._rankings = pd.DataFrame()
        except Exception as e:
            logger.warning("Could not load rankings snapshot: %s", e)
            self._rankings = pd.DataFrame()

    def _lookup_rank(self, player_name: str) -> float:
        """Return the current ATP rank for a player, or NaN if not found."""
        if self._rankings is None or self._rankings.empty:
            return float("nan")
        name_lower = player_name.lower().strip()
        # Try token-based fuzzy match (same logic as _lookup_player)
        mask = self._rankings["player_name_lower"].apply(
            lambda n: name_lower in n or any(tok in name_lower for tok in n.split() if len(tok) > 2)
        )
        matched = self._rankings[mask]
        if not matched.empty:
            return float(matched.iloc[0]["rank"])
        return float("nan")

    def _get_tournament_profiles(self) -> pd.DataFrame:
        """Load tournament profiles from /data/tournaments/ (cached)."""
        if self._tournament_profiles is None:
            try:
                df = self._writer.read_tournament_profiles()
                if "tournament_name" in df.columns:
                    df = df.set_index("tournament_name")
                self._tournament_profiles = df
                logger.info("Loaded %d tournament profiles.", len(self._tournament_profiles))
            except Exception as e:
                logger.warning("Could not load tournament profiles: %s", e)
                self._tournament_profiles = pd.DataFrame()
        return self._tournament_profiles

    # ------------------------------------------------------------------
    # SHAP explanations
    # ------------------------------------------------------------------

    def _explain(self, model, df: pd.DataFrame, top_n: int = 8) -> list[dict]:
        """Return top SHAP feature contributions for the winner model."""
        aligned = _align(model, df)
        X = aligned[model.feature_cols].fillna(0)

        explainer = _shap.TreeExplainer(model.model)
        shap_vals = explainer.shap_values(X)

        # For binary classifiers shap_values returns [neg_class, pos_class]
        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            shap_vals = shap_vals[1]  # class 1 = player_a wins

        vals = shap_vals[0]  # single row
        raw_vals = X.iloc[0]   # actual feature values for this row
        pairs = sorted(zip(model.feature_cols, vals), key=lambda x: abs(x[1]), reverse=True)

        LABEL_MAP = {
            # Elo
            "elo_diff":                            "Elo gap (A − B)",
            "elo_clay_diff":                       "Clay Elo gap",
            "elo_hard_diff":                       "Hard Elo gap",
            "elo_grass_diff":                      "Grass Elo gap",
            "player_a_elo_before":                 "A Elo rating",
            "player_b_elo_before":                 "B Elo rating",
            # Win-rate
            "win_rate_diff_career":                "Career win-rate gap",
            "win_rate_diff_last10":                "Last-10 win-rate gap",
            "win_rate_clay_diff":                  "Clay win-rate gap",
            "win_rate_hard_diff":                  "Hard win-rate gap",
            "win_rate_grass_diff":                 "Grass win-rate gap",
            "form_diff_last5":                     "Recent form gap (last 5)",
            # Serve
            "ia_serve1_vs_return":                 "A serve vs B return (career)",
            "ib_serve1_vs_return":                 "B serve vs A return (career)",
            "ia_serve1_vs_return_l10":             "A serve vs B return (last 10)",
            "ib_serve1_vs_return_l10":             "B serve vs A return (last 10)",
            "ia_clay_serve_vs_return":             "A serve vs B return on clay",
            "ib_clay_serve_vs_return":             "B serve vs A return on clay",
            "ia_hard_serve_vs_return":             "A serve vs B return on hard",
            "ib_hard_serve_vs_return":             "B serve vs A return on hard",
            "ia_grass_serve_vs_return":            "A serve vs B return on grass",
            "ib_grass_serve_vs_return":            "B serve vs A return on grass",
            "pa_career_service_pts_won_pct":       "A service pts won (career)",
            "pb_career_service_pts_won_pct":       "B service pts won (career)",
            "pa_last10_service_pts_won_pct":       "A service pts won (last 10)",
            "pb_last10_service_pts_won_pct":       "B service pts won (last 10)",
            "ace_rate_diff_l10":                   "Ace rate gap (last 10)",
            # H2H
            "h2h_win_rate_a":                      "H2H win rate (A vs B, overall)",
            "h2h_win_rate_a_surface":              "H2H win rate (A vs B, this surface)",
            "h2h_last5_win_rate_a":                "H2H win rate (last 5 meetings)",
            "h2h_total_matches":                   "H2H matches played",
            "h2h_days_since_last":                 "Days since last H2H meeting",
            # Surface form (last 20)
            "pa_surface_win_rate_last20":          "A win rate on surface (last 20)",
            "pb_surface_win_rate_last20":          "B win rate on surface (last 20)",
            "win_rate_surface_last20_diff":        "Surface win-rate gap (last 20)",
            # Opponent quality
            "pa_avg_opp_elo_won_last10":           "A avg opponent Elo (last 10 wins)",
            "pb_avg_opp_elo_won_last10":           "B avg opponent Elo (last 10 wins)",
            "avg_opp_elo_won_diff":                "Opponent quality gap (last 10 wins)",
            # Tournament history
            "pa_tournament_win_rate":              "A tournament win rate",
            "pb_tournament_win_rate":              "B tournament win rate",
            "pa_tournament_matches":               "A matches in this tournament",
            "pb_tournament_matches":               "B matches in this tournament",
            "tournament_win_rate_diff":            "Tournament win-rate gap",
            # ATP ranking
            "pa_atp_rank":                         "A current ATP rank",
            "pb_atp_rank":                         "B current ATP rank",
            "atp_rank_diff":                       "ATP rank gap (A − B, negative = A higher)",
            # Context
            "surface_clay":                        "Surface: clay",
            "surface_hard":                        "Surface: hard",
            "surface_grass":                       "Surface: grass",
            "tournament_category_ordinal":         "Tournament importance",
            "round_ordinal":                       "Round",
            "best_of_5":                           "Best of 5",
            # Tournament profile
            "t_avg_aces_per_match":                "Tournament avg aces/match",
            "t_avg_games_per_match":               "Tournament avg games/match",
            "t_avg_sets_played":                   "Tournament avg sets played",
            "t_tiebreak_pct":                      "Tournament tiebreak frequency",
            "t_avg_service_pts_won_pct":           "Tournament avg service pts won",
        }

        result = []
        for feat, val in pairs[:top_n]:
            if abs(val) < 0.001:
                continue
            label = LABEL_MAP.get(feat, feat.replace("_", " "))
            fv = raw_vals.get(feat, None)
            result.append({
                "feature":       feat,
                "label":         label,
                "value":         round(float(val), 4),
                "feature_value": round(float(fv), 4) if fv is not None and fv == fv else None,
                "favors":        "player_a" if val > 0 else "player_b",
            })
        return result
