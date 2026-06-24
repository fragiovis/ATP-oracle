"""Concrete model classes – one per prediction target.

Each class extends BaseModel and defines:
  - target_name / task / model_id
  - _build_estimator()
  - feature_include: explicit whitelist of feature columns for this task

Changes vs previous version:
  - All models include tournament features (t_* columns)
  - H2H features properly included in winner/tiebreak models
  - bo3/bo5 stratified stats removed (too sparse)
  - bp_saved_pct / bp_converted_pct removed (too noisy)
  - Surface masking applied at training time (see pipeline._apply_surface_masking)
    so each model's surface-specific features are valid at inference too
  - winner model uses h2h_win_rate_a, h2h_win_rate_a_surface, h2h_last5_win_rate_a
"""

from __future__ import annotations

import lightgbm as lgb
import xgboost as xgb

from config.settings import settings
from models.base import BaseModel


# ---------------------------------------------------------------------------
# Model 1: Winner
# ---------------------------------------------------------------------------

_WINNER_FEATURES = frozenset({
    # Context
    "best_of_5", "round_ordinal", "tournament_category_ordinal",
    "surface_clay", "surface_hard", "surface_grass",
    # Grand Slam flags
    "is_australian_open", "is_roland_garros", "is_wimbledon", "is_us_open",
    # Elo — absolute level + overall diff + per-surface diff
    "player_a_elo_before", "player_b_elo_before",
    "elo_diff",
    "elo_clay_diff", "elo_hard_diff", "elo_grass_diff",
    # Recent form
    "win_rate_diff_last10", "form_diff_last5",
    # Surface form (last 20, diff only — inactive surfaces NaN'd by masking)
    "win_rate_surface_last20_diff",
    # Opponent quality (diff only)
    "avg_opp_elo_won_diff",
    # Tournament-specific player history (diff + counts for calibration)
    "pa_tournament_win_rate", "pb_tournament_win_rate",
    "pa_tournament_matches", "pb_tournament_matches",
    "tournament_win_rate_diff",
    # H2H
    "h2h_total_matches",
    "h2h_win_rate_a",
    "h2h_win_rate_a_surface",
    "h2h_last5_win_rate_a",
    "h2h_days_since_last",
})


# ---------------------------------------------------------------------------
# Model 2: Tiebreak
# ---------------------------------------------------------------------------

_TIEBREAK_FEATURES = frozenset({
    "best_of_5",
    "surface_clay", "surface_hard", "surface_grass",
    # Elo diff per surface
    "elo_clay_diff", "elo_hard_diff", "elo_grass_diff",
    # Surface-specific serve stats (inactive surfaces NaN'd by masking)
    "pa_clay_career_aces_per_match",  "pa_clay_career_first_serve_pts_won_pct",  "pa_clay_career_service_pts_won_pct",
    "pa_hard_career_aces_per_match",  "pa_hard_career_first_serve_pts_won_pct",  "pa_hard_career_service_pts_won_pct",
    "pa_grass_career_aces_per_match", "pa_grass_career_first_serve_pts_won_pct", "pa_grass_career_service_pts_won_pct",
    "pb_clay_career_aces_per_match",  "pb_clay_career_first_serve_pts_won_pct",  "pb_clay_career_service_pts_won_pct",
    "pb_hard_career_aces_per_match",  "pb_hard_career_first_serve_pts_won_pct",  "pb_hard_career_service_pts_won_pct",
    "pb_grass_career_aces_per_match", "pb_grass_career_first_serve_pts_won_pct", "pb_grass_career_service_pts_won_pct",
    # Recent form serve
    "pa_last10_aces_per_match", "pa_last10_first_serve_pts_won_pct", "pa_last10_service_pts_won_pct",
    "pb_last10_aces_per_match", "pb_last10_first_serve_pts_won_pct", "pb_last10_service_pts_won_pct",
    # H2H tiebreak history
    "h2h_tiebreak_pct",
    "h2h_total_matches",
    # Tournament tiebreak tendency
    "t_tiebreak_pct",
    "t_avg_service_pts_won_pct",
    "t_n_matches",
})


# ---------------------------------------------------------------------------
# Model 3: Player Aces
# ---------------------------------------------------------------------------

_PLAYER_ACES_FEATURES = frozenset({
    "best_of_5",
    "surface_clay", "surface_hard", "surface_grass",
    # Career overall serve — pa and pb symmetric
    "pa_career_aces_per_match",         "pb_career_aces_per_match",
    "pa_career_first_serve_pts_won_pct","pb_career_first_serve_pts_won_pct",
    # BO3/BO5 ace rate
    "pa_bo3_career_aces_per_match",     "pb_bo3_career_aces_per_match",
    "pa_bo5_career_aces_per_match",     "pb_bo5_career_aces_per_match",
    # Surface-specific ace rate (inactive surfaces NaN'd by masking)
    "pa_clay_career_aces_per_match",    "pb_clay_career_aces_per_match",
    "pa_hard_career_aces_per_match",    "pb_hard_career_aces_per_match",
    "pa_grass_career_aces_per_match",   "pb_grass_career_aces_per_match",
    # Recent form
    "pa_last10_aces_per_match",         "pb_last10_aces_per_match",
    "pa_last10_first_serve_pts_won_pct","pb_last10_first_serve_pts_won_pct",
    # Return ability (affects aces conceded by each player)
    "pa_career_return_pts_won_pct",     "pb_career_return_pts_won_pct",
    "pa_last10_return_pts_won_pct",     "pb_last10_return_pts_won_pct",
    # Tournament ace tendency
    "t_avg_aces_per_match",
    "t_n_matches",
})


# ---------------------------------------------------------------------------
# Model 4: Total Aces
# ---------------------------------------------------------------------------

_TOTAL_ACES_FEATURES = frozenset({
    "best_of_5",
    "surface_clay", "surface_hard", "surface_grass",
    # Career overall
    "pa_career_aces_per_match",         "pb_career_aces_per_match",
    "pa_career_first_serve_pts_won_pct","pb_career_first_serve_pts_won_pct",
    # BO3/BO5 ace rate
    "pa_bo3_career_aces_per_match",     "pb_bo3_career_aces_per_match",
    "pa_bo5_career_aces_per_match",     "pb_bo5_career_aces_per_match",
    # Surface-specific (inactive NaN'd)
    "pa_clay_career_aces_per_match",    "pb_clay_career_aces_per_match",
    "pa_hard_career_aces_per_match",    "pb_hard_career_aces_per_match",
    "pa_grass_career_aces_per_match",   "pb_grass_career_aces_per_match",
    "pa_clay_career_first_serve_pts_won_pct",  "pb_clay_career_first_serve_pts_won_pct",
    "pa_hard_career_first_serve_pts_won_pct",  "pb_hard_career_first_serve_pts_won_pct",
    "pa_grass_career_first_serve_pts_won_pct", "pb_grass_career_first_serve_pts_won_pct",
    # Recent form
    "pa_last10_aces_per_match",         "pb_last10_aces_per_match",
    "pa_last10_first_serve_pts_won_pct","pb_last10_first_serve_pts_won_pct",
    # Return ability (affects aces conceded)
    "pa_career_return_pts_won_pct",     "pb_career_return_pts_won_pct",
    "pa_last10_return_pts_won_pct",     "pb_last10_return_pts_won_pct",
    # Tournament ace tendency
    "t_avg_aces_per_match",
    "t_avg_first_serve_pts_won_pct",
    "t_n_matches",
})


# ---------------------------------------------------------------------------
# Model 5: Total Games
# ---------------------------------------------------------------------------

_TOTAL_GAMES_FEATURES = frozenset({
    "best_of_5", "round_ordinal", "tournament_category_ordinal",
    "surface_clay", "surface_hard", "surface_grass",
    # Elo diff (match competitiveness proxy)
    "elo_diff",
    "elo_clay_diff", "elo_hard_diff", "elo_grass_diff",
    # Serve quality both players — drives game length
    "pa_career_first_serve_pts_won_pct", "pb_career_first_serve_pts_won_pct",
    "pa_career_service_pts_won_pct",     "pb_career_service_pts_won_pct",
    "pa_career_return_pts_won_pct",      "pb_career_return_pts_won_pct",
    "pa_last10_service_pts_won_pct",     "pb_last10_service_pts_won_pct",
    "pa_last10_return_pts_won_pct",      "pb_last10_return_pts_won_pct",
    # H2H match length history
    "h2h_avg_total_games",
    "h2h_avg_sets_played",
    # Tournament game-length tendency
    "t_avg_games_per_match",
    "t_avg_sets_played",
    "t_avg_service_pts_won_pct",
    "t_n_matches",
})


# ---------------------------------------------------------------------------
# Model 6: Sets Played (same signals as Total Games)
# ---------------------------------------------------------------------------

_SETS_PLAYED_FEATURES = _TOTAL_GAMES_FEATURES


# ---------------------------------------------------------------------------
# Estimator builders
# ---------------------------------------------------------------------------

def _lgb_clf(cfg):
    return lgb.LGBMClassifier(
        n_estimators=cfg.lgb_n_estimators,
        max_depth=cfg.lgb_max_depth,
        learning_rate=cfg.lgb_learning_rate,
        num_leaves=cfg.lgb_num_leaves,
        subsample=cfg.lgb_subsample,
        colsample_bytree=cfg.lgb_colsample_bytree,
        min_child_samples=cfg.lgb_min_child_samples,
        reg_alpha=cfg.lgb_reg_alpha,
        reg_lambda=cfg.lgb_reg_lambda,
        random_state=cfg.random_seed,
        n_jobs=-1,
        verbose=-1,
    )


def _lgb_reg(cfg):
    return lgb.LGBMRegressor(
        n_estimators=cfg.lgb_n_estimators,
        max_depth=cfg.lgb_max_depth,
        learning_rate=cfg.lgb_learning_rate,
        num_leaves=cfg.lgb_num_leaves,
        subsample=cfg.lgb_subsample,
        colsample_bytree=cfg.lgb_colsample_bytree,
        min_child_samples=cfg.lgb_min_child_samples,
        reg_alpha=cfg.lgb_reg_alpha,
        reg_lambda=cfg.lgb_reg_lambda,
        random_state=cfg.random_seed,
        n_jobs=-1,
        verbose=-1,
    )


def _xgb_reg(cfg):
    return xgb.XGBRegressor(
        n_estimators=cfg.xgb_n_estimators,
        max_depth=cfg.xgb_max_depth,
        learning_rate=cfg.xgb_learning_rate,
        subsample=cfg.xgb_subsample,
        colsample_bytree=cfg.xgb_colsample_bytree,
        min_child_weight=cfg.xgb_min_child_weight,
        reg_alpha=cfg.xgb_reg_alpha,
        reg_lambda=cfg.xgb_reg_lambda,
        random_state=cfg.random_seed,
        n_jobs=-1,
        tree_method="hist",
    )


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------

class WinnerModel(BaseModel):
    target_name = "winner_label"
    task = "classification"
    model_id = "winner"
    feature_include = _WINNER_FEATURES
    skip_nan_filter_for_whitelist = True

    def _build_estimator(self):
        return _lgb_clf(settings.model)


class TiebreakModel(BaseModel):
    target_name = "tiebreak_present_label"
    task = "classification"
    model_id = "tiebreak"
    use_prior_correction = True
    feature_include = _TIEBREAK_FEATURES

    def _build_estimator(self):
        return _lgb_clf(settings.model)


class PlayerAcesModel(BaseModel):
    target_name = "player_aces_target"
    task = "regression"
    model_id = "player_aces"
    feature_include = _PLAYER_ACES_FEATURES

    def _build_estimator(self):
        return _xgb_reg(settings.model)


class TotalAcesModel(BaseModel):
    target_name = "total_aces_target"
    task = "regression"
    model_id = "total_aces"
    feature_include = _TOTAL_ACES_FEATURES

    def _build_estimator(self):
        return _xgb_reg(settings.model)


class TotalGamesModel(BaseModel):
    target_name = "total_games_target"
    task = "regression"
    model_id = "total_games"
    feature_include = _TOTAL_GAMES_FEATURES

    def _build_estimator(self):
        return _xgb_reg(settings.model)


class SetsPlayedModel(BaseModel):
    target_name = "sets_played_target"
    task = "regression"
    model_id = "sets_played"
    feature_include = _SETS_PLAYED_FEATURES

    def _build_estimator(self):
        return _xgb_reg(settings.model)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_MODELS: list[type[BaseModel]] = [
    WinnerModel,
    TiebreakModel,
    PlayerAcesModel,
    TotalAcesModel,
    TotalGamesModel,
    SetsPlayedModel,
]


def build_all_models() -> list[BaseModel]:
    return [cls() for cls in ALL_MODELS]
