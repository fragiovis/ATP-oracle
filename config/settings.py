"""Central configuration for the ATP Oracle pipeline.

All tuneable parameters live here so every module can import from a
single source of truth instead of scattering magic numbers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------

class ScraperSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ATP_SCRAPER_")

    # Rate limiting
    requests_per_minute: int = Field(default=10, description="Max requests per minute")
    min_delay_seconds: float = Field(default=3.0, description="Min random delay between requests")
    max_delay_seconds: float = Field(default=8.0, description="Max random delay between requests")
    max_retries: int = Field(default=5, description="Max retry attempts per URL")
    backoff_multiplier: float = Field(default=2.0, description="Exponential backoff multiplier")
    max_parallel_requests: int = Field(default=1, description="Max concurrent requests (keep low)")

    # Retry-triggering HTTP status codes
    retry_status_codes: list[int] = Field(default=[429, 403, 500, 502, 503, 504])

    # HTML cache
    enable_cache: bool = Field(default=True, description="Cache raw HTML to disk")
    cache_dir: Path = Field(default=Path(".cache/html"), description="Path to HTML cache")
    cache_ttl_hours: int = Field(default=72, description="Cache TTL in hours")

    # Checkpointing
    checkpoint_dir: Path = Field(default=Path(".cache/checkpoints"))

    # (No proxies configured)

    # Scrape horizon
    years_back: int = Field(default=40, description="How many years of match history to scrape")

    # Top-N players to include in the current rankings snapshot (does NOT filter match history)
    player_universe_size: int = Field(default=200, description="Top N players in current rankings snapshot")

    # Logging
    log_requests: bool = Field(default=True)


# ---------------------------------------------------------------------------
# HDFS / Storage  (kept for reference / local fallback)
# ---------------------------------------------------------------------------

class HDFSSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ATP_HDFS_")

    namenode_host: str = Field(default="localhost")
    namenode_port: int = Field(default=9870)
    namenode_url: str = Field(default="http://localhost:9870")
    user: str = Field(default="hadoop")

    # Directory layout
    raw_rankings_path: str = Field(default="/data/raw/rankings")
    raw_players_path: str = Field(default="/data/raw/players")
    raw_matches_path: str = Field(default="/data/raw/matches")
    raw_match_stats_path: str = Field(default="/data/raw/match_stats")

    bronze_path: str = Field(default="/data/bronze")
    silver_path: str = Field(default="/data/silver")
    gold_features_path: str = Field(default="/data/gold/features")
    players_profiles_path: str = Field(default="/data/players")
    tournaments_path: str = Field(default="/data/tournaments")
    models_path: str = Field(default="/data/models")
    evaluation_path: str = Field(default="/data/evaluation")


# ---------------------------------------------------------------------------
# S3 Storage
# ---------------------------------------------------------------------------

class S3Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ATP_S3_")

    bucket: str = Field(default="tennis-predictor-959366206165-us-east-1-an")
    region: str = Field(default="us-east-1")
    # Optional key prefix inside the bucket (leave empty to use bucket root)
    prefix: str = Field(default="atp-oracle")

    # Directory layout (mirrors HDFS paths)
    raw_rankings_path: str = Field(default="/data/raw/rankings")
    raw_players_path: str = Field(default="/data/raw/players")
    raw_matches_path: str = Field(default="/data/raw/matches")
    raw_match_stats_path: str = Field(default="/data/raw/match_stats")

    bronze_path: str = Field(default="/data/bronze")
    silver_path: str = Field(default="/data/silver")
    gold_features_path: str = Field(default="/data/gold/features")
    players_profiles_path: str = Field(default="/data/players")
    tournaments_path: str = Field(default="/data/tournaments")
    models_path: str = Field(default="/data/models")
    evaluation_path: str = Field(default="/data/evaluation")


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

class FeatureSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ATP_FEATURES_")

    # Rolling windows (number of matches)
    rolling_windows: list[int] = Field(default=[5, 10, 25])

    # Time-based window (weeks)
    rolling_weeks: int = Field(default=52)

    # Surfaces to compute separate stats for
    surfaces: list[str] = Field(default=["clay", "hard", "grass"])

    # Minimum matches required for a feature to be considered valid
    min_matches_for_feature: int = Field(default=3)

    # Elo
    elo_k_factor: float = Field(default=32.0)
    elo_surface_k_factor: float = Field(default=24.0)
    elo_initial_rating: float = Field(default=1500.0)
    # Time decay: K-factor is multiplied by exp(-ln(2) * days_since_match / halflife).
    # Set to 0 to disable decay (uniform weighting).
    elo_time_decay_halflife_days: float = Field(default=365.0)


# ---------------------------------------------------------------------------
# Model Hyperparameters
# ---------------------------------------------------------------------------

class ModelSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ATP_MODEL_")

    random_seed: int = Field(default=42)

    # Time-based split fractions
    train_fraction: float = Field(default=0.70)
    val_fraction: float = Field(default=0.15)
    # test_fraction = 1 - train - val

    # Only use matches from this year onwards for training.
    # Set to 0 to use all data.
    min_train_year: int = Field(default=0)

    # Exponential recency weighting: matches decay by 0.5 every N years.
    # Set to 0 to disable (uniform weights).
    recency_half_life_years: float = Field(default=7.0)

    # XGBoost defaults (applied to all targets unless overridden)
    xgb_n_estimators: int = Field(default=500)
    xgb_max_depth: int = Field(default=6)
    xgb_learning_rate: float = Field(default=0.05)
    xgb_subsample: float = Field(default=0.8)
    xgb_colsample_bytree: float = Field(default=0.8)
    xgb_min_child_weight: int = Field(default=5)
    xgb_reg_alpha: float = Field(default=0.1)
    xgb_reg_lambda: float = Field(default=1.0)
    xgb_early_stopping_rounds: int = Field(default=50)

    # LightGBM defaults
    lgb_n_estimators: int = Field(default=500)
    lgb_max_depth: int = Field(default=6)
    lgb_learning_rate: float = Field(default=0.05)
    lgb_num_leaves: int = Field(default=63)
    lgb_subsample: float = Field(default=0.8)
    lgb_colsample_bytree: float = Field(default=0.8)
    lgb_min_child_samples: int = Field(default=20)
    lgb_reg_alpha: float = Field(default=0.1)
    lgb_reg_lambda: float = Field(default=1.0)
    lgb_early_stopping_rounds: int = Field(default=50)

    # Preferred model family per target (xgboost | lightgbm)
    winner_model: Literal["xgboost", "lightgbm"] = Field(default="lightgbm")
    deciding_set_model: Literal["xgboost", "lightgbm"] = Field(default="lightgbm")
    tiebreak_model: Literal["xgboost", "lightgbm"] = Field(default="lightgbm")
    first_set_model: Literal["xgboost", "lightgbm"] = Field(default="lightgbm")
    player_aces_model: Literal["xgboost", "lightgbm"] = Field(default="xgboost")
    total_aces_model: Literal["xgboost", "lightgbm"] = Field(default="xgboost")
    total_games_model: Literal["xgboost", "lightgbm"] = Field(default="xgboost")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ATP_AGENT_",
        env_file=Path(__file__).resolve().parent.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Groq model for NL understanding
    groq_model: str = Field(default="llama-3.3-70b-versatile")
    # API key: reads ATP_AGENT_GROQ_API_KEY or GROQ_API_KEY (no prefix) from env/.env
    groq_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("ATP_AGENT_GROQ_API_KEY", "GROQ_API_KEY"),
    )
    max_tokens: int = Field(default=1024)
    temperature: float = Field(default=0.0)

    # Default values injected when the user omits context fields
    default_best_of: int = Field(default=3)
    default_surface: str | None = Field(default=None)  # None → ask for clarification


# ---------------------------------------------------------------------------
# Root settings object (aggregates all sub-settings)
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ATP_")

    scraper: ScraperSettings = Field(default_factory=ScraperSettings)
    hdfs: HDFSSettings = Field(default_factory=HDFSSettings)
    s3: S3Settings = Field(default_factory=S3Settings)
    features: FeatureSettings = Field(default_factory=FeatureSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)


# Module-level singleton – import this everywhere:
#   from config.settings import settings
settings = Settings()
