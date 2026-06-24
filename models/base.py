"""Base class for all ATP Oracle prediction models.

Provides:
  - time-based train/val/test split
  - fit / predict / predict_proba interface
  - save / load via HDFS (pickle + metadata JSON)
  - feature importance logging
"""

from __future__ import annotations

import io
import json
import logging
import pickle
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base for all target-specific models."""

    # Subclasses must set these
    target_name: str = ""          # e.g. 'winner_label'
    task: str = ""                 # 'classification' | 'regression'
    model_id: str = ""             # e.g. 'winner', used for HDFS path

    # Subclasses may extend this to drop task-irrelevant features
    extra_exclude: set[str] = set()

    # If non-empty, only these features are considered (explicit whitelist).
    # Takes precedence over extra_exclude. NaN-rate filter still applied.
    feature_include: frozenset[str] = frozenset()

    # When True and feature_include is set, skip the NaN-rate filter so that
    # semantically-sparse columns (surface Elo diffs, H2H when no prior
    # meetings) are never dropped just because NaN rate > MAX_NAN_RATE.
    skip_nan_filter_for_whitelist: bool = False

    def __init__(self):
        self.model = None          # underlying sklearn-compatible estimator
        self.feature_cols: list[str] = []
        self.metadata: dict[str, Any] = {}
        # Decision threshold for binary classifiers. Tuned on the validation
        # split after fit; left as None for regressors / estimators without
        # predict_proba. None ⇒ use estimator's default predict().
        self.decision_threshold: float | None = None
        # Isotonic calibrator fitted on val set (classification only).
        # Corrects systematic over/under-confidence in raw probabilities.
        self.calibrator = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_estimator(self):
        """Return a freshly initialised sklearn-compatible estimator."""

    # ------------------------------------------------------------------
    # Data split
    # ------------------------------------------------------------------

    def time_split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split *df* into train / val / test using temporal ordering.
        Expects a 'year' column or date index.
        """
        cfg = settings.model
        if df.empty:
            raise ValueError("Empty feature DataFrame passed to time_split.")

        # Use 'year' column if available, else ordinal position
        if "year" in df.columns:
            # Drop pre-modern era rows before computing split boundaries.
            # Matches from before min_train_year use different playing styles,
            # court speeds, and ball/racket technology — they add noise.
            if cfg.min_train_year and "year" in df.columns:
                df = df[df["year"] >= cfg.min_train_year]
                if df.empty:
                    raise ValueError(f"No rows remain after min_train_year={cfg.min_train_year} filter.")
            years = sorted(df["year"].dropna().unique())
            n = len(years)
            train_end = years[int(n * cfg.train_fraction) - 1]
            val_end = years[int(n * (cfg.train_fraction + cfg.val_fraction)) - 1]
            train = df[df["year"] <= train_end]
            val = df[(df["year"] > train_end) & (df["year"] <= val_end)]
            test = df[df["year"] > val_end]
        else:
            n = len(df)
            i_train = int(n * cfg.train_fraction)
            i_val = int(n * (cfg.train_fraction + cfg.val_fraction))
            train, val, test = df.iloc[:i_train], df.iloc[i_train:i_val], df.iloc[i_val:]

        logger.info(
            "Split: train=%d val=%d test=%d (target=%s)", len(train), len(val), len(test),
            self.target_name
        )
        return train, val, test

    # ------------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------------

    # Features with NaN rate above this threshold are excluded from all models.
    # LightGBM/XGBoost handle NaN natively, but a feature that is missing in
    # >30% of rows contributes mostly noise and can distort predictions at inference
    # when the player has no data for that stat.
    MAX_NAN_RATE: float = 0.30

    # Correct for class-prior drift between train and val by rescaling the
    # training sample weights so the effective positive rate matches the
    # validation positive rate. Enable per-model for binary classifiers that
    # suffer from temporal drift (e.g. tiebreak rule changes).
    use_prior_correction: bool = False

    def _select_features(self, df: pd.DataFrame) -> list[str]:
        """Return feature columns for this model.

        If feature_include is set, use it as an explicit whitelist.
        Otherwise fall back to the exclude-based approach.
        NaN-rate filter applied in both cases.
        """
        _ALWAYS_EXCLUDE = {
            self.target_name, "match_id", "player_a_id", "player_b_id",
            "winner_id", "tournament_id", "tournament_name", "scoreline",
            "match_date", "surface", "round",
            # Other targets (never use targets as features)
            "winner_label", "tiebreak_present_label",
            "first_set_winner_label", "player_aces_target", "total_aces_target",
            "total_games_target", "sets_played_target",
        }

        numeric = df.select_dtypes(include=[np.number]).columns
        nan_rates = df[numeric].isna().mean()

        if self.feature_include:
            # Explicit whitelist: only columns in feature_include that exist
            # and are numeric. NaN-rate filter is skipped when
            # skip_nan_filter_for_whitelist=True (e.g. surface Elo diffs,
            # H2H features that are semantically sparse, not missing data).
            return [
                c for c in self.feature_include
                if c in numeric
                and c not in _ALWAYS_EXCLUDE
                and (self.skip_nan_filter_for_whitelist or nan_rates.get(c, 1.0) <= self.MAX_NAN_RATE)
            ]

        # Exclude-based fallback
        exclude = _ALWAYS_EXCLUDE | self.extra_exclude
        return [
            c for c in numeric
            if c not in exclude and nan_rates[c] <= self.MAX_NAN_RATE
        ]

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, features: pd.DataFrame) -> None:
        """Fit the model on the training split."""
        cfg = settings.model

        # Drop rows where target is missing
        df = features.dropna(subset=[self.target_name]).copy()
        if df.empty:
            raise ValueError(f"No rows with target={self.target_name}")

        train, val, test = self.time_split(df)

        self.feature_cols = self._select_features(train)
        logger.info("Features: %d columns", len(self.feature_cols))

        # LightGBM and XGBoost handle NaN natively — do NOT fillna.
        # Filling with a sentinel like -999 breaks their internal NaN routing
        # and treats missing values as extremely negative real numbers.
        X_train = train[self.feature_cols]
        y_train = train[self.target_name]
        X_val = val[self.feature_cols]
        y_val = val[self.target_name]

        estimator = self._build_estimator()

        # Recency weighting: give exponentially more weight to recent matches.
        # weight(year) = 2^((year - max_year) / half_life)
        # so that a match from `half_life` years ago has weight 0.5.
        sample_weight = None
        if cfg.recency_half_life_years > 0 and "year" in train.columns:
            max_year = float(train["year"].max())
            half_life = cfg.recency_half_life_years
            years_ago = max_year - train["year"].fillna(max_year).astype(float)
            sample_weight = np.power(2.0, -years_ago / half_life).values
            # Normalise so mean weight = 1 (keeps effective learning rate stable)
            sample_weight = sample_weight / sample_weight.mean()
            logger.info(
                "Recency weights for %s: half_life=%.1f years, "
                "weight range [%.3f, %.3f]",
                self.model_id, half_life, sample_weight.min(), sample_weight.max(),
            )

        # Prior-correction sample weights: rescale positive/negative samples so
        # that the weighted training prior equals the validation prior. Only
        # applied for binary classifiers where the subclass opted in.
        if (
            self.use_prior_correction
            and self.task == "classification"
            and y_train.nunique() == 2
            and not y_val.empty
        ):
            p_train = float(y_train.mean())
            p_val = float(y_val.mean())
            if 0.0 < p_train < 1.0 and 0.0 < p_val < 1.0:
                w_pos = p_val / p_train
                w_neg = (1.0 - p_val) / (1.0 - p_train)
                prior_w = np.where(y_train == 1, w_pos, w_neg)
                sample_weight = prior_w if sample_weight is None else sample_weight * prior_w
                logger.info(
                    "Prior correction for %s: p_train=%.3f p_val=%.3f  w_pos=%.3f w_neg=%.3f",
                    self.model_id, p_train, p_val, w_pos, w_neg,
                )

        # LightGBM / XGBoost support early stopping with eval_set
        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        if hasattr(estimator, "fit"):
            try:
                estimator.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    **fit_kwargs,
                )
            except TypeError:
                estimator.fit(X_train, y_train, **fit_kwargs)

        self.model = estimator

        # Tune decision threshold on the validation split for classifiers that
        # return probabilities. Distributional drift between train and test
        # (e.g. rule changes for tiebreak) makes the 0.5 default suboptimal —
        # picking the threshold that maximises validation accuracy recovers a
        # lot of that loss without touching the estimator itself.
        if self.task == "classification" and hasattr(estimator, "predict_proba"):
            try:
                self.decision_threshold = self._tune_threshold(X_val, y_val)
                logger.info(
                    "Tuned decision threshold for %s: %.3f",
                    self.model_id, self.decision_threshold,
                )
            except Exception as exc:
                logger.warning("Threshold tuning failed for %s: %s", self.model_id, exc)
                self.decision_threshold = None

            # Fit isotonic calibrator on val set.
            # Maps raw model probabilities to empirical win rates, correcting
            # systematic over-confidence (model says 75% → actual 69%).
            try:
                from sklearn.isotonic import IsotonicRegression
                raw_val_probs = estimator.predict_proba(X_val)[:, 1]
                self.calibrator = IsotonicRegression(out_of_bounds="clip")
                self.calibrator.fit(raw_val_probs, y_val.to_numpy())
                # Log calibration quality
                cal_probs = self.calibrator.predict(raw_val_probs)
                raw_err = float(np.mean(np.abs(raw_val_probs - y_val.to_numpy())))
                cal_err = float(np.mean(np.abs(cal_probs - y_val.to_numpy())))
                logger.info(
                    "Calibrator fitted for %s: mean abs error raw=%.4f → calibrated=%.4f",
                    self.model_id, raw_err, cal_err,
                )
            except Exception as exc:
                logger.warning("Calibration failed for %s: %s", self.model_id, exc)
                self.calibrator = None

        self.metadata = {
            "target_name": self.target_name,
            "task": self.task,
            "model_id": self.model_id,
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "n_features": len(self.feature_cols),
            "random_seed": cfg.random_seed,
            "decision_threshold": self.decision_threshold,
        }

        self._log_feature_importance()

    # ------------------------------------------------------------------
    # Threshold tuning
    # ------------------------------------------------------------------

    def _tune_threshold(self, X_val: pd.DataFrame, y_val: pd.Series) -> float:
        """Pick the probability threshold that maximises accuracy on the
        validation split. Search over a 0.05-step grid in [0.05, 0.95]."""
        if y_val.empty:
            return 0.5
        proba = self.model.predict_proba(X_val)[:, 1]
        y = y_val.to_numpy()
        best_thr, best_acc = 0.5, -1.0
        for thr in np.arange(0.05, 0.96, 0.01):
            acc = ((proba >= thr).astype(int) == y).mean()
            if acc > best_acc:
                best_acc, best_thr = acc, float(thr)
        return round(best_thr, 3)

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_in = X[self.feature_cols]
        return self.model.predict(X_in)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self.model, "predict_proba"):
            raise NotImplementedError(f"{self.model_id} does not support predict_proba.")
        X_in = X[self.feature_cols]
        raw = self.model.predict_proba(X_in)
        if self.calibrator is not None:
            p1 = self.calibrator.predict(raw[:, 1])
            return np.column_stack([1 - p1, p1])
        return raw

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def to_bytes(self) -> bytes:
        payload = {
            "model": self.model,
            "feature_cols": self.feature_cols,
            "metadata": self.metadata,
            "decision_threshold": self.decision_threshold,
            "calibrator": self.calibrator,
        }
        return pickle.dumps(payload)

    def from_bytes(self, data: bytes) -> None:
        payload = pickle.loads(data)
        self.model = payload["model"]
        self.feature_cols = payload["feature_cols"]
        self.metadata = payload["metadata"]
        self.decision_threshold = payload.get("decision_threshold")
        self.calibrator = payload.get("calibrator")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_feature_importance(self) -> None:
        if not hasattr(self.model, "feature_importances_"):
            return
        imp = sorted(
            zip(self.feature_cols, self.model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
        top = imp[:15]
        logger.info("Top features (%s): %s", self.model_id, top)
