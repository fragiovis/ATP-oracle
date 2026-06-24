"""Evaluate trained models on the held-out test split.

Metrics:
  Classification (winner, deciding_set, tiebreak): Accuracy, ROC-AUC, Log-Loss
  Regression (player_aces, total_aces, total_games): MAE, RMSE, R²

Test split: temporally the most recent years (defined by val/test fractions in settings).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from config.settings import settings
from models.targets import ALL_MODELS
from models.trainer import ModelTrainer

logger = logging.getLogger(__name__)


def evaluate_all(features: pd.DataFrame) -> dict[str, dict]:
    trainer = ModelTrainer()
    results: dict[str, dict] = {}

    for model_cls in ALL_MODELS:
        try:
            model = trainer.load_latest_model(model_cls)
        except FileNotFoundError:
            logger.warning("Model not found: %s — skipping.", model_cls.model_id)
            continue

        target = model.target_name
        df = features.dropna(subset=[target]).copy()
        if df.empty:
            logger.warning("No rows for target %s — skipping.", target)
            continue

        # Reproduce the same time split used during training
        _, _, test = model.time_split(df)
        if test.empty:
            logger.warning("Empty test split for %s — skipping.", target)
            continue

        # Align feature columns
        missing = [c for c in model.feature_cols if c not in test.columns]
        for c in missing:
            test[c] = np.nan
        # LightGBM / XGBoost handle NaN natively. Training uses raw NaN, so the
        # evaluator MUST not substitute a sentinel here — doing so breaks the
        # model's internal NaN routing and silently corrupts test predictions.
        X_test = test[model.feature_cols]
        y_test = test[target]

        if model.task == "classification":
            # Use the model's tuned decision threshold when present (classifiers
            # may select a non-0.5 threshold on the validation split); otherwise
            # fall back to the estimator's default predict().
            threshold = getattr(model, "decision_threshold", None)
            if threshold is not None and hasattr(model.model, "predict_proba"):
                y_prob_full = model.model.predict_proba(X_test)[:, 1]
                y_pred = (y_prob_full >= threshold).astype(int)
            else:
                y_pred = model.model.predict(X_test)
            metrics: dict = {
                "n_test": len(test),
                "accuracy": round(accuracy_score(y_test, y_pred), 4),
            }
            if hasattr(model.model, "predict_proba"):
                y_prob = model.model.predict_proba(X_test)[:, 1]
                metrics["roc_auc"]  = round(roc_auc_score(y_test, y_prob), 4)
                metrics["log_loss"] = round(log_loss(y_test, y_prob), 4)
            # Baseline: always predict majority class
            baseline = max(y_test.mean(), 1 - y_test.mean())
            metrics["baseline_accuracy"] = round(baseline, 4)
            metrics["lift_vs_baseline"]  = round(metrics["accuracy"] - baseline, 4)

        else:  # regression
            y_pred = model.model.predict(X_test)
            mae  = mean_absolute_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred) ** 0.5
            metrics = {
                "n_test": len(test),
                "mae":    round(mae, 3),
                "rmse":   round(rmse, 3),
                "r2":     round(r2_score(y_test, y_pred), 4),
                "mean_target": round(float(y_test.mean()), 3),
            }

        results[model.model_id] = metrics
        logger.info("%-15s  %s", model.model_id, metrics)

    return results
