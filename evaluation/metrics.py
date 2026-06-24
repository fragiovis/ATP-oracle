"""Compute evaluation metrics for each model type.

Classification: accuracy, ROC-AUC, log-loss, Brier score.
Regression:     MAE, RMSE, R².

Also supports breakdown by surface, tournament category, and ranking-diff bucket.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None) -> dict[str, float]:
    """Compute classification metrics. y_proba should be P(class=1)."""
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            metrics["log_loss"] = float(log_loss(y_true, y_proba))
            metrics["brier_score"] = float(brier_score_loss(y_true, y_proba))
        except Exception as e:
            logger.warning("Could not compute proba-based metrics: %s", e)
    return metrics


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class ModelEvaluator:
    """Evaluate a model on test data with breakdowns."""

    def evaluate(
        self,
        model,
        test_df: pd.DataFrame,
        breakdown_cols: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Parameters
        ----------
        model         : fitted BaseModel instance
        test_df       : feature DataFrame that includes the target column
        breakdown_cols: columns to group by for breakdown metrics

        Returns
        -------
        dict with 'overall' metrics and per-group breakdowns
        """
        target = model.target_name
        task = model.task

        df = test_df.dropna(subset=[target]).copy()
        if df.empty:
            logger.warning("No test rows for target=%s", target)
            return {}

        y_true = df[target].values
        y_pred = model.predict(df)
        y_proba = None
        if task == "classification" and hasattr(model.model, "predict_proba"):
            try:
                proba = model.predict_proba(df)
                y_proba = proba[:, 1] if proba.ndim == 2 else proba
            except Exception:
                pass

        result: dict[str, Any] = {}
        if task == "classification":
            result["overall"] = classification_metrics(y_true, y_pred, y_proba)
        else:
            result["overall"] = regression_metrics(y_true, y_pred)

        # Breakdowns
        if breakdown_cols:
            for col in breakdown_cols:
                if col not in df.columns:
                    continue
                result[f"by_{col}"] = {}
                for group_val, group_df in df.groupby(col):
                    gt = group_df[target].values
                    gp = model.predict(group_df)
                    gproba = None
                    if y_proba is not None:
                        idx = group_df.index
                        mask = df.index.isin(idx)
                        gproba = y_proba[mask] if mask.any() else None
                    if task == "classification":
                        result[f"by_{col}"][str(group_val)] = classification_metrics(gt, gp, gproba)
                    else:
                        result[f"by_{col}"][str(group_val)] = regression_metrics(gt, gp)

        return result
