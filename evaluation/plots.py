"""Diagnostic plots for model evaluation.

Generates and saves:
  - Calibration curves (for classifiers)
  - ROC curves (for classifiers)
  - Residual plots (for regressors)
  - Feature importance bar charts
"""

from __future__ import annotations

import io
import logging

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def calibration_plot(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> bytes:
    """Return PNG bytes of a calibration curve."""
    from sklearn.calibration import calibration_curve
    fig, ax = plt.subplots(figsize=(6, 5))
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
    ax.plot(mean_pred, frac_pos, "s-", label="Model")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve")
    ax.legend()
    return _fig_to_bytes(fig)


def roc_curve_plot(y_true: np.ndarray, y_proba: np.ndarray) -> bytes:
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC Curve")
    ax.legend()
    return _fig_to_bytes(fig)


def residual_plot(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Residuals") -> bytes:
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(y_pred, residuals, alpha=0.4, s=10)
    axes[0].axhline(0, color="r", linestyle="--")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")
    axes[0].set_title(f"{title} – residuals vs predicted")

    axes[1].hist(residuals, bins=40, edgecolor="black")
    axes[1].set_xlabel("Residual")
    axes[1].set_title(f"{title} – residual distribution")
    plt.tight_layout()
    return _fig_to_bytes(fig)


def feature_importance_plot(feature_cols: list[str], importances: np.ndarray, top_n: int = 20) -> bytes:
    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, max(4, top_n // 3)))
    ax.barh([feature_cols[i] for i in idx], importances[idx])
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importances (top {top_n})")
    plt.tight_layout()
    return _fig_to_bytes(fig)


def _fig_to_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
