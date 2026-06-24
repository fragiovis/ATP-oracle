"""Run evaluation for all trained models and persist results to HDFS."""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from config.settings import settings
from evaluation.metrics import ModelEvaluator
from evaluation.plots import (
    calibration_plot,
    feature_importance_plot,
    residual_plot,
    roc_curve_plot,
)
from models.targets import ALL_MODELS
from models.trainer import ModelTrainer
from storage.hdfs_client import HDFSClient
from storage.writer import StorageWriter

logger = logging.getLogger(__name__)

BREAKDOWN_COLS = ["surface", "tournament_category_ordinal"]


class EvaluationRunner:
    def __init__(self):
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        self.client = HDFSClient()
        self.eval_path = settings.hdfs.evaluation_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all(self, features: pd.DataFrame, run_id: str | None = None) -> dict:
        """Evaluate all models and write results to HDFS."""
        from datetime import datetime
        run_id = run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        all_results: dict = {}

        for model_cls in ALL_MODELS:
            model_id = model_cls.model_id
            logger.info("Evaluating model: %s", model_id)
            try:
                model = self.trainer.load_latest_model(model_cls)
            except FileNotFoundError:
                logger.warning("No trained model found for %s – skipping.", model_id)
                continue

            # Test split: last portion
            target = model.target_name
            df = features.dropna(subset=[target]).copy()
            if df.empty:
                continue

            _, _, test_df = model.time_split(df)
            if test_df.empty:
                logger.warning("Empty test split for %s", model_id)
                continue

            metrics = self.evaluator.evaluate(model, test_df, BREAKDOWN_COLS)
            all_results[model_id] = metrics

            # Save metrics JSON
            metrics_path = f"{self.eval_path}/{run_id}/{model_id}/metrics.json"
            self.client.write_json(metrics, metrics_path)

            # Save plots
            self._save_plots(model, test_df, run_id)

        # Summary
        summary_path = f"{self.eval_path}/{run_id}/evaluation_summary.json"
        self.client.write_json(all_results, summary_path)
        logger.info("Evaluation complete – results saved under %s/%s", self.eval_path, run_id)
        return all_results

    # ------------------------------------------------------------------
    # Plot generation
    # ------------------------------------------------------------------

    def _save_plots(self, model, test_df: pd.DataFrame, run_id: str) -> None:
        target = model.target_name
        task = model.task
        mid = model.model_id

        y_true = test_df[target].values
        y_pred = model.predict(test_df)

        plots_dir = f"{self.eval_path}/{run_id}/{mid}"

        if task == "classification" and hasattr(model.model, "predict_proba"):
            try:
                proba = model.predict_proba(test_df)
                y_proba = proba[:, 1] if proba.ndim == 2 else proba

                self.client.write_bytes(
                    calibration_plot(y_true, y_proba),
                    f"{plots_dir}/calibration.png",
                )
                self.client.write_bytes(
                    roc_curve_plot(y_true, y_proba),
                    f"{plots_dir}/roc_curve.png",
                )
            except Exception as e:
                logger.warning("Could not generate classification plots for %s: %s", mid, e)
        elif task == "regression":
            try:
                self.client.write_bytes(
                    residual_plot(y_true, y_pred, title=mid),
                    f"{plots_dir}/residuals.png",
                )
            except Exception as e:
                logger.warning("Could not generate residual plot for %s: %s", mid, e)

        # Feature importance
        if hasattr(model.model, "feature_importances_"):
            try:
                self.client.write_bytes(
                    feature_importance_plot(model.feature_cols, model.model.feature_importances_),
                    f"{plots_dir}/feature_importance.png",
                )
            except Exception as e:
                logger.warning("Could not generate feature importance plot for %s: %s", mid, e)
