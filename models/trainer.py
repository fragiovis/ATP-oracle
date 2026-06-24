"""Train all models and persist them to S3 (with local cache fallback)."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from config.settings import settings
from models.targets import build_all_models
from storage.s3_client import S3Client
from features.pipeline import FeaturePipeline

logger = logging.getLogger(__name__)

LOCAL_MODEL_CACHE = Path(".cache/models")


class ModelTrainer:
    def __init__(self):
        self.client = S3Client()
        self.models_path = settings.s3.models_path

    # ------------------------------------------------------------------
    # Local cache helpers
    # ------------------------------------------------------------------

    def _local_model_path(self, run_id: str, model_id: str) -> Path:
        return LOCAL_MODEL_CACHE / run_id / model_id / "model.pkl"

    def _save_local(self, model, run_id: str) -> None:
        path = self._local_model_path(run_id, model.model_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(model.to_bytes())
        logger.debug("Local cache saved: %s", path)

    def _load_local_latest(self, model_class):
        """Load the most recently saved local model (by run_id dir name)."""
        if not LOCAL_MODEL_CACHE.exists():
            raise FileNotFoundError("No local model cache found.")
        run_dirs = sorted(
            [p for p in LOCAL_MODEL_CACHE.iterdir() if p.is_dir()],
            key=lambda p: p.name,
            reverse=True,
        )
        instance = model_class()
        for run_dir in run_dirs:
            pkl = run_dir / instance.model_id / "model.pkl"
            if pkl.exists():
                instance.from_bytes(pkl.read_bytes())
                logger.info(
                    "Loaded %s from local cache (run %s)", instance.model_id, run_dir.name
                )
                return instance
        raise FileNotFoundError(f"No local cache found for {instance.model_id}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_all(self, features: pd.DataFrame) -> dict[str, dict]:
        """Train every target model and save to S3 + local cache."""
        models = build_all_models()
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        results: dict[str, dict] = {}

        # If the winner label is severely skewed (e.g. Sackmann ordering),
        # randomise player roles to balance the binary target prior before
        # training. This avoids models learning the trivial "player_a wins"
        # artefact when gold features were generated from source data that
        # always listed the winner first.
        if "winner_label" in features.columns:
            p = float(features["winner_label"].dropna().mean())
            if p > 0.85 or p < 0.15:
                logger.info(
                    "Detected winner_label skew (%.3f) — applying role randomisation to balance.",
                    p,
                )
                features = FeaturePipeline._randomise_roles(features, seed=settings.model.random_seed)
                p2 = float(features["winner_label"].dropna().mean())
                logger.info("Post-randomise winner_label mean: %.3f", p2)

        for m in models:
            logger.info("Training model: %s (target=%s)", m.model_id, m.target_name)
            try:
                m.fit(features)
                self._save_local(m, run_id)   # always save locally first
                self._save_model(m, run_id)   # then push to S3
                results[m.model_id] = {**m.metadata, "status": "ok", "run_id": run_id}
            except Exception as exc:
                logger.error("Training failed for %s: %s", m.model_id, exc)
                results[m.model_id] = {"status": "error", "error": str(exc)}

        summary_path = f"{self.models_path}/{run_id}/training_summary.json"
        self.client.write_json(results, summary_path)
        logger.info("Training complete. Run ID: %s", run_id)
        return results

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def _save_model(self, model, run_id: str) -> None:
        model_dir = f"{self.models_path}/{run_id}/{model.model_id}"
        self.client.write_bytes(model.to_bytes(), f"{model_dir}/model.pkl")
        self.client.write_json(model.metadata,   f"{model_dir}/metadata.json")
        logger.info("Saved %s → %s", model.model_id, model_dir)

    def load_model(self, model_class, run_id: str):
        instance = model_class()
        model_dir = f"{self.models_path}/{run_id}/{instance.model_id}"
        data = self.client.read_bytes(f"{model_dir}/model.pkl")
        instance.from_bytes(data)
        return instance

    def load_latest_model(self, model_class):
        """Load from S3; fall back to local cache if credentials expired."""
        try:
            run_id = self.client.latest_subdir(self.models_path)
            if run_id is None:
                raise FileNotFoundError(f"No saved models found under {self.models_path}")
            instance = model_class()
            pkl_path = f"{self.models_path}/{run_id}/{instance.model_id}/model.pkl"
            data = self.client.read_bytes(pkl_path)
            instance.from_bytes(data)
            logger.info("Loaded %s from S3 run %s", instance.model_id, run_id)
            return instance
        except Exception as e:
            if "AccessDenied" in str(e) or "credentials" in str(e).lower() or "ExpiredToken" in str(e):
                logger.warning(
                    "S3 access denied for %s – falling back to local cache. (%s)",
                    model_class.__name__, type(e).__name__,
                )
                return self._load_local_latest(model_class)
            raise
