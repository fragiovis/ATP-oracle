"""Seed the local storage backend from the legacy .cache/ files.

Copies the gold features, profiles and cached models into the S3-style
layout under settings.storage.local_root, so ATP_STORAGE_BACKEND=local
works without re-running the whole pipeline.

Usage:  python scripts/seed_local_storage.py [--overwrite]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from config.settings import settings

CACHE = Path(".cache")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = settings.s3
    root = Path(settings.storage.local_root)
    mapping = {
        CACHE / "pre_match_features.parquet":       f"{cfg.gold_features_path}/pre_match_features.parquet",
        CACHE / "player_profiles.parquet":          f"{cfg.players_profiles_path}/player_profiles.parquet",
        CACHE / "player_tournament_history.parquet": f"{cfg.players_profiles_path}/player_tournament_history.parquet",
        CACHE / "tournament_profiles.parquet":      f"{cfg.tournaments_path}/tournament_profiles.parquet",
        CACHE / "rankings_snapshot.parquet":        f"{cfg.silver_path}/rankings_snapshot_current_top200.parquet",
    }
    model_cache = CACHE / "models"
    if model_cache.is_dir():
        for pkl in model_cache.glob("*/*/model.pkl"):
            rel = pkl.relative_to(model_cache).as_posix()
            mapping[pkl] = f"{cfg.models_path}/{rel}"

    for src, dst_path in mapping.items():
        dst = root / dst_path.lstrip("/")
        if not src.exists():
            print(f"skip (missing)  {src}")
            continue
        if dst.exists() and not args.overwrite:
            print(f"skip (exists)   {dst}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"copied          {src} → {dst}")


if __name__ == "__main__":
    main()
