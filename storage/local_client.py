"""Local filesystem storage client – drop-in replacement for S3Client.

Same interface as S3Client/HDFSClient, so the pipeline runs without AWS.
Paths map onto a local root directory:
  /data/silver/matches.parquet  →  <local_root>/data/silver/matches.parquet
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config.settings import settings

logger = logging.getLogger(__name__)


class LocalClient:
    """Filesystem wrapper with the same interface as S3Client."""

    def __init__(self, root: Path | str | None = None):
        self._root = Path(root or settings.storage.local_root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        logger.info("Local storage client ready: %s", self._root)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _path(self, path: str) -> Path:
        """Convert an absolute-style path to a file under the local root."""
        return self._root / path.lstrip("/")

    def _prepare(self, path: str) -> Path:
        p = self._path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # ------------------------------------------------------------------
    # Existence / listing
    # ------------------------------------------------------------------

    def mkdir(self, path: str) -> None:
        self._path(path).mkdir(parents=True, exist_ok=True)

    def exists(self, path: str) -> bool:
        return self._path(path).exists()

    def list_files(self, directory: str) -> list[str]:
        base = self._path(directory)
        if not base.is_dir():
            return []
        return sorted(
            "/" + p.relative_to(self._root).as_posix()
            for p in base.rglob("*") if p.is_file()
        )

    def delete(self, path: str, recursive: bool = False) -> None:
        p = self._path(path)
        if p.is_dir():
            if recursive:
                shutil.rmtree(p)
            else:
                p.rmdir()
        elif p.exists():
            p.unlink()

    # ------------------------------------------------------------------
    # JSON / JSONL
    # ------------------------------------------------------------------

    def write_json(self, data: Any, path: str) -> None:
        p = self._prepare(path)
        p.write_text(json.dumps(data, default=str, ensure_ascii=False), encoding="utf-8")
        logger.debug("Wrote JSON → %s", p)

    def read_json(self, path: str) -> Any:
        return json.loads(self._path(path).read_text(encoding="utf-8"))

    def write_jsonlines(self, records: list[dict], path: str) -> None:
        p = self._prepare(path)
        p.write_text("\n".join(json.dumps(r, default=str) for r in records), encoding="utf-8")
        logger.debug("Wrote %d JSONL records → %s", len(records), p)

    def read_jsonlines(self, path: str) -> list[dict]:
        text = self._path(path).read_text(encoding="utf-8")
        return [json.loads(line) for line in text.strip().split("\n") if line.strip()]

    # ------------------------------------------------------------------
    # Parquet
    # ------------------------------------------------------------------

    def write_parquet(self, df: pd.DataFrame, path: str) -> None:
        if df.empty:
            logger.warning("Skipping empty DataFrame for %s", path)
            return
        p = self._prepare(path)
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, p, compression="snappy")
        logger.debug("Wrote Parquet (%d rows) → %s", len(df), p)

    def read_parquet(self, path: str) -> pd.DataFrame:
        p = self._path(path)
        if not p.exists():
            raise FileNotFoundError(f"{p} not found")
        return pq.read_table(p).to_pandas()

    def read_parquet_dir(self, directory: str) -> pd.DataFrame:
        files = [f for f in self.list_files(directory) if f.endswith(".parquet")]
        if not files:
            return pd.DataFrame()
        return pd.concat([self.read_parquet(f) for f in files], ignore_index=True)

    # ------------------------------------------------------------------
    # Raw bytes (model artifacts)
    # ------------------------------------------------------------------

    def write_bytes(self, data: bytes, path: str) -> None:
        p = self._prepare(path)
        p.write_bytes(data)
        logger.debug("Wrote bytes (%d B) → %s", len(data), p)

    def read_bytes(self, path: str) -> bytes:
        p = self._path(path)
        if not p.exists():
            raise FileNotFoundError(f"{p} not found")
        return p.read_bytes()

    # ------------------------------------------------------------------
    # Latest run helper
    # ------------------------------------------------------------------

    def latest_subdir(self, parent: str) -> str | None:
        """Return the lexicographically last subdirectory under parent."""
        base = self._path(parent)
        if not base.is_dir():
            return None
        subdirs = sorted(p.name for p in base.iterdir() if p.is_dir())
        return subdirs[-1] if subdirs else None
