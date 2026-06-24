"""Thin wrapper around the `hdfs` library for reading/writing to HDFS.

Provides:
  - write_parquet(df, hdfs_path)    – write a pandas DataFrame as Parquet
  - read_parquet(hdfs_path)         – read a Parquet file from HDFS
  - write_json(data, hdfs_path)     – write raw JSON (for bronze/raw layers)
  - list_files(hdfs_dir)            – list files under an HDFS directory
  - mkdir(hdfs_dir)                 – create directory (idempotent)
  - exists(hdfs_path)               – check existence
"""

from __future__ import annotations

import io
import json
import logging
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from hdfs import InsecureClient
from hdfs.util import HdfsError

from config.settings import settings

logger = logging.getLogger(__name__)


class HDFSClient:
    """Wrapper around the HDFS InsecureClient with convenience methods."""

    def __init__(self):
        cfg = settings.hdfs
        self._client = InsecureClient(cfg.namenode_url, user=cfg.user)
        logger.debug("HDFS client connected: %s (user=%s)", cfg.namenode_url, cfg.user)

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------

    def mkdir(self, hdfs_path: str) -> None:
        """Create directory and all parents (idempotent)."""
        try:
            self._client.makedirs(hdfs_path, permission=755)
        except HdfsError as e:
            if "already exists" not in str(e):
                raise

    def exists(self, hdfs_path: str) -> bool:
        try:
            status = self._client.status(hdfs_path, strict=False)
            return status is not None
        except HdfsError:
            return False

    def list_files(self, hdfs_dir: str) -> list[str]:
        """Return list of file paths under *hdfs_dir*."""
        try:
            return [
                f"{hdfs_dir}/{name}"
                for name in self._client.list(hdfs_dir)
            ]
        except HdfsError as e:
            logger.warning("Could not list %s: %s", hdfs_dir, e)
            return []

    def delete(self, hdfs_path: str, recursive: bool = False) -> None:
        self._client.delete(hdfs_path, recursive=recursive)

    # ------------------------------------------------------------------
    # JSON (raw / bronze)
    # ------------------------------------------------------------------

    def write_json(self, data: Any, hdfs_path: str) -> None:
        """Write a Python object as JSON (newline-delimited if list)."""
        self.mkdir("/".join(hdfs_path.split("/")[:-1]))
        raw = json.dumps(data, default=str, ensure_ascii=False)
        with self._client.write(hdfs_path, overwrite=True, encoding="utf-8") as writer:
            writer.write(raw)
        logger.debug("Wrote JSON → %s", hdfs_path)

    def read_json(self, hdfs_path: str) -> Any:
        with self._client.read(hdfs_path, encoding="utf-8") as reader:
            return json.loads(reader.read())

    def write_jsonlines(self, records: list[dict], hdfs_path: str) -> None:
        """Write a list of dicts as newline-delimited JSON."""
        self.mkdir("/".join(hdfs_path.split("/")[:-1]))
        lines = "\n".join(json.dumps(r, default=str) for r in records)
        with self._client.write(hdfs_path, overwrite=True, encoding="utf-8") as writer:
            writer.write(lines)
        logger.debug("Wrote %d JSONL records → %s", len(records), hdfs_path)

    # ------------------------------------------------------------------
    # Parquet (silver / gold)
    # ------------------------------------------------------------------

    def write_parquet(self, df: pd.DataFrame, hdfs_path: str) -> None:
        """Write a pandas DataFrame to HDFS as Parquet."""
        if df.empty:
            logger.warning("Skipping empty DataFrame for %s", hdfs_path)
            return
        self.mkdir("/".join(hdfs_path.split("/")[:-1]))
        table = pa.Table.from_pandas(df, preserve_index=False)
        buf = io.BytesIO()
        pq.write_table(table, buf, compression="snappy")
        buf.seek(0)
        with self._client.write(hdfs_path, overwrite=True) as writer:
            writer.write(buf.read())
        logger.debug("Wrote Parquet (%d rows) → %s", len(df), hdfs_path)

    def read_parquet(self, hdfs_path: str) -> pd.DataFrame:
        """Read a Parquet file from HDFS into a pandas DataFrame."""
        with self._client.read(hdfs_path) as reader:
            buf = io.BytesIO(reader.read())
        return pq.read_table(buf).to_pandas()

    def read_parquet_dir(self, hdfs_dir: str) -> pd.DataFrame:
        """Read all Parquet files in a directory and concatenate."""
        files = [f for f in self.list_files(hdfs_dir) if f.endswith(".parquet")]
        if not files:
            return pd.DataFrame()
        dfs = [self.read_parquet(f) for f in files]
        return pd.concat(dfs, ignore_index=True)

    # ------------------------------------------------------------------
    # Model artifact helpers (pickle / bytes)
    # ------------------------------------------------------------------

    def write_bytes(self, data: bytes, hdfs_path: str) -> None:
        self.mkdir("/".join(hdfs_path.split("/")[:-1]))
        with self._client.write(hdfs_path, overwrite=True) as writer:
            writer.write(data)
        logger.debug("Wrote bytes (%d B) → %s", len(data), hdfs_path)

    def read_bytes(self, hdfs_path: str) -> bytes:
        with self._client.read(hdfs_path) as reader:
            return reader.read()
