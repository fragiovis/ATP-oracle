"""S3 storage client – drop-in replacement for HDFSClient.

Provides the same write_parquet / read_parquet / write_jsonlines / write_bytes /
read_bytes interface so the rest of the pipeline doesn't need to change.

S3 key layout mirrors the HDFS path layout:
  /data/silver/matches.parquet  →  s3://<bucket>/data/silver/matches.parquet
  (leading slash stripped automatically)
"""

from __future__ import annotations

import io
import json
import logging
from typing import Any

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config.settings import settings

logger = logging.getLogger(__name__)


class S3Client:
    """Thin boto3 wrapper with the same interface as HDFSClient."""

    def __init__(self):
        cfg = settings.s3
        self._bucket = cfg.bucket
        self._prefix = cfg.prefix.strip("/")
        self._s3 = boto3.client("s3", region_name=cfg.region)
        logger.info("S3 client ready: s3://%s/%s", self._bucket, self._prefix)

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    def _key(self, path: str) -> str:
        """Convert an absolute-style path to an S3 key."""
        p = path.lstrip("/")
        return f"{self._prefix}/{p}" if self._prefix else p

    # ------------------------------------------------------------------
    # Existence / listing
    # ------------------------------------------------------------------

    def exists(self, path: str) -> bool:
        try:
            self._s3.head_object(Bucket=self._bucket, Key=self._key(path))
            return True
        except self._s3.exceptions.ClientError:
            return False
        except Exception:
            return False

    def list_files(self, directory: str) -> list[str]:
        prefix = self._key(directory).rstrip("/") + "/"
        paginator = self._s3.get_paginator("list_objects_v2")
        paths = []
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                # Return paths in the same absolute-style format
                rel = key[len(self._prefix):].lstrip("/") if self._prefix else key
                paths.append("/" + rel)
        return paths

    def delete(self, path: str, recursive: bool = False) -> None:
        if recursive:
            prefix = self._key(path).rstrip("/") + "/"
            paginator = self._s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    self._s3.delete_object(Bucket=self._bucket, Key=obj["Key"])
        else:
            self._s3.delete_object(Bucket=self._bucket, Key=self._key(path))

    # ------------------------------------------------------------------
    # JSON / JSONL
    # ------------------------------------------------------------------

    def write_json(self, data: Any, path: str) -> None:
        body = json.dumps(data, default=str, ensure_ascii=False).encode("utf-8")
        self._s3.put_object(Bucket=self._bucket, Key=self._key(path), Body=body)
        logger.debug("Wrote JSON → s3://%s/%s", self._bucket, self._key(path))

    def read_json(self, path: str) -> Any:
        resp = self._s3.get_object(Bucket=self._bucket, Key=self._key(path))
        return json.loads(resp["Body"].read().decode("utf-8"))

    def write_jsonlines(self, records: list[dict], path: str) -> None:
        lines = "\n".join(json.dumps(r, default=str) for r in records)
        self._s3.put_object(
            Bucket=self._bucket, Key=self._key(path),
            Body=lines.encode("utf-8"),
        )
        logger.debug("Wrote %d JSONL records → s3://%s/%s", len(records), self._bucket, self._key(path))

    # ------------------------------------------------------------------
    # Parquet
    # ------------------------------------------------------------------

    def write_parquet(self, df: pd.DataFrame, path: str) -> None:
        if df.empty:
            logger.warning("Skipping empty DataFrame for %s", path)
            return
        table = pa.Table.from_pandas(df, preserve_index=False)
        buf = io.BytesIO()
        pq.write_table(table, buf, compression="snappy")
        buf.seek(0)
        self._s3.put_object(
            Bucket=self._bucket, Key=self._key(path), Body=buf.read()
        )
        logger.debug("Wrote Parquet (%d rows) → s3://%s/%s", len(df), self._bucket, self._key(path))

    def read_parquet(self, path: str) -> pd.DataFrame:
        key = self._key(path)
        try:
            resp = self._s3.get_object(Bucket=self._bucket, Key=key)
        except self._s3.exceptions.NoSuchKey:
            raise FileNotFoundError(f"s3://{self._bucket}/{key} not found")
        except Exception as e:
            # boto3 raises ClientError with code NoSuchKey
            if "NoSuchKey" in str(e) or "404" in str(e):
                raise FileNotFoundError(f"s3://{self._bucket}/{key} not found")
            raise
        buf = io.BytesIO(resp["Body"].read())
        return pq.read_table(buf).to_pandas()

    def read_parquet_dir(self, directory: str) -> pd.DataFrame:
        files = [f for f in self.list_files(directory) if f.endswith(".parquet")]
        if not files:
            return pd.DataFrame()
        return pd.concat([self.read_parquet(f) for f in files], ignore_index=True)

    # ------------------------------------------------------------------
    # Raw bytes (model artifacts)
    # ------------------------------------------------------------------

    def write_bytes(self, data: bytes, path: str) -> None:
        self._s3.put_object(Bucket=self._bucket, Key=self._key(path), Body=data)
        logger.debug("Wrote bytes (%d B) → s3://%s/%s", len(data), self._bucket, self._key(path))

    def read_bytes(self, path: str) -> bytes:
        key = self._key(path)
        try:
            resp = self._s3.get_object(Bucket=self._bucket, Key=key)
        except Exception as e:
            if "NoSuchKey" in str(e) or "404" in str(e):
                raise FileNotFoundError(f"s3://{self._bucket}/{key} not found")
            raise
        return resp["Body"].read()

    # ------------------------------------------------------------------
    # Latest run helper (mirrors HDFS trainer pattern)
    # ------------------------------------------------------------------

    def latest_subdir(self, parent: str) -> str | None:
        """Return the lexicographically last 'subdirectory' under parent."""
        prefix = self._key(parent).rstrip("/") + "/"
        paginator = self._s3.get_paginator("list_objects_v2")
        subdirs: set[str] = set()
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix, Delimiter="/"):
            for cp in page.get("CommonPrefixes", []):
                name = cp["Prefix"].rstrip("/").split("/")[-1]
                subdirs.add(name)
        return sorted(subdirs)[-1] if subdirs else None
