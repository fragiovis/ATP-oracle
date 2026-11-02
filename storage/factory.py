"""Pick the storage client according to ATP_STORAGE_BACKEND ("s3" | "local")."""

from __future__ import annotations

from config.settings import settings


def get_storage_client():
    if settings.storage.backend == "local":
        from storage.local_client import LocalClient
        return LocalClient()
    from storage.s3_client import S3Client
    return S3Client()
