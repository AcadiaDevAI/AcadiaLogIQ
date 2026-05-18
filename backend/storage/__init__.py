"""
Storage package — abstractions over where uploaded file bytes live.

Two concrete providers ship today:

* ``LocalStorageProvider`` — writes to ``settings.UPLOAD_DIR`` (legacy).
* ``S3StorageProvider``    — writes to ``settings.S3_UPLOAD_BUCKET`` and
                             issues presigned PUT URLs for direct
                             browser uploads (Phase 1).

The factory functions below let callers pick the right provider without
hard-coding the choice. The legacy ``/upload`` route in ``backend/api.py``
intentionally keeps a hard-coded ``LocalStorageProvider`` instance so it
remains a safe fallback while the S3 path is being rolled out.
"""

from __future__ import annotations

from typing import Optional

from backend.config import settings
from backend.storage.base import StorageProvider
from backend.storage.local_storage import LocalStorageProvider
from backend.storage.s3_storage import S3StorageProvider


def _is_s3_enabled() -> bool:
    """Return True iff settings request S3 *and* a bucket is configured."""
    storage_type = (getattr(settings, "STORAGE_TYPE", "local") or "local").lower()
    bucket = (getattr(settings, "S3_UPLOAD_BUCKET", "") or "").strip()
    return storage_type == "s3" and bool(bucket)


def get_s3_upload_provider() -> Optional[S3StorageProvider]:
    """Return a configured ``S3StorageProvider`` or None if S3 uploads
    are disabled / not configured. Callers (e.g. the /upload/presign
    route) should 503 when this returns None."""
    if not _is_s3_enabled():
        return None
    return S3StorageProvider(
        bucket_name=settings.S3_UPLOAD_BUCKET.strip(),
        prefix=(getattr(settings, "S3_KEY_PREFIX", "tenants") or "tenants").strip("/"),
    )


def get_upload_storage_provider() -> StorageProvider:
    """Return the active upload storage provider (S3 when configured,
    Local otherwise). Always returns a provider — never None."""
    s3 = get_s3_upload_provider()
    if s3 is not None:
        return s3
    return LocalStorageProvider()


__all__ = [
    "StorageProvider",
    "LocalStorageProvider",
    "S3StorageProvider",
    "get_s3_upload_provider",
    "get_upload_storage_provider",
]
