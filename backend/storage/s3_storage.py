"""
S3 storage implementation.

This provider participates in two flows:

1) Server-side writes via ``save_bytes`` (used by legacy code paths and
   any future ingestion job that needs to write to S3 directly).
2) Browser-direct uploads via presigned PUT URLs
   (``generate_presigned_put_url``) — see ``backend/uploads/`` and the
   ``/upload/presign`` + ``/upload/finalize`` routes. In this flow the
   API never streams the file bytes; the browser uploads straight to
   S3 and the API only confirms the object exists (``head_object``)
   before kicking ingestion.

Bytes can be fetched back for ingestion via ``read_bytes(s3://...)``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from backend.config import settings
from backend.storage.base import StorageProvider

logger = logging.getLogger("acadia-log-iq")


def _parse_s3_uri(storage_uri: str) -> Optional[Tuple[str, str]]:
    """Split ``s3://bucket/key`` → ``(bucket, key)``. Returns None on miss."""
    if not storage_uri or not storage_uri.startswith("s3://"):
        return None
    body = storage_uri[len("s3://"):]
    if "/" not in body:
        return None
    bucket, key = body.split("/", 1)
    if not bucket or not key:
        return None
    return bucket, key


class S3StorageProvider(StorageProvider):
    """S3-backed storage provider.

    Construction is cheap (just stores the bucket + region). The boto3
    client is created lazily on first use so module import never fails
    when AWS credentials are not yet resolved (e.g. local dev without
    a profile configured).
    """

    def __init__(self, bucket_name: str, prefix: str = "uploads"):
        self.bucket_name = bucket_name
        # Strip leading/trailing slashes so prefix concatenation stays
        # predictable regardless of how operators format the env value.
        self.prefix = (prefix or "").strip("/")
        self._client: Any = None

    # ── boto3 client (lazy) ──────────────────────────────────────────
    @property
    def client(self) -> Any:
        if self._client is None:
            # SigV4 is required for presigned URLs against KMS-encrypted
            # buckets and is the modern default — pin it explicitly so
            # behavior doesn't drift across boto3 versions.
            self._client = boto3.client(
                "s3",
                region_name=settings.AWS_REGION,
                config=BotoConfig(signature_version="s3v4"),
            )
        return self._client

    # ── StorageProvider contract ─────────────────────────────────────
    def save_bytes(self, relative_name: str, content: bytes) -> str:
        """Write bytes via PutObject (server-side). Primarily used by
        non-browser code paths; browser uploads use presigned PUT."""
        key = f"{self.prefix}/{relative_name}" if self.prefix else relative_name
        self.client.put_object(Bucket=self.bucket_name, Key=key, Body=content)
        return f"s3://{self.bucket_name}/{key}"

    def delete(self, storage_uri: str) -> None:
        parsed = _parse_s3_uri(storage_uri)
        if not parsed:
            return
        bucket, key = parsed
        self.client.delete_object(Bucket=bucket, Key=key)

    def resolve_local_path(self, storage_uri: str) -> Optional[Path]:
        # S3 objects are remote — callers must use read_bytes() instead.
        return None

    def read_bytes(self, storage_uri: str) -> bytes:
        """Fetch object bytes for ingestion. Ingestion jobs typically
        write these bytes to a NamedTemporaryFile so existing parsers
        (which expect a local Path) work unchanged."""
        parsed = _parse_s3_uri(storage_uri)
        if not parsed:
            raise ValueError(f"Not an s3:// URI: {storage_uri}")
        bucket, key = parsed
        try:
            obj = self.client.get_object(Bucket=bucket, Key=key)
            return obj["Body"].read()
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "Unknown")
            raise FileNotFoundError(
                f"S3 object not readable: s3://{bucket}/{key} ({code})"
            ) from exc

    # ── Presigned PUT + verification helpers ─────────────────────────
    def generate_presigned_put_url(
        self,
        *,
        key: str,
        content_type: str = "application/octet-stream",
        content_length_max: Optional[int] = None,
        expires_in: int = 300,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Issue a one-shot presigned PUT URL scoped to a single key.

        ``content_type`` must match the ``Content-Type`` header the
        browser sends — boto3 signs it into the URL so a mismatch will
        be rejected by S3 as SignatureDoesNotMatch.

        ``content_length_max`` is currently informational; AWS enforces
        size limits via bucket policy or PostObject conditions (we use
        the simpler put_object form here for PUT semantics). The
        frontend should pre-validate size before requesting a URL.
        """
        params: Dict[str, Any] = {
            "Bucket": self.bucket_name,
            "Key": key,
            "ContentType": content_type,
        }
        if extra_params:
            params.update(extra_params)
        try:
            return self.client.generate_presigned_url(
                ClientMethod="put_object",
                Params=params,
                ExpiresIn=int(expires_in),
                HttpMethod="PUT",
            )
        except ClientError as exc:
            logger.warning("[s3] presign failed key=%s err=%s", key, exc)
            raise

    def head_object(self, key: str) -> Dict[str, Any]:
        """Return object metadata or raise ``FileNotFoundError`` on 404."""
        try:
            head = self.client.head_object(Bucket=self.bucket_name, Key=key)
            return {
                "content_length": int(head.get("ContentLength", 0)),
                "content_type": head.get("ContentType", ""),
                "etag": (head.get("ETag") or "").strip('"'),
                "last_modified": head.get("LastModified"),
                "version_id": head.get("VersionId"),
            }
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if code in ("404", "NoSuchKey", "NotFound") or status == 404:
                raise FileNotFoundError(
                    f"S3 object not found: s3://{self.bucket_name}/{key}"
                ) from exc
            raise

    # Convenience builder so callers don't reach into bucket_name.
    def make_storage_uri(self, key: str) -> str:
        return f"s3://{self.bucket_name}/{key}"
