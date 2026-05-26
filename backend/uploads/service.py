"""
Upload pipeline orchestration.

Pure-Python service functions called by the HTTP routes in
``backend/api.py``. Keeping the orchestration out of the route layer
lets us unit-test the logic without spinning up FastAPI and keeps the
HTTP file thin.

Two responsibilities:

* ``issue_presigned_upload`` — validates the request, builds an S3 key,
  records a pending ``ingestion_jobs`` row, returns a presigned URL.
* ``finalize_upload``        — HEADs the S3 object to confirm bytes
  landed, idempotently records the ``documents`` row, returns the
  storage URI so the route can schedule ingestion.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Tuple

from fastapi import HTTPException

from backend.config import settings
from backend.storage.s3_storage import S3StorageProvider
from backend.vector_store import (
    create_ingestion_job,
    get_ingestion_job,
    update_ingestion_job,
)
from backend.uploads.keys import (
    build_upload_key,
    derive_tenant_slug,
    new_job_id,
    sanitize_filename,
)
from backend.uploads.schemas import (
    FinalizeUploadRequest,
    FinalizeUploadResponse,
    PresignUploadRequest,
    PresignUploadResponse,
)

logger = logging.getLogger("acadia-log-iq")


# ── Validation helpers ──────────────────────────────────────────────


def _validate_extension(filename: str) -> str:
    """Return the lower-cased extension, raising 400 on disallowed types."""
    ext = Path(filename or "").suffix[1:].lower()
    if not ext or ext not in settings.ALLOWED_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Type '{ext}' not allowed. Allowed: {settings.ALLOWED_FILE_TYPES}",
        )
    return ext


def _validate_size(size_bytes: int) -> None:
    """Reject oversize uploads early so we don't waste a presigned URL."""
    if size_bytes <= 0:
        return  # client did not declare; size will be checked at finalize
    max_bytes = int(settings.MAX_FILE_SIZE_MB) * 1024 * 1024
    if size_bytes > max_bytes:
        size_mb = size_bytes / (1024 * 1024)
        raise HTTPException(
            status_code=400,
            detail=f"File {size_mb:.1f}MB exceeds {settings.MAX_FILE_SIZE_MB}MB limit",
        )


def _resolve_doc_kind(doc_kind: str) -> str:
    """Coerce unknown values to 'ticket' (matches legacy /upload)."""
    raw = (doc_kind or "").strip().lower()
    return raw if raw in settings.VALID_DOC_KINDS else "ticket"


# ── Presign ─────────────────────────────────────────────────────────


def issue_presigned_upload(
    *,
    req: PresignUploadRequest,
    user_id: str,
    s3: S3StorageProvider,
) -> PresignUploadResponse:
    """Validate the request, mint a key, persist a pending job, return URL."""

    _validate_extension(req.filename)
    if req.size_bytes is not None:
        _validate_size(req.size_bytes)

    job_id = new_job_id()
    file_id = str(uuid.uuid4())
    tenant_slug = derive_tenant_slug(user_id)
    doc_kind = _resolve_doc_kind(req.doc_kind)

    # Flat S3 layout: tenants/{username}/{filename}. Same filename
    # twice = S3 overwrite (intentional — matches desktop folder UX).
    key = build_upload_key(
        prefix=settings.S3_KEY_PREFIX,
        tenant_slug=tenant_slug,
        filename=req.filename,
    )
    storage_uri = s3.make_storage_uri(key)

    expires_in = int(getattr(settings, "S3_PRESIGN_PUT_EXPIRY_SECONDS", 300))
    upload_url = s3.generate_presigned_put_url(
        key=key,
        content_type=req.content_type,
        expires_in=expires_in,
    )

    # ``owner_id`` is the raw authenticated user_id (matches what the
    # legacy /upload route and the rest of the app key off — sidebar
    # listing, RAG access filtering, etc.). The tenant_slug is *only*
    # used for the S3 path, not for ownership semantics.
    owner_id = (user_id or "").strip() or "anonymous"

    # Record a pending ingestion job so /upload_status/{job_id} works
    # immediately. The file_hash is unknown until finalize; legacy
    # consumers tolerate empty hash because they read it from the doc.
    create_ingestion_job(
        job_id=job_id,
        file_id=file_id,
        owner_id=owner_id,
        file_name=sanitize_filename(req.filename),
        file_type=req.file_type,
        file_hash="",
    )
    # Mark the job as awaiting upload so /upload_status reflects reality
    # between presign and finalize. update_ingestion_job silently drops
    # unknown columns, so this remains forward-compatible.
    update_ingestion_job(job_id, status="pending_upload")

    logger.info(
        "[upload.presign] job=%s owner=%s tenant_slug=%s key=%s exp=%ds",
        job_id, owner_id, tenant_slug, key, expires_in,
    )

    return PresignUploadResponse(
        job_id=job_id,
        file_id=file_id,
        bucket=s3.bucket_name,
        key=key,
        storage_uri=storage_uri,
        upload_url=upload_url,
        expires_in=expires_in,
        required_headers={"Content-Type": req.content_type},
    )


# ── Finalize ────────────────────────────────────────────────────────


def finalize_upload(
    *,
    req: FinalizeUploadRequest,
    user_id: str,
    s3: S3StorageProvider,
) -> Tuple[FinalizeUploadResponse, dict]:
    """HEAD the S3 object, verify ownership, return (response, job dict).

    The route layer is responsible for scheduling the ingestion
    background task — we return the job dict so the route has every
    field it needs (filename, file_id, file_type, doc_kind, etc.)
    without re-querying.
    """

    job = get_ingestion_job(req.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_not_found")

    owner_id = (user_id or "").strip() or "anonymous"
    if str(job.get("owner_id") or "") != owner_id:
        # Defense-in-depth: the presigned URL itself is scoped to the
        # caller's prefix; this catches the case where someone tries to
        # finalize a stranger's job_id.
        raise HTTPException(status_code=403, detail="job_owner_mismatch")

    # Confirm S3 actually has the object before we tell the rest of the
    # pipeline the upload succeeded. Cheap (single HEAD) and prevents
    # us from kicking ingestion on a stale or never-completed PUT.
    try:
        head = s3.head_object(req.key)
    except FileNotFoundError:
        raise HTTPException(status_code=409, detail="object_not_in_s3")

    _validate_size(int(head["content_length"]))

    storage_uri = s3.make_storage_uri(req.key)

    # ``file_hash`` is not whitelisted on the ingestion_jobs UPDATE path
    # in the current schema; we keep the S3 ETag in logs only. The
    # ``documents`` row that the indexer writes carries content-hash
    # already (calculated on the bytes we re-read for parsing).
    update_ingestion_job(req.job_id, status="queued")

    logger.info(
        "[upload.finalize] job=%s owner=%s key=%s size=%d etag=%s",
        req.job_id, owner_id, req.key, head["content_length"], head.get("etag", ""),
    )

    return (
        FinalizeUploadResponse(
            job_id=req.job_id,
            file_id=str(job.get("file_id") or ""),
            status="queued",
            storage_uri=storage_uri,
            size_bytes=int(head["content_length"]),
        ),
        dict(job),
    )
