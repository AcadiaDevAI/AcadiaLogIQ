"""
Pydantic request/response models for the presigned-upload pipeline.

These deliberately mirror the field names of the legacy
``UploadResponse`` (``job_id`` / ``file_id``) so frontend state-management
code can be reused across both upload paths during the cut-over.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


# ── Presign ─────────────────────────────────────────────────────────


class PresignUploadRequest(BaseModel):
    """Browser tells the API what it wants to upload; gets back a URL."""

    filename: str = Field(min_length=1, max_length=255)
    content_type: str = Field(
        default="application/octet-stream",
        max_length=120,
        description="MUST match the Content-Type the browser will PUT.",
    )
    size_bytes: Optional[int] = Field(
        default=None,
        ge=0,
        description="Optional — the API rejects sizes over the configured cap.",
    )
    file_type: str = Field(
        default="kb",
        pattern="^(kb)$",
        description="Storage classification (kept identical to legacy /upload).",
    )
    doc_kind: str = Field(
        default="ticket",
        max_length=40,
        description="Sprint 3-PREP-B doc_kind whitelist.",
    )

    model_config = ConfigDict(extra="ignore")


class PresignUploadResponse(BaseModel):
    """API tells the browser where + how to upload."""

    job_id: str
    file_id: str
    bucket: str
    key: str
    storage_uri: str
    upload_url: str
    expires_in: int
    required_headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Headers the browser must send with the PUT for the "
                    "presigned signature to validate (notably Content-Type).",
    )

    model_config = ConfigDict(extra="ignore")


# ── Finalize ────────────────────────────────────────────────────────


class FinalizeUploadRequest(BaseModel):
    """Browser confirms upload landed; API verifies + kicks ingestion."""

    job_id: str = Field(min_length=1, max_length=64)
    key: str = Field(min_length=1, max_length=2048)
    sha256: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Optional client-computed SHA-256 for integrity audit. "
                    "Not verified server-side today (S3 ETag is the ground truth).",
    )

    model_config = ConfigDict(extra="ignore")


class FinalizeUploadResponse(BaseModel):
    job_id: str
    file_id: str
    status: str
    storage_uri: str
    size_bytes: int

    model_config = ConfigDict(extra="ignore")
