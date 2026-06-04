"""FastAPI router for the Escalation Procedures KB.

Endpoints
---------
``GET  /escalation/status``   — readiness + per-section chunk counts.
``POST /escalation/upload``   — one-shot upload + parse + embed.
``POST /escalation/ask``      — section-scoped grounded Q&A.
``GET  /escalation/debug``    — diagnostic snapshot of UPLOAD_DIR.

Typical flow: the modal opens, hits ``/status``. If the KB is already
loaded the picker renders immediately; otherwise the modal shows an
upload dragger that POSTs to ``/upload``. After the upload succeeds
the modal refreshes status and lands on the picker.

All routes are Clerk-auth gated via :func:`lazy_auth_dependency`.
"""

from __future__ import annotations

import logging
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from backend._lazy_auth import lazy_auth_dependency
from backend.observability.rate_limit import limiter

from .bedrock_client import embed_text
from .bootstrap import _iter_pdfs, _name_matches_kb, _upload_root, ensure_kb_loaded, scan_report
from .parser import parse_pdf
from .qa import answer as answer_question
from .sections import KB_FILENAME, SECTION_IDS
from .store import delete_kb, save_kb


logger = logging.getLogger("acadia-log-iq")


router = APIRouter(prefix="/escalation", tags=["escalation"])


# ── Schemas ─────────────────────────────────────────────────────────


class StatusResponse(BaseModel):
    ready: bool
    filename: Optional[str] = None
    updated_at: Optional[str] = None
    sections: dict = Field(default_factory=dict)
    error: Optional[str] = None
    # When the KB isn't loaded we surface a quick diagnostic snapshot so
    # the modal can show the user exactly what the backend can see on
    # disk. ``pdfs_seen`` is the list of PDF filenames discovered
    # anywhere under ``upload_dir`` — empty means the backend can't see
    # any uploaded file at all (volume mount / wrong host path).
    upload_dir: Optional[str] = None
    pdfs_seen: List[str] = Field(default_factory=list)


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    text: str
    model_config = ConfigDict(extra="ignore")


class AskRequest(BaseModel):
    section: Literal["cisco", "microsoft", "verizon", "att", "vendor_dispatch"]
    question: str = Field(min_length=1, max_length=2000)
    history: List[HistoryTurn] = Field(default_factory=list)
    model_config = ConfigDict(extra="ignore")


class AskSource(BaseModel):
    page: int
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: List[AskSource]


class UploadResponse(BaseModel):
    ok: bool
    filename: str
    total_chunks: int
    sections: dict


class DeleteResponse(BaseModel):
    ok: bool
    deleted_index: bool
    deleted_files: List[str]


# ── Routes ──────────────────────────────────────────────────────────


@router.get("/status", response_model=StatusResponse)
async def status(
    request: Request,
    user_id: Optional[str] = Depends(lazy_auth_dependency),
) -> StatusResponse:
    return StatusResponse(**ensure_kb_loaded(user_id=user_id))


@router.post("/upload", response_model=UploadResponse)
@limiter.limit("10/minute")
async def upload(
    request: Request,
    file: UploadFile = File(...),
    user_id: Optional[str] = Depends(lazy_auth_dependency),
) -> UploadResponse:
    """One-shot ingest. Parses + embeds + persists ``kb.json`` so the
    very next ``/status`` call reports ready and the modal jumps to the
    picker without a refresh."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF uploads are accepted.")

    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file payload.")

    try:
        chunks, sections_summary = parse_pdf(content)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        logger.warning("[escalation] parse_pdf failed: %s", exc)
        raise HTTPException(500, "Failed to parse the PDF.")

    if not chunks:
        raise HTTPException(400, "No content extracted from the PDF.")

    embedded: List[dict] = []
    for chunk in chunks:
        try:
            vector = embed_text(chunk.text)
        except Exception as exc:
            logger.warning(
                "[escalation] embedding failed (section=%s page=%s): %s",
                chunk.section, chunk.page, exc,
            )
            raise HTTPException(502, "Embedding service failed mid-ingest.")
        embedded.append({
            "section": chunk.section,
            "page": chunk.page,
            "text": chunk.text,
            "embedding": vector,
        })

    saved = save_kb(
        filename=file.filename or KB_FILENAME,
        sections=sections_summary,
        chunks=embedded,
    )
    logger.info(
        "[escalation] uploaded KB %s (%d chunks, sections=%s)",
        saved.get("filename"),
        len(embedded),
        {sid: meta.get("chunks", 0) for sid, meta in sections_summary.items()},
    )

    return UploadResponse(
        ok=True,
        filename=saved.get("filename") or (file.filename or KB_FILENAME),
        total_chunks=len(embedded),
        sections=sections_summary,
    )


@router.delete("/kb", response_model=DeleteResponse)
@limiter.limit("10/minute")
async def delete_kb_route(
    request: Request,
    user_id: Optional[str] = Depends(lazy_auth_dependency),
) -> DeleteResponse:
    """Wipe the indexed KB JSON and any matching source PDF on disk so
    the modal goes back to the upload step. S3-backed copies are left
    alone — operators should manage those out of band."""
    deleted_index = False
    try:
        deleted_index = delete_kb()
    except OSError as exc:
        raise HTTPException(500, f"Failed to delete kb.json: {exc}")

    deleted_files: List[str] = []
    try:
        root = _upload_root()
        if root.is_dir():
            for pdf in _iter_pdfs(root):
                if _name_matches_kb(pdf.name):
                    try:
                        pdf.unlink()
                        deleted_files.append(str(pdf))
                    except OSError as exc:
                        logger.warning(
                            "[escalation] failed to delete source PDF %s: %s",
                            pdf, exc,
                        )
    except OSError as exc:
        logger.warning("[escalation] upload-dir scan failed during delete: %s", exc)

    logger.info(
        "[escalation] KB deleted (index=%s, files=%d)",
        deleted_index, len(deleted_files),
    )
    return DeleteResponse(
        ok=True,
        deleted_index=deleted_index,
        deleted_files=deleted_files,
    )


@router.get("/debug")
async def debug(
    request: Request,
    user_id: Optional[str] = Depends(lazy_auth_dependency),
) -> dict:
    """Diagnostic view of what the backend sees in UPLOAD_DIR.

    Returns the resolved upload directory, whether it exists, and the
    list of PDFs the scanner found with a per-file ``matches_kb`` flag.
    Use this to confirm the Escalation PDF really is where it needs
    to be (and named matchably) before debugging anything deeper.
    """
    return scan_report()


@router.post("/ask", response_model=AskResponse)
@limiter.limit("60/minute")
async def ask(
    request: Request,
    payload: AskRequest,
    user_id: Optional[str] = Depends(lazy_auth_dependency),
) -> AskResponse:
    if payload.section not in SECTION_IDS:
        raise HTTPException(400, "Unknown section.")

    state = ensure_kb_loaded(user_id=user_id)
    if not state.get("ready"):
        raise HTTPException(
            503,
            state.get("error")
            or "Escalation Procedures KB is not loaded on the server.",
        )

    try:
        result = answer_question(
            section_id=payload.section,
            question=payload.question,
            history=[t.model_dump() for t in payload.history],
        )
    except Exception as exc:
        logger.warning(
            "[escalation] ask failed (section=%s): %s",
            payload.section, exc,
        )
        raise HTTPException(500, "Failed to generate an answer.")

    return AskResponse(
        answer=result["answer"],
        sources=[AskSource(**s) for s in result.get("sources", [])],
    )
