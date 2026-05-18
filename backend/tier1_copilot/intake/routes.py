"""Sprint 9 — Universal Intake FastAPI router.

Endpoints:
  POST /intake/extract                     extract candidates
  POST /intake/extraction/{id}/feedback    log engineer pick/edit/reject
  GET  /intake/health                      catalog state probe
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from backend._lazy_auth import lazy_auth_dependency
from backend.config import settings
from backend.tier1_copilot.intake.audit import (
    log_extraction,
    log_extraction_feedback,
)
from backend.tier1_copilot.intake.catalogs import get_intake_catalogs
from backend.tier1_copilot.intake.diversifier import diversify
from backend.tier1_copilot.intake.extractor import extract_candidates
from backend.tier1_copilot.intake.schemas import (
    ExtractRequest,
    ExtractResponse,
    ExtractionFeedbackRequest,
    ExtractionFeedbackResponse,
    IntakeHealthResponse,
    VALID_SOURCE_TYPES,
)
from backend.tier1_copilot.intake.validator import validate_candidate

logger = logging.getLogger("acadia-log-iq")

router = APIRouter(prefix="/intake", tags=["universal-intake"])


@router.post("/extract", response_model=ExtractResponse)
async def extract(
    req: ExtractRequest,
    user_id: Optional[str] = Depends(lazy_auth_dependency),
) -> ExtractResponse:
    raw = (req.raw_text or "").strip()
    if not raw:
        raise HTTPException(
            status_code=422, detail={"error": "raw_text cannot be empty"},
        )
    cap = int(getattr(settings, "INTAKE_MAX_RAW_CHARS", 10000))
    if len(raw) > cap:
        raise HTTPException(
            status_code=422,
            detail={"error": f"raw_text exceeds {cap} char limit"},
        )
    if req.source not in VALID_SOURCE_TYPES:
        raise HTTPException(
            status_code=422,
            detail={"error": f"source must be one of {list(VALID_SOURCE_TYPES)}"},
        )

    catalogs = get_intake_catalogs()

    raw_candidates = extract_candidates(
        raw_text=raw,
        source_type=req.source,
        catalogs=catalogs,
        n_candidates=int(getattr(settings, "INTAKE_MAX_CANDIDATES", 5)),
    )
    if not raw_candidates:
        return ExtractResponse(
            extraction_id=None,
            candidates=[],
            error="extraction_failed_please_fill_manually",
        )

    # Sprint 9.2 — pass raw_text so the validator's substring grounding
    # can reject hallucinated extractions (fields whose evidence isn't
    # a verbatim substring of the engineer's pasted content). When
    # raw_text is omitted, the validator is permissive (Sprint 9
    # behaviour) so older callers keep working.
    validated = [
        validate_candidate(c, catalogs, raw_text=raw)
        for c in raw_candidates
    ]
    diversified = diversify(
        validated,
        max_cards=int(getattr(settings, "INTAKE_MAX_CARDS", 4)),
    )

    extraction_id = log_extraction(
        source=req.source,
        raw_text=raw,
        candidates=diversified,
        session_id=req.session_id,
    )

    logger.info(
        "[intake] extract source=%s cands=%d sess=%s",
        req.source, len(diversified), req.session_id,
    )
    return ExtractResponse(
        extraction_id=extraction_id,
        candidates=diversified,
        error=None,
    )


@router.post(
    "/extraction/{extraction_id}/feedback",
    response_model=ExtractionFeedbackResponse,
)
async def extraction_feedback(
    extraction_id: str,
    body: ExtractionFeedbackRequest,
    user_id: Optional[str] = Depends(lazy_auth_dependency),
) -> ExtractionFeedbackResponse:
    ok = log_extraction_feedback(
        extraction_id=extraction_id,
        picked_index=body.picked_index,
        edits=body.edits,
        was_rejected=body.was_rejected,
    )
    return ExtractionFeedbackResponse(
        ok=bool(ok), extraction_id=extraction_id,
    )


@router.get("/health", response_model=IntakeHealthResponse)
async def health() -> IntakeHealthResponse:
    catalogs = get_intake_catalogs(lazy_build=False)
    s, a, t, c = catalogs.health_snapshot()
    return IntakeHealthResponse(
        ok=True,
        flag_on=True,
        catalogs_built=catalogs.is_built(),
        severities=s,
        asset_families=a,
        alert_types=t,
        customers=c,
    )
