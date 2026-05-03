"""Sprint 10 — Tier-1 Resolution Journey FastAPI router.

All endpoints under prefix `/tier1/journey`. Every endpoint short-
circuits with `404 detail="tier1_journey_flag_off"` when
`LOGIQ_TIER1_JOURNEY_BACKEND` is False.

`/initial` runs Stage 0 + 1A + 1B aggregators in sequence and returns
the bundle for the always-visible first paint. Per-stage GETs share an
in-process cohort cache keyed on session_id (TTL +
LRU-evict-at-1024) so Stages 2-5 don't re-fetch the same metadata.

Telemetry POST writes to `tier1_journey_events` fire-and-forget.
"""
from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, Header, HTTPException, Request

from backend.config import settings

from .schemas import (
    JourneyEventRequest,
    JourneyEventResponse,
    JourneyInitial,
    PivotInsights,
    ResumeStateResponse,
    SearchKBHandoffRequest,
    SearchKBHandoffResponse,
    Stage2HistoricalMatches,
    Stage3TroubleshootingApproach,
    Stage4SearchKB,
)
from .stage0_confidence import compute_stage0
from .stage1_smoking_gun import build_stage1a
from .stage1_do_not_chase import build_stage1b
from .stage2_historical import build_stage2
from .stage3_troubleshooting import build_stage3
from .stage4_kb_handoff import build_stage4
from .stage4_search_kb_handoff import create_chat_session_with_handoff
from .stage5_escalation import build_journey_escalation_package
from .telemetry import record_event
from .ticket_loader import load_cohort_metadata


logger = logging.getLogger("acadia-log-iq")


router = APIRouter(prefix="/tier1/journey", tags=["tier1-journey"])


# ─────────────────────────────────────────────────────────────
# In-process cohort cache (keyed on session_id)
# ─────────────────────────────────────────────────────────────
_COHORT_CACHE_MAX = 1024
_cohort_cache: "OrderedDict[str, Tuple[float, List[Dict[str, Any]]]]" = OrderedDict()


def _cache_get(session_id: str) -> Optional[List[Dict[str, Any]]]:
    if not session_id:
        return None
    entry = _cohort_cache.get(session_id)
    if entry is None:
        return None
    expires_at, cohort = entry
    if time.time() >= expires_at:
        _cohort_cache.pop(session_id, None)
        return None
    _cohort_cache.move_to_end(session_id)
    return cohort


def _cache_put(session_id: str, cohort: List[Dict[str, Any]]) -> None:
    ttl = int(getattr(settings, "TIER1_JOURNEY_BUNDLE_CACHE_TTL_SECONDS", 600))
    _cohort_cache[session_id] = (time.time() + ttl, cohort)
    _cohort_cache.move_to_end(session_id)
    while len(_cohort_cache) > _COHORT_CACHE_MAX:
        _cohort_cache.popitem(last=False)


def _reset_cache() -> None:
    """Test helper. Production code should never call this."""
    _cohort_cache.clear()


def _load_or_cache(session_id: str) -> List[Dict[str, Any]]:
    """Single-flight cohort fetch. Cache miss → DB → cache."""
    cached = _cache_get(session_id)
    if cached is not None:
        return cached
    cohort = load_cohort_metadata(session_id)
    _cache_put(session_id, cohort)
    return cohort


# ─────────────────────────────────────────────────────────────
# Flag gate (mirrors Sprint 7/8 pattern)
# ─────────────────────────────────────────────────────────────
def _require_flag() -> None:
    if not getattr(settings, "LOGIQ_TIER1_JOURNEY_BACKEND", False):
        raise HTTPException(status_code=404, detail="tier1_journey_flag_off")


# Sprint 10.6 §4 — lazy wrapper around backend.api.auth_dependency.
# Module-level `from backend.api import auth_dependency` would
# circular-import: api.py mounts this router at line 443, which fires
# BEFORE auth_dependency is defined further down at line 520. The
# wrapper defers the import to request time. FastAPI injects `request`
# and `x_api_key` into the wrapper signature directly; we forward both
# to the lazy-imported real function.
async def _lazy_auth_dependency(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> Optional[str]:
    from backend.api import auth_dependency
    return await auth_dependency(request, x_api_key)


# ─────────────────────────────────────────────────────────────
# /initial — what paints on first journey load
# Sprint 10.2 — Stage 1A + 1B merged into pivot_insights wrapper.
# Stage 0 returns Best-Ticket Distillation (was ConfidenceLead).
# ─────────────────────────────────────────────────────────────
@router.get("/{session_id}/initial", response_model=JourneyInitial)
async def get_initial(session_id: str) -> JourneyInitial:
    """Stage 0 (Best-Ticket Distillation) + merged PivotInsights —
    eager, always visible."""
    _require_flag()

    cohort = _load_or_cache(session_id)
    lookback = int(getattr(settings, "TIER1_JOURNEY_LOOKBACK_MONTHS", 18))
    threshold = float(getattr(settings, "TIER1_JOURNEY_STAGE0_DOMINANT_THRESHOLD", 0.4))

    stage_0 = compute_stage0(cohort, lookback_months=lookback)
    smoking_gun = build_stage1a(cohort, min_frequency_ratio=threshold)
    do_not_chase = build_stage1b(cohort)

    # Sprint 10.3 §5.2 — diagnostic-grade log. From one line you can
    # tell whether each panel got populated and why.
    logger.info(
        "[journey.initial] sid=%s top5=%d cohort=%d "
        "s1a_derived=%s s1b_entries=%d s1b_reason=%s "
        "s0_strength=%s",
        session_id[:12],
        len(cohort),
        stage_0.cohort_size,
        smoking_gun.derived_from,
        len(do_not_chase.entries),
        do_not_chase.reason,
        stage_0.evidence_strength,
    )

    return JourneyInitial(
        session_id=session_id,
        stage_0=stage_0,
        pivot_insights=PivotInsights(
            smoking_gun=smoking_gun,
            do_not_chase=do_not_chase,
        ),
    )


# ─────────────────────────────────────────────────────────────
# Sprint 10.2 — /pivot-insights — merged Smoking Gun + Do Not
# Chase. Same data as embedded in /initial, exposed standalone
# for callers that want just the Stage 1 panel content.
# ─────────────────────────────────────────────────────────────
@router.get("/{session_id}/pivot-insights", response_model=PivotInsights)
async def get_pivot_insights(session_id: str) -> PivotInsights:
    _require_flag()
    cohort = _load_or_cache(session_id)
    threshold = float(getattr(settings, "TIER1_JOURNEY_STAGE0_DOMINANT_THRESHOLD", 0.4))
    return PivotInsights(
        smoking_gun=build_stage1a(cohort, min_frequency_ratio=threshold),
        do_not_chase=build_stage1b(cohort),
    )


# ─────────────────────────────────────────────────────────────
# /stage-2 — Historical Matches
# ─────────────────────────────────────────────────────────────
@router.get("/{session_id}/stage-2", response_model=Stage2HistoricalMatches)
async def get_stage_2(session_id: str) -> Stage2HistoricalMatches:
    _require_flag()
    cohort = _load_or_cache(session_id)
    return build_stage2(cohort)


# ─────────────────────────────────────────────────────────────
# /stage-3 — Troubleshooting Approach
# ─────────────────────────────────────────────────────────────
@router.get("/{session_id}/stage-3", response_model=Stage3TroubleshootingApproach)
async def get_stage_3(session_id: str) -> Stage3TroubleshootingApproach:
    _require_flag()
    cohort = _load_or_cache(session_id)
    return build_stage3(cohort)


# ─────────────────────────────────────────────────────────────
# /stage-4 — Search KB / SOP handoff payload
# ─────────────────────────────────────────────────────────────
@router.get("/{session_id}/stage-4", response_model=Stage4SearchKB)
async def get_stage_4(session_id: str) -> Stage4SearchKB:
    """Pre-filled chat message + allowed_doc_kinds. The frontend reads
    the alert payload from the engineer's intake form; this endpoint
    just supplies the dominant_root_cause line from Stage 0 so the
    handoff message stays accurate."""
    _require_flag()
    cohort = _load_or_cache(session_id)

    # Pull the engineer's original alert from tier1_sessions.alert_payload
    severity = asset = alert = notes = None
    try:
        from sqlalchemy import text
        from backend.db.connection import engine
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT alert_payload FROM tier1_sessions WHERE id = :id"
                ),
                {"id": session_id},
            ).mappings().first()
        if row and isinstance(row.get("alert_payload"), dict):
            ap = row["alert_payload"]
            severity = ap.get("severity")
            asset = ap.get("asset_name")
            alert = ap.get("alert_type")
            notes = ap.get("notes")
    except Exception as exc:
        logger.warning("[journey.stage4] alert_payload read failed: %s", exc)

    # Dominant root cause from Stage 0 profile signature (cheap re-compute)
    lookback = int(getattr(settings, "TIER1_JOURNEY_LOOKBACK_MONTHS", 18))
    stage_0 = compute_stage0(cohort, lookback_months=lookback)
    dominant = stage_0.profile_match if not stage_0.sparse else None

    return build_stage4(
        severity=severity,
        asset_name=asset,
        alert_type=alert,
        notes=notes,
        dominant_root_cause=dominant,
    )


# ─────────────────────────────────────────────────────────────
# /stage-5 — Escalation package (reuses Sprint 7 generator)
# ─────────────────────────────────────────────────────────────
@router.get("/{session_id}/stage-5")
async def get_stage_5(session_id: str):
    """Returns the existing Sprint 7 `Tier1EscalationPackage` shape with
    the journey-traversal log appended to `what_was_tried`."""
    _require_flag()
    cohort = _load_or_cache(session_id)

    # Top-1 ticket metadata is the basis for the package (matches
    # Sprint 7 behaviour). Empty cohort → package builder degrades.
    top_md = cohort[0] if cohort else {}

    # Read alert_payload from the session row.
    alert_payload: Dict[str, Any] = {}
    try:
        from sqlalchemy import text
        from backend.db.connection import engine
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT alert_payload FROM tier1_sessions WHERE id = :id"
                ),
                {"id": session_id},
            ).mappings().first()
        if row and isinstance(row.get("alert_payload"), dict):
            alert_payload = row["alert_payload"]
    except Exception as exc:
        logger.warning("[journey.stage5] alert_payload read failed: %s", exc)

    try:
        package = build_journey_escalation_package(
            session_id=session_id,
            ticket_metadata=top_md,
            alert_payload=alert_payload,
        )
    except Exception as exc:
        logger.error("[journey.stage5] package build failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="escalation_build_failed")

    return package


# ─────────────────────────────────────────────────────────────
# Sprint 10.2 — /search-kb-handoff — creates a real chat session,
# auto-submits the journey's prefilled question, and either invokes
# /ask (when SOP/KB corpus exists) or inserts an upload-prompt
# assistant message (when corpus is empty).
# ─────────────────────────────────────────────────────────────
@router.post(
    "/{session_id}/search-kb-handoff",
    response_model=SearchKBHandoffResponse,
)
async def search_kb_handoff(
    session_id: str,
    req: Optional[SearchKBHandoffRequest] = None,
    user_id: Optional[str] = Depends(_lazy_auth_dependency),
) -> SearchKBHandoffResponse:
    """Sprint 10.6 §4 — owner_id is now derived from the authenticated
    request, not a hardcoded constant. The previous implementation
    stamped owner_id="tier1-journey" on the new chat_sessions row,
    which made the row invisible to the engineer's authenticated
    GET /chat/sessions/{id} (filters by owner_id) — the 404 the user
    reported. Aligning ownership eliminates that 404 entirely."""
    _require_flag()

    cohort = _load_or_cache(session_id)
    lookback = int(getattr(settings, "TIER1_JOURNEY_LOOKBACK_MONTHS", 18))
    stage_0 = compute_stage0(cohort, lookback_months=lookback)
    dominant = stage_0.profile_match if not stage_0.sparse else None

    # Read the engineer's intake-form alert from tier1_sessions.alert_payload.
    severity = asset = alert = notes = None
    try:
        from sqlalchemy import text as _text
        from backend.db.connection import engine as _engine
        with _engine.connect() as conn:
            row = conn.execute(
                _text("SELECT alert_payload FROM tier1_sessions WHERE id = :id"),
                {"id": session_id},
            ).mappings().first()
        if row and isinstance(row.get("alert_payload"), dict):
            ap = row["alert_payload"]
            severity = ap.get("severity")
            asset = ap.get("asset_name")
            alert = ap.get("alert_type")
            notes = ap.get("notes")
    except Exception as exc:
        logger.warning("[journey.stage4.handoff] alert_payload read failed: %s", exc)

    handoff = build_stage4(
        severity=severity,
        asset_name=asset,
        alert_type=alert,
        notes=notes,
        dominant_root_cause=dominant,
    )

    # Sprint 10.6 §4.4 — owner_id MUST match the authenticated user's
    # id so GET /chat/sessions/{id} (which filters by owner_id) finds
    # the new row. Falls back to "anonymous" when Clerk is disabled,
    # matching backend.api._normalize_owner_id's contract.
    owner_id = user_id or "anonymous"

    try:
        from backend.db.connection import engine as _engine
    except Exception:
        _engine = None

    # Sprint 11 — `prefilled_message_override` lets a per-step "Ask in
    # chat" link (Stage 0 / Stage 3) carry an arbitrary step text into
    # the new chat session instead of the journey's Stage 4 default
    # message. Empty / whitespace overrides fall back to the default so
    # an accidentally-blank payload doesn't create a useless empty chat.
    override = (req.prefilled_message_override or "").strip() if req else ""
    prefilled_message = override or handoff.prefilled_message

    try:
        result = create_chat_session_with_handoff(
            journey_session_id=session_id,
            owner_id=owner_id,
            prefilled_message=prefilled_message,
            engine=_engine,
            ask_fn=None,  # frontend fires /ask after SET_SESSION lands
        )
    except Exception as exc:
        logger.error(
            "[journey.stage4.handoff] failed sid=%s err=%s",
            session_id, exc, exc_info=True,
        )
        raise HTTPException(status_code=500, detail="search_kb_handoff_failed")

    return SearchKBHandoffResponse(**result)


# ─────────────────────────────────────────────────────────────
# /event — telemetry POST
# ─────────────────────────────────────────────────────────────
@router.post("/{session_id}/event", response_model=JourneyEventResponse)
async def post_event(
    session_id: str,
    req: JourneyEventRequest,
) -> JourneyEventResponse:
    _require_flag()
    ok = record_event(
        session_id=session_id,
        stage=req.stage,
        event_type=req.event_type,
        payload=req.payload,
    )
    return JourneyEventResponse(ok=bool(ok))


# ─────────────────────────────────────────────────────────────
# Sprint 10.7 §3.1 — /resume-state
#
# Returns the engineer's last-viewed stage, derived from the most
# recent `stage_advanced` row in tier1_journey_events. Read-only,
# idempotent, and never raises — failures degrade to "stage_0", the
# pre-10.7 default behaviour. Auth-gated like /search-kb-handoff so
# we don't leak per-session navigation history to unauthenticated
# callers.
# ─────────────────────────────────────────────────────────────
@router.get(
    "/{session_id}/resume-state",
    response_model=ResumeStateResponse,
)
async def get_resume_state(
    session_id: str,
    user_id: Optional[str] = Depends(_lazy_auth_dependency),
) -> ResumeStateResponse:
    _require_flag()

    try:
        from sqlalchemy import text as _text
        from backend.db.connection import engine as _engine
        with _engine.connect() as conn:
            row = conn.execute(
                _text(
                    """
                    SELECT stage, created_at
                    FROM tier1_journey_events
                    WHERE session_id = :sid
                      AND event_type = 'stage_advanced'
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                ),
                {"sid": session_id},
            ).mappings().first()
    except Exception as exc:
        logger.warning(
            "[journey.resume_state] read failed sid=%s: %s — defaulting to stage_0",
            session_id, exc,
        )
        return ResumeStateResponse(session_id=session_id)

    if not row:
        # No advance events yet → engineer is still on Stage 0.
        return ResumeStateResponse(session_id=session_id)

    stage = row.get("stage") or "stage_0"
    # Sprint 10.7 — guard against legacy rows that pre-date the
    # current six-stage taxonomy (stage_1a / stage_1b were merged into
    # pivot_insights in Sprint 10.2). Pydantic's Literal validation
    # would 500 on those values; map to the closest current stage.
    if stage in ("stage_1a", "stage_1b"):
        stage = "pivot_insights"

    last_at = row.get("created_at")
    last_at_str: Optional[str] = None
    if last_at is not None:
        try:
            last_at_str = last_at.isoformat()
        except Exception:
            last_at_str = str(last_at)

    return ResumeStateResponse(
        session_id=session_id,
        current_stage=stage,
        last_event_at=last_at_str,
    )
