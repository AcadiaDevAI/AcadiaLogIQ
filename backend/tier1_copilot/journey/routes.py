"""Sprint 10 — Tier-1 Resolution Journey FastAPI router.

All endpoints under prefix `/tier1/journey`.

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
    EscalationHandoffNoteRequest,
    EscalationHandoffNoteResponse,
    EscalationRouting,
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
from .stage0_environment import build_environment_profile
from .stage1_smoking_gun import build_stage1a
from .stage1_do_not_chase import build_stage1b
from .stage1_do_not_chase_synthesis import synthesize_do_not_chase
from .stage2_historical import build_stage2
from .stage3_troubleshooting import build_stage3
from .stage4_kb_handoff import build_stage4
from .stage4_search_kb_handoff import create_chat_session_with_handoff
from .stage3_consolidated import build_consolidated_ledger
from .stage5_escalation import (
    build_journey_escalation_package,
    compute_journey_time_metrics,
    fetch_traversal_log,
)
from .stage5_handoff_note import generate_handoff_note
from .stage5_routing import build_escalation_routing
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
    _consolidated_cache.clear()


def _load_or_cache(session_id: str) -> List[Dict[str, Any]]:
    """Single-flight cohort fetch. Cache miss → DB → cache.
    Sprint 13.24 PERF — when the cohort is reloaded from DB (cache
    miss / TTL expiry), invalidate the dependent consolidated-ledger
    cache so the next /stage-3 or /escalation-handoff-note rebuilds
    from the fresh cohort. Stale consolidated-vs-fresh-cohort would
    only matter on a re-ingest mid-session — rare but cheap to guard.
    """
    cached = _cache_get(session_id)
    if cached is not None:
        return cached
    cohort = load_cohort_metadata(session_id)
    _cache_put(session_id, cohort)
    _consolidated_invalidate(session_id)
    return cohort


# ─────────────────────────────────────────────────────────────
# Sprint 13.24 PERF — session-keyed cache for the LLM-synthesised
# consolidated ledger. The ledger is built from the cohort and is
# DETERMINISTIC for a given cohort, so caching by session_id is
# safe (cohort is fixed for the life of the session). Two routes
# call `build_consolidated_ledger`:
#   * /stage-3 (build_stage3 → builder)
#   * /escalation-handoff-note (when stage_3 was visited)
# Without this cache, the second route runs a duplicate ~5s LLM
# call. With it, both routes share one warm result.
# Bypass via `force=True` to support a Regenerate UI affordance.
# ─────────────────────────────────────────────────────────────
_CONSOLIDATED_TTL_SECONDS = 1800   # 30 min — cohort doesn't change within a journey
_CONSOLIDATED_CACHE_MAX = 256
_consolidated_cache: "OrderedDict[str, Tuple[float, Tuple[List[Any], bool]]]" = OrderedDict()


def _consolidated_get(session_id: str):
    entry = _consolidated_cache.get(session_id)
    if not entry:
        return None
    expires_at, payload = entry
    if time.time() >= expires_at:
        _consolidated_cache.pop(session_id, None)
        return None
    _consolidated_cache.move_to_end(session_id)
    return payload


def _consolidated_put(session_id: str, payload):
    _consolidated_cache[session_id] = (
        time.time() + _CONSOLIDATED_TTL_SECONDS, payload,
    )
    _consolidated_cache.move_to_end(session_id)
    while len(_consolidated_cache) > _CONSOLIDATED_CACHE_MAX:
        _consolidated_cache.popitem(last=False)


def _consolidated_invalidate(session_id: str) -> None:
    _consolidated_cache.pop(session_id, None)


def get_or_compute_consolidated_ledger(
    session_id: str,
    cohort: List[Dict[str, Any]],
    *,
    force: bool = False,
):
    """Cache-aware wrapper around `build_consolidated_ledger`.
    Returns the same ``(steps, used_fallback)`` tuple the underlying
    builder returns.
    """
    if not force:
        cached = _consolidated_get(session_id)
        if cached is not None:
            return cached
    payload = build_consolidated_ledger(cohort)
    _consolidated_put(session_id, payload)
    return payload


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

    cohort = _load_or_cache(session_id)
    lookback = int(getattr(settings, "TIER1_JOURNEY_LOOKBACK_MONTHS", 18))
    threshold = float(getattr(settings, "TIER1_JOURNEY_STAGE0_DOMINANT_THRESHOLD", 0.4))

    environment_profile = build_environment_profile(cohort)
    stage_0 = compute_stage0(cohort, lookback_months=lookback)
    smoking_gun = build_stage1a(cohort, min_frequency_ratio=threshold)
    do_not_chase = build_stage1b(cohort)
    # Sprint 13.3 — LLM polish pass for "What NOT to chase" entries.
    # Failure-open: any LLM error returns the verbatim Python output
    # with synthesis_skipped=True.
    # Sprint 13.24 PERF — call SKIPPED. The "What NOT to chase"
    # panel was suppressed at the user's request in Sprint 13.10
    # (PivotInsightsPanel render commented). Running the LLM polish
    # pass on data that no UI surface renders cost ~5s on every
    # /initial fetch. Reinstate by un-commenting the line below
    # together with the PivotInsightsPanel JSX in
    # ResolutionJourney.js.
    # do_not_chase = synthesize_do_not_chase(do_not_chase, cohort)

    # Sprint 10.3 §5.2 — diagnostic-grade log. From one line you can
    # tell whether each panel got populated and why.
    # Sprint 12.4 — env_empty added so the new lead-in panel's
    # coverage is visible in the same line.
    logger.info(
        "[journey.initial] sid=%s top5=%d cohort=%d "
        "env_empty=%s env_with_data=%d "
        "s1a_derived=%s s1b_entries=%d s1b_reason=%s "
        "s0_strength=%s",
        session_id[:12],
        len(cohort),
        stage_0.cohort_size,
        environment_profile.empty,
        environment_profile.tickets_with_data,
        smoking_gun.derived_from,
        len(do_not_chase.entries),
        do_not_chase.reason,
        stage_0.evidence_strength,
    )

    return JourneyInitial(
        session_id=session_id,
        environment_profile=environment_profile,
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
    cohort = _load_or_cache(session_id)
    return build_stage2(cohort)


# ─────────────────────────────────────────────────────────────
# /stage-3 — Troubleshooting Approach
# ─────────────────────────────────────────────────────────────
@router.get("/{session_id}/stage-3", response_model=Stage3TroubleshootingApproach)
async def get_stage_3(session_id: str) -> Stage3TroubleshootingApproach:
    cohort = _load_or_cache(session_id)
    # Sprint 13.24 PERF — share the consolidated-ledger LLM call with
    # /escalation-handoff-note via the session-keyed cache. First
    # call ~5s; subsequent calls (this route or the handoff note)
    # are near-instant. New Ticket clears via _consolidated_invalidate.
    consolidated_steps, consolidated_skipped = get_or_compute_consolidated_ledger(
        session_id, cohort,
    )
    return build_stage3(
        cohort,
        consolidated_steps=consolidated_steps,
        consolidated_synthesis_skipped=consolidated_skipped,
    )


# ─────────────────────────────────────────────────────────────
# /stage-4 — Search KB / SOP handoff payload
# ─────────────────────────────────────────────────────────────
@router.get("/{session_id}/stage-4", response_model=Stage4SearchKB)
async def get_stage_4(session_id: str) -> Stage4SearchKB:
    """Pre-filled chat message + allowed_doc_kinds. The frontend reads
    the alert payload from the engineer's intake form; this endpoint
    just supplies the dominant_root_cause line from Stage 0 so the
    handoff message stays accurate."""
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
# Sprint 12.7 — /escalation-routing — aggregated routing & vendor
# data for the Operational Handoff card. Read-only, derived from
# the same cohort as /stage-5; runs cheap (no LLM, no DB beyond the
# already-cached cohort load).
# ─────────────────────────────────────────────────────────────
@router.get(
    "/{session_id}/escalation-routing",
    response_model=EscalationRouting,
)
async def get_escalation_routing(session_id: str) -> EscalationRouting:
    """Aggregated Resolution_Groups + Team_Path + Vendor_OEM_Engagement
    across the cohort. Returned as a separate endpoint so the existing
    /stage-5 (Sprint 7 Tier1EscalationPackage) wire shape stays
    byte-identical for the chat-Escalate consumers."""
    cohort = _load_or_cache(session_id)
    return build_escalation_routing(cohort)


# ─────────────────────────────────────────────────────────────
# Sprint 12.7 — /escalation-handoff-note — LLM-generated Tier-2
# handoff note. Triggered by the "Generate Tier 2 Escalation Handoff"
# button on the Operational Handoff card. Failure-open: never raises;
# falls back to a deterministic template-fill on LLM error so the
# button always returns a usable note.
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# Sprint 13.17 — POST /escalation-handoff-note reactivated with a
# richer context payload. The note now reflects WHAT TIER-1 actually
# did during the journey (Stage 3's consolidated read-only steps +
# traversal log) rather than just dumping cohort raw text. Failure-
# open: any LLM error returns a deterministic template-fill so the
# panel never blanks.
# ─────────────────────────────────────────────────────────────
@router.post(
    "/{session_id}/escalation-handoff-note",
    response_model=EscalationHandoffNoteResponse,
)
async def post_escalation_handoff_note(
    session_id: str,
    req: Optional[EscalationHandoffNoteRequest] = None,
) -> EscalationHandoffNoteResponse:

    # Sprint 13.30 — Regenerate is a hard reload. Read `force` BEFORE
    # the cohort fetch so we can pop the cohort cache and let
    # _load_or_cache rebuild from DB. _load_or_cache already
    # invalidates the consolidated cache on a cohort cache miss
    # (see `_consolidated_invalidate(session_id)` inside
    # _load_or_cache), so this single pop cascades to bust both
    # caches with one line.
    #
    # Why hoist this above _load_or_cache: in steady state the cohort
    # is fixed for the life of a journey, but if a re-ingest or a
    # deferred chunk-write happens mid-journey, the engineer's
    # Regenerate click should pick that up — not be served stale
    # cached cohort data. Auto-fetches (force=False) keep the cohort
    # cache for the perf win; only the explicit Regenerate click hits
    # this code path.
    force_regen = bool(req and getattr(req, "force", False))
    if force_regen:
        _cohort_cache.pop(session_id, None)

    cohort = _load_or_cache(session_id)

    # ── Sprint 13.19 — full journey-driven dynamic gating ──
    # The handoff note must reflect EXACTLY what Tier-1 did. Three
    # signals drive the "what they did" picture:
    #   1. tier1_journey_events — which stages they rendered (Stage
    #      3 = Guided Troubleshooting Workflow, Stage 4 = Search KB).
    #   2. The same events table — kb_chat_engaged fires after the
    #      first /ask round-trip in a Stage-4-spawned chat session,
    #      so we can tell "opened the panel" from "actually used it".
    #   3. The POST body — `attempted_step_numbers` carries the
    #      Stage 3 checkbox state lifted from the frontend. Empty
    #      list = engineer didn't tick anything (or escalated
    #      without ticking). We honour that strictly.
    traversal_log = fetch_traversal_log(session_id)

    stage2_label = "Related Incidents & Probable Causes"
    stage3_label = "Guided Troubleshooting Workflow"
    stage4_label = "Knowledge Base & SOP Reference"
    stage2_visited = any(
        (entry.get("step") or "").strip() == stage2_label
        for entry in (traversal_log or [])
    )
    stage3_visited = any(
        (entry.get("step") or "").strip() == stage3_label
        for entry in (traversal_log or [])
    )
    stage4_visited = any(
        (entry.get("step") or "").strip() == stage4_label
        for entry in (traversal_log or [])
    )

    # kb_chat_engaged is logged as an event_type, not a step label.
    # fetch_traversal_log rolls per-stage; we re-read the raw events
    # table for this single signal so we don't widen the helper.
    kb_chat_engaged = False
    try:
        from sqlalchemy import text as _t
        from backend.db.connection import engine as _e
        with _e.connect() as conn:
            row = conn.execute(
                _t(
                    "SELECT 1 FROM tier1_journey_events "
                    "WHERE session_id = :sid AND event_type = 'kb_chat_engaged' "
                    "LIMIT 1"
                ),
                {"sid": session_id},
            ).first()
        kb_chat_engaged = row is not None
    except Exception as _kb_exc:
        logger.warning(
            "[journey.handoff_note] kb_chat_engaged lookup failed sid=%s "
            "err=%s — assuming False", session_id, _kb_exc,
        )

    # Sprint 13.19 — `attempted_step_numbers` from the POST body
    # filters the consolidated ledger down to what the engineer
    # ACTUALLY ticked. When stage 3 wasn't visited at all, we skip
    # build_consolidated_ledger entirely (saves an LLM call).
    # Sprint 13.30 — `force_regen` already captured above; only read
    # `attempted_step_numbers` here.
    attempted_step_numbers: List[int] = []
    if req is not None and isinstance(req.attempted_step_numbers, list):
        attempted_step_numbers = [
            int(n) for n in req.attempted_step_numbers
            if isinstance(n, (int, float)) and int(n) > 0
        ]

    # Sprint 13.31 — escalation-reason selection from the modal. Drives
    # the "Reason for Escalation" block in the note. Modal enforces
    # ≥1 selection on the client; keep server-side coercion permissive
    # so any odd whitespace / empty entry is filtered cleanly.
    escalation_reasons: List[str] = []
    if req is not None and isinstance(req.escalation_reasons, list):
        escalation_reasons = [
            r.strip() for r in req.escalation_reasons
            if isinstance(r, str) and r.strip()
        ]

    attempted_steps = []
    if stage3_visited and attempted_step_numbers:
        # Sprint 13.24 PERF — same session-keyed cache as /stage-3.
        # When the engineer just came from Stage 3, this is a hot
        # cache hit and the LLM call is skipped entirely.
        # Regenerate button passes `force=True` to bypass cache.
        all_consolidated, _ = get_or_compute_consolidated_ledger(
            session_id, cohort, force=force_regen,
        )
        attempted_set = set(attempted_step_numbers)
        attempted_steps = [
            s for s in all_consolidated if s.step_number in attempted_set
        ]
        logger.info(
            "[journey.handoff_note] sid=%s — stage3_visited=True "
            "ticked=%d total_consolidated=%d",
            session_id, len(attempted_steps), len(all_consolidated),
        )
    else:
        logger.info(
            "[journey.handoff_note] sid=%s — stage3_visited=%s "
            "ticked_in_body=%d → no diagnostic bullets",
            session_id, stage3_visited, len(attempted_step_numbers),
        )

    # Stage 5 escalation routing — Resolution_Groups + Team_Path +
    # Vendor_OEM_Engagement aggregations. Drives the "Escalation
    # Routing & Vendor/OEM Engagement" section.
    routing = build_escalation_routing(cohort)

    # Sprint 13.18 — Contact Details. Reuse the Sprint 7
    # Tier1EscalationPackage that /stage-5 already builds; pull
    # affected_customer + customer_contacts + vendor_contacts +
    # directory_contacts from it. Failure-open: any error returns
    # an empty contacts_payload so the note still renders.
    contacts_payload: Dict[str, Any] = {}
    try:
        from sqlalchemy import text as _t
        from backend.db.connection import engine as _e
        alert_payload: Dict[str, Any] = {}
        with _e.connect() as conn:
            row = conn.execute(
                _t("SELECT alert_payload FROM tier1_sessions WHERE id = :id"),
                {"id": session_id},
            ).mappings().first()
        if row and isinstance(row.get("alert_payload"), dict):
            alert_payload = row["alert_payload"]
        top_md = cohort[0] if cohort else {}
        package = build_journey_escalation_package(
            session_id=session_id,
            ticket_metadata=top_md,
            alert_payload=alert_payload,
        )
        contacts_payload = {
            "affected_customer": getattr(package, "affected_customer", None),
            "customer_contacts": getattr(package, "customer_contacts", []) or [],
            "vendor_contacts": getattr(package, "vendor_contacts", []) or [],
            "directory_contacts": getattr(package, "directory_contacts", []) or [],
        }
    except Exception as _contacts_exc:
        logger.warning(
            "[journey.handoff_note] contacts fetch failed sid=%s err=%s "
            "— shipping note without contact-details enrichment",
            session_id, _contacts_exc,
        )
        contacts_payload = {}

    # Sprint 13.26 — gather per-stage durations + chat-session time
    # totals for the opener bullets. Failure-open: the helper
    # returns whatever it could compute and never raises, so a DB
    # hiccup just renders bullets without time suffixes.
    time_metrics = compute_journey_time_metrics(session_id)

    note, used_fallback = generate_handoff_note(
        cohort,
        attempted_steps=attempted_steps,
        stage_2_visited=stage2_visited,
        stage_3_visited=stage3_visited,
        stage_4_visited=stage4_visited,
        kb_chat_engaged=kb_chat_engaged,
        routing=routing,
        contacts_payload=contacts_payload,
        time_metrics=time_metrics,
        escalation_reasons=escalation_reasons,
    )
    return EscalationHandoffNoteResponse(note=note, used_fallback=used_fallback)


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

    # Sprint 12.1 — per-bullet Ask-in-Chat carries the bullet's source
    # Incident_Number so the chat session can be scoped to that one
    # ticket. NULL when Stage 4 / Search-in-KB invokes the handoff —
    # global retrieval behavior is preserved in that case.
    scope_incident_id = (
        (req.scope_incident_id or "").strip() if req else ""
    ) or None

    try:
        result = create_chat_session_with_handoff(
            journey_session_id=session_id,
            owner_id=owner_id,
            prefilled_message=prefilled_message,
            engine=_engine,
            ask_fn=None,  # frontend fires /ask after SET_SESSION lands
            scope_incident_id=scope_incident_id,
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
