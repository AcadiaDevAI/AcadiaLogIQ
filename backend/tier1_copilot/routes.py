"""FastAPI router for Tier-1 Alert Copilot.

Three endpoints:
  POST /tier1/analyze   — main intake → answer flow
  POST /tier1/feedback  — 👍/👎 + 5 follow-up actions
  GET  /tier1/health    — probe (flag state, alias size, cache size)

The router is ONLY mounted when LOGIQ_TIER1_COPILOT_BACKEND is True
(see backend/api.py app creation block). When the flag is off the
module's URL space is not registered, so FastAPI returns 404 for
every /tier1/* path naturally — no per-endpoint guard needed.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Callable, Optional

from fastapi import APIRouter, HTTPException

from backend.config import settings
from backend.tier1_copilot.alias_dictionary import get_alias_dictionary
from backend.tier1_copilot.cache import (
    cache_size,
    get_cached_answer,
    set_cached_answer,
)
from backend.tier1_copilot.context_extractor import extract_compact_context
from backend.tier1_copilot.feedback import handle_follow_up_action, record_feedback
from backend.tier1_copilot.normalizer import normalize_alert
from backend.tier1_copilot.prompt_builder import (
    build_prompt,
    parse_answer,
    template_fallback,
)
from backend.tier1_copilot.retrieval import retrieve_top_matches
from backend.tier1_copilot.schemas import (
    Tier1ActionLogRequest,
    Tier1ActionLogResponse,
    Tier1AnalyzeRequest,
    Tier1AnalyzeResponse,
    Tier1AnswerSection,
    Tier1DeeperDiagnosticsRequest,
    Tier1DeeperDiagnosticsResponse,
    Tier1EscalationPackageRequest,
    Tier1EscalationPackageResponse,
    Tier1ExplainRequest,
    Tier1ExplainResponse,
    Tier1FeedbackRequest,
    Tier1FeedbackResponse,
    Tier1HealthResponse,
    Tier1MatchIndexRequest,
    Tier1SessionCreateRequest,
    Tier1SessionStatus,
)
from backend.tier1_copilot.diagnostics.deeper_diagnostics import (
    build_deeper_diagnostics,
)
from backend.tier1_copilot.diagnostics.escalation_package import build_package
from backend.tier1_copilot.diagnostics.explain_recommendation import build_explain
from backend.tier1_copilot.diagnostics.stuck_detector import (
    elapsed_seconds,
    should_nudge,
)
from backend.tier1_copilot.feedback import record_event
from backend.tier1_copilot.session_state.tier1_session import (
    append_what_tried,
    create_session,
    get_session,
    increment_thumbs_down,
    mark_escalated,
    mark_stuck_shown,
    touch_activity,
    update_match_index,
)

logger = logging.getLogger("acadia-log-iq")

router = APIRouter(prefix="/tier1", tags=["tier1-copilot"])


# ─────────────────────────────────────────────────────────────
# /tier1/analyze
# ─────────────────────────────────────────────────────────────
@router.post("/analyze", response_model=Tier1AnalyzeResponse)
async def analyze(req: Tier1AnalyzeRequest) -> Tier1AnalyzeResponse:
    # Defense-in-depth: the router is only mounted when the flag is on,
    # but this check lets tests exercise the flag-off path without
    # rebuilding the app.
    if not getattr(settings, "LOGIQ_TIER1_COPILOT_BACKEND", False):
        raise HTTPException(status_code=404, detail="tier1_flag_off")

    alias_dict = get_alias_dictionary()
    normalized = normalize_alert(req, alias_dict)
    signature_hash = normalized["signature_hash"]
    alert_input = req.model_dump()
    sprint7_on = bool(getattr(settings, "LOGIQ_TIER1_PROGRESSIVE_BACKEND", False))

    # ── Cache check ─────────────────────────────────────────
    cached = get_cached_answer(signature_hash)
    if cached:
        logger.info(
            "[tier1_copilot] cache hit sig=%s", signature_hash[:10],
        )
        answer_dict = cached["answer_json"] or {}
        answer = _coerce_answer_section(answer_dict)

        # Sprint 7 — even on a cache hit, give the client a fresh session
        # so arrow pagination / deeper-diag / escalation endpoints have
        # somewhere to read from. The cached payload doesn't carry top-5
        # IDs (they are retrieval-time state), so we leave that list
        # empty for cache hits; the Workspace will re-call /analyze with
        # a fresh signature if the engineer wants the full rank list.
        sess_id = None
        started_at = None
        if sprint7_on:
            sess = create_session(
                alert_signature=normalized["alert_signature"],
                alert_payload=alert_input,
                top_5_match_ids=[],
            )
            if sess is not None:
                sess_id = sess.id
                started_at = sess.created_at.isoformat()

        return Tier1AnalyzeResponse(
            matched_incident=(answer_dict.get("matched_incident")
                              if isinstance(answer_dict, dict) else None),
            confidence=cached.get("confidence", "Medium"),
            similar_count=(answer_dict.get("similar_count", 0)
                           if isinstance(answer_dict, dict) else 0),
            answer=answer,
            cache_hit=True,
            response_id=signature_hash,
            top_5_match_ids=[],
            session_id=sess_id,
            started_at=started_at,
        )

    # ── Retrieval ───────────────────────────────────────────
    engine = _lazy_engine()
    embed_fn = _lazy_embed_fn()
    matches = retrieve_top_matches(
        normalized=normalized,
        alert_input=alert_input,
        engine=engine,
        embed_fn=embed_fn,
    )
    best = matches[0] if matches else None
    similar_count = len(matches)
    confidence = best["confidence"] if best else "None"
    matched_incident = None
    compact_ctx = {}
    matched_chunk_id: Optional[str] = None

    if best:
        matched_chunk_id = best.get("chunk_id")
        compact_ctx = extract_compact_context(best.get("metadata_json") or {})
        matched_incident = compact_ctx.get("incident_number")

    # ── Sprint 7 session (created once retrieval has run) ───
    sess_id: Optional[str] = None
    started_at: Optional[str] = None
    top_5_ids: list = []
    if sprint7_on:
        top_5_ids = [
            c.get("chunk_id") for c in matches[:int(
                getattr(settings, "TIER1_TOP_N_MATCHES", 5)
            )] if c.get("chunk_id")
        ]
        sess = create_session(
            alert_signature=normalized["alert_signature"],
            alert_payload=alert_input,
            top_5_match_ids=top_5_ids,
        )
        if sess is not None:
            sess_id = sess.id
            started_at = sess.created_at.isoformat()

    # ── None-confidence fast path ───────────────────────────
    if confidence == "None" or not best:
        fallback = template_fallback(
            alert_input=alert_input,
            compact_ctx=compact_ctx,
            similar_count=similar_count,
            confidence="None",
        )
        response_id = uuid.uuid4().hex[:16]
        logger.info(
            "[tier1_copilot] no-evidence fallback sig=%s", signature_hash[:10],
        )
        return Tier1AnalyzeResponse(
            matched_incident=None,
            confidence="None",
            similar_count=0,
            answer=fallback,
            cache_hit=False,
            response_id=response_id,
            top_5_match_ids=top_5_ids,
            session_id=sess_id,
            started_at=started_at,
        )

    # ── LLM prompt + parse (1 retry + template fallback) ────
    prompt = build_prompt(
        alert_input=alert_input,
        compact_ctx=compact_ctx,
        similar_count=similar_count,
    )
    raw = _invoke_haiku(prompt)
    parsed = parse_answer(raw) if raw else None

    if parsed is None:
        stricter = (
            prompt
            + "\n\nSTRICT: reproduce ALL 8 section headers exactly as listed, "
            "each on its own line, in the given order. Do not skip any header."
        )
        raw2 = _invoke_haiku(stricter)
        parsed = parse_answer(raw2) if raw2 else None

    if parsed is None:
        logger.warning(
            "[tier1_copilot] LLM output malformed twice, template fallback sig=%s",
            signature_hash[:10],
        )
        parsed = template_fallback(
            alert_input=alert_input,
            compact_ctx=compact_ctx,
            similar_count=similar_count,
            confidence=confidence,
        )

    # ── Persist to cache + return ──────────────────────────
    payload = {
        "matched_incident": matched_incident,
        "similar_count": similar_count,
        "answer": parsed.model_dump(),
    }
    set_cached_answer(
        signature_hash=signature_hash,
        alert_signature=normalized["alert_signature"],
        answer=payload,
        confidence=confidence,
        matched_chunk_id=matched_chunk_id,
    )

    response_id = signature_hash
    logger.info(
        "[tier1_copilot] analyze done sig=%s conf=%s inc=%s",
        signature_hash[:10], confidence, matched_incident,
    )
    return Tier1AnalyzeResponse(
        matched_incident=matched_incident,
        confidence=confidence,
        similar_count=similar_count,
        answer=parsed,
        cache_hit=False,
        response_id=response_id,
        top_5_match_ids=top_5_ids,
        session_id=sess_id,
        started_at=started_at,
    )


# ─────────────────────────────────────────────────────────────
# /tier1/feedback
# ─────────────────────────────────────────────────────────────
@router.post("/feedback", response_model=Tier1FeedbackResponse)
async def feedback(req: Tier1FeedbackRequest) -> Tier1FeedbackResponse:
    if not getattr(settings, "LOGIQ_TIER1_COPILOT_BACKEND", False):
        raise HTTPException(status_code=404, detail="tier1_flag_off")

    ok = record_feedback(
        response_id=req.response_id,
        session_id=req.session_id,
        helpful=req.helpful,
        follow_up_action=req.follow_up_action,
    )
    action = handle_follow_up_action(
        response_id=req.response_id,
        action=req.follow_up_action,
    )

    # Sprint 7 — count negative feedback on the session so stuck
    # detection can trip on "2× 👎". Silent no-op when the session
    # doesn't exist (Sprint 6 flows pass a chat session_id that isn't
    # in tier1_sessions).
    if (
        getattr(settings, "LOGIQ_TIER1_PROGRESSIVE_BACKEND", False)
        and req.helpful is False
        and (req.session_id or "").startswith("sess_")
    ):
        increment_thumbs_down(req.session_id)

    return Tier1FeedbackResponse(
        ok=ok,
        response_id=req.response_id,
        action_taken=action,
    )


# ─────────────────────────────────────────────────────────────
# /tier1/health
# ─────────────────────────────────────────────────────────────
@router.get("/health", response_model=Tier1HealthResponse)
async def health() -> Tier1HealthResponse:
    flag_on = bool(getattr(settings, "LOGIQ_TIER1_COPILOT_BACKEND", False))
    ad = get_alias_dictionary()
    return Tier1HealthResponse(
        ok=flag_on,
        alias_term_count=ad.term_count(),
        cache_size=cache_size(),
        flag_on=flag_on,
    )


# ═════════════════════════════════════════════════════════════
# Sprint 7 — progressive workflow endpoints (all gated behind
# LOGIQ_TIER1_PROGRESSIVE_BACKEND; flag-off → 404)
# ═════════════════════════════════════════════════════════════
def _require_sprint7() -> None:
    if not getattr(settings, "LOGIQ_TIER1_PROGRESSIVE_BACKEND", False):
        raise HTTPException(status_code=404, detail="tier1_progressive_flag_off")


@router.post("/session", response_model=Tier1SessionStatus)
async def create_session_endpoint(
    req: Tier1SessionCreateRequest,
) -> Tier1SessionStatus:
    """Explicit session creation — /analyze also creates implicitly.
    Exposed so a client that pre-loads a saved alert can spin up a
    session without running retrieval again."""
    _require_sprint7()
    sess = create_session(
        alert_signature=req.alert_signature,
        alert_payload=req.alert_payload,
        top_5_match_ids=req.top_5_match_ids,
    )
    if sess is None:
        raise HTTPException(status_code=503, detail="session_create_failed")
    from datetime import datetime, timezone
    return Tier1SessionStatus(
        session_id=sess.id,
        elapsed_seconds=elapsed_seconds(sess, datetime.now(timezone.utc)),
        stuck_nudge=False,
        thumbs_down_count=sess.thumbs_down_count,
        current_match_index=sess.current_match_index,
        escalated=sess.escalated,
        resolved=sess.resolved,
    )


@router.get("/session/{session_id}/status", response_model=Tier1SessionStatus)
async def session_status(session_id: str) -> Tier1SessionStatus:
    _require_sprint7()
    sess = get_session(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="session_not_found")
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    nudge = should_nudge(sess, now)
    if nudge:
        mark_stuck_shown(session_id)
        record_event(
            response_id=sess.alert_signature[:40] or session_id,
            session_id=session_id,
            event_type="stuck_nudge",
            session_elapsed_seconds=elapsed_seconds(sess, now),
        )
    return Tier1SessionStatus(
        session_id=sess.id,
        elapsed_seconds=elapsed_seconds(sess, now),
        stuck_nudge=nudge,
        thumbs_down_count=sess.thumbs_down_count,
        current_match_index=sess.current_match_index,
        escalated=sess.escalated,
        resolved=sess.resolved,
    )


@router.post("/session/{session_id}/match-index", response_model=Tier1SessionStatus)
async def swap_match_index(
    session_id: str, req: Tier1MatchIndexRequest,
) -> Tier1SessionStatus:
    _require_sprint7()
    sess = get_session(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="session_not_found")
    # Clamp to top_5 length so the frontend can't push past the end.
    max_idx = max(0, len(sess.top_5_match_ids) - 1)
    clamped = max(0, min(req.match_index, max_idx))
    update_match_index(session_id, clamped)
    record_event(
        response_id=session_id,
        session_id=session_id,
        event_type="match_cycle",
        session_elapsed_seconds=None,
    )
    sess = get_session(session_id) or sess
    from datetime import datetime, timezone
    return Tier1SessionStatus(
        session_id=sess.id,
        elapsed_seconds=elapsed_seconds(sess, datetime.now(timezone.utc)),
        stuck_nudge=False,
        thumbs_down_count=sess.thumbs_down_count,
        current_match_index=sess.current_match_index,
        escalated=sess.escalated,
        resolved=sess.resolved,
    )


@router.post(
    "/session/{session_id}/action", response_model=Tier1ActionLogResponse,
)
async def log_action(
    session_id: str, req: Tier1ActionLogRequest,
) -> Tier1ActionLogResponse:
    _require_sprint7()
    entry: dict = {"step": req.step, "result": req.result}
    if req.note:
        entry["note"] = req.note
    out = append_what_tried(session_id, entry)
    if out is None:
        raise HTTPException(status_code=404, detail="session_not_found")
    record_event(
        response_id=session_id,
        session_id=session_id,
        event_type="action_log",
        follow_up_action=None,
    )
    return Tier1ActionLogResponse(ok=True, what_tried=out)


@router.post(
    "/deeper-diagnostics", response_model=Tier1DeeperDiagnosticsResponse,
)
async def deeper_diagnostics(
    req: Tier1DeeperDiagnosticsRequest,
) -> Tier1DeeperDiagnosticsResponse:
    _require_sprint7()
    sess = get_session(req.session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="session_not_found")

    ticket_md = _load_ticket_metadata_for_session(sess, req.matched_incident)
    if ticket_md is None:
        raise HTTPException(status_code=404, detail="matched_ticket_not_found")

    touch_activity(req.session_id)
    severity = (sess.alert_payload.get("severity") or "P3") if sess.alert_payload else "P3"
    return build_deeper_diagnostics(
        ticket_metadata=ticket_md,
        severity=severity,
        llm_formatter=_invoke_haiku,
    )


@router.post(
    "/escalation-package", response_model=Tier1EscalationPackageResponse,
)
async def escalation_package(
    req: Tier1EscalationPackageRequest,
) -> Tier1EscalationPackageResponse:
    _require_sprint7()
    sess = get_session(req.session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="session_not_found")

    ticket_md = _load_ticket_metadata_for_session(sess, req.matched_incident)
    if ticket_md is None:
        raise HTTPException(status_code=404, detail="matched_ticket_not_found")

    engine = _lazy_engine()
    package = build_package(
        ticket_metadata=ticket_md,
        alert_payload=sess.alert_payload or {},
        session_what_tried=sess.what_tried,
        client_what_tried=req.what_tried,
        engine=engine,
        related_incidents=_incident_numbers_for_session(sess),
    )
    mark_escalated(req.session_id)
    record_event(
        response_id=req.session_id,
        session_id=req.session_id,
        event_type="escalation_open",
    )
    return Tier1EscalationPackageResponse(package=package)


@router.post("/explain", response_model=Tier1ExplainResponse)
async def explain(req: Tier1ExplainRequest) -> Tier1ExplainResponse:
    _require_sprint7()
    sess = get_session(req.session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="session_not_found")

    top_matches = _rerun_retrieval_for_session(sess)
    if not top_matches:
        raise HTTPException(status_code=404, detail="no_candidates")

    # Select target chunk if caller specified a specific incident;
    # otherwise default to rank-1.
    target_chunk_id: Optional[str] = None
    if req.matched_incident:
        for c in top_matches:
            md = c.get("metadata_json") or {}
            meta = md.get("Metadata") or {}
            if (
                isinstance(meta, dict)
                and str(meta.get("Incident_Number") or "") == str(req.matched_incident)
            ):
                target_chunk_id = c.get("chunk_id")
                break

    return build_explain(
        alert_payload=sess.alert_payload or {},
        top_matches=top_matches,
        target_chunk_id=target_chunk_id,
    )


# ─────────────────────────────────────────────────────────────
# Sprint 7 — internal helpers
# ─────────────────────────────────────────────────────────────
def _load_ticket_metadata_for_session(
    sess: Any, matched_incident: Optional[str],
) -> Optional[dict]:
    """Resolve the ticket the user is currently looking at.

    Order of preference:
      1. Explicit matched_incident from the request body.
      2. top_5_match_ids[current_match_index] if populated.
      3. None (caller returns 404).
    """
    engine = _lazy_engine()
    try:
        from sqlalchemy import text
    except Exception:
        return None

    if matched_incident:
        try:
            with engine.connect() as conn:
                row = conn.execute(
                    text(
                        """
                        SELECT metadata_json
                        FROM chunks
                        WHERE metadata_json->'Metadata'->>'Incident_Number' = :inc
                        LIMIT 1
                        """
                    ),
                    {"inc": matched_incident},
                ).mappings().first()
            if row:
                md = row["metadata_json"]
                return md if isinstance(md, dict) else None
        except Exception as exc:
            logger.warning("[tier1_copilot:sprint7] ticket lookup by incident failed: %s", exc)

    ids = list(sess.top_5_match_ids or [])
    if not ids:
        return None
    idx = max(0, min(int(sess.current_match_index or 0), len(ids) - 1))
    cid = ids[idx]
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT metadata_json FROM chunks WHERE id = :c"),
                {"c": cid},
            ).mappings().first()
        if row:
            md = row["metadata_json"]
            return md if isinstance(md, dict) else None
    except Exception as exc:
        logger.warning("[tier1_copilot:sprint7] ticket lookup by chunk failed: %s", exc)
    return None


def _incident_numbers_for_session(sess: Any) -> list:
    """Pull the Incident_Number of every ticket in top_5_match_ids, in
    order. Used by the escalation package's relevant_tickets list."""
    ids = list(sess.top_5_match_ids or [])
    if not ids:
        return []
    try:
        from sqlalchemy import bindparam, text
        engine = _lazy_engine()
        sql = text(
            """
            SELECT id, metadata_json->'Metadata'->>'Incident_Number' AS inc
            FROM chunks
            WHERE id IN :cids
            """
        ).bindparams(bindparam("cids", expanding=True))
        with engine.connect() as conn:
            rows = conn.execute(sql, {"cids": tuple(ids)}).mappings().all()
    except Exception as exc:
        logger.warning("[tier1_copilot:sprint7] incident lookup failed: %s", exc)
        return []
    by_id = {r["id"]: r["inc"] for r in rows if r.get("inc")}
    return [by_id[c] for c in ids if c in by_id]


def _rerun_retrieval_for_session(sess: Any) -> list:
    """Cheap re-retrieval for /explain. We don't store the
    _score_components on the session (they're heavy), so re-running
    retrieve_top_matches is the accurate way to produce the same ranked
    candidates + components the original /analyze used.

    Graceful: returns [] on any error. Explain endpoint returns 404 in
    that case, which is a better UX than partial score output."""
    try:
        alias_dict = get_alias_dictionary()
        from pydantic import BaseModel
        payload = dict(sess.alert_payload or {})
        if "severity" not in payload:
            return []
        from backend.tier1_copilot.schemas import Tier1AnalyzeRequest
        # Reconstruct a request so normalize_alert can accept it.
        req = Tier1AnalyzeRequest(
            severity=payload.get("severity", "P3"),
            asset_name=payload.get("asset_name", ""),
            alert_type=payload.get("alert_type", ""),
            customer=payload.get("customer"),
            location=payload.get("location"),
            technology=payload.get("technology"),
            ip_or_device_id=payload.get("ip_or_device_id"),
            error_code=payload.get("error_code"),
            notes=payload.get("notes"),
            session_id=sess.id,
        )
    except Exception:
        return []

    normalized = normalize_alert(req, alias_dict)
    try:
        return retrieve_top_matches(
            normalized=normalized,
            alert_input=req.model_dump(),
            engine=_lazy_engine(),
            embed_fn=_lazy_embed_fn(),
        )
    except Exception as exc:
        logger.warning("[tier1_copilot:sprint7] rerun retrieval failed: %s", exc)
        return []


# ─────────────────────────────────────────────────────────────
# Helpers — lazy imports keep the module import-safe at app boot
# ─────────────────────────────────────────────────────────────
def _lazy_engine() -> Any:
    from backend.db.connection import engine
    return engine


def _lazy_embed_fn() -> Optional[Callable[[str], list]]:
    """Return a callable str→list[float] that uses the Titan embed model
    via the same boto3 client the parent app already configured. None
    if Bedrock isn't reachable — the retrieval layer handles this
    gracefully (skips the vector pass)."""
    try:
        from backend import api as _api
        bedrock = getattr(_api, "bedrock", None)
        if bedrock is None:
            return None
    except Exception:
        return None

    import json

    def _embed(text: str) -> list:
        body = json.dumps({"inputText": text or ""}).encode("utf-8")
        try:
            resp = bedrock.invoke_model(
                modelId=settings.BEDROCK_EMBED_MODEL,
                body=body,
                accept="application/json",
                contentType="application/json",
            )
            payload = json.loads(resp["body"].read().decode("utf-8"))
            return list(payload.get("embedding") or [])
        except Exception as exc:
            logger.warning("[tier1_copilot] embed call failed: %s", exc)
            return []

    return _embed


def _invoke_haiku(prompt: str) -> str:
    """Call Haiku via the shared agents.base.invoke_llm pipeline.

    Returns the model text on success, empty string on failure. No
    exceptions propagate — the route handler has a deterministic
    template fallback for the twice-failed case."""
    try:
        from backend.agents.base import TokenBudget, invoke_llm
        from backend import api as _api
    except Exception as exc:
        logger.warning("[tier1_copilot] import LLM stack failed: %s", exc)
        return ""

    budget = TokenBudget(max_total=4000)
    try:
        step = invoke_llm(
            prompt=prompt,
            model="haiku",
            max_tokens=600,
            budget=budget,
            agent_name="tier1_copilot",
            generate_fn=getattr(_api, "safe_generate", None),
            bedrock_client=getattr(_api, "bedrock", None),
        )
    except Exception as exc:
        logger.warning("[tier1_copilot] invoke_llm raised: %s", exc)
        return ""

    return (step.output or "").strip() if step.success else ""


def _coerce_answer_section(answer_dict: Any) -> Tier1AnswerSection:
    """Rebuild a Tier1AnswerSection from a cached payload dict."""
    if isinstance(answer_dict, dict) and "answer" in answer_dict \
            and isinstance(answer_dict["answer"], dict):
        payload = answer_dict["answer"]
    elif isinstance(answer_dict, dict):
        payload = answer_dict
    else:
        payload = {}
    try:
        return Tier1AnswerSection(**{
            k: v for k, v in payload.items()
            if k in Tier1AnswerSection.model_fields
        })
    except Exception:
        return Tier1AnswerSection()


# ═════════════════════════════════════════════════════════════
# Sprint 8 — rank-N match endpoint for arrow pagination
# ═════════════════════════════════════════════════════════════
@router.get(
    "/session/{session_id}/match/{match_index}",
    response_model=Tier1AnalyzeResponse,
)
async def get_session_match(
    session_id: str, match_index: int,
) -> Tier1AnalyzeResponse:
    """Return the rank-N match in Tier1AnalyzeResponse shape.

    Gated behind BOTH Sprint 6 (router mount) and Sprint 7 (the session
    must exist) AND Sprint 8 (LOGIQ_TIER1_UX_FIXES_BACKEND). Sprint 7
    only stored `top_5_match_ids` — we recompute scores on the fly via
    the existing `_rerun_retrieval_for_session` helper so no schema
    migration is needed and Sprint 7's /analyze state machine stays
    intact.
    """
    if not getattr(settings, "LOGIQ_TIER1_UX_FIXES_BACKEND", False):
        raise HTTPException(status_code=404, detail="tier1_ux_fixes_flag_off")

    sess = get_session(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="session_not_found")

    match_ids = list(sess.top_5_match_ids or [])
    total = len(match_ids)
    if match_index < 0 or match_index >= total:
        raise HTTPException(status_code=422, detail="match_index_out_of_range")

    # Re-rank on the fly. This returns the same candidate objects the
    # original /analyze call ranked, including the _score_components
    # dict that /tier1/explain reads — so confidence bands here stay
    # identical to what the engineer saw on the first call.
    top_matches = _rerun_retrieval_for_session(sess)
    # Map chunk_id → ranked candidate so we can pick the one the
    # session's top_5 list points at (the re-rank order may differ
    # slightly if /analyze and this call see different timestamps, but
    # top_5_match_ids is the authoritative navigation list).
    target_chunk = match_ids[match_index]
    candidate = next(
        (c for c in top_matches if c.get("chunk_id") == target_chunk),
        None,
    )

    ticket_md = None
    if candidate and isinstance(candidate.get("metadata_json"), dict):
        ticket_md = candidate["metadata_json"]
    else:
        ticket_md = _load_ticket_metadata_by_chunk_id(target_chunk)

    if not isinstance(ticket_md, dict):
        raise HTTPException(status_code=404, detail="matched_ticket_not_found")

    # Sprint 6 path — compact context + cached answer or fresh Haiku.
    compact_ctx = extract_compact_context(ticket_md)
    meta = ticket_md.get("Metadata") or {}
    meta = meta if isinstance(meta, dict) else {}
    matched_incident = meta.get("Incident_Number") if isinstance(meta, dict) else None
    similar_count = total

    # Confidence derives from the re-ranked candidate's final_score when
    # we have one; if re-rank didn't return the target (cold DB /
    # missing row) we fall back to "Medium" so the UI still renders
    # non-alarmingly — the same graceful-fallback pattern Sprint 7 uses.
    if candidate and "final_score" in candidate:
        from backend.tier1_copilot.retrieval import confidence_band
        confidence = confidence_band(float(candidate.get("final_score", 0.0)))
    else:
        confidence = "Medium"

    prompt = build_prompt(
        alert_input=sess.alert_payload or {},
        compact_ctx=compact_ctx,
        similar_count=similar_count,
    )
    raw = _invoke_haiku(prompt)
    parsed = parse_answer(raw) if raw else None
    if parsed is None:
        # One stricter retry — same pattern /analyze uses.
        stricter = (
            prompt
            + "\n\nSTRICT: reproduce ALL 8 section headers exactly as "
            "listed, each on its own line, in the given order."
        )
        raw2 = _invoke_haiku(stricter)
        parsed = parse_answer(raw2) if raw2 else None
    if parsed is None:
        parsed = template_fallback(
            alert_input=sess.alert_payload or {},
            compact_ctx=compact_ctx,
            similar_count=similar_count,
            confidence=confidence,
        )

    # Persist the navigation so stuck-detection sees the latest activity.
    update_match_index(session_id, match_index)
    touch_activity(session_id)

    response_id = f"{sess.id}:m{match_index}"
    logger.info(
        "[tier1_copilot:sprint8] match %d/%d sess=%s inc=%s conf=%s",
        match_index + 1, total, session_id, matched_incident, confidence,
    )
    return Tier1AnalyzeResponse(
        matched_incident=matched_incident,
        confidence=confidence,
        similar_count=similar_count,
        answer=parsed,
        cache_hit=False,
        response_id=response_id,
        top_5_match_ids=match_ids,
        session_id=session_id,
        started_at=sess.created_at.isoformat(),
        match_index=match_index,
        total_matches=total,
    )


def _load_ticket_metadata_by_chunk_id(chunk_id: str) -> Optional[dict]:
    """Direct chunk fetch fallback when the re-rank didn't surface the
    requested candidate (e.g., the ticket was removed from the corpus
    after /analyze ran). Returns None on any failure."""
    if not chunk_id:
        return None
    try:
        from sqlalchemy import text
        engine = _lazy_engine()
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT metadata_json FROM chunks WHERE id = :c"),
                {"c": chunk_id},
            ).mappings().first()
        if row:
            md = row["metadata_json"]
            return md if isinstance(md, dict) else None
    except Exception as exc:
        logger.warning(
            "[tier1_copilot:sprint8] direct chunk fetch failed c=%s: %s",
            chunk_id, exc,
        )
    return None
