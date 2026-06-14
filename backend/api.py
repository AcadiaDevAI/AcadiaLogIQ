import asyncio
import json
import logging
import hashlib
import re
import sys
import uuid
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path 
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Set

import boto3
from botocore.config import Config as BotoConfig
from fastapi import (
    BackgroundTasks, Depends, FastAPI, File, Form, Header,
    HTTPException, Query, Request, UploadFile, status,
)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, ConfigDict, Field, field_validator

# ─────────────────────────────────────────────────────────────
# AWS Secrets Manager bootstrap.
#
# Must run BEFORE ``from backend.config import settings`` — pydantic's
# Settings() snapshots os.environ at module-load time, so any secrets
# we want pydantic to see must already be present in os.environ.
#
# This call fetches the JSON secret blob from AWS Secrets Manager and
# populates os.environ with every key inside. If AWS is unreachable
# (network, missing IAM perm, or AWS_SECRETS_DISABLED=true), it
# transparently falls back to the legacy backend/.env file via
# python-dotenv so laptop development keeps working.
#
# See backend/core/secrets.py for the full design.
# ─────────────────────────────────────────────────────────────
from backend.core.secrets import bootstrap_environment

bootstrap_environment()

from backend.config import settings
from backend.clerk_auth import clerk_auth_dependency, is_clerk_enabled, get_clerk_user_display
from backend.storage.local_storage import LocalStorageProvider
from backend.vector_store import (
    count_pg_chunks,
    create_ingestion_job,
    delete_all_chat_sessions,
    delete_chat_session,
    delete_document_and_chunks,
    get_bm25_index,
    get_chat_session,
    get_ingestion_job,
    insert_document_and_chunks,
    insert_rejected_document_row,
    list_active_files,
    list_active_files_all,
    list_chat_sessions,
    pgvector_search,
    purge_orphan_chunks_db,
    rebuild_bm25_from_postgres,
    reset_pg_data,
    save_message_to_session,
    update_ingestion_job,
    update_message_feedback,
    upsert_user,
    get_user_by_clerk_id,
)
from backend.services.contextual_ingestion_service import process_document
from backend.services.journey_retrieval_context import (
    build_query_prefix as _journey_ctx_prefix,
    load_journey_context as _load_journey_ctx,
)
from backend.vector_store import find_duplicate_by_hash, find_version_candidates
from backend.retrieval.orchestrator import (
    retrieve as orchestrator_retrieve,
    _extract_identifiers,
)
from backend.retrieval.metadata_sql import (
    detect_aggregation_intent,
    detect_aggregation_intent_v2,
    run_aggregation,
)
from backend.routing.model_router import route_and_generate
from backend.vector_store import get_recent_session_messages
from backend.agents.orchestrator import (
    should_escalate_to_agents,
    run_agent_pipeline,
    resolve_mode_doc_kinds,
)
from backend.agents.mode_selector import resolve_mode, MODE_AUTO, MODE_HYBRID, MODE_MULTI_AGENT
from backend.agents.step_retriever import build_step_retriever
from backend.routing.complexity_classifier import classify_complexity
from backend.routing.cross_cutting_detector import detect_cross_cutting_analytical
from backend.routing.intent_detector import detect_intent, INTENT_GENERAL
from backend.services.trivial_responder import match_trivial_response
from backend.services.query_rewriter import rewrite_query
from backend.services.triage_classifier import classify_triage
from backend.routing.answer_cache import ANSWER_CACHE
from backend.routing.chunk_limiter import limit_chunks, DEFAULT_MAX_CHUNKS
from backend.routing.input_guard import check_input, GUARD_REASON_OK
from backend.routing.stage_enforcer import (
    enforce_stage,
    normalize_stage,
    STAGE_GENERAL,
    STAGE_TICKETS,
    STAGE_DOCS,
)
from backend.services.session_mode_state import (
    get_session_mode,
    set_session_mode,
    reset_session_mode,
    patch_session_mode,
    SessionMode,
    MODE_TROUBLESHOOTING,
    MODE_TICKET_HANDLING,
    MODE_ESCALATION,
    MODE_VENDOR_OEM,
)
from backend.routing.evidence_checker import check_evidence
from backend.validation.validator import validate_answer
from backend.retrieval.query_expansion import (
    expand_query,
    extract_glossary_from_chunks,
    get_glossary_store,
    has_sufficient_document_support_v2,
    rebuild_glossary_from_postgres,
)
import time as _time


if sys.version_info < (3, 11):
    raise RuntimeError("Python 3.11+ required")


# ─────────────────────────────────────────────────────────────
# Observability — structured logging + request_id correlation.
#
# Replaces the legacy ``logging.basicConfig(...)`` call. Behaviour
# stays backward-compatible for every existing ``logger.info(...)``
# call in this codebase — they keep working unchanged but now emit
# JSON lines (in prod) with ``request_id`` / ``user_id`` correlation
# fields injected automatically by the ContextFilter inside
# ``observability/log_setup.py``.
#
# Formatter selection (env var ``LOG_FORMAT``):
#   * ``json`` (default)  — newline-delimited JSON for CloudWatch.
#   * ``text``            — human-readable for laptop dev.
#
# See ``backend/observability/__init__.py`` for the public surface.
# ─────────────────────────────────────────────────────────────
from backend.observability import (
    configure_logging,
    RequestContextMiddleware,
    init_sentry,
)

configure_logging(level=settings.LOG_LEVEL, fmt=settings.LOG_FORMAT)
logger = logging.getLogger("acadia-log-iq")

# Sentry MUST be initialised before any logger.warning/error/exception
# fires that we want captured. We init here at module load (right
# after logging is up) so the auth-config CRITICAL line in the
# lifespan startup still flows into Sentry on a misconfigured deploy.
# DSN-empty → silent no-op; the rest of boot is unaffected.
init_sentry(
    dsn=settings.SENTRY_DSN,
    environment=settings.SENTRY_ENV,
    release=settings.SENTRY_RELEASE,
    traces_sample_rate=settings.SENTRY_TRACES_SAMPLE_RATE,
)


CHARS_PER_TOKEN = 3

TOKEN_BUDGET = {
    "MODEL_MAX_TOKENS": 32_768,
    "MAX_LOG_CONTEXT_CHARS": 30_000,
    "MAX_KB_CONTEXT_CHARS": 30_000,
    "MAX_GENERATION_TOKENS": 2_048,
    "MAX_SINGLE_CHUNK_CHARS": 6_000,
    "PROMPT_OVERHEAD_CHARS": 1_500,
}

_MAX_PROMPT_TOKENS = TOKEN_BUDGET["MODEL_MAX_TOKENS"] - TOKEN_BUDGET["MAX_GENERATION_TOKENS"]
MAX_TOTAL_PROMPT_CHARS = (_MAX_PROMPT_TOKENS * CHARS_PER_TOKEN) - TOKEN_BUDGET["PROMPT_OVERHEAD_CHARS"]

CHUNK_CONFIG = {
    "LINES_PER_CHUNK": 80,
    "MAX_CHUNK_CHARS": 6_000,
    "OVERLAP_LINES": 10,
}

HYBRID_CONFIG = {
    "VECTOR_WEIGHT": 0.6,
    "BM25_WEIGHT": 0.4,
    "VECTOR_CANDIDATES": 20,
    "BM25_CANDIDATES": 20,
    "RERANK_TOP_K_LOG": 6,
    "RERANK_TOP_K_KB": 5,
}

# ---------------------------------------------------------------------------
# Follow-up context inheritance (Fix 5)
# ---------------------------------------------------------------------------
# A bare follow-up like "what QA gaps?" contains no ticket ID, so the embedding
# and BM25 terms drift to an unrelated chunk. We carry forward explicit
# entities from the recent assistant turns and prepend them to the
# retrieval-only query. The LLM prompt and chat persistence still see the
# original user text — the enrichment exists solely to steer retrieval.
_FOLLOWUP_TICKET_RE = re.compile(r"\bINC-\d+\b", re.I)
_FOLLOWUP_ENTERPRISE_RE = re.compile(r"\bEnterprise-\d+\b", re.I)
_FOLLOWUP_NEBULA_RE = re.compile(r"\bNebula-Corp\b", re.I)
_FOLLOWUP_PATTERNS = (
    _FOLLOWUP_TICKET_RE,
    _FOLLOWUP_ENTERPRISE_RE,
    _FOLLOWUP_NEBULA_RE,
)

# Goal 3 — cross-cutting / aggregation tokens. When the query contains any
# of these, the user wants a multi-record view and enrichment would bias
# retrieval toward whatever ticket was last discussed.
_CROSS_CUTTING_RE = re.compile(
    r"\b(?:"
    r"across|all|every|each|common|patterns|trends|compare|"
    r"how\s+many|list\s+all|show\s+all|rank|ranking"
    r")\b",
    re.IGNORECASE,
)
# "which X had ..." — narrow but still cross-cutting over a cohort.
_WHICH_X_HAD_RE = re.compile(r"\bwhich\s+\w+\s+(?:had|have|has)\b", re.IGNORECASE)


def _classify_followup_intent(query: str) -> str:
    """Classify the current query into one of three follow-up intents.

    Returns:
        "self_contained" — already names an identifier (any schema) or a
            tracked named entity (Enterprise-N / Nebula-Corp). Don't enrich.
        "cross_cutting"  — multi-record question (across / common / rank /
            compare / how many / ...). Enriching with one entity biases
            retrieval toward a single record; skip.
        "narrow_followup" — everything else. Short bare follow-ups go here
            and get one entity carried from the most recent assistant turn.
    """
    q = (query or "").strip()
    if not q:
        return "narrow_followup"

    # Self-contained: explicit entity in the current turn.
    if _FOLLOWUP_TICKET_RE.search(q):
        return "self_contained"
    if _FOLLOWUP_ENTERPRISE_RE.search(q):
        return "self_contained"
    if _FOLLOWUP_NEBULA_RE.search(q):
        return "self_contained"
    # Config-driven identifiers (PROJ-001, KB-123, ...).
    try:
        if _extract_identifiers(q):
            return "self_contained"
    except Exception:
        pass

    # Cross-cutting: multi-record question. Enrichment would pollute.
    if _CROSS_CUTTING_RE.search(q) or _WHICH_X_HAD_RE.search(q):
        return "cross_cutting"

    return "narrow_followup"


def _enrich_query_with_history(
    query: str,
    recent_messages: Optional[List[Dict[str, Any]]],
) -> Tuple[str, List[str]]:
    """
    DEPRECATED (Brief 3, Phase 3): the /ask pipeline now runs an LLM
    query rewriter upstream that resolves pronouns, ordinals, ellipsis,
    and filter swaps — a strict superset of this function's behavior.

    Kept as a no-op for any legacy callers so nothing breaks. Remove in
    a future cleanup pass once we're sure nothing outside /ask uses it.
    """
    return query, []


# Brief 4 / Opt 2 helper. Conservative: when any listed aggregation signal
# appears on one side but not the other, we must re-run the classifier on
# the rewritten query since the Tier-2 verdict (customer/priority/sla/op)
# may have flipped. A mismatch on any signal returns True.
_AGGREGATION_FLIP_SIGNALS = (
    "all", "how many", "list", "count", "top", "bottom", "by customer",
    "total", "missed sla", "p1", "p2", "p3", "p4",
)


def _might_flip_aggregation(original: str, rewritten: str, intent) -> bool:
    a = (original or "").lower()
    b = (rewritten or "").lower()
    for s in _AGGREGATION_FLIP_SIGNALS:
        if (s in a) != (s in b):
            return True
    return False


storage = LocalStorageProvider()

# Phase 1 — S3 direct-upload pipeline (presigned PUT). Initialised once at
# module load. Returns None when STORAGE_TYPE != "s3" or no bucket is
# configured — the /upload/presign + /upload/finalize routes return 503
# in that case. The legacy /upload (multipart) route is unaffected.
from backend.storage import get_s3_upload_provider

s3_upload_provider = get_s3_upload_provider()
if s3_upload_provider is not None:
    logger.info(
        "[upload] S3 direct-upload pipeline enabled bucket=%s prefix=%s",
        s3_upload_provider.bucket_name, s3_upload_provider.prefix,
    )
else:
    logger.info("[upload] S3 direct-upload pipeline DISABLED (STORAGE_TYPE != 's3' or bucket not set)")


class JobInfo(TypedDict, total=False):
    job_id: str
    status: str
    processed_chunks: int
    total_chunks: int
    successful_chunks: int
    file: Optional[str]
    file_type: Optional[str]
    file_size_mb: float
    file_hash: str
    error: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]
    owner_id: Optional[str]
    file_id: Optional[str]


bm25 = None


def _make_bedrock_client():
    # Timeouts are sourced from settings so operators can tune them in
    # .env without touching code. Defaults were lowered in the LLM-
    # timeout iteration: read_timeout 120s → 45s, max_attempts 10 → 3.
    # Previously a single hung Bedrock call could leave a /ask request
    # waiting up to 20 minutes (10 × 120s); now the per-attempt budget
    # is bounded and the safe_generate path falls back to Haiku on
    # ReadTimeoutError / ThrottlingException via llm_timeout_guard.
    boto_cfg = BotoConfig(
        retries={"max_attempts": settings.LLM_MAX_ATTEMPTS, "mode": "adaptive"},
        read_timeout=settings.LLM_READ_TIMEOUT_S,
        connect_timeout=settings.LLM_CONNECT_TIMEOUT_S,
        tcp_keepalive=True,
    )
    kwargs = {
        "service_name": "bedrock-runtime",
        "region_name": settings.AWS_REGION,
        "config": boto_cfg,
    }
    if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
        kwargs["aws_access_key_id"] = settings.AWS_ACCESS_KEY_ID
        kwargs["aws_secret_access_key"] = settings.AWS_SECRET_ACCESS_KEY
        if settings.AWS_SESSION_TOKEN:
            kwargs["aws_session_token"] = settings.AWS_SESSION_TOKEN
    return boto3.client(**kwargs)


bedrock = _make_bedrock_client()


def _make_ses_client():
    ses_region = settings.SES_REGION or settings.AWS_REGION
    kwargs = {
        "service_name": "ses",
        "region_name": ses_region,
    }
    if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
        kwargs["aws_access_key_id"] = settings.AWS_ACCESS_KEY_ID
        kwargs["aws_secret_access_key"] = settings.AWS_SECRET_ACCESS_KEY
        if settings.AWS_SESSION_TOKEN:
            kwargs["aws_session_token"] = settings.AWS_SESSION_TOKEN
    return boto3.client(**kwargs)


ses_client = None
try:
    if settings.SES_ENABLED.lower() == "true":
        ses_client = _make_ses_client()
        logger.info("SES ready")
except Exception as e:
    logger.warning("SES init failed: %s", e)


def send_feedback_email(subject: str, body_text: str, body_html: str) -> bool:
    if not ses_client:
        return False
    try:
        ses_client.send_email(
            Source=settings.SES_SENDER_EMAIL,
            Destination={"ToAddresses": [settings.SES_FEEDBACK_RECIPIENT]},
            Message={
                "Subject": {"Data": subject, "Charset": "UTF-8"},
                "Body": {
                    "Text": {"Data": body_text, "Charset": "UTF-8"},
                    "Html": {"Data": body_html, "Charset": "UTF-8"},
                },
            },
        )
        return True
    except Exception as e:
        logger.exception("SES send_email failed: %s", e)
        return False


# def _resolve_user_display(user_id: Optional[str]) -> dict:
#     return get_clerk_user_display(user_id)

def _resolve_user_display(user_id: Optional[str]) -> dict:
    """
    Resolve user display name and email.
    First checks the local users table (populated by useAutoRegister),
    then falls back to Clerk API if not found locally.
    """
    if not user_id:
        return {"email": "anonymous", "name": "Anonymous", "user_id": "anonymous"}

    # Try local DB first (fast, no API call)
    try:
        local_user = get_user_by_clerk_id(user_id)
        if local_user and local_user.get("email"):
            return {
                "email": local_user["email"],
                "name": local_user.get("full_name") or local_user["email"].split("@")[0],
                "user_id": user_id,
            }
    except Exception:
        pass

    # Fall back to Clerk API
    return get_clerk_user_display(user_id)


def _normalize_owner_id(user_id: Optional[str]) -> str:
    return user_id or "anonymous"


def _get_active_indexed_file_ids(user_id: Optional[str]) -> Set[str]:
    """Get ALL indexed files across all users (files are shared)."""
    return {f["id"] for f in list_active_files_all() if f.get("status") == "indexed"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bm25
    logger.info("Starting API...")

    # ── Auth configuration: fail loud on misconfiguration ──
    # Strict mode — Clerk MUST be configured. If CLERK_ENABLED is on
    # but a key is missing, OR CLERK_ENABLED is "true" string-but-not-
    # really, we surface a CRITICAL log here so the misconfig is
    # impossible to miss in the boot log. Protected requests will still
    # 503 (see auth_dependency); this just makes the operator-facing
    # warning arrive at process-start instead of first-request-time.
    clerk_enabled_setting = getattr(settings, 'CLERK_ENABLED', 'NOT SET')
    secret_set = bool(getattr(settings, 'CLERK_SECRET_KEY', ''))
    publishable_set = bool(getattr(settings, 'CLERK_PUBLISHABLE_KEY', ''))
    clerk_on = is_clerk_enabled()
    logger.info("=== AUTH CONFIG ===")
    logger.info("  CLERK_ENABLED setting: %s", clerk_enabled_setting)
    logger.info("  CLERK_SECRET_KEY set: %s", secret_set)
    logger.info("  CLERK_PUBLISHABLE_KEY set: %s", publishable_set)
    logger.info("  is_clerk_enabled() = %s", clerk_on)
    if not clerk_on:
        # Mark the situation in the loudest terms possible. Every
        # protected route below will 503 until the operator fixes it.
        logger.critical(
            "  *** CLERK AUTH IS NOT CONFIGURED — every protected "
            "endpoint will return 503. Set CLERK_ENABLED=true, "
            "CLERK_SECRET_KEY, and CLERK_PUBLISHABLE_KEY in the secret "
            "store and restart. ***"
        )
    elif str(clerk_enabled_setting).lower() == "true" and not (secret_set and publishable_set):
        # is_clerk_enabled() already covers this case (returns False)
        # but keep an explicit branch so operators get a single,
        # specific diagnostic instead of just "not enabled".
        logger.critical(
            "  *** CLERK_ENABLED=true but a Clerk key is missing "
            "(secret=%s publishable=%s) — fix the secret store. ***",
            secret_set, publishable_set,
        )
    logger.info("===================")

    bm25 = get_bm25_index()
    doc_count = rebuild_bm25_from_postgres()
    logger.info("BM25 ready from PostgreSQL: %d docs", doc_count)

    # Rebuild glossary from document content (learns abbreviations automatically)
    glossary_count = rebuild_glossary_from_postgres()
    logger.info("Glossary store ready: %d acronyms learned from documents", glossary_count)

    # ── Periodic glossary refresher ────────────────────────────────
    # The glossary store is an in-process cache of a Postgres truth
    # (document_metadata + chunks). Without a refresher each worker
    # would be frozen at boot time — a new document ingested by
    # worker A would not produce expanded acronyms on worker B until
    # the next deploy. We refresh every GLOSSARY_REFRESH_SECONDS so
    # drift across replicas is bounded.
    #
    # Set GLOSSARY_REFRESH_SECONDS=0 to disable (used by tests and
    # one-shot tooling that doesn't want a background loop).
    refresher_task: Optional[asyncio.Task] = None
    refresh_interval = int(getattr(settings, "GLOSSARY_REFRESH_SECONDS", 0) or 0)
    if refresh_interval > 0:
        async def _glossary_refresher_loop():
            while True:
                try:
                    await asyncio.sleep(refresh_interval)
                    # ``rebuild_glossary_from_postgres`` is sync + does a
                    # DB scan; offload to a thread so the event loop
                    # stays responsive. Failure-open — a single bad
                    # refresh logs a WARNING and keeps the previous
                    # in-memory state.
                    try:
                        n = await asyncio.to_thread(rebuild_glossary_from_postgres)
                        logger.info(
                            "[glossary] refresh OK: %d acronyms", n,
                        )
                    except Exception as refresh_exc:
                        logger.warning(
                            "[glossary] refresh failed (%s) — "
                            "keeping previous in-memory state",
                            refresh_exc,
                        )
                except asyncio.CancelledError:
                    # Graceful shutdown — propagate so the task exits
                    # cleanly instead of hanging the lifespan teardown.
                    raise
        refresher_task = asyncio.create_task(
            _glossary_refresher_loop(), name="glossary-refresher",
        )
        logger.info(
            "[glossary] refresher scheduled every %ds", refresh_interval,
        )
    else:
        logger.info("[glossary] refresher disabled (GLOSSARY_REFRESH_SECONDS=0)")

    yield

    # ── Shutdown — cancel the refresher first so it doesn't try to
    # run a DB query after the engine has been disposed by FastAPI.
    if refresher_task is not None:
        refresher_task.cancel()
        try:
            await refresher_task
        except (asyncio.CancelledError, Exception):
            pass
    logger.info("Shutting down...")


app = FastAPI(
    title="Acadia's Log IQ API",
    description="AI log analysis — Hybrid Search + Re-ranking",
    version="2.3.0",
    lifespan=lifespan,
)

# Sprint 6 — Tier-1 Alert Copilot router mount.
try:
    from backend.tier1_copilot.routes import router as _tier1_router
    app.include_router(_tier1_router)
    logger.info("[tier1_copilot] router mounted at /tier1")
except Exception as _tier1_exc:
    logger.warning(
        "[tier1_copilot] failed to mount router (module disabled): %s",
        _tier1_exc,
    )

# Sprint 10 — Tier-1 Resolution Journey router mount.
try:
    from backend.tier1_copilot.journey.routes import router as _journey_router
    app.include_router(_journey_router)
    logger.info("[tier1_journey] router mounted at /tier1/journey")
except Exception as _journey_exc:
    logger.warning(
        "[tier1_journey] failed to mount router (module disabled): %s",
        _journey_exc,
    )

# Sprint 9 — Universal Intake router mount.
try:
    from backend.tier1_copilot.intake.routes import router as _intake_router
    app.include_router(_intake_router)
    logger.info("[intake] router mounted at /intake")
except Exception as _intake_exc:
    logger.warning(
        "[intake] failed to mount router (module disabled): %s",
        _intake_exc,
    )

# Sprint 13.32 — RCA-from-incident-number flow. Standalone surface
# triggered by the sidebar's "RCA" button. No flag gate by design —
# the route is self-contained and adds no risk to other flows. Mount
# is wrapped in try/except so an import error in the new module can
# never block app startup.
try:
    from backend.tier1_copilot.rca.routes import router as _rca_router
    app.include_router(_rca_router)
    logger.info("[rca] router mounted at /rca")
except Exception as _rca_exc:
    logger.warning(
        "[rca] failed to mount router (module disabled): %s",
        _rca_exc,
    )

# Sprint — Gap Analysis flow (LogIQ Gap Analysis + Blameless Post-Mortem).
# Independent fork of the RCA module; same parallel-LLM pattern but
# different prompts, response shape, and (optionally) different
# Bedrock model. Strict-auth gated. Mount in a try/except so a
# downstream regression in the module never blocks startup.
try:
    from backend.tier1_copilot.gap_analysis.routes import router as _gap_analysis_router
    app.include_router(_gap_analysis_router)
    logger.info("[gap_analysis] router mounted at /gap-analysis")
except Exception as _gap_exc:
    logger.warning(
        "[gap_analysis] failed to mount router (module disabled): %s",
        _gap_exc,
    )

# Phase 1 — async job polling endpoint (``GET /jobs/{id}``).
# The RCA + Gap Analysis routes now return 202 + job_id on
# cache-miss; the frontend hits /jobs/{id} every 2 s to track
# status, then re-POSTs the original route (which returns 200 from
# cache once the worker is done).
try:
    from backend.jobs.routes import router as _jobs_router
    app.include_router(_jobs_router)
    logger.info("[jobs] router mounted at /jobs")
except Exception as _jobs_exc:
    logger.warning(
        "[jobs] failed to mount router (module disabled): %s",
        _jobs_exc,
    )

# Escalation Procedures KB — independent feature, lives at /escalation.
# One consolidated PDF, four section-scoped chatbots. Zero overlap with
# the main chat / RCA / Gap pipeline.
try:
    from backend.escalation.routes import router as _escalation_router
    app.include_router(_escalation_router)
    logger.info("[escalation] router mounted at /escalation")
except Exception as _escalation_exc:
    logger.warning(
        "[escalation] failed to mount router (module disabled): %s",
        _escalation_exc,
    )

# Ticket Filter — independent feature, lives at /ticket-filter.
# Two-field dropdown filter (SLA_Target_Met + Resolution_Quality_Score)
# over historical tickets. Zero dependency on RCA / Gap / Chat code;
# its own router, its own SQL helpers, its own response schema.
# Mount wrapped in try/except so a regression in this module can
# never block app startup.
try:
    from backend.tier1_copilot.ticket_filter.routes import router as _ticket_filter_router
    app.include_router(_ticket_filter_router)
    logger.info("[ticket_filter] router mounted at /ticket-filter")
except Exception as _ticket_filter_exc:
    logger.warning(
        "[ticket_filter] failed to mount router (module disabled): %s",
        _ticket_filter_exc,
    )

# Phase 4 — shared per-user rate limiter (see
# backend/observability/rate_limit.py). The Limiter is now defined
# in a separate module so sub-routers (RCA, Gap Analysis, jobs) can
# import it without creating a circular dependency on api.py.
from slowapi.middleware import SlowAPIMiddleware
from backend.observability.rate_limit import limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
# SlowAPIMiddleware is required for the @limiter.limit decorator to
# fire under ASGI / Starlette (which is what FastAPI uses). Without
# this middleware the decorator is a silent no-op — exactly the
# misconfiguration the Phase 4 smoke caught.
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    # allow_origins=[
    #      "http://18.233.93.19:8501",
    #     "http://localhost:8501",
    #     "http://127.0.0.1:8501",
    #     "http://localhost:8001",
    #     "http://127.0.0.1:8001",
    #     "http://localhost:3000",
    #     "http://127.0.0.1:3000",
    #     "http://localhost:8000",
    #     "http://127.0.0.1:8000",
    # ]
    # 
allow_origins=[
    "http://localhost:8501",
    "http://127.0.0.1:8501",

    "http://localhost:3000",
    "http://127.0.0.1:3000",

    "http://localhost:8000",
    "http://127.0.0.1:8000",

    "http://localhost:8001",
    "http://127.0.0.1:8001",

   
    "http://100.48.5.177",
    "http://100.48.5.177:8501",
    "http://100.48.5.177:3000",
],
    allow_credentials=True,
    # Narrowed from "*" so the browser preflight can't be tricked into
    # whitelisting arbitrary methods/headers. Add new entries here as
    # endpoints grow.
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=[
        "Authorization",   # Clerk Bearer JWT
        "Content-Type",    # application/json + multipart
        "Accept",
        # Note: X-API-Key was intentionally removed. The strict
        # Clerk-only ``auth_dependency`` ignores it; allowing it here
        # would mislead callers into thinking it still works.
    ],
    expose_headers=["X-Processing-Time", "X-Request-ID"],
    max_age=600,
)


# ─────────────────────────────────────────────────────────────
# Request-id correlation middleware.
#
# Mints (or accepts) ``X-Request-ID`` for every inbound request,
# binds it onto a ContextVar so every log line — application or
# uvicorn.access — carries the same correlation field, and echoes
# the ID back via the response header so the caller can quote it
# in a support ticket.
#
# Registered AFTER ``CORSMiddleware`` because FastAPI runs the
# *most-recently-added* middleware first on inbound; this puts the
# ContextVar binding closest to the request edge, before any user
# handler runs.
# ─────────────────────────────────────────────────────────────
app.add_middleware(RequestContextMiddleware)


# ─────────────────────────────────────────────────────────────
# Strict Clerk-only auth.
#
# Earlier revisions of this file accepted X-API-Key as a fallback when
# Clerk was disabled — that opened a silent "anonymous mode" hole if
# CLERK_ENABLED, CLERK_SECRET_KEY, or CLERK_PUBLISHABLE_KEY were ever
# misconfigured. The fallback is gone. Every protected route MUST
# present a valid Clerk Bearer JWT.
#
# Misconfigured deployment? auth_dependency below returns 503 instead
# of silently allowing access. Operators get a loud failure mode.
#
# The legacy ``verify_api_key`` function and the ``X-API-Key`` header
# parameter are retained on this signature only to keep the symbol
# importable for older sub-modules (none currently use it). They are
# functional no-ops — the header is ignored.
# ─────────────────────────────────────────────────────────────


async def auth_dependency(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> str:
    """Resolve the authenticated user's Clerk ID, or raise.

    Always returns a non-empty ``user_id`` on success.

    Raises
    ------
    HTTPException(401) — JWT missing/invalid/expired (via clerk_auth)
    HTTPException(503) — Clerk not configured (operator misconfiguration)
    """
    if not is_clerk_enabled():
        # Loud failure rather than silent open access. If you see this
        # in prod, check CLERK_ENABLED / CLERK_SECRET_KEY /
        # CLERK_PUBLISHABLE_KEY are all present in AWS Secrets Manager.
        logger.error(
            "auth_dependency: Clerk auth not configured (CLERK_ENABLED=%r, "
            "CLERK_SECRET_KEY set=%s, CLERK_PUBLISHABLE_KEY set=%s) — "
            "rejecting request to %s",
            getattr(settings, "CLERK_ENABLED", None),
            bool(getattr(settings, "CLERK_SECRET_KEY", None)),
            bool(getattr(settings, "CLERK_PUBLISHABLE_KEY", None)),
            request.url.path,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured on this server.",
        )

    user_id = await clerk_auth_dependency(request)
    if not user_id:
        # Defense in depth — clerk_auth_dependency should always raise
        # 401 on bad tokens, but guard against a refactor regression.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please sign in.",
        )
    # Bind the verified user_id onto the request-scoped ContextVar so
    # every downstream log line (Bedrock call, DB write, error trace)
    # carries the same correlation field as the access log. We bind
    # AFTER Clerk validates the JWT so an unauthenticated caller can
    # never spoof user_id in the logs by sending a header. ContextVars
    # auto-reset when the asyncio Context unwinds — no manual cleanup
    # needed inside a request handler.
    from backend.observability import bind_user_id  # local import: avoid circular
    bind_user_id(user_id)
    logger.debug("Auth: clerk user_id=%s", user_id)
    return user_id


class ClarificationSelection(BaseModel):
    """Payload sent when user clicks an Interactive Clarifier option (Brief 6)."""
    clarification_id: str
    selected_option_id: str
    free_text: Optional[str] = None
    model_config = ConfigDict(extra="ignore")


class Question(BaseModel):
    q: str = Field(min_length=1, max_length=1000)
    session_id: Optional[str] = None
    mode: Optional[str] = Field(
        default=MODE_AUTO,
        description="Routing mode: 'auto' (default, 5-gate escalation), 'hybrid' (force standard RAG), or 'multi_agent' (force agent pipeline)",
    )
    stage: Optional[str] = Field(
        default=STAGE_GENERAL,
        description="Conversation stage: 'general' (default), 'tickets' (ticket-only retrieval), or 'docs' (docs flow w/ unresolved escalation)",
    )
    # Brief 6 — present only when the user clicks a clarification option.
    clarification_response: Optional[ClarificationSelection] = None
    # Sprint 10 — optional cohort-scoping filter for the Tier-1
    # Resolution Journey's Stage 4 KB handoff. When non-empty, the
    # retrieval call is restricted to chunks whose metadata_json
    # ->>'doc_kind' matches one of the listed values (e.g. ["sop","kb"]).
    # When None or empty, retrieval falls through to Sprint 3C's
    # mode-derived doc_kinds filter exactly as before.
    allowed_doc_kinds: Optional[List[str]] = None
    model_config = ConfigDict(extra="ignore")

    @field_validator("q")
    @classmethod
    def validate_q(cls, v):
        v = (v or "").strip()
        if not v:
            raise ValueError("Empty")
        return v


class UploadResponse(BaseModel):
    job_id: str
    file_id: str
    message: str
    file_hash: Optional[str] = None
    model_config = ConfigDict(extra="ignore")


class JobStatus(BaseModel):
    job_id: str
    status: str
    processed_chunks: int = 0
    total_chunks: Optional[int] = None
    successful_chunks: Optional[int] = None
    file: Optional[str] = None
    file_type: Optional[str] = None
    file_size_mb: Optional[float] = None
    file_hash: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    model_config = ConfigDict(extra="ignore")


class ClarificationOptionDTO(BaseModel):
    """Frontend-facing Interactive Clarifier option (Brief 6)."""
    id: str
    label: str
    model_config = ConfigDict(extra="ignore")


class AnswerResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float = Field(ge=0, le=1)
    processing_time_ms: Optional[int] = None
    context_stats: Optional[Dict] = None
    session_id: Optional[str] = None
    # Brief 6 — populated when clarification is needed instead of an answer
    needs_clarification: Optional[bool] = None
    clarification_id: Optional[str] = None
    clarification_options: Optional[List[ClarificationOptionDTO]] = None
    clarification_context: Optional[str] = None
    # Sprint 3B — "low" when the top ticket-history chunk score is below
    # LOW_SIMILARITY_THRESHOLD, "normal" otherwise. Omitted (None) when
    # Sprint 3B flag is off OR retrieval wasn't applicable (e.g. cached
    # and trivial-short-circuit paths). Frontend tolerates None → no
    # banner. Pre-3B shape preserved byte-identical in the off path.
    confidence_band: Optional[str] = None
    model_config = ConfigDict(extra="ignore")


class ChatMessage(BaseModel):
    role: str
    content: str
    sources: Optional[Dict] = None
    timestamp: str
    feedback: Optional[str] = None


class ChatSession(BaseModel):
    id: str
    title: str
    messages: List[ChatMessage]
    created_at: str
    updated_at: str


class FileInfo(BaseModel):
    id: str
    name: str
    file_type: str
    size_mb: float
    status: str
    job_id: Optional[str] = None
    uploaded_at: str
    owner_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Interactive Clarifier (Brief 6) — in-process clarification store
# ---------------------------------------------------------------------------
# Maps clarification_id → {session_id, options, created_at}. 10-minute TTL.
# Good enough for single-instance MVP; move to DB for horizontal scaling.
_CLARIFICATION_STORE: Dict[str, Dict[str, Any]] = {}
_CLARIFICATION_STORE_TTL_SECONDS: int = 600


def _prune_clarification_store() -> None:
    """Drop entries older than TTL. Called opportunistically."""
    now = _time.time()
    expired = [
        cid for cid, rec in _CLARIFICATION_STORE.items()
        if now - rec.get("created_at", 0) > _CLARIFICATION_STORE_TTL_SECONDS
    ]
    for cid in expired:
        _CLARIFICATION_STORE.pop(cid, None)


def _store_clarification(
    *,
    session_id: str,
    options: List[Dict[str, Any]],
    original_query: str,
) -> str:
    """Persist a clarification record and return its id."""
    _prune_clarification_store()
    clarification_id = uuid.uuid4().hex
    _CLARIFICATION_STORE[clarification_id] = {
        "session_id": session_id,
        "options": [
            {
                "id": o.get("id"),
                "label": o.get("label"),
                "refined_query": o.get("refined_query") or "",
                "record_ref": o.get("record_ref"),
            }
            for o in options
        ],
        "original_query": original_query,
        "created_at": _time.time(),
    }
    return clarification_id


def _expand_clarification_selection(
    *,
    session_id: str,
    selection: ClarificationSelection,
    fallback_query: str,
) -> str:
    """
    Look up the stored clarification record and return the refined_query
    for the selected option. Falls back to fallback_query on any problem
    (expired, missing, wrong session).
    """
    _prune_clarification_store()
    rec = _CLARIFICATION_STORE.get(selection.clarification_id)
    if not rec:
        logger.info(
            "[interactive_clarifier] selection: clarification_id %s expired/missing",
            selection.clarification_id,
        )
        return fallback_query
    if rec.get("session_id") != session_id:
        logger.info("[interactive_clarifier] selection: session mismatch")
        return fallback_query

    # opt_other with free_text → use the free text directly
    if selection.selected_option_id == "opt_other":
        return (selection.free_text or "").strip() or fallback_query

    for opt in rec.get("options", []):
        if opt.get("id") == selection.selected_option_id:
            refined = (opt.get("refined_query") or "").strip()
            return refined or fallback_query

    return fallback_query


def safe_embed(text: str) -> Optional[List[float]]:
    if not text or not text.strip():
        return None
    try:
        body = json.dumps({"inputText": text[: settings.MAX_CHARS]}).encode("utf-8")
        resp = bedrock.invoke_model(
            modelId=settings.BEDROCK_EMBED_MODEL,
            body=body,
            accept="application/json",
            contentType="application/json",
        )
        payload = json.loads(resp["body"].read().decode("utf-8"))
        emb = payload.get("embedding")
        return emb if isinstance(emb, list) else None
    except Exception as e:
        logger.exception("Embed failed: %s", e)
        return None


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def _invoke_primary_llm(prompt: str, max_tokens: int) -> str:
    """
    The "raw" primary-model invocation, factored out of safe_generate so
    the timeout guard can call it as a primary_fn. ANY Bedrock-side
    exception is allowed to propagate from here — the guard inspects
    the exception type to decide whether to fall back to Haiku.

    The prompt truncation logic stays here because it's specific to
    Mistral's MODEL_MAX_TOKENS budget; Haiku has a much larger context
    and doesn't need it.
    """
    max_prompt_tokens = TOKEN_BUDGET["MODEL_MAX_TOKENS"] - max_tokens - 200
    max_prompt_chars = max_prompt_tokens * CHARS_PER_TOKEN

    if len(prompt) > max_prompt_chars:
        logger.warning("TRUNCATING: %d -> %d chars", len(prompt), max_prompt_chars)
        marker = "\nANSWER:"
        pos = prompt.rfind(marker)
        if pos > 0:
            tail = prompt[pos:]
            prompt = (
                prompt[: max_prompt_chars - len(tail) - 80]
                + "\n\n[... context truncated ...]\n"
                + tail
            )
        else:
            prompt = prompt[:max_prompt_chars] + "\n\n[... truncated ...]\n"

    body = json.dumps(
        {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "top_p": 0.9,
        }
    ).encode("utf-8")

    resp = bedrock.invoke_model(
        modelId=settings.BEDROCK_LLM_MODEL,
        body=body,
        accept="application/json",
        contentType="application/json",
    )
    payload = json.loads(resp["body"].read().decode("utf-8"))

    if isinstance(payload, dict):
        if "outputs" in payload and payload["outputs"]:
            return (payload["outputs"][0].get("text") or "").strip() or "No response."
        if "generation" in payload:
            return str(payload["generation"]).strip()
        if "outputText" in payload:
            return str(payload["outputText"]).strip()
    return "No response generated."


def safe_generate(prompt: str, max_tokens: int = None) -> str:
    """
    Chat-path LLM entry point.

    Calls the primary model (Mistral via `_invoke_primary_llm`). If the
    Bedrock client raises a *recoverable* error (read timeout, throttle,
    connection drop), `llm_timeout_guard` retries once on Claude
    Haiku 4.5. If both fail, the user gets a polite decline message
    instead of waiting indefinitely.

    Non-recoverable errors (e.g. our own ValidationException) still hit
    the catch-all below and surface the generic "Error generating..."
    string. That preserves the function's existing contract: callers
    always get a non-empty string back.
    """
    if max_tokens is None:
        max_tokens = TOKEN_BUDGET["MAX_GENERATION_TOKENS"]
    try:
        from backend.services.llm_timeout_guard import generate_with_fallback
        text, model_label = generate_with_fallback(
            prompt=prompt,
            max_tokens=max_tokens,
            primary_fn=_invoke_primary_llm,
        )
        if model_label != "primary":
            # Fallback or decline — log so we can monitor frequency.
            logger.info("safe_generate result via model=%s", model_label)
        return text
    except Exception as e:
        logger.exception("Generation failed: %s", e)
        return "Error generating response. Please try again."


def hybrid_search(
    query: str,
    query_embedding: List[float],
    file_type: str,
    n_results: int = 10,
    allowed_file_ids: Optional[Set[str]] = None,
    owner_id: Optional[str] = None,
) -> List[Tuple[str, str, Dict, float]]:
    v_weight = HYBRID_CONFIG["VECTOR_WEIGHT"]
    b_weight = HYBRID_CONFIG["BM25_WEIGHT"]
    rrf_k = 60
    owner_id = _normalize_owner_id(owner_id)

    vector_results = {}
    try:
        vector_hits = pgvector_search(
            query_embedding=query_embedding,
            n_results=HYBRID_CONFIG["VECTOR_CANDIDATES"],
            allowed_file_ids=list(allowed_file_ids) if allowed_file_ids else None,
        )

        rank = 0
        for hit in vector_hits:
            doc_id = hit["id"]
            meta = hit["metadata"]
            meta_file_id = meta.get("file_id")

            if allowed_file_ids is not None and meta_file_id not in allowed_file_ids:
                continue
            # NOTE: owner_id filtering removed — files are shared across all users

            rank += 1
            vector_results[doc_id] = {
                "text": hit["text"],
                "metadata": meta,
                "rank": rank,
                "similarity": max(0.0, 1.0 - float(hit["distance"])),
            }

    except Exception as e:
        logger.warning("Vector search failed (%s): %s", file_type, e)

    bm25_results = {}
    try:
        if bm25 and bm25.size > 0:
            raw_bm25 = bm25.search(
                query,
                n_results=HYBRID_CONFIG["BM25_CANDIDATES"],
                file_type=file_type,
            )
            rank = 0
            for doc_id, text_value, meta, score in raw_bm25:
                meta_file_id = meta.get("file_id")

                if allowed_file_ids is not None and meta_file_id not in allowed_file_ids:
                    continue
                # NOTE: owner_id filtering removed — files are shared across all users

                rank += 1
                bm25_results[doc_id] = {
                    "text": text_value,
                    "metadata": meta,
                    "rank": rank,
                    "bm25_score": score,
                }

                if rank >= HYBRID_CONFIG["BM25_CANDIDATES"]:
                    break

    except Exception as e:
        logger.warning("BM25 search failed: %s", e)

    combined: Dict[str, float] = {}
    all_data: Dict[str, dict] = {}

    for doc_id, data in vector_results.items():
        combined[doc_id] = combined.get(doc_id, 0) + v_weight / (rrf_k + data["rank"])
        all_data[doc_id] = data

    for doc_id, data in bm25_results.items():
        combined[doc_id] = combined.get(doc_id, 0) + b_weight / (rrf_k + data["rank"])
        if doc_id not in all_data:
            all_data[doc_id] = data

    sorted_ids = sorted(combined, key=lambda x: combined[x], reverse=True)

    return [
        (did, all_data[did]["text"], all_data[did]["metadata"], combined[did])
        for did in sorted_ids[:n_results]
    ]


def rerank_chunks(
    query: str,
    chunks: List[Tuple[str, str, Dict, float]],
    top_k: int = 6,
) -> List[Tuple[str, str, Dict, float]]:
    if not chunks or len(chunks) <= 1:
        return chunks[:top_k]
    # Brief 4 / Opt 4: skip the Mistral call entirely when the candidate
    # set is below the configured minimum — nothing to re-order.
    if (
        getattr(settings, "SKIP_RERANK_ON_TINY_RESULTS_ENABLED", False)
        and len(chunks) < int(getattr(settings, "RERANK_MIN_CHUNKS", 3))
    ):
        logger.info(
            "[rerank] skipped (api.rerank_chunks) — only %d chunks (threshold=%d)",
            len(chunks), int(settings.RERANK_MIN_CHUNKS),
        )
        return chunks[:top_k]

    candidates = chunks[: min(len(chunks), 12)]

    previews = []
    for i, (_, text, meta, _) in enumerate(candidates):
        preview = text[:600].replace("\n", " ").strip()
        src = meta.get("source", "?")
        previews.append(f"[{i + 1}] (source: {src}) {preview}")

    rerank_prompt = f"""Rate each chunk's relevance to the question (0=irrelevant, 10=perfect match).
Question: {query}

Chunks:
{chr(10).join(previews)}

Respond ONLY with a JSON array: [{{"chunk":1,"score":8}}, ...]"""

    try:
        resp = safe_generate(rerank_prompt, max_tokens=512)
        raw = resp.strip()
        if "```" in raw:
            raw = raw.split("```")[1] if "```json" not in raw else raw.split("```json")[1].split("```")[0]
        start, end = raw.find("["), raw.rfind("]") + 1
        if start < 0 or end <= start:
            raise ValueError("No JSON array found")

        scores = json.loads(raw[start:end])

        scored = []
        for item in scores:
            idx = item.get("chunk", 0) - 1
            relevance = item.get("score", 0)
            if 0 <= idx < len(candidates):
                did, text, meta, orig = candidates[idx]
                final = (relevance / 10.0) * 0.7 + orig * 30 * 0.3
                scored.append((did, text, meta, final))

        scored_ids = {s[0] for s in scored}
        for c in candidates:
            if c[0] not in scored_ids:
                scored.append(c)

        scored.sort(key=lambda x: x[3], reverse=True)
        return scored[:top_k]

    except Exception as e:
        logger.warning("Re-rank failed (%s), using hybrid scores", e)
        return candidates[:top_k]


def truncate_chunk(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    t = text[:max_chars]
    nl = t.rfind("\n")
    if nl > max_chars * 0.7:
        t = t[:nl]
    return t + "\n[... truncated ...]"


def assemble_context(
    ranked: List[Tuple[str, str, Dict, float]],
    max_total_chars: int,
    max_sources: int = 5,
) -> Tuple[str, List[str]]:
    max_chunk = TOKEN_BUDGET["MAX_SINGLE_CHUNK_CHARS"]
    if not ranked:
        return "", []

    parts = []
    source_scores: Dict[str, float] = {}
    total = 0

    for _, text, meta, score in ranked:
        if not text or not text.strip():
            continue

        chunk = truncate_chunk(text.strip(), max_chunk)
        src = meta.get("source", "unknown")
        entry = f"[Source: {src}]\n{chunk}"

        if total + len(entry) > max_total_chars:
            remaining = max_total_chars - total
            if remaining > 300:
                parts.append(f"[Source: {src}]\n{truncate_chunk(chunk, remaining - 60)}")
                source_scores[src] = source_scores.get(src, 0.0) + score
            break

        parts.append(entry)
        source_scores[src] = source_scores.get(src, 0.0) + score
        total += len(entry)

    if not source_scores:
        return "\n\n".join(parts), []

    ranked_sources = sorted(source_scores.items(), key=lambda x: x[1], reverse=True)
    top_score = ranked_sources[0][1]
    threshold = top_score * 0.40
    final_sources = [src for src, score in ranked_sources if score >= threshold]
    return "\n\n".join(parts), final_sources[:max_sources]

def has_sufficient_document_support(
    question: str,
    ranked: List[Tuple[str, str, Dict, float]],
    min_score: float = 0.12,
    min_keyword_hits: int = 1,
    expanded_keywords: Optional[Set[str]] = None,
) -> bool:
    """
    Document-aware support check. Uses glossary-expanded keywords so that
    a query for "DHCP" also matches chunks containing "Dynamic Host Configuration Protocol"
    (if the document's glossary defined that mapping).
    """
    return has_sufficient_document_support_v2(
        question=question,
        ranked=ranked,
        min_score=min_score,
        min_keyword_hits=min_keyword_hits,
        expanded_keywords=expanded_keywords,
    )


def calculate_file_hash_bytes(content: bytes) -> str:
    sha = hashlib.sha256()
    sha.update(content)
    return sha.hexdigest()


def extract_text_from_pdf(fp: Path) -> str:
    try:
        import fitz

        parts = []
        with fitz.open(str(fp)) as doc:
            for i, page in enumerate(doc):
                t = page.get_text("text")
                if t and t.strip():
                    parts.append(f"--- Page {i+1} ---\n{t}")
        return "\n\n".join(parts) if parts else ""
    except ImportError:
        pass

    try:
        import pdfplumber

        parts = []
        with pdfplumber.open(str(fp)) as pdf:
            for i, page in enumerate(pdf.pages):
                t = page.extract_text()
                if t and t.strip():
                    parts.append(f"--- Page {i+1} ---\n{t}")
        return "\n\n".join(parts) if parts else ""
    except ImportError:
        raise RuntimeError("No PDF library. pip install PyMuPDF")


def extract_text_from_docx(fp: Path) -> str:
    try:
        from docx import Document
    except ImportError:
        raise RuntimeError("pip install python-docx")

    doc = Document(str(fp))
    parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    for ti, table in enumerate(doc.tables):
        rows = [" | ".join(c.text.strip() for c in r.cells) for r in table.rows]
        if rows:
            parts.append(f"--- Table {ti + 1} ---")
            parts.extend(rows)
    return "\n".join(parts)


def extract_text(fp: Path) -> str:
    ext = fp.suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(fp)
    if ext == ".docx":
        return extract_text_from_docx(fp)
    return fp.read_text(encoding="utf-8", errors="ignore")


def iter_text_chunks(text, max_chars=None, lines_per=None, overlap=None):
    max_chars = max_chars or CHUNK_CONFIG["MAX_CHUNK_CHARS"]
    lines_per = lines_per or CHUNK_CONFIG["LINES_PER_CHUNK"]
    overlap = overlap or CHUNK_CONFIG["OVERLAP_LINES"]
    lines = text.splitlines(keepends=True)
    if not lines:
        return
    buf, sz = [], 0
    for ln in lines:
        buf.append(ln)
        sz += len(ln)
        if len(buf) >= lines_per or sz >= max_chars:
            yield "".join(buf)
            if overlap > 0 and len(buf) > overlap:
                buf = buf[-overlap:]
                sz = sum(len(l) for l in buf)
            else:
                buf.clear()
                sz = 0
    if buf:
        yield "".join(buf)


def iter_line_chunks(fp: Path, lines_per=None):
    lines_per = lines_per or CHUNK_CONFIG["LINES_PER_CHUNK"]
    ext = fp.suffix.lower()
    if ext in (".pdf", ".docx"):
        text = extract_text(fp)
        if not text or not text.strip():
            return
        for c in iter_text_chunks(text, lines_per=lines_per):
            yield c
        return

    text = fp.read_text(encoding="utf-8", errors="ignore")
    if text and text.strip():
        for c in iter_text_chunks(text, lines_per=lines_per):
            yield c


# async def index_file_job(
#     job_id: str,
#     storage_uri: str,
#     filename: str,
#     file_type: str,
#     file_id: str,
#     owner_id: Optional[str],
#     file_size_mb: float,
# ):
#     owner_id = _normalize_owner_id(owner_id)
#     update_ingestion_job(job_id, status="running")

#     try:
#         local_path = storage.resolve_local_path(storage_uri)
#         if not local_path or not local_path.exists():
#             raise RuntimeError(f"Stored file path is not readable: {storage_uri}")

#         job = get_ingestion_job(job_id)
#         file_hash = job["file_hash"] if job else ""

#         processed = process_document(
#             local_path=local_path,
#             filename=filename,
#             file_type=file_type,
#             owner_id=owner_id,
#             fingerprint=file_hash,
#             exact_duplicate_lookup=find_duplicate_by_hash,
#             version_candidate_lookup=find_version_candidates,
#         )

#         if processed["status"] == "exact_duplicate":
#             update_ingestion_job(
#                 job_id,
#                 status="done",
#                 processed_chunks="0",
#                 total_chunks="0",
#                 successful_chunks="0",
#                 error=None,
#                 completed_at=datetime.now(timezone.utc),
#             )
#             return

#         chunk_rows = []
#         bm25_ids, bm25_docs, bm25_metas = [], [], []

#         total_chunks = len(processed["chunk_rows"])

#         embed_inputs: List[Tuple[int, Dict[str, Any], str]] = []
#         for idx, row in enumerate(processed["chunk_rows"]):
#             embed_text = row.get("contextualized_content") or row["content"]
#             embed_inputs.append((idx, row, embed_text))

#         embeddings_map: Dict[int, List[float]] = {}
#         embed_workers = min(settings.EMBED_CONCURRENCY, max(1, total_chunks))

#         with ThreadPoolExecutor(max_workers=embed_workers) as executor:
#             future_to_idx = {
#                 executor.submit(safe_embed, text): idx
#                 for idx, _row, text in embed_inputs
#             }
#             for future in as_completed(future_to_idx):
#                 idx = future_to_idx[future]
#                 try:
#                     emb = future.result()
#                     if emb:
#                         embeddings_map[idx] = emb
#                 except Exception:
#                     pass

#         for idx, row, embed_text in embed_inputs:
#             emb = embeddings_map.get(idx)
#             if not emb:
#                 continue

#             chunk_id = f"{file_id}:{job_id}:{idx}"
#             row["id"] = chunk_id
#             row["embedding"] = emb
#             chunk_rows.append(row)

#             bm25_ids.append(chunk_id)
#             bm25_docs.append(embed_text)
#             bm25_metas.append(
#                 {
#                     "file_id": file_id,
#                     "owner_id": owner_id,
#                     "source": filename,
#                     "file_type": file_type,
#                     "section_heading": row.get("section_heading"),
#                     "chunk_type": row.get("chunk_type"),
#                     "summary": row.get("summary"),
#                     "labels_json": row.get("labels_json", {}),
#                     "metadata_json": row.get("metadata_json", {}),
#                 }
#             )

#             if (len(chunk_rows)) % settings.BATCH_SIZE == 0:
#                 update_ingestion_job(
#                     job_id,
#                     processed_chunks=str(len(chunk_rows)),
#                     total_chunks=str(total_chunks),
#                     successful_chunks=str(len(chunk_rows)),
#                 )

#         inserted = insert_document_and_chunks(
#             document_id=file_id,
#             filename=filename,
#             fingerprint=file_hash,
#             chunk_rows=chunk_rows,
#             owner_id=owner_id,
#             file_type=file_type,
#             storage_uri=storage_uri,
#             file_size_mb=file_size_mb,
#             metadata=processed["document_metadata"],
#             version_decision=processed["version_decision"],
#         )

#         if bm25 and inserted["status"] == "inserted" and bm25_ids:
#             bm25.remove_documents_by_file_id(file_id)
#             bm25.add_documents_batch(bm25_ids, bm25_docs, bm25_metas)

#         update_ingestion_job(
#             job_id,
#             status="done",
#             processed_chunks=str(total_chunks),
#             total_chunks=str(total_chunks),
#             successful_chunks=str(len(chunk_rows)),
#             error=None,
#             completed_at=datetime.now(timezone.utc),
#         )

#     except Exception as exc:
#         logger.exception("Indexing failed for %s: %s", filename, exc)
#         update_ingestion_job(
#             job_id,
#             status="error",
#             error=str(exc),
#             completed_at=datetime.now(timezone.utc),
#         )   

async def index_file_job(
    job_id: str,
    storage_uri: str,
    filename: str,
    file_type: str,
    file_id: str,
    owner_id: Optional[str],
    file_size_mb: float,
    doc_kind: str = "ticket",   # Sprint 3-PREP-B
):
    owner_id = _normalize_owner_id(owner_id)
    update_ingestion_job(job_id, status="running")
    t_total_start = _time.perf_counter()

    # Visibility — confirm the filename we'll write into documents.name
    # actually arrived non-empty. Aligns with the [upload.finalize]
    # scheduling log so you can grep one job_id across the pipeline.
    logger.info(
        "[index_file_job] start job=%s file_id=%s filename=%r file_type=%s "
        "size_mb=%.2f doc_kind=%s storage_uri=%s",
        job_id, file_id, filename, file_type, file_size_mb, doc_kind, storage_uri,
    )

    # When the upload came in via the S3 presigned-PUT pipeline,
    # ``storage_uri`` is an ``s3://`` URI rather than a local path. The
    # parser still needs a Path on disk, so we download the object to a
    # NamedTemporaryFile here and clean it up in the outer finally
    # block below. The legacy ``local://`` path is byte-identical.
    _s3_temp_file: Optional[Path] = None

    try:
        if storage_uri.startswith("s3://"):
            if s3_upload_provider is None:
                raise RuntimeError(
                    "Got s3:// storage_uri but S3 upload pipeline is "
                    f"not configured: {storage_uri}"
                )
            import tempfile as _tempfile
            logger.info(
                "[upload.s3] downloading object for ingestion uri=%s",
                storage_uri,
            )
            try:
                s3_bytes = await asyncio.to_thread(
                    s3_upload_provider.read_bytes, storage_uri,
                )
            except Exception as _s3_exc:
                # Spell out the most common cause — missing s3:GetObject
                # on the backend's IAM principal. Without this line the
                # boto3 ClientError detail gets swallowed in the generic
                # "Indexing failed" log below and operators stare at the
                # error column wondering why.
                raise RuntimeError(
                    f"S3 GetObject failed for {storage_uri}: {_s3_exc}. "
                    f"Verify the backend IAM identity has s3:GetObject "
                    f"on arn:aws:s3:::{s3_upload_provider.bucket_name}/*"
                ) from _s3_exc
            suffix = Path(filename).suffix or ".bin"
            tf = _tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            try:
                tf.write(s3_bytes)
            finally:
                tf.close()
            local_path = Path(tf.name)
            _s3_temp_file = local_path
            logger.info(
                "[upload.s3] downloaded %d bytes uri=%s temp=%s",
                len(s3_bytes), storage_uri, local_path,
            )
        else:
            local_path = storage.resolve_local_path(storage_uri)
        if not local_path or not local_path.exists():
            raise RuntimeError(f"Stored file path is not readable: {storage_uri}")

        job = get_ingestion_job(job_id)
        # S3 path: file_hash isn't computed at presign time (we don't
        # have the bytes); fall back to S3 ETag from the job row when
        # the legacy file_hash column is empty so duplicate detection
        # still has something to key off. The ETag is only equal to the
        # MD5 for single-part uploads — sufficient for files <100MB.
        file_hash = (job["file_hash"] if job else "") or ""
        if not file_hash and storage_uri.startswith("s3://") and _s3_temp_file is not None:
            file_hash = calculate_file_hash_bytes(s3_bytes)
            logger.info(
                "[upload.s3] computed local file_hash for job=%s hash=%s",
                job_id, file_hash[:12],
            )

        # ── Phase 1: Parse + contextual enrichment (Haiku LLM calls) ──
        t_parse_start = _time.perf_counter()

        processed = await asyncio.to_thread(
            process_document,
            local_path=local_path,
            filename=filename,
            file_type=file_type,
            owner_id=owner_id,
            fingerprint=file_hash,
            exact_duplicate_lookup=find_duplicate_by_hash,
            version_candidate_lookup=find_version_candidates,
            doc_kind=doc_kind,                          # Sprint 3-PREP-B
        )

        t_parse_end = _time.perf_counter()
        logger.info(
            "[PERF] %s — Parse + metadata: %.1fs (%d chunks)",
            filename, t_parse_end - t_parse_start, len(processed.get("chunk_rows", []))
        )

        # Sprint 2.9 — malformed-JSON rejection. process_document returns
        # a {"status": "rejected", ...} dict BEFORE any Haiku / embedding /
        # DB-chunk work when the JSON validator rejects the file.
        # Persist a rejected documents row so the admin UI can show a
        # red-dot badge, mark the ingestion job failed with a human-friendly
        # error detail, and return — no chunks, no embeddings.
        if processed.get("status") == "rejected" and processed.get("ingestion_status") == "invalid_json":
            detail = (
                f"Invalid JSON at line {processed.get('error_line', '?')}"
                f" col {processed.get('error_col', '?')}: "
                f"{processed.get('error_reason', 'parse error')}"
            )
            try:
                await asyncio.to_thread(
                    insert_rejected_document_row,
                    document_id=file_id,
                    owner_id=owner_id,
                    filename=filename,
                    file_type=file_type,
                    ingestion_status="invalid_json",
                    ingestion_error=detail,
                )
            except Exception as row_exc:
                logger.warning("[json_validator] could not persist rejected row: %s", row_exc)
            update_ingestion_job(
                job_id,
                status="error",
                processed_chunks="0",
                total_chunks="0",
                successful_chunks="0",
                error=detail,
                completed_at=datetime.now(timezone.utc),
            )
            logger.info(
                "[json_validator] rejected %s in %.1fs",
                filename, _time.perf_counter() - t_total_start,
            )
            return

        if processed["status"] == "exact_duplicate":
            update_ingestion_job(
                job_id,
                status="done",
                processed_chunks="0",
                total_chunks="0",
                successful_chunks="0",
                error=None,
                completed_at=datetime.now(timezone.utc),
            )
            logger.info("[PERF] %s — Exact duplicate, skipped in %.1fs",
                        filename, _time.perf_counter() - t_total_start)
            return

        chunk_rows = []
        bm25_ids, bm25_docs, bm25_metas = [], [], []

        total_chunks = len(processed["chunk_rows"])

        # ── Phase 2: Concurrent embedding (Bedrock API calls) ──
        t_embed_start = _time.perf_counter()

        embed_inputs: List[Tuple[int, Dict[str, Any], str]] = []
        for idx, row in enumerate(processed["chunk_rows"]):
            embed_text = row.get("contextualized_content") or row["content"]
            embed_inputs.append((idx, row, embed_text))

        embed_workers = min(settings.EMBED_CONCURRENCY, max(1, total_chunks))

        def _run_embeddings() -> Dict[int, List[float]]:
            emb_map: Dict[int, List[float]] = {}
            with ThreadPoolExecutor(max_workers=embed_workers) as executor:
                future_to_idx = {
                    executor.submit(safe_embed, text): idx
                    for idx, _row, text in embed_inputs
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        emb = future.result()
                        if emb:
                            emb_map[idx] = emb
                    except Exception:
                        pass
            return emb_map

        embeddings_map: Dict[int, List[float]] = await asyncio.to_thread(_run_embeddings)

        t_embed_end = _time.perf_counter()
        logger.info(
            "[PERF] %s — Embedding: %.1fs (%d/%d succeeded, %d workers)",
            filename, t_embed_end - t_embed_start,
            len(embeddings_map), total_chunks, embed_workers
        )

        # ── Phase 3: Build chunk rows + BM25 entries ──
        for idx, row, embed_text in embed_inputs:
            emb = embeddings_map.get(idx)
            if not emb:
                continue

            chunk_id = f"{file_id}:{job_id}:{idx}"
            row["id"] = chunk_id
            row["embedding"] = emb
            chunk_rows.append(row)

            bm25_ids.append(chunk_id)
            bm25_docs.append(embed_text)
            bm25_metas.append(
                {
                    "file_id": file_id,
                    "owner_id": owner_id,
                    "source": filename,
                    "file_type": file_type,
                    "section_heading": row.get("section_heading"),
                    "chunk_type": row.get("chunk_type"),
                    "summary": row.get("summary"),
                    "labels_json": row.get("labels_json", {}),
                    "metadata_json": row.get("metadata_json", {}),
                }
            )

        # Single progress update after embedding (not per-batch)
        update_ingestion_job(
            job_id,
            processed_chunks=str(len(chunk_rows)),
            total_chunks=str(total_chunks),
            successful_chunks=str(len(chunk_rows)),
        )

        # ── Phase 4: DB insert ──
        t_db_start = _time.perf_counter()

        inserted = await asyncio.to_thread(
            insert_document_and_chunks,
            document_id=file_id,
            filename=filename,
            fingerprint=file_hash,
            chunk_rows=chunk_rows,
            owner_id=owner_id,
            file_type=file_type,
            storage_uri=storage_uri,
            file_size_mb=file_size_mb,
            metadata=processed["document_metadata"],
            version_decision=processed["version_decision"],
            doc_kind=processed.get("doc_kind", "ticket"),   # Sprint 3-PREP-A
        )

        t_db_end = _time.perf_counter()
        logger.info(
            "[PERF] %s — DB insert: %.1fs (%d chunks)",
            filename, t_db_end - t_db_start, len(chunk_rows)
        )

        # ── Phase 5: BM25 index update ──
        t_bm25_start = _time.perf_counter()

        if bm25 and inserted["status"] == "inserted" and bm25_ids:
            def _update_bm25() -> None:
                bm25.remove_documents_by_file_id(file_id)
                bm25.add_documents_batch(bm25_ids, bm25_docs, bm25_metas)
            await asyncio.to_thread(_update_bm25)

        t_bm25_end = _time.perf_counter()

        # ── Phase 5b: Learn glossary/abbreviations from this document ──
        try:
            doc_glossary = extract_glossary_from_chunks(chunk_rows)
            if doc_glossary:
                get_glossary_store().add_document_glossary(file_id, doc_glossary)
                # Also store in metadata_json for persistence across restarts
                logger.info(
                    "Learned %d abbreviations from %s (e.g. %s)",
                    len(doc_glossary), filename, list(doc_glossary.keys())[:5],
                )
        except Exception as e:
            logger.warning("Glossary extraction failed (non-fatal): %s", e)

        update_ingestion_job(
            job_id,
            status="done",
            processed_chunks=str(total_chunks),
            total_chunks=str(total_chunks),
            successful_chunks=str(len(chunk_rows)),
            error=None,
            completed_at=datetime.now(timezone.utc),
        )

        t_total_end = _time.perf_counter()
        logger.info(
            "[PERF] %s — TOTAL: %.1fs | parse=%.1fs embed=%.1fs db=%.1fs bm25=%.1fs | %d chunks",
            filename,
            t_total_end - t_total_start,
            t_parse_end - t_parse_start,
            t_embed_end - t_embed_start,
            t_db_end - t_db_start,
            t_bm25_end - t_bm25_start,
            len(chunk_rows),
        )

    except Exception as exc:
        logger.exception("Indexing failed for %s: %s", filename, exc)
        update_ingestion_job(
            job_id,
            status="error",
            error=str(exc),
            completed_at=datetime.now(timezone.utc),
        )
    finally:
        # Clean up the temp file we downloaded from S3 (if any). Local
        # path uploads have no temp file and skip this branch entirely.
        if _s3_temp_file is not None:
            try:
                _s3_temp_file.unlink(missing_ok=True)
            except Exception as _cleanup_exc:
                logger.warning(
                    "[upload.s3] temp file cleanup failed path=%s err=%s",
                    _s3_temp_file, _cleanup_exc,
                )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    import time

    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000.0
    response.headers["X-Processing-Time"] = f"{ms:.2f}ms"
    # The human-readable ``msg`` stays identical to the legacy format
    # for grep continuity. The structured ``extra`` payload promotes
    # method / route / status / duration_ms to top-level JSON fields,
    # which CloudWatch Logs Insights treats as queryable columns.
    # ``request_id`` / ``user_id`` are injected automatically by the
    # ContextFilter — no need to pass them here.
    logger.info(
        "%s %s -> %s (%.2fms)",
        request.method, request.url.path, response.status_code, ms,
        extra={
            "method": request.method,
            "route": request.url.path,
            "status": response.status_code,
            "duration_ms": round(ms, 2),
        },
    )
    return response


# ─────────────────────────────────────────────────────────────
# Build identity — captured once at module load so each /health
# response carries deterministic deploy fingerprints.
#
# Source priority:
#   1. ``GIT_SHA`` / ``BUILD_TIMESTAMP`` env vars  → set by the deploy
#      pipeline (e.g. redeploy.sh exports them).
#   2. ``.git/HEAD`` walk                          → laptop dev fallback.
#   3. literal "unknown"                           → fully decoupled run.
#
# The /health response also includes a Sentry-enabled boolean so
# operators can confirm error monitoring is active without poking the
# Sentry dashboard.
# ─────────────────────────────────────────────────────────────


def _resolve_git_sha() -> str:
    """Return the current commit SHA (short form) or 'unknown'.

    Tries ``GIT_SHA`` env var first (set by CI / deploy scripts), then
    falls back to reading ``.git/HEAD`` so laptop runs still produce
    a useful value. Never raises.
    """
    import os
    env_sha = (os.environ.get("GIT_SHA") or "").strip()
    if env_sha:
        return env_sha[:12]
    try:
        from pathlib import Path
        head = Path(__file__).resolve().parent.parent / ".git" / "HEAD"
        if head.exists():
            ref = head.read_text(encoding="utf-8").strip()
            if ref.startswith("ref: "):
                ref_path = head.parent / ref[5:]
                if ref_path.exists():
                    return ref_path.read_text(encoding="utf-8").strip()[:12]
            return ref[:12]
    except Exception:
        pass
    return "unknown"


_BUILD_INFO = {
    "git_sha": _resolve_git_sha(),
    # BUILD_TIMESTAMP is exported by redeploy.sh and the docker-compose
    # build args (kept in sync with the frontend bundle stamp).
    "build_timestamp": (
        (__import__("os").environ.get("BUILD_TIMESTAMP") or "unknown").strip()
    ),
    "boot_time": datetime.now(timezone.utc).isoformat(),
}


# ─────────────────────────────────────────────────────────────
# Liveness probe — deliberately cheap.
#
# The ALB target group + Fargate container health check both poll
# this endpoint every 30 s × N containers. The full ``/health`` route
# below queries the DB and reads BM25 + glossary state — fine for an
# operator probe but expensive at the ALB cadence. The liveness path
# stays tight: a static 200 + a process uptime field.
#
# Readiness ("am I ready to serve traffic?") is what ``/health``
# answers. ALB uses liveness; CI smoke tests and operators use
# /health for the richer view.
# ─────────────────────────────────────────────────────────────
@app.get("/health/live")
async def health_live():
    return {
        "status": "ok",
        "uptime_since": _BUILD_INFO.get("boot_time"),
        "git_sha": _BUILD_INFO.get("git_sha"),
    }


@app.get("/health")
async def health_check():
    chunk_count = 0
    try:
        chunk_count = count_pg_chunks()
    except Exception:
        pass

    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": settings.BEDROCK_LLM_MODEL,
        # Build identity — useful for "is the new version actually
        # deployed?" diagnostics and for cross-referencing CloudWatch
        # log lines with a specific commit.
        "build": _BUILD_INFO,
        "services": {
            "vector_store": f"{chunk_count} chunks" if chunk_count >= 0 else "uninitialized",
            "bm25_index": f"{bm25.size} docs" if bm25 else "uninitialized",
            "glossary_store": f"{get_glossary_store().size} acronyms (learned from docs)",
            "bedrock": "available",
        },
        "search_mode": "hybrid (pgvector + BM25 + re-ranking + doc-aware query expansion)",
        # Strict Clerk-only auth. If clerk isn't enabled the backend
        # rejects every protected request with 503 — see auth_dependency.
        "auth_mode": "clerk" if is_clerk_enabled() else "misconfigured",
        # Observability flags — confirm at a glance whether error
        # monitoring is hooked up on this deploy.
        "observability": {
            "log_format": settings.LOG_FORMAT,
            "sentry_enabled": bool((settings.SENTRY_DSN or "").strip()),
        },
    }


# =========================================================
# Auth Debug — call this to diagnose config issues
# =========================================================
@app.get("/auth/debug")
async def auth_debug(
    request: Request,
    user_id: Optional[str] = Depends(auth_dependency),
):
    """Diagnostic endpoint — auth configuration + current JWT state.

    Auth REQUIRED. This endpoint discloses internal configuration
    (Clerk key presence, JWT issuer, JWKS URL) that's helpful to
    operators but should never be visible to the public internet.
    Setting ``Depends(auth_dependency)`` here means a caller must
    already hold a valid Clerk JWT before they can inspect the
    config — which closes the info-disclosure surface.
    """
    clerk_enabled = is_clerk_enabled()
    has_bearer = bool(request.headers.get("Authorization", "").startswith("Bearer "))

    result = {
        "clerk_enabled": clerk_enabled,
        "clerk_enabled_setting": getattr(settings, 'CLERK_ENABLED', 'NOT SET'),
        "clerk_secret_key_set": bool(getattr(settings, 'CLERK_SECRET_KEY', '')),
        "clerk_publishable_key_set": bool(getattr(settings, 'CLERK_PUBLISHABLE_KEY', '')),
        "request_has_bearer_token": has_bearer,
        "auth_mode": "clerk",
        "caller_user_id": user_id,
    }

    # The caller already passed auth_dependency above, so we know the
    # token is valid. Re-decode here only to surface the issuer / sub
    # claims for diagnostic display.
    if has_bearer:
        try:
            from backend.clerk_auth import extract_bearer_token, verify_clerk_token
            token = extract_bearer_token(request)
            if token:
                payload = verify_clerk_token(token)
                result["jwt_valid"] = True
                result["jwt_user_id"] = payload.get("sub")
                result["jwt_issuer"] = payload.get("iss")
        except Exception as e:
            result["jwt_valid"] = False
            result["jwt_error"] = str(e)

    return result


@app.get("/me")
async def get_current_user(request: Request, user_id: Optional[str] = Depends(auth_dependency)):
    # Reaching this point means ``auth_dependency`` already verified the
    # Clerk JWT — anonymous fallback no longer exists, so the response
    # always reflects an authenticated caller.
    payload = getattr(request.state, "clerk_payload", {})
    return {
        "authenticated": True,
        "user_id": user_id,
        "issuer": payload.get("iss"),
        "auth_mode": "clerk",
    }


# =========================================================
# Multi-User: Registration & Account Management
# =========================================================

class RegisterRequest(BaseModel):
    email: Optional[str] = None
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    model_config = ConfigDict(extra="ignore")


@app.post("/auth/register-or-login")
async def register_or_login(
    req: RegisterRequest,
    user_id: Optional[str] = Depends(auth_dependency),
):
    if not user_id:
        raise HTTPException(401, "Authentication required")

    user = upsert_user(
        clerk_id=user_id,
        email=req.email,
        full_name=req.full_name,
        avatar_url=req.avatar_url,
    )
    return {
        "status": "ok",
        "user_id": user["clerk_id"],
        "email": user.get("email"),
        "full_name": user.get("full_name"),
        "created_at": str(user["created_at"]),
        "is_new": user["is_new"],
    }


@app.get("/auth/profile")
async def get_profile(
    user_id: Optional[str] = Depends(auth_dependency),
):
    if not user_id:
        raise HTTPException(401, "Authentication required")

    user = get_user_by_clerk_id(user_id)
    if not user:
        raise HTTPException(404, "User not found. Please sign in again.")

    return {
        "user_id": user["clerk_id"],
        "email": user.get("email"),
        "full_name": user.get("full_name"),
        "avatar_url": user.get("avatar_url"),
        "created_at": str(user["created_at"]),
        "last_login_at": str(user["last_login_at"]) if user.get("last_login_at") else None,
    }


@app.delete("/auth/delete-account")
async def delete_account(
    user_id: Optional[str] = Depends(auth_dependency),
):
    if not user_id:
        raise HTTPException(401, "Authentication required")

    from backend.db.connection import SessionLocal
    from sqlalchemy import text as sa_text

    with SessionLocal() as db:
        result = db.execute(
            sa_text("SELECT * FROM delete_user_data(:oid)"),
            {"oid": user_id},
        ).mappings().first()
        db.commit()

    if bm25:
        rebuild_bm25_from_postgres()

    return {
        "status": "deleted",
        "user_id": user_id,
        "details": dict(result) if result else {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/reset")
async def reset_all(user_id: Optional[str] = Depends(auth_dependency)):
    deleted_chunks = reset_pg_data()
    if bm25:
        bm25.clear()

    return {
        "status": "success",
        "message": "All data deleted",
        "deleted_chunks": deleted_chunks,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/purge_orphans")
async def purge_orphan_chunks(user_id: Optional[str] = Depends(auth_dependency)):
    deleted = purge_orphan_chunks_db()
    bm25_count = rebuild_bm25_from_postgres()

    return {
        "status": "purged",
        "orphans_deleted": deleted,
        "chunks_remaining": count_pg_chunks(),
        "bm25_rebuilt": bm25_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ────────────────────────────────────────────────────────────────
# v2 Bug #5 — Admin ingestion diagnostics
# ────────────────────────────────────────────────────────────────
# verify-ingestion reports chunk / embedding / vocab counts so operators
# can detect the "chunks exist but not queryable" failure mode. reindex
# rebuilds the learned vocabulary from existing chunks — cheaper than
# re-running the full LLM pipeline and directly targets the scenario
# where a file was ingested BEFORE the v2 JSON-aware learner deployed.
# Both are gated by INGESTION_VERIFICATION_ENABLED so the routes fail
# closed when the feature flag is flipped off.

@app.get("/admin/verify-ingestion/{file_id}")
async def admin_verify_ingestion(
    file_id: str,
    user_id: Optional[str] = Depends(auth_dependency),
):
    if not getattr(settings, "INGESTION_VERIFICATION_ENABLED", False):
        raise HTTPException(404, "Ingestion verification disabled")

    from sqlalchemy import text as _sql_text
    from backend.db.connection import engine as _engine

    issues: List[str] = []
    chunks_count = 0
    embeddings_count = 0
    identifiers_learned = 0
    file_exists = False
    file_name: Optional[str] = None

    try:
        with _engine.connect() as conn:
            row = conn.execute(
                _sql_text(
                    "SELECT id::text AS id, name, status FROM documents "
                    "WHERE id::text = :fid LIMIT 1"
                ),
                {"fid": file_id},
            ).mappings().first()
            if row:
                file_exists = True
                file_name = row.get("name")

            chunks_count = conn.execute(
                _sql_text(
                    "SELECT COUNT(*)::int FROM chunks WHERE document_id::text = :fid"
                ),
                {"fid": file_id},
            ).scalar() or 0

            embeddings_count = conn.execute(
                _sql_text(
                    "SELECT COUNT(*)::int FROM chunks "
                    "WHERE document_id::text = :fid AND embedding IS NOT NULL"
                ),
                {"fid": file_id},
            ).scalar() or 0

            identifiers_learned = conn.execute(
                _sql_text(
                    "SELECT COUNT(*)::int FROM learned_vocabulary "
                    "WHERE first_seen_file = :fid AND token_type = 'identifier'"
                ),
                {"fid": file_id},
            ).scalar() or 0
    except Exception as exc:
        logger.warning("[admin/verify-ingestion] query failed: %s", exc)
        issues.append(f"QUERY_ERROR — {exc}")

    if not file_exists:
        issues.append("FILE_NOT_FOUND — no row in documents table")
    if file_exists and chunks_count == 0:
        issues.append("NO_CHUNKS — file exists but has zero chunks")
    if chunks_count > 0 and identifiers_learned == 0:
        issues.append("NO_IDENTIFIERS — vocab learning produced no identifiers")
    if chunks_count > 0 and embeddings_count != chunks_count:
        issues.append(
            f"EMBEDDING_GAP — {chunks_count} chunks but {embeddings_count} embeddings"
        )

    return {
        "file_id": file_id,
        "file_name": file_name,
        "chunks_count": chunks_count,
        "embeddings_count": embeddings_count,
        "identifiers_learned": identifiers_learned,
        "issues": issues,
        "healthy": len(issues) == 0,
    }


@app.post("/admin/reindex/{file_id}")
async def admin_reindex(
    file_id: str,
    user_id: Optional[str] = Depends(auth_dependency),
):
    if not getattr(settings, "INGESTION_VERIFICATION_ENABLED", False):
        raise HTTPException(404, "Ingestion verification disabled")

    from sqlalchemy import text as _sql_text
    from backend.db.connection import engine as _engine

    # Load existing chunk content + file name.
    file_name: Optional[str] = None
    combined_text_parts: List[str] = []
    try:
        with _engine.connect() as conn:
            row = conn.execute(
                _sql_text(
                    "SELECT name FROM documents WHERE id::text = :fid LIMIT 1"
                ),
                {"fid": file_id},
            ).mappings().first()
            if not row:
                raise HTTPException(404, f"File not found: {file_id}")
            file_name = row["name"]

            chunk_rows = conn.execute(
                _sql_text(
                    "SELECT COALESCE(contextualized_content, content) AS body "
                    "FROM chunks WHERE document_id::text = :fid"
                ),
                {"fid": file_id},
            ).mappings().all()
            combined_text_parts = [r["body"] for r in chunk_rows if r.get("body")]
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("[admin/reindex] chunk read failed: %s", exc)
        raise HTTPException(500, f"Reindex failed during chunk read: {exc}")

    if not combined_text_parts:
        return {
            "status": "no_content",
            "file_id": file_id,
            "message": "No chunks found — nothing to re-learn from. Re-upload the file.",
        }

    combined_text = "\n".join(combined_text_parts)

    try:
        from backend.services.vocabulary_learner import (
            learn_from_content,
            persist as _vocab_persist,
            reload_cache as _vocab_reload_cache,
        )
        learned = learn_from_content(combined_text, file_id)
        rows = _vocab_persist(learned, file_id)
        _vocab_reload_cache()
    except Exception as exc:
        logger.warning("[admin/reindex] vocab rebuild failed: %s", exc)
        raise HTTPException(500, f"Reindex failed during vocab learn: {exc}")

    return {
        "status": "reindexed",
        "file_id": file_id,
        "file_name": file_name,
        "chunks_scanned": len(combined_text_parts),
        "vocab_rows_upserted": rows,
        "identifiers": len(learned.get("identifiers", [])),
        "field_names": len(learned.get("field_names", [])),
        "enum_values": len(learned.get("enum_values", [])),
    }


@app.post("/upload", response_model=UploadResponse)
@limiter.limit("100/minute")
async def upload(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    file_type: str = Query(default="kb", pattern="^(kb)$"),
    # doc_kind default switched from "ticket" to "" — an empty value
    # asks the ingestion service to auto-detect from content (PDF/DOCX
    # → kb, JSON/CSV → ticket). An explicit value from the form still
    # wins after whitelist validation below.
    doc_kind: str = Form(default=""),
    user_id: Optional[str] = Depends(auth_dependency),
):
    ext = Path(file.filename).suffix[1:].lower() if file.filename else ""
    if not ext or ext not in settings.ALLOWED_FILE_TYPES:
        raise HTTPException(400, f"Type '{ext}' not allowed. Allowed: {settings.ALLOWED_FILE_TYPES}")

    # Sprint 3-PREP-B — validate doc_kind against the whitelist.
    # Explicit valid value wins; empty / invalid → None so the ingestion
    # service runs content detection on the downloaded file. The earlier
    # "ticket" coerce caused every KB PDF to land as doc_kind=ticket,
    # silently breaking the KB-search / Discuss-with-LogIQ separation.
    _raw_kind = (doc_kind or "").strip().lower()
    if _raw_kind in settings.VALID_DOC_KINDS:
        resolved_doc_kind = _raw_kind
    else:
        resolved_doc_kind = None  # auto-detect downstream

    owner_id = _normalize_owner_id(user_id)
    job_id = uuid.uuid4().hex
    file_id = str(uuid.uuid4())

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(400, f"File {size_mb:.1f}MB exceeds {settings.MAX_FILE_SIZE_MB}MB limit")

    relative_name = f"{job_id}_{Path(file.filename).name}"
    storage_uri = storage.save_bytes(relative_name, content)
    file_hash = calculate_file_hash_bytes(content)

    # Phase 5 — ingestion runs on the worker fleet, not in-process.
    # Behaviour is gated on ``settings.INGESTION_VIA_WORKER``:
    #
    #   true  (default, production posture): enqueue an
    #         ingest_document job; the worker container does the work.
    #   false (rollback escape hatch):       use the legacy
    #         ``background_tasks.add_task(index_file_job, ...)`` path
    #         — heavy work runs in the API process. Lets ops roll back
    #         the migration via env flag without a redeploy if staging
    #         surfaces a regression.
    if settings.INGESTION_VIA_WORKER:
        from backend.jobs.ingestion_queue import enqueue_ingestion_job
        enqueue_ingestion_job(
            job_id=job_id,
            file_id=file_id,
            owner_id=owner_id,
            file_name=file.filename,
            file_type=file_type,
            file_hash=file_hash,
            payload={
                "job_id": job_id,
                "storage_uri": storage_uri,
                "filename": file.filename,
                "file_type": file_type,
                "file_id": file_id,
                "owner_id": owner_id,
                "file_size_mb": size_mb,
                "doc_kind": resolved_doc_kind,
            },
        )
    else:
        # Legacy in-process path. Identical to the pre-Phase-5
        # implementation — create the row, schedule the async task.
        # Note: this branch DOES NOT scale past 2-3 concurrent
        # ingests on a single API container; if you find yourself
        # toggling here under load, redeploy with the worker path on.
        create_ingestion_job(
            job_id=job_id,
            file_id=file_id,
            owner_id=owner_id,
            file_name=file.filename,
            file_type=file_type,
            file_hash=file_hash,
        )
        background_tasks.add_task(
            index_file_job,
            job_id,
            storage_uri,
            file.filename,
            file_type,
            file_id,
            owner_id,
            size_mb,
            resolved_doc_kind,
        )
        logger.warning(
            "[upload] INGESTION_VIA_WORKER=false — running ingest in-process. "
            "This is a temporary fallback; flip back to true once staged.",
        )

    return UploadResponse(
        job_id=job_id,
        file_id=file_id,
        message="Uploaded. Processing started.",
        file_hash=file_hash,
    )


# ─────────────────────────────────────────────────────────────
# Phase 1 — Presigned-PUT S3 upload pipeline
#
# Two endpoints replace the multipart POST /upload above when the
# operator sets STORAGE_TYPE=s3 + S3_UPLOAD_BUCKET in the backend env
# AND the frontend toggles REACT_APP_UPLOAD_VIA_S3=true. Until both
# are flipped, this is dead code from the frontend's perspective —
# the legacy /upload route stays the production hot path.
#
# Flow:
#   1) Browser POSTs /upload/presign  → gets {upload_url, key, job_id}
#   2) Browser PUTs file bytes directly to upload_url (no API hop)
#   3) Browser POSTs /upload/finalize → API HEADs S3 + schedules ingest
#
# Why two endpoints?  The API never touches the file bytes — bandwidth
# stays on the S3 path, and FastAPI stays horizontally scalable.
# ─────────────────────────────────────────────────────────────

from backend.uploads.schemas import (
    FinalizeUploadRequest,
    FinalizeUploadResponse,
    PresignUploadRequest,
    PresignUploadResponse,
)
from backend.uploads.service import (
    finalize_upload as _svc_finalize_upload,
    issue_presigned_upload as _svc_issue_presigned_upload,
)


def _require_s3_pipeline() -> Any:
    """Return the configured S3 provider or 503 with an actionable error.

    Centralised so both routes give the same response when the operator
    forgot to set STORAGE_TYPE / S3_UPLOAD_BUCKET in env.
    """
    if s3_upload_provider is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "S3 upload pipeline not configured. Set STORAGE_TYPE=s3 "
                "and S3_UPLOAD_BUCKET in backend/.env, then restart."
            ),
        )
    return s3_upload_provider


@app.post("/upload/presign", response_model=PresignUploadResponse)
@limiter.limit("100/minute")
async def upload_presign(
    request: Request,
    payload: PresignUploadRequest,
    user_id: Optional[str] = Depends(auth_dependency),
):
    """Issue a one-shot presigned PUT URL for a single file.

    The returned ``upload_url`` MUST be PUT to with the exact same
    ``Content-Type`` header listed in ``required_headers`` — otherwise
    S3 rejects the signature.
    """
    s3 = _require_s3_pipeline()
    return _svc_issue_presigned_upload(
        req=payload,
        user_id=_normalize_owner_id(user_id),
        s3=s3,
    )


@app.post("/upload/finalize", response_model=FinalizeUploadResponse)
@limiter.limit("100/minute")
async def upload_finalize(
    request: Request,
    payload: FinalizeUploadRequest,
    background_tasks: BackgroundTasks,
    user_id: Optional[str] = Depends(auth_dependency),
):
    """Verify the S3 object exists and schedule background ingestion.

    Idempotent on the API side via job-owner check; the underlying
    ingestion ``index_file_job`` is itself safe to re-enter (it
    deduplicates by file hash + version).
    """
    s3 = _require_s3_pipeline()
    response, job = _svc_finalize_upload(
        req=payload,
        user_id=_normalize_owner_id(user_id),
        s3=s3,
    )

    # ── Filename resolution (belt-and-suspenders) ──
    # The job row's file_name was set at presign time, but defend
    # against the (rare) case where the column came back empty by
    # reconstructing the filename from the trailing segment of the
    # S3 key. Without this, documents.name lands as '' and the
    # sidebar shows a size-only entry.
    resolved_filename = (job.get("file_name") or "").strip()
    if not resolved_filename:
        # The S3 key is "{prefix}/{tenant_slug}/{filename}" — last
        # path segment is always the user-supplied filename.
        resolved_filename = (payload.key or "").rsplit("/", 1)[-1] or "uploaded_file"
        logger.warning(
            "[upload.finalize] job=%s had empty file_name in DB; "
            "reconstructed from S3 key tail: %s",
            payload.job_id, resolved_filename,
        )
    logger.info(
        "[upload.finalize] scheduling ingest job=%s filename=%r uri=%s",
        payload.job_id, resolved_filename, response.storage_uri,
    )

    # Phase 5 ingestion routing — same flag as the legacy /upload
    # route (``settings.INGESTION_VIA_WORKER``):
    #
    #   true  → attach payload + leave row pending; worker claims it.
    #   false → legacy ``background_tasks.add_task(index_file_job, ...)``.
    #
    # The row's other fields (file_name, file_type, owner_id, hash)
    # were populated at presign time, so both branches operate on
    # the same DB row — only the "what runs the ingest" answer differs.
    # NOTE on doc_kind: we deliberately pass None here (was hardcoded
    # "ticket" previously). None lets the ingestion service's content
    # detector run against the downloaded file and decide between
    # "ticket" (JSON / CSV / TSV) and "kb" (PDF / DOCX / TXT) based on
    # magic bytes + a JSON-parse probe. An explicit doc_kind from the
    # presign request body would still take precedence; that pathway
    # is not yet plumbed here, which is fine — the detector now does
    # the right thing for all current upload routes.
    if settings.INGESTION_VIA_WORKER:
        from backend.jobs.ingestion_queue import attach_payload_and_enqueue
        attach_payload_and_enqueue(
            job_id=payload.job_id,
            payload={
                "job_id": payload.job_id,
                "storage_uri": response.storage_uri,
                "filename": resolved_filename,
                "file_type": job.get("file_type") or "kb",
                "file_id": response.file_id,
                "owner_id": _normalize_owner_id(user_id),
                "file_size_mb": response.size_bytes / (1024 * 1024),
                "doc_kind": None,  # auto-detect from content in ingestion svc
            },
        )
    else:
        # Legacy in-process path. The ingestion_jobs row is already
        # created (at presign time); we just kick off the async task.
        background_tasks.add_task(
            index_file_job,
            payload.job_id,
            response.storage_uri,
            resolved_filename,
            job.get("file_type") or "kb",
            response.file_id,
            _normalize_owner_id(user_id),
            response.size_bytes / (1024 * 1024),
            None,  # doc_kind — auto-detect in ingestion service
        )
        logger.warning(
            "[upload.finalize] INGESTION_VIA_WORKER=false — running ingest "
            "in-process. Temporary fallback; flip back to true after staging.",
        )

    return response


@app.get("/upload/diagnose/{job_id}")
async def upload_diagnose(
    job_id: str,
    user_id: Optional[str] = Depends(auth_dependency),
):
    """End-to-end pipeline snapshot for a given upload job.

    Returns ingestion-job state, whether the S3 object is present, and
    whether a documents row was written. Saves you from cross-checking
    three places by hand when an upload "succeeds in S3" but doesn't
    appear in the sidebar.
    """
    job = get_ingestion_job(job_id)
    if not job:
        raise HTTPException(404, "job_not_found")

    owner_id = _normalize_owner_id(user_id)
    if str(job.get("owner_id") or "") != owner_id:
        raise HTTPException(404, "job_not_found")

    file_id = job.get("file_id")
    file_name = job.get("file_name") or ""

    # ── S3 presence check (only when an S3 upload is configured) ──
    s3_state: Dict[str, Any] = {"checked": False}
    if s3_upload_provider is not None and file_name:
        # Reconstruct the expected key from job metadata. This is a
        # best-effort probe — operators who hand-deleted the object
        # will see "missing" here, which is exactly the diagnosis they
        # need.
        from backend.uploads.keys import build_upload_key, derive_tenant_slug

        try:
            slug = derive_tenant_slug(owner_id)
            key = build_upload_key(
                prefix=settings.S3_KEY_PREFIX,
                tenant_slug=slug,
                filename=file_name,
            )
            head = s3_upload_provider.head_object(key)
            s3_state = {
                "checked": True,
                "present": True,
                "bucket": s3_upload_provider.bucket_name,
                "key": key,
                "size_bytes": head["content_length"],
                "etag": head["etag"],
                "content_type": head.get("content_type", ""),
            }
        except FileNotFoundError:
            s3_state = {
                "checked": True,
                "present": False,
                "bucket": s3_upload_provider.bucket_name,
                "key": key,
            }
        except Exception as exc:
            s3_state = {
                "checked": True,
                "present": None,
                "error": str(exc),
            }

    # ── Documents row check ──────────────────────────────────────
    doc_state: Dict[str, Any] = {"present": False}
    if file_id:
        try:
            from backend.db.connection import SessionLocal
            from sqlalchemy import text as _sql_text
            with SessionLocal() as db:
                row = db.execute(
                    _sql_text(
                        """
                        SELECT id::text AS id, name, status, ingestion_status,
                               ingestion_error
                        FROM documents
                        WHERE id = :id
                        """
                    ),
                    {"id": file_id},
                ).mappings().first()
            if row:
                doc_state = {
                    "present": True,
                    **dict(row),
                }
        except Exception as exc:
            doc_state = {"present": None, "error": str(exc)}

    return {
        "job": dict(job),
        "s3": s3_state,
        "documents_row": doc_state,
        "next_hint": _diagnose_hint(job, s3_state, doc_state),
    }


def _diagnose_hint(job: dict, s3_state: dict, doc_state: dict) -> str:
    """One-line plain-English summary of where the file is stuck."""
    status = (job.get("status") or "").lower()
    if status == "pending_upload" and not s3_state.get("present"):
        return "Browser never finished the PUT to S3 — call /upload/finalize after the PUT completes."
    if status == "pending_upload" and s3_state.get("present"):
        return "S3 has the object but /upload/finalize was never called — frontend bug."
    if status == "queued" and not doc_state.get("present"):
        return "Ingestion task queued but not yet running — wait or check backend logs for BackgroundTask exceptions."
    if status == "running":
        return "Ingestion in progress — re-check in 30-60s."
    if status == "error":
        return f"Ingestion failed: {job.get('error') or '(no error detail)'}"
    if status == "done" and not doc_state.get("present"):
        return "Ingestion completed but no documents row — likely exact_duplicate of an earlier upload; check duplicate detection."
    if status == "done" and doc_state.get("present"):
        return "All good — file should appear in the sidebar."
    return f"Unhandled state: status={status} s3_present={s3_state.get('present')} doc_present={doc_state.get('present')}"


@app.get("/upload_status/{job_id}", response_model=JobStatus)
async def upload_status(job_id: str, user_id: Optional[str] = Depends(auth_dependency)):
    job = get_ingestion_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    owner_id = _normalize_owner_id(user_id)
    if job.get("owner_id", "anonymous") != owner_id:
        raise HTTPException(404, "Job not found")

    return JobStatus(**job)


@app.get("/files")
async def list_files(user_id: Optional[str] = Depends(auth_dependency)):
    """List ALL active files across all users (files are shared)."""
    files = list_active_files_all()
    return {"files": files, "total": len(files)}


@app.delete("/files/{file_id}")
async def delete_file(file_id: str, user_id: Optional[str] = Depends(auth_dependency)):
    files = {f["id"]: f for f in list_active_files_all()}
    if file_id not in files:
        raise HTTPException(404, "File not found")

    deleted_chunks = delete_document_and_chunks(file_id)

    if bm25:
        bm25.remove_documents_by_file_id(file_id)

    # Brief 5 / Part 1 — wipe cached answers sourced from this file so
    # stale content doesn't keep serving after delete.
    semantic_rows_invalidated = 0
    try:
        from backend.services.semantic_cache import invalidate_by_source_file
        semantic_rows_invalidated = invalidate_by_source_file(file_id)
    except Exception as _sem_exc:
        logger.warning("[semantic_cache] invalidate_by_source_file wrapper failed: %s", _sem_exc)

    return {
        "status": "deleted",
        "file_id": file_id,
        "filename": files[file_id]["name"],
        "deleted_chunks": deleted_chunks,
        "semantic_cache_invalidated": semantic_rows_invalidated,
    }


@app.get("/chat/sessions")
async def list_sessions(user_id: Optional[str] = Depends(auth_dependency)):
    owner_id = _normalize_owner_id(user_id)
    return {"sessions": list_chat_sessions(owner_id)}


@app.get("/chat/sessions/{session_id}")
async def get_session(session_id: str, user_id: Optional[str] = Depends(auth_dependency)):
    owner_id = _normalize_owner_id(user_id)
    session = get_chat_session(session_id, owner_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return session


@app.delete("/chat/sessions/{session_id}")
async def delete_session(session_id: str, user_id: Optional[str] = Depends(auth_dependency)):
    owner_id = _normalize_owner_id(user_id)
    ok = delete_chat_session(session_id, owner_id)
    if not ok:
        raise HTTPException(404, "Session not found")
    return {"status": "deleted", "session_id": session_id}


@app.delete("/chat/sessions")
async def delete_all_sessions(user_id: Optional[str] = Depends(auth_dependency)):
    owner_id = _normalize_owner_id(user_id)
    deleted_count = delete_all_chat_sessions(owner_id)
    return {"status": "cleared", "deleted_count": deleted_count}


# ─────────────────────────────────────────────────────────────
# Guided Workflow — session mode state endpoints (Sprint 1).
# All three require an authenticated user and verify the session
# belongs to that user via the existing get_chat_session pattern.
# ─────────────────────────────────────────────────────────────
class SetSessionModeRequest(BaseModel):
    selected_mode: str
    sub_mode: Optional[str] = None
    form_data: Optional[Dict[str, Any]] = None


class SessionModeResponse(BaseModel):
    ok: bool
    mode: Dict[str, Any]
    reason: Optional[str] = None


@app.get("/chat/sessions/{session_id}/mode", response_model=SessionModeResponse)
async def api_get_session_mode(
    session_id: str,
    user_id: Optional[str] = Depends(auth_dependency),
):
    """Read the current mode-state snapshot for a session."""
    owner_id = _normalize_owner_id(user_id)
    session = get_chat_session(session_id, owner_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    snap = get_session_mode(session_id)
    if not snap.is_valid:
        return SessionModeResponse(ok=False, mode={}, reason="db_read_failed")
    return SessionModeResponse(ok=True, mode=snap.to_dict())


@app.post("/chat/sessions/{session_id}/mode", response_model=SessionModeResponse)
async def api_set_session_mode(
    session_id: str,
    payload: SetSessionModeRequest,
    user_id: Optional[str] = Depends(auth_dependency),
):
    """Lock a session to a selected mode (+ optional sub-mode / form data)."""
    owner_id = _normalize_owner_id(user_id)
    session = get_chat_session(session_id, owner_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    snap = set_session_mode(
        session_id=session_id,
        selected_mode=payload.selected_mode,
        sub_mode=payload.sub_mode,
        form_data=payload.form_data,
    )
    if not snap.is_valid:
        return SessionModeResponse(
            ok=False, mode={}, reason="invalid_mode_or_db_write_failed",
        )
    return SessionModeResponse(ok=True, mode=snap.to_dict())


@app.post("/chat/sessions/{session_id}/context/reset", response_model=SessionModeResponse)
async def api_reset_session_context(
    session_id: str,
    user_id: Optional[str] = Depends(auth_dependency),
):
    """Clear the mode-state for a session. Session itself is preserved."""
    owner_id = _normalize_owner_id(user_id)
    session = get_chat_session(session_id, owner_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    snap = reset_session_mode(session_id)
    if not snap.is_valid:
        return SessionModeResponse(ok=False, mode={}, reason="db_write_failed")
    return SessionModeResponse(ok=True, mode=snap.to_dict())


# ─────────────────────────────────────────────────────────────
# Sprint 2 — Partial mode/form patch (customer form, tech form, etc.)
# Whitelists the same fields as patch_session_mode.
# ─────────────────────────────────────────────────────────────
class PatchSessionFormRequest(BaseModel):
    customer_name: Optional[str] = None
    technology_domain: Optional[str] = None
    ticket_id: Optional[str] = None
    issue_summary: Optional[str] = None
    form_data: Optional[Dict[str, Any]] = None


@app.post(
    "/chat/sessions/{session_id}/mode/form",
    response_model=SessionModeResponse,
)
async def api_patch_session_form(
    session_id: str,
    payload: PatchSessionFormRequest,
    user_id: Optional[str] = Depends(auth_dependency),
):
    """Apply a partial patch to mode-state fields (Sprint 2 forms)."""
    owner_id = _normalize_owner_id(user_id)
    session = get_chat_session(session_id, owner_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    updates: Dict[str, Any] = {}
    for key in (
        "customer_name", "technology_domain", "ticket_id", "issue_summary", "form_data",
    ):
        val = getattr(payload, key, None)
        if val is not None:
            updates[key] = val

    if not updates:
        snap = get_session_mode(session_id)
        if not snap.is_valid:
            return SessionModeResponse(ok=False, mode={}, reason="db_read_failed")
        return SessionModeResponse(ok=True, mode=snap.to_dict(), reason="noop")

    snap = patch_session_mode(session_id, updates)
    if not snap.is_valid:
        return SessionModeResponse(ok=False, mode={}, reason="db_write_failed")
    return SessionModeResponse(ok=True, mode=snap.to_dict())


# ─────────────────────────────────────────────────────────────
# Sprint 4 — Fingerprint-First Expert Copilot
#
#   POST /fingerprint/lookup
#     Body: { session_id, fingerprint }
#     - Validates fingerprint against FINGERPRINT_REGEX (422 on mismatch).
#     - Calls retrieve_by_fingerprint — on miss returns {match: false}.
#     - On hit: builds findings from the gold-ticket metadata_json and
#       calls run_composer with voice_override="expert_copilot". Writes
#       entered_via='fingerprint' + original_fingerprint to the session
#       and saves both the user "turn" and assistant answer to chat
#       history so the transcript is coherent.
#
#   POST /fingerprint/skip
#     Body: { session_id }
#     - Marks entered_via='skip'. Leaves selected_mode NULL so the
#       frontend can then show the normal mode picker.
# ─────────────────────────────────────────────────────────────
class FingerprintLookupRequest(BaseModel):
    # session_id may be empty on the user's first interaction — the
    # endpoint will create a new session via save_message_to_session
    # (same pattern as /ask). Callers should treat the session_id
    # returned in the response as authoritative.
    session_id: Optional[str] = None
    fingerprint: str


class FingerprintSkipRequest(BaseModel):
    session_id: Optional[str] = None


class FingerprintLookupResponse(BaseModel):
    match: bool
    session_id: Optional[str] = None
    answer: Optional[str] = None
    fingerprint: Optional[str] = None
    reason: Optional[str] = None


def _build_fingerprint_findings(metadata_json: Dict[str, Any]) -> List[str]:
    """Render the gold-ticket rich metadata into findings blocks that the
    Expert Copilot voice can structure into its Phase 1/2/3 sections.

    We intentionally split by section so the Composer can cite each one
    (the voice's branching-diagnostics block needs Symptom_Solution_Mapping
    verbatim; the Phase 3 RaC snippet comes from remediation_payload)."""
    import json as _json

    parts: List[str] = []
    meta = metadata_json.get("Metadata") or {}
    if meta:
        header = metadata_json.get("Header") or meta.get("Header") or ""
        parts.append(
            "[Gold ticket header]\n"
            + (header or "(no header)")
            + "\nFingerprints: "
            + ", ".join(meta.get("Fingerprints") or [])
        )
    ssm = metadata_json.get("Symptom_Solution_Mapping")
    if isinstance(ssm, dict) and ssm:
        parts.append(
            "[Symptom_Solution_Mapping]\n" + _json.dumps(ssm, indent=2, ensure_ascii=False)
        )
    sop = metadata_json.get("Operational_SOP")
    if isinstance(sop, dict) and sop:
        parts.append(
            "[Operational_SOP]\n" + _json.dumps(sop, indent=2, ensure_ascii=False)
        )
    kb = metadata_json.get("Knowledge_Base")
    if kb:
        parts.append(
            "[Knowledge_Base]\n" + _json.dumps(kb, indent=2, ensure_ascii=False)
        )
    rem = metadata_json.get("remediation_payload")
    if isinstance(rem, dict) and rem:
        parts.append(
            "[remediation_payload]\n" + _json.dumps(rem, indent=2, ensure_ascii=False)
        )
    return parts or ["[No rich metadata available for this fingerprint]"]


@app.post("/fingerprint/lookup", response_model=FingerprintLookupResponse)
@limiter.limit("30/minute")
async def api_fingerprint_lookup(
    request: Request,
    payload: FingerprintLookupRequest,
    user_id: Optional[str] = Depends(auth_dependency),
):
    """Sprint 4 landing-screen handler: resolve a fingerprint code to the
    Expert Copilot answer for its highest-quality gold ticket."""
    # Pass-through: the regex gate has been intentionally removed so the
    # raw trimmed input reaches retrieve_by_fingerprint and the JSONB `?`
    # exact-match lookup. Non-existent shapes just return a miss, same as
    # any other no-match fingerprint. An empty body still short-circuits
    # as 422 since there is nothing to look up.
    fp_raw = (payload.fingerprint or "").strip()
    if not fp_raw:
        raise HTTPException(status_code=422, detail="fingerprint_empty")

    owner_id = _normalize_owner_id(user_id)

    # Existing-session case: verify ownership. First-interaction case
    # (session_id empty): skip the check — save_message_to_session will
    # create a new session on the user's first turn below (same pattern
    # /ask uses).
    incoming_sid = (payload.session_id or "").strip() or None
    if incoming_sid:
        session = get_chat_session(incoming_sid, owner_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

    # Import here to dodge any import-order surprises and keep the
    # retrieval package optional at startup.
    from backend.retrieval.orchestrator import retrieve_by_fingerprint

    # Sprint 5 — request the chunk_id alongside the metadata so the
    # Expert Copilot answer cache (keyed by chunks.id) is reachable.
    # The tuple form is additive; when flag-off the chunk_id is simply
    # ignored below. Sprint 4 contract for other callers of
    # retrieve_by_fingerprint is unchanged (they don't pass the kwarg).
    metadata_json, chunk_id = retrieve_by_fingerprint(
        fp_raw, return_chunk_id=True,
    )

    # Persist the user-authored "turn" FIRST so a new session gets
    # created on the miss path too (the transcript will just show the
    # fingerprint probe with no assistant reply, which is intentional —
    # the frontend routes the user to the "no match" screen next).
    user_turn = f"[Fingerprint lookup] {fp_raw}"
    session_id = save_message_to_session(
        session_id=incoming_sid,
        role="user",
        content=user_turn,
        owner_id=owner_id,
    )

    if metadata_json is None:
        # Mark the session as having attempted fingerprint entry so the
        # audit trail still records the user's path, even on miss.
        patch_session_mode(
            session_id,
            {"entered_via": "fingerprint", "original_fingerprint": fp_raw},
        )
        return FingerprintLookupResponse(
            match=False,
            session_id=session_id,
            fingerprint=fp_raw,
            reason="no_gold_ticket_match",
        )

    # Lock the session to troubleshooting-mode + audit columns so /ask
    # follow-ups keep the Expert Copilot context. Done BEFORE the compose
    # call so the composer reads the updated session_mode snapshot.
    patch_session_mode(
        session_id,
        {
            "selected_mode": "troubleshooting",
            "entered_via": "fingerprint",
            "original_fingerprint": fp_raw,
        },
    )
    session_mode_snap = get_session_mode(session_id)

    from backend.agents.base import TokenBudget
    from backend.agents.composer import run_composer, run_hybrid_expert_pipeline
    from backend.agents.expert_copilot_template import (
        is_gold_schema_ticket,
        render_header_and_fingerprints,
        render_kb_citations,
        render_phase_2_branching,
        render_phase_3_remediation,
    )
    from backend.vector_store import (
        get_cached_expert_answer,
        set_cached_expert_answer,
    )

    budget = TokenBudget(max_total=settings.AGENT_MAX_TOTAL_TOKENS)
    findings = _build_fingerprint_findings(metadata_json)
    header_name = (metadata_json.get("Header") or fp_raw)[:120]

    # ── Sprint 5 — Template-First Expert Copilot + Answer Cache ──
    # Applies ONLY to gold-schema JSON tickets. Non-gold retrievals
    # (PDFs, Word, KBs, contacts, partial tickets) ALWAYS fall through
    # to the Sprint 4 run_composer path below.
    gold_schema = is_gold_schema_ticket(metadata_json)
    answer: Optional[str] = None

    if gold_schema:
        try:
            # Cache check — fast path. chunk_id may be None if the
            # retriever couldn't identify the row; in that case skip
            # straight to the hybrid render (no cache write either).
            cached = get_cached_expert_answer(chunk_id) if chunk_id else None
            if cached:
                logger.info(
                    "[sprint5] cache_hit chunk_id=%s fp=%s chars=%d",
                    chunk_id, fp_raw, len(cached),
                )
                answer = cached
            else:
                pre_rendered = {
                    "header": render_header_and_fingerprints(metadata_json),
                    "phase_2": render_phase_2_branching(metadata_json),
                    "phase_3": render_phase_3_remediation(metadata_json),
                    "kb_citations": render_kb_citations(metadata_json),
                }
                answer = run_hybrid_expert_pipeline(
                    json_ticket=metadata_json,
                    pre_rendered_sections=pre_rendered,
                    budget=budget,
                    generate_fn=safe_generate,
                    bedrock_client=bedrock,
                )
                if chunk_id and answer:
                    set_cached_expert_answer(chunk_id, answer)
                logger.info(
                    "[sprint5] cache_miss chunk_id=%s fp=%s cached_answer_chars=%d",
                    chunk_id, fp_raw, len(answer or ""),
                )
        except Exception as exc:
            # Any Sprint 5 failure degrades to the Sprint 4 LLM path so
            # the user still gets an answer. No user-visible error.
            logger.exception(
                "[sprint5] hybrid pipeline failed, falling back to Sprint 4: %s",
                exc,
            )
            answer = None

    if not answer:
        try:
            compose_result = run_composer(
                query=f"Walk me through resolving fingerprint {fp_raw}.",
                findings=findings,
                source_names=[header_name],
                budget=budget,
                generate_fn=safe_generate,
                bedrock_client=bedrock,
                session_mode=session_mode_snap,
                voice_override="expert_copilot",
            )
            answer = (compose_result.output or "").strip()
        except Exception as exc:
            logger.exception("[fingerprint_lookup] compose failed fp=%s: %s", fp_raw, exc)
            return FingerprintLookupResponse(
                match=True,
                session_id=session_id,
                fingerprint=fp_raw,
                answer="",
                reason="compose_failed",
            )

    save_message_to_session(
        session_id=session_id,
        role="assistant",
        content=answer,
        owner_id=owner_id,
    )

    return FingerprintLookupResponse(
        match=True,
        session_id=session_id,
        fingerprint=fp_raw,
        answer=answer,
    )


@app.post("/fingerprint/skip", response_model=SessionModeResponse)
async def api_fingerprint_skip(
    payload: FingerprintSkipRequest,
    user_id: Optional[str] = Depends(auth_dependency),
):
    """Sprint 4: user clicked Skip on the landing screen. Records the
    audit trail and returns the refreshed mode snapshot (selected_mode
    stays NULL so the frontend shows the regular mode picker next).

    If no session_id is passed (user hasn't created a session yet), the
    endpoint returns ok=True with an empty mode payload — the audit
    write will happen when the eventual mode-selector write creates the
    session. This keeps Skip cheap and idempotent."""
    incoming_sid = (payload.session_id or "").strip() or None
    if not incoming_sid:
        return SessionModeResponse(ok=True, mode={}, reason="no_session_yet")

    owner_id = _normalize_owner_id(user_id)
    session = get_chat_session(incoming_sid, owner_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    snap = patch_session_mode(
        incoming_sid,
        {"entered_via": "skip", "original_fingerprint": None},
    )
    if not snap.is_valid:
        return SessionModeResponse(ok=False, mode={}, reason="db_write_failed")
    return SessionModeResponse(ok=True, mode=snap.to_dict())


@app.post("/ask", response_model=AnswerResponse)
@limiter.limit("30/minute")
async def ask(request: Request, req: Question, user_id: Optional[str] = Depends(auth_dependency)):
    import time

    start = time.perf_counter()

    if count_pg_chunks() < 0:
        raise HTTPException(503, "Vector store not ready")

    owner_id = _normalize_owner_id(user_id)

    session_id = save_message_to_session(
        session_id=req.session_id,
        role="user",
        content=req.q,
        owner_id=owner_id,
    )

    # Capture clarifier-refined flag before clarification_response is consumed
    # below; pattern analytics uses this signal downstream.
    _pattern_is_clarifier_refined = req.clarification_response is not None

    # ── Interactive Clarifier — expand refined-query follow-ups ──
    if req.clarification_response is not None:
        try:
            expanded_q = _expand_clarification_selection(
                session_id=session_id,
                selection=req.clarification_response,
                fallback_query=req.q,
            )
            if expanded_q and expanded_q.strip() != req.q.strip():
                logger.info(
                    "[interactive_clarifier] expanded selection %s -> %r",
                    req.clarification_response.selected_option_id, expanded_q[:120],
                )
                req = req.model_copy(update={"q": expanded_q, "clarification_response": None})
        except Exception as _clarify_exc:
            logger.warning(
                "[interactive_clarifier] selection expansion failed (%s) - using raw q",
                _clarify_exc,
            )

    # ── Trivial input short-circuit (greetings / thanks / ack / bye / small-talk) ──
    trivial = match_trivial_response(req.q)
    if trivial is not None:
        trivial_category, trivial_response_text = trivial
        logger.info(
            "[trivial] category=%s query=%r — zero-LLM response",
            trivial_category, req.q[:80],
        )
        save_message_to_session(
            session_id=session_id,
            role="assistant",
            content=trivial_response_text,
            owner_id=owner_id,
            sources={"docs": []},
        )
        return AnswerResponse(
            answer=trivial_response_text,
            sources=[],
            confidence=1.0,
            processing_time_ms=int((time.perf_counter() - start) * 1000),
            session_id=session_id,
            context_stats={
                "trivial_short_circuit": True,
                "trivial_category": trivial_category,
                "cache_hit": False,
            },
        )

    # ── Input guard (safety / prompt-injection / size) ──
    try:
        guard_result = check_input(req.q)
    except Exception as _guard_exc:
        logger.warning("Input guard wrapper raised (%s) — failing open", _guard_exc)
        guard_result = None

    if guard_result is not None and guard_result.flagged:
        save_message_to_session(
            session_id=session_id,
            role="assistant",
            content=guard_result.canned_response,
            owner_id=owner_id,
            sources={"docs": []},
        )
        return AnswerResponse(
            answer=guard_result.canned_response,
            sources=[],
            confidence=0.0,
            processing_time_ms=int((time.perf_counter() - start) * 1000),
            session_id=session_id,
            context_stats={
                "trivial_short_circuit": False,
                "input_guard_flagged": True,
                "input_guard_reason": guard_result.reason,
                "input_guard_phrase": guard_result.matched_phrase,
                "cache_hit": False,
            },
        )

    # ── Brief 5 / Part 3 — 3-layer input guardrails (injection/secret/PII) ──
    # Runs after the lightweight regex input_guard above so we preserve the
    # existing canned-rejection path for legacy patterns, then applies the
    # deeper BLOCK/SCRUB verdicts from the new guardrail service.
    _pii_scrubbed_query: Optional[str] = None
    _pii_scrub_matches: List[str] = []
    if settings.INPUT_GUARDRAILS_ENABLED:
        try:
            from backend.services.input_guardrails import (
                check_input as _guardrail_check,
                GuardrailVerdict as _GV,
                safe_rejection_message as _safe_reject,
            )
            _guard = _guardrail_check(req.q)
        except Exception as _gr_exc:
            logger.warning("[guardrail] wrapper raised (%s) — failing open", _gr_exc)
            _guard = None

        if _guard is not None and _guard.verdict == _GV.BLOCK:
            _central = bool(getattr(_guard, "scrub_invalidates_query", False))
            if _central:
                logger.info(
                    "[guardrail] block (pii, central): patterns=%s reason=%s",
                    _guard.matched_patterns, _guard.reason,
                )
            else:
                logger.info(
                    "[guardrail] blocked query: category=%s layer=%s reason=%s",
                    _guard.category, _guard.layer, _guard.reason,
                )
            _block_msg = _safe_reject(_guard.category, guard=_guard)
            save_message_to_session(
                session_id=session_id,
                role="assistant",
                content=_block_msg,
                owner_id=owner_id,
                sources={"docs": []},
            )
            return AnswerResponse(
                answer=_block_msg,
                sources=[],
                confidence=1.0,
                processing_time_ms=int((time.perf_counter() - start) * 1000),
                session_id=session_id,
                context_stats={
                    "guardrail_blocked": True,
                    "guardrail_category": _guard.category,
                    "guardrail_layer": _guard.layer,
                    "guardrail_reason": _guard.reason,
                    "model_used": "guardrail_block",
                    "cache_hit": False,
                },
            )
        if _guard is not None and _guard.verdict == _GV.SCRUB:
            _pii_scrubbed_query = _guard.scrubbed_query
            _pii_scrub_matches = list(_guard.matched_patterns or [])
            logger.info(
                "[guardrail] query scrubbed: removed=%s",
                _pii_scrub_matches,
            )

    # ── Brief 3: same-session query rewriter ──
    # Resolves pronouns, ordinals, ellipsis, and filter swaps against
    # the last few turns BEFORE aggregation detect / retrieval run. Every
    # downstream stage consumes `effective_query`. The session record still
    # stores req.q verbatim so the UI continues to render the user's words.
    #
    # Brief 4 / Opt 2: when PARALLEL_REWRITE_AND_CLASSIFY_ENABLED, the
    # rewriter and the aggregation classifier run concurrently on req.q.
    # If the rewrite ends up changing an aggregation-significant signal
    # (customer, priority, sla, "how many"/"list"/"all", etc.), we
    # re-run the classifier synchronously on the rewritten query so the
    # downstream fast-path stays correct.
    # Brief 5 / Part 3 — pipeline runs on the PII-scrubbed text when the
    # guardrail returned SCRUB; req.q is preserved for session display and
    # auditing. Everything downstream (rewriter, aggregation, retrieval)
    # sees _guarded_q so the scrubbed tokens never reach downstream LLMs.
    _guarded_q = _pii_scrubbed_query if _pii_scrubbed_query is not None else req.q

    recent_msgs_all = get_recent_session_messages(session_id, owner_id)
    _parallel_agg_intent = None  # populated only on the parallel path
    _parallel_path_used = False
    _triage_result = None  # Brief 4 / Opt 3
    if (
        settings.QUERY_REWRITER_ENABLED
        and getattr(settings, "PARALLEL_REWRITE_AND_CLASSIFY_ENABLED", False)
    ):
        _parallel_path_used = True
        _par_skip_agg = bool(re.search(r"\bINC-\d+\b", _guarded_q, re.IGNORECASE))
        _triage_on = bool(getattr(settings, "MERGED_TRIAGE_ENABLED", False))
        _workers = 2 + (1 if _triage_on else 0)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=_workers) as _pool:
                _rw_fut = _pool.submit(
                    rewrite_query, _guarded_q, recent_msgs_all
                )
                _ag_fut = (
                    _pool.submit(detect_aggregation_intent_v2, _guarded_q)
                    if not _par_skip_agg else None
                )
                _tr_fut = (
                    _pool.submit(classify_triage, _guarded_q) if _triage_on else None
                )
                rewrite = _rw_fut.result(timeout=5.0)
                _parallel_agg_intent = (
                    _ag_fut.result(timeout=5.0) if _ag_fut is not None else None
                )
                if _tr_fut is not None:
                    try:
                        _triage_result = _tr_fut.result(timeout=5.0)
                    except Exception as _tr_exc:
                        logger.warning("[triage] future error: %s", _tr_exc)
                        _triage_result = None
        except Exception as _parallel_exc:
            logger.warning(
                "[parallel] fallback to sequential (%s: %s)",
                type(_parallel_exc).__name__, _parallel_exc,
            )
            _parallel_path_used = False
            rewrite = rewrite_query(_guarded_q, recent_messages=recent_msgs_all)
            _parallel_agg_intent = None
            _triage_result = None

        effective_query = rewrite.rewritten_query
        rewrite_reason = rewrite.reason
        rewrite_confidence = rewrite.confidence
        rewrite_applied = rewrite.was_rewritten

        # Reconciliation: if the rewrite changed an aggregation signal, the
        # classifier verdict computed against req.q may be stale. Re-run.
        if (
            _parallel_path_used
            and rewrite_applied
            and _might_flip_aggregation(_guarded_q, effective_query, _parallel_agg_intent)
        ):
            logger.info(
                "[parallel] rewrite invalidated classifier verdict — re-classifying on %r",
                effective_query[:120],
            )
            _parallel_agg_intent = (
                detect_aggregation_intent_v2(effective_query)
                if not re.search(r"\bINC-\d+\b", effective_query, re.IGNORECASE)
                else None
            )
    elif settings.QUERY_REWRITER_ENABLED:
        rewrite = rewrite_query(_guarded_q, recent_messages=recent_msgs_all)
        effective_query = rewrite.rewritten_query
        rewrite_reason = rewrite.reason
        rewrite_confidence = rewrite.confidence
        rewrite_applied = rewrite.was_rewritten
    else:
        effective_query = _guarded_q
        rewrite_reason = "disabled"
        rewrite_confidence = 0.0
        rewrite_applied = False

    # ── Sprint 2: Context-break detection (non-blocking signal) ──
    # Piggybacks on the regex phrase detector + the existing triage
    # verdict (zero new LLM calls). When the session has an active
    # locked mode, we emit a soft hint in context_stats so the UI can
    # offer the Continue / Start-new modal. The user's turn still
    # runs end-to-end — we never short-circuit the answer pipeline.
    _context_break_hit: Optional[Dict[str, Any]] = None
    try:
        _cb_snap = get_session_mode(session_id)
    except Exception:
        _cb_snap = None
    if (
        _cb_snap is not None
        and getattr(_cb_snap, "is_valid", False)
        and getattr(_cb_snap, "conversation_context_active", False)
    ):
        try:
            from backend.services.context_break_phrases import (
                detect_context_break as _detect_cb,
            )
            _cb_match = _detect_cb(req.q or "")
        except Exception as _cb_exc:
            logger.warning("[context_break] regex detect failed: %s", _cb_exc)
            _cb_match = None
        _cb_llm = False
        _cb_llm_reason = ""
        if (
            _triage_result is not None
            and getattr(_triage_result, "is_valid", False)
        ):
            _cb_llm = bool(getattr(_triage_result, "context_break", False))
            _cb_llm_reason = str(getattr(_triage_result, "context_break_reason", "") or "")
        if _cb_match is not None or _cb_llm:
            _context_break_hit = {
                "detected": True,
                "source": (
                    "phrase+llm" if (_cb_match is not None and _cb_llm)
                    else ("phrase" if _cb_match is not None else "llm")
                ),
                "category": (
                    getattr(_cb_match, "category", "")
                    if _cb_match is not None else ""
                ),
                "matched_phrase": (
                    getattr(_cb_match, "matched_phrase", "")
                    if _cb_match is not None else ""
                ),
                "llm_reason": _cb_llm_reason,
                "active_mode": _cb_snap.selected_mode,
                "active_sub_mode": _cb_snap.sub_mode,
            }
            logger.info(
                "[context_break] hit source=%s mode=%s phrase=%r llm=%s",
                _context_break_hit["source"], _cb_snap.selected_mode,
                _context_break_hit["matched_phrase"], _cb_llm,
            )

    active_file_ids = _get_active_indexed_file_ids(user_id)

    # ── Answer cache lookup (fail-safe: any error returns miss) ──
    cache_lookup = ANSWER_CACHE.get(
        query=effective_query,
        owner_id=owner_id,
        active_file_ids=active_file_ids or [],
    )
    if cache_lookup.hit and cache_lookup.payload:
        cached = cache_lookup.payload
        save_message_to_session(
            session_id=session_id,
            role="assistant",
            content=cached.get("answer", ""),
            owner_id=owner_id,
            sources={"docs": cached.get("sources", [])},
        )
        stats = dict(cached.get("context_stats") or {})
        stats["cache_hit"] = True
        stats["cache_key"] = cache_lookup.key[:12]
        return AnswerResponse(
            answer=cached.get("answer", ""),
            sources=cached.get("sources", []),
            confidence=float(cached.get("confidence", 0.0)),
            processing_time_ms=int((time.perf_counter() - start) * 1000),
            session_id=session_id,
            context_stats=stats,
        )

    # ── Brief 5 / Part 1 — Semantic answer cache lookup ──
    # Runs AFTER exact-match miss, BEFORE aggregation/retrieval/LLM spend.
    # Fail-safe: any error returns None (treated as miss).
    if settings.SEMANTIC_CACHE_ENABLED and active_file_ids:
        try:
            from backend.services.semantic_cache import lookup as _sem_lookup
            _sem_hit = _sem_lookup(effective_query, active_file_ids=active_file_ids)
        except Exception as _sem_exc:
            logger.warning("[semantic_cache] lookup wrapper raised: %s", _sem_exc)
            _sem_hit = None
        if _sem_hit is not None:
            logger.info(
                "[semantic_cache] serving cached answer (sim=%.3f, hit_count=%d, id=%s)",
                _sem_hit.similarity, _sem_hit.hit_count, _sem_hit.cache_id,
            )
            _sem_sources = _sem_hit.answer_metadata.get("sources", []) or []
            save_message_to_session(
                session_id=session_id,
                role="assistant",
                content=_sem_hit.answer_text,
                owner_id=owner_id,
                sources={"docs": _sem_sources},
            )
            return AnswerResponse(
                answer=_sem_hit.answer_text,
                sources=_sem_sources,
                confidence=float(_sem_hit.answer_metadata.get("confidence", 0.95)),
                processing_time_ms=int((time.perf_counter() - start) * 1000),
                session_id=session_id,
                context_stats={
                    "semantic_cache_hit": True,
                    "semantic_cache_id": _sem_hit.cache_id,
                    "semantic_cache_similarity": round(_sem_hit.similarity, 4),
                    "semantic_cache_hit_count": _sem_hit.hit_count,
                    "semantic_cached_query": _sem_hit.cached_query,
                    "model_used": "semantic_cache",
                    "cache_hit": False,
                    "original_query": req.q,
                    "effective_query": effective_query,
                },
            )

    if not active_file_ids:
        answer = "No indexed files are currently available. Please upload a document first."
        save_message_to_session(
            session_id=session_id,
            role="assistant",
            content=answer,
            owner_id=owner_id,
            sources={"docs": []},
        )
        return AnswerResponse(
            answer=answer,
            sources=[],
            confidence=0.0,
            processing_time_ms=int((time.perf_counter() - start) * 1000),
            session_id=session_id,
           )

    # ── Cross-Cutting Analytical Router ──
    # "What are common root causes across all tickets?" / "What process
    # improvements were recommended most frequently?" look like aggregation
    # to the SQL classifier but need synthesis across ticket content, not a
    # list of IDs. Detect analytical intent with multi-signal categories
    # (pattern + insight + scope + frequency + synthesis) and skip the
    # aggregation fast-path, forcing the agent pipeline below. Fails closed:
    # any query that doesn't clear the confidence threshold drops through
    # the existing agg_classifier unchanged.
    force_agent_mode = False
    analytical_mode = None
    if getattr(settings, "CROSS_CUTTING_ANALYTICAL_ROUTING_ENABLED", True):
        _cc = detect_cross_cutting_analytical(effective_query)
        if _cc.is_analytical:
            force_agent_mode = True
            analytical_mode = _cc.mode
            logger.info(
                "[cross_cutting] analytical=True conf=%.2f mode=%s "
                "categories=%s signals=%s",
                _cc.confidence, _cc.mode,
                _cc.matched_categories, _cc.signals,
            )
        else:
            logger.info(
                "[cross_cutting] non-analytical conf=%.2f (below threshold)",
                _cc.confidence,
            )

    # ── Fix 6: SQL aggregation fast-path ──
    # "How many Nebula-Corp tickets?" / "Which tickets missed SLA?" are set
    # operations, not semantic search. If the query parses as an aggregation
    # AND at least one ticket is in scope, answer directly from JSONB — no
    # embeddings, no LLM. Falls through to full RAG when the detector misses
    # or the SQL returns nothing.
    # Goal 4: skip aggregation when the query names a specific INC-\d+ ticket —
    # those belong to retrieval / ticket_id_exact, never to bulk aggregation.
    # Cross-cutting: also skip when analytical intent was detected above;
    # force_agent_mode routes directly to the agent pipeline for synthesis.
    if _parallel_path_used:
        # Reuse the classifier verdict produced concurrently with the rewriter.
        # If effective_query mentions a specific INC-\d+, drop the verdict so
        # identifier_exact retrieval handles it (matches sequential behavior).
        if re.search(r"\bINC-\d+\b", effective_query, re.IGNORECASE) or force_agent_mode:
            _agg_intent = None
        else:
            _agg_intent = _parallel_agg_intent
    else:
        _agg_intent = (
            detect_aggregation_intent_v2(effective_query)
            if (
                not re.search(r"\bINC-\d+\b", effective_query, re.IGNORECASE)
                and not force_agent_mode
            )
            else None
        )
    if _agg_intent:
        _ticket_ids_in_scope = {
            f["id"] for f in list_active_files_all()
            if f.get("id") in active_file_ids
            and str(f.get("file_type", "")).lower() in {"ticket", "tickets", "incident"}
        }
        if _ticket_ids_in_scope:
            _agg_result = run_aggregation(_agg_intent, owner_id, _ticket_ids_in_scope)
            if _agg_result.count > 0:
                logger.info("[aggregation] SQL fast-path: %d results", _agg_result.count)
                save_message_to_session(
                    session_id=session_id,
                    role="assistant",
                    content=_agg_result.prose_summary,
                    owner_id=owner_id,
                    sources={"docs": []},
                )
                return AnswerResponse(
                    answer=_agg_result.prose_summary,
                    sources=[],
                    confidence=1.0,
                    processing_time_ms=int((time.perf_counter() - start) * 1000),
                    session_id=session_id,
                    context_stats={
                        "aggregation_fast_path": True,
                        "aggregation_operation": _agg_intent.operation,
                        "aggregation_count": _agg_result.count,
                        "aggregation_filters": _agg_result.filters_applied,
                        "cache_hit": False,
                        "original_query": req.q,
                        "effective_query": effective_query,
                        "query_rewrite_applied": rewrite_applied,
                        "query_rewrite_reason": rewrite_reason,
                        "query_rewrite_confidence": round(rewrite_confidence, 3),
                    },
                )

    # q_emb = safe_embed(req.q)
    # if not q_emb:
    #     raise HTTPException(500, "Embedding failed")

    # # ── Query Expansion: resolve abbreviations learned from documents ──
    # expanded = expand_query(req.q)
    # if expanded.acronyms_found:
    #     logger.info(
    #         "Query expansion: '%s' -> acronyms=%s",
    #         req.q, list(expanded.acronyms_found.keys()),
    #     )
    #     # Re-embed with expanded text for better semantic match
    #     expanded_emb = safe_embed(expanded.expanded_text)
    #     if expanded_emb:
    #         q_emb = expanded_emb
    
    # ── Follow-up enrichment: DEPRECATED by Brief 3 query rewriter. The
    # rewriter upstream already produces a self-contained effective_query,
    # so this call is a no-op returning its input. Kept for legacy reasons.
    _recent_for_followup = recent_msgs_all
    retrieval_query, _carried_entities = _enrich_query_with_history(
        effective_query, _recent_for_followup,
    )

    # ── Query Expansion: normalize + resolve abbreviations ──
    expanded = expand_query(retrieval_query)
    if expanded.acronyms_found:
        logger.info(
            "Query expansion: '%s' -> acronyms=%s",
            retrieval_query, list(expanded.acronyms_found.keys()),
        )

    # Always embed the expanded/normalized text (not raw query)
    # This handles: "toDC" → "to DC", underscores → spaces, acronym expansion
    embed_text = expanded.expanded_text
    if embed_text.lower() != retrieval_query.lower():
        logger.info(
            "Embedding normalized query: '%s' (original: '%s')",
            embed_text, retrieval_query,
        )

    q_emb = safe_embed(embed_text)
    if not q_emb:
        # Fallback: try the pre-expansion retrieval query if expanded fails.
        q_emb = safe_embed(retrieval_query)
    if not q_emb:
        raise HTTPException(500, "Embedding failed")

    # Sprint 3C — escalation mode routes retrieval to the contact_customer
    # corpus. resolve_mode_doc_kinds returns None when the 3C flag is off
    # or when the active mode is not "escalation", preserving the default
    # all-corpora behavior for every other code path.
    _sprint3c_doc_kinds: Optional[List[str]] = None
    try:
        _sprint3c_mode_snap = get_session_mode(session_id)
    except Exception as _sm3c_exc:
        logger.warning("[sprint3c] session_mode read failed: %s", _sm3c_exc)
        _sprint3c_mode_snap = None
    if _sprint3c_mode_snap is not None and getattr(
        _sprint3c_mode_snap, "is_valid", False
    ):
        _sprint3c_doc_kinds = resolve_mode_doc_kinds(_sprint3c_mode_snap)

    # Sprint 10 — request-level override for Stage 4 KB handoff. When the
    # client passes a non-empty `allowed_doc_kinds`, prefer it over the
    # mode-derived list. Empty / None falls through to Sprint 3C behaviour.
    _effective_doc_kinds = _sprint3c_doc_kinds
    if getattr(req, "allowed_doc_kinds", None):
        _effective_doc_kinds = list(req.allowed_doc_kinds)

    # Sprint 10 follow-up — persisted-session fallback (migration 050).
    # The frontend (useChatHandoff.js, Stage4SearchKBHandoff.js) only
    # attaches `allowed_doc_kinds` on the AUTO-FIRED first message of a
    # Search-in-KB chat. Subsequent user-typed turns in the same chat
    # session went through ChatArea with no `allowed_doc_kinds`, so the
    # backend was losing KB-search context after turn 1 — composer voice
    # degraded to "default" and the KB-Search system prompt
    # (backend/routing/kb_search_prompt.py) never activated.
    #
    # Read the value persisted by create_chat_session_with_handoff ONLY
    # when the request didn't carry an explicit override AND no session-
    # mode-derived value is present. Result: the chat session "remembers"
    # it's a KB-search session for the life of the row, no matter what
    # the frontend sends on follow-ups.
    if not _effective_doc_kinds and session_id:
        try:
            from sqlalchemy import text as _dk_text
            from backend.db.connection import engine as _dk_engine
            with _dk_engine.connect() as _conn:
                _row = _conn.execute(
                    _dk_text(
                        "SELECT allowed_doc_kinds FROM chat_sessions "
                        "WHERE id = :sid"
                    ),
                    {"sid": session_id},
                ).mappings().first()
            _persisted = _row.get("allowed_doc_kinds") if _row else None
            if isinstance(_persisted, list) and _persisted:
                _effective_doc_kinds = [str(k) for k in _persisted if k]
                logger.info(
                    "[doc_kinds] session=%s using persisted allowed_doc_kinds=%s",
                    session_id, _effective_doc_kinds,
                )
        except Exception as _dk_exc:
            # Safe-by-default: a failure here just means the chat falls
            # back to global retrieval. NEVER let a column-missing /
            # malformed-JSON error break /ask.
            logger.warning(
                "[doc_kinds] persisted-session lookup failed for chat=%s: %s",
                session_id, _dk_exc,
            )

    # ── Sprint 11 — journey-aware retrieval enrichment ──
    # When this chat session originated from a Stage 0 / Stage 3
    # "Ask in chat" link (Sprint 11 surfaces) the chat session metadata
    # carries journey_session_id. Look up the journey's intake context
    # and prepend a short "[Context: asset=… alert=… customer=… …]"
    # prefix to the BM25 / keyword query so retrieval narrows toward
    # the right corner of the corpus.
    #
    # We DO NOT touch:
    #   - raw_query   → orchestrator uses for identifier extraction
    #     (regex on user text); enrichment would create false IDs
    #   - q_emb       → vector channel keeps working off the original
    #     question's embedding; adding context would require a second
    #     Titan call we don't yet need to spend
    #   - LLM prompt + saved chat message → unchanged, so the user
    #     sees their bare question both in the chat thread and in any
    #     downstream answer rendering
    #
    # Failure-open: helper returns None on any DB / parsing error and
    # /ask continues with the unenriched query. Flag-gated; defaults
    # ON (LOGIQ_JOURNEY_CHAT_RETRIEVAL_CONTEXT in config.py).
    bm25_query_text = expanded.expanded_text
    if getattr(settings, "LOGIQ_JOURNEY_CHAT_RETRIEVAL_CONTEXT", True):
        from backend.db.connection import SessionLocal as _SL
        _journey_ctx = _load_journey_ctx(
            chat_session_id=session_id, db_session_factory=_SL,
        )
        _ctx_prefix = _journey_ctx_prefix(_journey_ctx)
        if _ctx_prefix:
            bm25_query_text = _ctx_prefix + bm25_query_text
            logger.info(
                "[journey_context] enriched chat=%s journey=%s fields=%s",
                session_id,
                (_journey_ctx or {}).get("_journey_session_id"),
                sorted(k for k in (_journey_ctx or {}).keys() if not k.startswith("_")),
            )

    # ── Sprint 12.1 — per-bullet "Ask in Chat" scope lookup ──
    # When the chat session was created via the Stage 0 per-bullet
    # handoff (Stage0BestTicketDistillation.js → /search-kb-handoff),
    # chat_sessions.scope_incident_id holds the source ticket the
    # bullet was tagged with. Honor that scope by passing it to the
    # orchestrator so retrieval is filtered to that one ticket's
    # chunks. NULL = global Search-in-KB behavior (unchanged path).
    #
    # Failure-open: any read error leaves scope as None — chat falls
    # back to today's behavior rather than breaking the demo.
    _scope_incident_ids = None
    try:
        from sqlalchemy import text as _scope_sql_text
        from backend.db.connection import engine as _scope_engine
        with _scope_engine.connect() as _conn:
            _row = _conn.execute(
                _scope_sql_text(
                    "SELECT scope_incident_id FROM chat_sessions WHERE id = :sid"
                ),
                {"sid": session_id},
            ).mappings().first()
        if _row and _row.get("scope_incident_id"):
            _scope_incident_ids = [str(_row["scope_incident_id"]).strip()]
            logger.info(
                "[scope] chat=%s scoped to incident=%s",
                session_id, _scope_incident_ids[0],
            )
    except Exception as _scope_exc:
        logger.warning(
            "[scope] lookup failed for chat=%s err=%s — falling back to global",
            session_id, _scope_exc,
        )

    # ── Sprint 12.2 — scoped vs global retrieval branch ──
    # When the chat session is scoped to a single Incident_Number,
    # delegate to the dedicated scoped-retrieval module. Otherwise
    # take the original global hybrid path. The two paths are
    # entirely separate code routes; the global orchestrator is
    # never aware of scope, and the scoped module never touches
    # BM25 / fusion / rerank. This eliminates the "filter-too-late"
    # empty-result class of bug while keeping Search-in-KB
    # byte-identical to its pre-scope behavior.
    if _scope_incident_ids:
        # ── Sprint 13.15 — JSON-first scoped chat ──
        # When a ticket is in scope, try to answer from its FULL
        # structured JSON before falling back to chunk retrieval.
        # Engineers ask in natural language ("affected assets", not
        # "Affected_Assets"); the LLM maps phrasings → JSON fields
        # semantically with the entire ticket as context. Cost is
        # trivial (~3 K input tokens / question) because Sprint 11's
        # full-fidelity ingest already stamps the entire source
        # ticket on every chunk's metadata_json. Failure-open: any
        # error or empty answer falls through to today's chunk
        # path below. Search-KB / unscoped flows are untouched.
        try:
            from backend.retrieval.scoped_full_ticket import (
                build_source_for_full_ticket,
                fetch_full_ticket_json,
                synthesize_scoped_answer,
            )
            _full_record = fetch_full_ticket_json(_scope_incident_ids[0])
            if _full_record is not None:
                _json_answer = synthesize_scoped_answer(req.q, _full_record)
                if _json_answer:
                    # Persistence shape (rich dicts) vs response shape
                    # (flat string list) — AnswerResponse.sources is
                    # typed as List[str], so we pass just the file
                    # name. The saved chat_messages.sources keeps the
                    # rich dict so the frontend's source-detail
                    # expansion still has all metadata.
                    _json_sources_rich = build_source_for_full_ticket(_full_record)
                    _json_sources_for_response = [_full_record.document_name]
                    save_message_to_session(
                        session_id=session_id,
                        role="assistant",
                        content=_json_answer,
                        owner_id=owner_id,
                        sources={"docs": _json_sources_rich},
                    )
                    logger.info(
                        "[scope_json] chat=%s scope=%s — JSON-first "
                        "answer served (chars=%d, source=%s)",
                        session_id,
                        _scope_incident_ids[0],
                        len(_json_answer),
                        _full_record.document_name,
                    )
                    return AnswerResponse(
                        answer=_json_answer,
                        sources=_json_sources_for_response,
                        confidence=1.0,
                        processing_time_ms=int((time.perf_counter() - start) * 1000),
                        session_id=session_id,
                        context_stats={
                            "scope_incident_id": _scope_incident_ids[0],
                            "scoped_path": "json_first",
                            "scoped_source_document": _full_record.document_name,
                            "cache_hit": False,
                        },
                    )
        except Exception as _scope_json_exc:
            logger.warning(
                "[scope_json] JSON-first synthesis failed for chat=%s "
                "scope=%s err=%s — falling back to chunk path",
                session_id, _scope_incident_ids[0], _scope_json_exc,
            )

        # Fallback (or default when JSON synthesis was empty/None):
        # today's chunk-based scoped retrieval. Same behaviour as
        # before Sprint 13.15 from this point onward.
        from backend.retrieval.scoped_retrieval import retrieve_within_incident
        retrieval = retrieve_within_incident(
            scope_incident_id=_scope_incident_ids[0],
            query_embedding=q_emb,
            allowed_file_ids=active_file_ids,
        )
    else:
        retrieval = orchestrator_retrieve(
            query=bm25_query_text,
            raw_query=req.q,
            query_embedding=q_emb,
            owner_id=owner_id,
            allowed_file_ids=active_file_ids,
            file_type="kb",
            generate_fn=safe_generate,
            bm25_search_fn=bm25.search if bm25 and bm25.size > 0 else None,
            vector_search_fn=pgvector_search,
            doc_kinds=_effective_doc_kinds,
        )

    # ── Clean "not found" short-circuit when an identifier was asked for
    # but no such record is indexed. The retrieval layer signals this via
    # search_mode in {"identifier_not_found", "ticket_id_not_found"}; we
    # short-circuit here — BEFORE validator, agents, and model routing —
    # so nothing downstream can override the signal and surface the canned
    # false-refusal message.
    if retrieval.stats.get("search_mode") in (
        "identifier_not_found",
        "ticket_id_not_found",
    ):
        requested_pairs = retrieval.stats.get("requested_identifiers") or []
        requested = [cid for (cid, _itype) in requested_pairs]
        if not requested:
            requested = retrieval.stats.get("requested_ticket_ids") or []
        if not requested and retrieval.stats.get("ticket_id"):
            requested = [retrieval.stats["ticket_id"]]
        ids_str = ", ".join(requested) if requested else "The requested record"
        answer = (
            f"{ids_str} was not found in the indexed documents. "
            "If you expected this record to be available, please upload "
            "the corresponding data."
        )
        save_message_to_session(
            session_id=session_id,
            role="assistant",
            content=answer,
            owner_id=owner_id,
            sources={"docs": []},
        )
        return AnswerResponse(
            answer=answer,
            sources=[],
            confidence=1.0,
            processing_time_ms=int((time.perf_counter() - start) * 1000),
            session_id=session_id,
            context_stats={
                "ticket_id_not_found": True,
                "identifier_not_found": True,
                "requested_ticket_ids": requested,
                "cache_hit": False,
                **retrieval.stats,
            },
        )

    # ── Sprint 12.1 — Scoped chat polite no-answer short-circuit ──
    # When this chat session is scoped to a single Incident_Number
    # (per-bullet "Ask in Chat" handoff from Stage 0) and retrieval
    # returns zero chunks from that ticket for the user's question,
    # the user wants a natural, polite acknowledgement — NOT a
    # general-knowledge answer (the corpus is the source of truth)
    # and NOT a generic "no results" string.
    #
    # We call Claude Haiku via safe_generate with a tight prompt
    # that:
    #   - Names the source incident explicitly
    #   - States the question is not covered in that ticket's data
    #   - Suggests Search-in-KB as the next step
    #   - Forbids answering from general knowledge
    #
    # Failure-open: any LLM error falls through to a deterministic
    # template so the chat still responds politely. Demo-day safety.
    if _scope_incident_ids and not retrieval.ranked:
        _scoped_id = _scope_incident_ids[0]
        # Conversational chat reply — not a letter. The previous prompt
        # was producing "Dear Engineer / Best regards / [Your Name]"
        # signature blocks because the framing implied formal
        # correspondence. This prompt explicitly forbids any salutation,
        # signoff, or signature placeholder so the output reads like a
        # natural chat message.
        _polite_prompt = (
            f"You are a chat assistant. The current chat is restricted "
            f"to one source ticket only: {_scoped_id}. A search across "
            f"that ticket's content returned no relevant passages for "
            f"the user's latest message.\n\n"
            f"User's message:\n{req.q}\n\n"
            f"Reply with ONE short, plain-text chat sentence (max two "
            f"sentences) that:\n"
            f"- mentions {_scoped_id} by name,\n"
            f"- says the answer wasn't found in that ticket,\n"
            f"- suggests using Search KB for a wider lookup.\n\n"
            f"Hard rules:\n"
            f"- Plain conversational prose, like a chat reply.\n"
            f"- No salutation (no 'Dear ...', no 'Hi ...').\n"
            f"- No signoff (no 'Best regards', no 'Thanks', no '[Your Name]', "
            f"no '[Your Role]').\n"
            f"- No bullet points, no headings, no markdown.\n"
            f"- Do NOT answer from general knowledge or from other tickets.\n"
            f"- Do NOT invent commands or configuration values.\n\n"
            f"Reply:"
        )
        try:
            _polite_answer = safe_generate(_polite_prompt, max_tokens=160).strip()
            # Defensive scrub — strip residual letter framing if the
            # model still slips one in. Cheap belt-and-suspenders.
            for _bad in (
                "Dear Engineer,", "Dear Engineer", "Dear engineer,",
                "Best regards,", "Best regards", "Sincerely,",
                "[Your Name]", "[Your Role]", "[Team]",
            ):
                _polite_answer = _polite_answer.replace(_bad, "").strip()
            if not _polite_answer:
                raise RuntimeError("empty LLM response after scrub")
        except Exception as _polite_exc:
            logger.warning(
                "[scope_no_answer] LLM polite reply failed for chat=%s "
                "scope=%s err=%s — using template fallback",
                session_id, _scoped_id, _polite_exc,
            )
            _polite_answer = (
                f"I couldn't find that in {_scoped_id}. Try Search KB to "
                f"look across the wider knowledge base."
            )
        save_message_to_session(
            session_id=session_id,
            role="assistant",
            content=_polite_answer,
            owner_id=owner_id,
            sources={"docs": []},
        )
        logger.info(
            "[scope_no_answer] chat=%s scope=%s — returned polite redirect",
            session_id, _scoped_id,
        )
        return AnswerResponse(
            answer=_polite_answer,
            sources=[],
            confidence=1.0,
            processing_time_ms=int((time.perf_counter() - start) * 1000),
            session_id=session_id,
            context_stats={
                "scoped_no_answer": True,
                "scope_incident_id": _scoped_id,
                "cache_hit": False,
                **retrieval.stats,
            },
        )

    # ── Fallback: try variant queries if primary retrieval found nothing ──
    # if not retrieval.ranked and len(expanded.variants) > 1:
    #     logger.info("Primary retrieval empty — trying %d variant queries", len(expanded.variants) - 1)
    #     for variant in expanded.variants[1:]:
    #         v_emb = safe_embed(variant)
    #         if not v_emb:
    #             continue
    #         variant_retrieval = orchestrator_retrieve(
    #             query=variant,
    #             query_embedding=v_emb,
    #             owner_id=owner_id,
    #             allowed_file_ids=active_file_ids,
    #             file_type="kb",
    #             generate_fn=safe_generate,
    #             bm25_search_fn=bm25.search if bm25 and bm25.size > 0 else None,
    #             vector_search_fn=pgvector_search,
    #         )
    #         if variant_retrieval.ranked:
    #             retrieval = variant_retrieval
    #             logger.info("Variant query succeeded: '%s'", variant[:80])
    #             break
    # ── Fallback: try variant queries if primary retrieval is empty OR weak ──
    # "Weak" = retrieval returned results but document support check fails,
    # meaning the chunks don't actually contain the query's key terms.
    # This catches cases like "toDC" where normalization ("to DC") would
    # retrieve from the correct document.
    #
    # Sprint 12.2 fix — variant retry calls the *unscoped* global
    # orchestrator, which would silently overwrite `retrieval` with
    # chunks from any ticket in the corpus. For a scope-locked chat
    # that's a data breach (chat answers leaking from other tickets),
    # so we skip variant retry entirely when scope is set. The
    # in-scope cosine ranking from `retrieve_within_incident` is the
    # only retrieval result this chat is allowed to use.
    should_try_variants = (
        not _scope_incident_ids
        and len(expanded.variants) > 1
        and (
            not retrieval.ranked
            or not has_sufficient_document_support(
                effective_query, retrieval.ranked,
                expanded_keywords=expanded.expanded_keywords,
            )
        )
    )
    if should_try_variants:
        logger.info(
            "Trying %d variant queries (primary %s)",
            len(expanded.variants) - 1,
            "empty" if not retrieval.ranked else "weak support",
        )
        for variant in expanded.variants[1:]:
            v_emb = safe_embed(variant)
            if not v_emb:
                continue
            variant_retrieval = orchestrator_retrieve(
                query=variant,
                query_embedding=v_emb,
                owner_id=owner_id,
                allowed_file_ids=active_file_ids,
                file_type="kb",
                generate_fn=safe_generate,
                bm25_search_fn=bm25.search if bm25 and bm25.size > 0 else None,
                vector_search_fn=pgvector_search,
            )
            if variant_retrieval.ranked:
                # Check if variant results are better than what we have
                variant_support = has_sufficient_document_support(
                    variant, variant_retrieval.ranked,
                    expanded_keywords=expanded.expanded_keywords,
                )
                if variant_support or not retrieval.ranked:
                    retrieval = variant_retrieval
                    logger.info("Variant query improved results: '%s'", variant[:80])
                    break

    # ── Empty-result fallback chain ──────────────────────────────────
    # Last line of defence against hallucinations: if BOTH the primary
    # retrieve AND the glossary-variant retry returned zero chunks, the
    # next code paths would feed an empty context to the LLM and we'd
    # get a confidently-wrong answer. The fallback module runs:
    #   1. Haiku-generated semantic rewrites + orchestrator retry
    #   2. BM25-only sweep on the original query
    #   3. Marks the result as "empty_after_fallback" so the short-
    #      circuit handler below returns a canned decline message
    #      instead of calling the LLM.
    #
    # Gated by settings.ENABLE_EMPTY_RESULT_FALLBACK and disabled
    # automatically when the chat is scope-locked to a single ticket
    # (the existing scoped polite-no-answer path below handles that).
    if (
        settings.ENABLE_EMPTY_RESULT_FALLBACK
        and not _scope_incident_ids
        and not retrieval.ranked
    ):
        from backend.retrieval.empty_result_fallback import attempt_fallback
        retrieval = attempt_fallback(
            original_query=effective_query,
            original_retrieval=retrieval,
            retrieve_fn=orchestrator_retrieve,
            embed_fn=safe_embed,
            bm25_search_fn=(bm25.search if bm25 and bm25.size > 0 else None),
            owner_id=owner_id,
            allowed_file_ids=active_file_ids,
            file_type="kb",
            generate_fn=safe_generate,
            vector_search_fn=pgvector_search,
            doc_kinds=_effective_doc_kinds,
        )

    # ── Graceful decline short-circuit ───────────────────────────────
    # Fires only when the fallback chain explicitly gave up. NEVER
    # falls through to the LLM with empty context.
    from backend.retrieval.empty_result_fallback import (
        SEARCH_MODE_EMPTY_AFTER_FALLBACK,
    )
    if retrieval.stats.get("search_mode") == SEARCH_MODE_EMPTY_AFTER_FALLBACK:
        decline_msg = settings.EMPTY_FALLBACK_DECLINE_MESSAGE
        logger.info(
            "[fallback_chain] declining gracefully — query=%r "
            "rewrites_tried=%d bm25_tried=%s",
            (effective_query or req.q)[:120],
            retrieval.stats.get("fallback_rewrites_tried", 0),
            retrieval.stats.get("fallback_bm25_tried", False),
        )
        save_message_to_session(
            session_id=session_id,
            role="assistant",
            content=decline_msg,
            owner_id=owner_id,
            sources={"docs": []},
        )
        return AnswerResponse(
            answer=decline_msg,
            sources=[],
            confidence=1.0,
            processing_time_ms=int((time.perf_counter() - start) * 1000),
            session_id=session_id,
            context_stats={
                "empty_after_fallback": True,
                "cache_hit": False,
                **retrieval.stats,
            },
        )

    doc_ranked_original = retrieval.ranked
    # ── Chunk limiter: cap chunks before context/LLM/agents ──
    _limit = limit_chunks(doc_ranked_original, max_chunks=DEFAULT_MAX_CHUNKS)
    doc_ranked = _limit.limited
    chunk_original_count = _limit.original_count
    chunk_limited_count = _limit.limited_count
    chunk_limit_applied = _limit.applied

    # ── Stage enforcement (filter by stage + track unresolved follow-ups) ──
    # Intent detected on the effective (rewritten) query so pronoun-style
    # follow-ups route to the same stage as the resolved subject.
    _pre_intent = detect_intent(effective_query)
    stage_norm = normalize_stage(req.stage)
    try:
        stage_result = enforce_stage(
            stage=stage_norm,
            session_id=session_id,
            ranked_chunks=doc_ranked,
            intent_name=_pre_intent.intent,
        )
    except Exception as _stage_exc:
        logger.warning("Stage enforcer wrapper raised (%s) — passing through", _stage_exc)
        from backend.routing.stage_enforcer import StageResult as _SR
        stage_result = _SR(
            stage=stage_norm,
            filtered_chunks=doc_ranked,
            original_chunk_count=len(doc_ranked),
            filtered_chunk_count=len(doc_ranked),
            notes=[f"stage_wrapper_error: {_stage_exc}"],
        )
    doc_ranked = stage_result.filtered_chunks

    # ── Interactive Clarifier — post-retrieval, pre-agent-pipeline ──
    if getattr(settings, "INTERACTIVE_CLARIFIER_ENABLED", False) and doc_ranked:
        try:
            from backend.services.interactive_clarifier import try_clarify as _try_clarify
            _triage_conf = None
            try:
                if _triage_result is not None and getattr(_triage_result, "is_valid", False):
                    _triage_conf = float(getattr(_triage_result, "confidence", 0.0) or 0.0)
            except Exception:
                _triage_conf = None
            # Sprint 2.7 Bug C — pass clarification-chain count so Clarifier
            # caps the loop at one round per logical question. Count=1 when
            # this /ask is the refined submission following a prior pick.
            _session_clarif_count = 1 if _pattern_is_clarifier_refined else 0
            _clarify_res = _try_clarify(
                query=effective_query,
                ranked_chunks=doc_ranked,
                triage_confidence=_triage_conf,
                carried_identifiers=list(_carried_entities or []),
                session_id=session_id,
                recent_messages=recent_msgs_all,
                session_clarif_count=_session_clarif_count,
            )
            if _clarify_res.needs_clarification:
                _clarif_id = _store_clarification(
                    session_id=session_id,
                    options=[
                        {
                            "id": o.id,
                            "label": o.label,
                            "refined_query": o.refined_query,
                            "record_ref": o.record_ref,
                        }
                        for o in _clarify_res.options
                    ],
                    original_query=req.q,
                )
                _clarif_dtos = [
                    ClarificationOptionDTO(
                        id=o.id,
                        label=o.label,
                        refined_query=o.refined_query,
                        record_ref=o.record_ref,
                    )
                    for o in _clarify_res.options
                ]
                _clarif_assistant_text = (
                    _clarify_res.context_summary
                    or "Can you clarify which of these you meant?"
                )
                save_message_to_session(
                    session_id=session_id,
                    role="assistant",
                    content=_clarif_assistant_text,
                    owner_id=owner_id,
                    sources={
                        "docs": [],
                        "clarification_presented": True,
                        "clarification_id": _clarif_id,
                        "ambiguity_score": _clarify_res.ambiguity_score,
                    },
                )
                logger.info(
                    "[interactive_clarifier] returning clarification_id=%s opts=%d",
                    _clarif_id, len(_clarif_dtos),
                )
                return AnswerResponse(
                    answer=_clarif_assistant_text,
                    sources=[],
                    confidence=0.0,
                    processing_time_ms=int((time.perf_counter() - start) * 1000),
                    session_id=session_id,
                    context_stats={
                        "clarification_presented": True,
                        "clarification_id": _clarif_id,
                        "ambiguity_score": _clarify_res.ambiguity_score,
                        "cache_hit": False,
                    },
                    needs_clarification=True,
                    clarification_id=_clarif_id,
                    clarification_options=_clarif_dtos,
                    clarification_context=_clarify_res.context_summary or None,
                )
            else:
                logger.info(
                    "[interactive_clarifier] no clarification skip=%s score=%.2f",
                    _clarify_res.skip_reason or "n/a", _clarify_res.ambiguity_score,
                )
        except Exception as _clarify_trigger_exc:
            logger.warning(
                "[interactive_clarifier] trigger failed (%s) - continuing normally",
                _clarify_trigger_exc,
            )

    max_ctx_chars = TOKEN_BUDGET["MAX_LOG_CONTEXT_CHARS"] + TOKEN_BUDGET["MAX_KB_CONTEXT_CHARS"]

    # Brief 4 / Opt 5: compress single identifier_exact chunks down to just
    # the sections the query asks about. Safe — non-ticket chunks and broad
    # queries pass through unchanged; only applies when exactly one chunk
    # came back from the identifier_exact short-circuit.
    if (
        getattr(settings, "CONTEXT_COMPRESSION_ENABLED", False)
        and len(doc_ranked) == 1
        and retrieval.stats.get("search_mode") == "identifier_exact"
    ):
        from backend.retrieval.context_builder import compress_chunk_for_query
        _cid, _ctext, _cmeta, _cscore = doc_ranked[0]
        _compressed, _was = compress_chunk_for_query(_ctext, effective_query)
        if _was:
            logger.info(
                "[compress] chunk reduced from %d to %d chars (%d%%) query=%r",
                len(_ctext), len(_compressed),
                int(100 * len(_compressed) / max(1, len(_ctext))),
                effective_query[:60],
            )
            doc_ranked = [(_cid, _compressed, _cmeta, _cscore)]

    doc_ctx, doc_src = assemble_context(doc_ranked, max_ctx_chars)

    if not doc_ctx or not has_sufficient_document_support(effective_query, doc_ranked, expanded_keywords=expanded.expanded_keywords):
        # Sprint 12.2 fix — for scope-locked chats the conversational
        # fallback below is unsafe: `run_conversational_fallback` is
        # given NO ticket context and would emit a free-form LLM
        # answer drawn from training data only (the breach +
        # hallucination the engineer is reporting). When this chat
        # is locked to one incident, never invoke it; redirect to
        # the same polite no-answer path the empty-retrieval branch
        # uses above so the reply stays grounded in the scope.
        if _scope_incident_ids:
            _scoped_id = _scope_incident_ids[0]
            _polite_prompt = (
                f"You are a chat assistant. The current chat is restricted "
                f"to one source ticket only: {_scoped_id}. The retrieved "
                f"passages from that ticket do not contain enough relevant "
                f"information to answer the user's latest message.\n\n"
                f"User's message:\n{req.q}\n\n"
                f"Reply with ONE short, plain-text chat sentence (max two "
                f"sentences) that:\n"
                f"- mentions {_scoped_id} by name,\n"
                f"- says the answer wasn't found in that ticket,\n"
                f"- suggests using Search KB for a wider lookup.\n\n"
                f"Hard rules:\n"
                f"- Plain conversational prose, like a chat reply.\n"
                f"- No salutation (no 'Dear ...', no 'Hi ...').\n"
                f"- No signoff (no 'Best regards', no 'Thanks', no '[Your Name]', "
                f"no '[Your Role]').\n"
                f"- No bullet points, no headings, no markdown.\n"
                f"- Do NOT answer from general knowledge or from other tickets.\n"
                f"- Do NOT invent commands or configuration values.\n\n"
                f"Reply:"
            )
            try:
                _polite_answer = safe_generate(_polite_prompt, max_tokens=160).strip()
                for _bad in (
                    "Dear Engineer,", "Dear Engineer", "Dear engineer,",
                    "Best regards,", "Best regards", "Sincerely,",
                    "[Your Name]", "[Your Role]", "[Team]",
                ):
                    _polite_answer = _polite_answer.replace(_bad, "").strip()
                if not _polite_answer:
                    raise RuntimeError("empty LLM response after scrub")
            except Exception as _polite_exc:
                logger.warning(
                    "[scope_no_answer] LLM polite reply failed (insufficient "
                    "support) chat=%s scope=%s err=%s — using template fallback",
                    session_id, _scoped_id, _polite_exc,
                )
                _polite_answer = (
                    f"I couldn't find that in {_scoped_id}. Try Search KB to "
                    f"look across the wider knowledge base."
                )
            save_message_to_session(
                session_id=session_id,
                role="assistant",
                content=_polite_answer,
                owner_id=owner_id,
                sources={"docs": []},
            )
            logger.info(
                "[scope_no_answer] chat=%s scope=%s — insufficient support, "
                "returned polite redirect (suppressed unscoped fallback)",
                session_id, _scoped_id,
            )
            return AnswerResponse(
                answer=_polite_answer,
                sources=[],
                confidence=1.0,
                processing_time_ms=int((time.perf_counter() - start) * 1000),
                session_id=session_id,
                context_stats={
                    "scoped_no_answer": True,
                    "scoped_no_answer_reason": "insufficient_support",
                    "scope_incident_id": _scoped_id,
                    "cache_hit": False,
                    **retrieval.stats,
                },
            )

        # No supporting docs — run the conversational fallback (Clarifier +
        # support-engineer-style reply). Never raises; legacy text on failure.
        from backend.agents.fallback_responder import run_conversational_fallback
        _fallback_prior = recent_msgs_all
        fb = run_conversational_fallback(
            query=effective_query,
            source_names=list(retrieval.stats.get("doc_sources", []) or []),
            generate_fn=safe_generate,
            bedrock_client=bedrock,
            prior_messages=_fallback_prior,
        )
        answer = fb.answer
        save_message_to_session(
            session_id=session_id,
            role="assistant",
            content=answer,
            owner_id=owner_id,
            sources={"docs": []},
        )
        return AnswerResponse(
            answer=answer,
            sources=[],
            confidence=0.0,
            processing_time_ms=int((time.perf_counter() - start) * 1000),
            session_id=session_id,
            context_stats={
                "active_file_count": len(active_file_ids),
                "grounded_answer": False,
                "fallback_mode": fb.mode,
                "clarifying": fb.clarifying,
                **retrieval.stats,
            },
        )

    retrieval_confidence = min(0.3 + len(doc_ranked) * 0.1, 1.0)
    recent_msgs = recent_msgs_all

    # ── Layer 3 Pattern Analytics (selective, gated, cache-aware) ──
    # Runs only when the classifier deems the query benefits from historical
    # pattern insights. Every integration point is try/except wrapped so that
    # any failure here leaves the standard response pipeline untouched.
    pattern_context: Optional[Dict[str, Any]] = None
    # Sprint 2 — wire the active session mode into pattern analytics so
    # PATTERN_ANALYTICS_FORCE_ENABLE_IN_TROUBLESHOOTING_MODE can trigger.
    _pattern_session_mode: Optional[str] = None
    try:
        _mode_snap = get_session_mode(session_id)
        if getattr(_mode_snap, "is_valid", False):
            _pattern_session_mode = _mode_snap.selected_mode
    except Exception as _mode_exc:
        logger.warning("[session_mode] read failed in pattern path: %s", _mode_exc)
        _pattern_session_mode = None
    try:
        from backend.services.pattern_analytics import enrich_if_needed as _pattern_enrich
        from backend.db.queries import load_similar_tickets_for_topic as _load_similar

        pattern_context = _pattern_enrich(
            query=effective_query,
            retrieved_chunks=doc_ranked,
            session_mode=_pattern_session_mode,
            is_clarifier_refined=_pattern_is_clarifier_refined,
            organization_id=owner_id,
            similar_tickets_loader=_load_similar,
        )
    except Exception as _pa_exc:
        logger.warning(
            "[pattern_analytics] enrichment failed: %s (continuing without)", _pa_exc,
        )
        pattern_context = None

    # ── Complexity classification (always cheap; no LLM call) ──
    complexity = classify_complexity(
        query=effective_query,
        ranked_chunks=doc_ranked,
        retrieval_confidence=retrieval_confidence,
        source_count=len(set(doc_src)),
        context_chars=len(doc_ctx),
    )

    # Brief 4 / Opt 3: high-confidence LLM triage overrides complexity tier
    # (used by resolve_mode below). We keep the heuristic score because it
    # drives the numeric threshold comparison; the tier is the categorical
    # gate. Only override when confidence >= 0.7 AND verdicts disagree.
    if (
        _triage_result is not None
        and getattr(_triage_result, "is_valid", False)
        and _triage_result.confidence >= 0.7
        and _triage_result.complexity
        and _triage_result.complexity != complexity.tier
    ):
        logger.info(
            "[triage] overriding complexity tier: heuristic=%s -> llm=%s (conf=%.2f)",
            complexity.tier, _triage_result.complexity, _triage_result.confidence,
        )
        complexity.tier = _triage_result.complexity
        if _triage_result.complexity == "complex":
            complexity.score = max(complexity.score, settings.AGENT_COMPLEXITY_THRESHOLD)
        elif _triage_result.complexity == "simple":
            complexity.score = min(complexity.score, settings.COMPLEXITY_SIMPLE_THRESHOLD)

    # Placeholder routing slot filled either by hybrid call or agent path
    routing = None
    agent_result = None
    requested_mode = req.mode or MODE_AUTO

    # ── Intent detection (additive; never overrides an explicit mode) ──
    intent_result = detect_intent(effective_query)
    effective_mode = requested_mode
    intent_upgrade = False
    if (
        requested_mode == MODE_AUTO
        and intent_result.matched
        and intent_result.suggested_mode
    ):
        effective_mode = intent_result.suggested_mode
        intent_upgrade = True
        logger.info(
            "Intent '%s' upgrading auto -> %s",
            intent_result.intent, intent_result.suggested_mode,
        )

    # Stage escalation wins over auto + intent (but not over explicit hybrid)
    stage_enforced_mode = stage_result.enforced_mode
    if stage_enforced_mode and requested_mode != MODE_HYBRID:
        effective_mode = stage_enforced_mode

    should_agent, agent_reason = resolve_mode(
        mode=effective_mode,
        query=effective_query,
        complexity_score=complexity.score,
        complexity_tier=complexity.tier,
        source_count=len(set(doc_src)),
    )
    if intent_upgrade:
        agent_reason = f"{agent_reason} [intent={intent_result.intent}]"
    if stage_enforced_mode and requested_mode != MODE_HYBRID:
        agent_reason = f"{agent_reason} [stage={stage_result.stage}:{stage_result.escalate_reason}]"

    # Sprint 12.2 fix — the agent pipeline builds a per-step retriever
    # from the *unscoped* `orchestrator_retrieve`, so any agent run
    # inside a scope-locked chat would issue follow-up retrievals that
    # span the entire corpus and re-introduce the leak this fix is
    # closing. Force the simple grounded path for scoped chats — the
    # ticket's chunks are a small, fully-known set; agent decomposition
    # adds no value here. Search-in-KB and other unscoped flows are
    # untouched.
    if _scope_incident_ids and (should_agent or force_agent_mode):
        logger.info(
            "[scope] suppressing agent pipeline for scoped chat=%s scope=%s "
            "(would have used: %s)",
            session_id, _scope_incident_ids[0], agent_reason,
        )
        should_agent = False
        force_agent_mode = False
        agent_reason = (
            f"{agent_reason} [scope_locked={_scope_incident_ids[0]} "
            f"agents_suppressed]"
        )

    if should_agent or force_agent_mode:
        if force_agent_mode and not should_agent:
            agent_reason = (
                f"{agent_reason} [cross_cutting=True mode={analytical_mode}]"
            )
        logger.info("Escalating to agent pipeline: %s", agent_reason)

        # ── Build the per-step hybrid retriever the Analyst will use ──
        try:
            step_retriever_fn = build_step_retriever(
                owner_id=owner_id,
                allowed_file_ids=active_file_ids,
                file_type="kb",
                embed_fn=safe_embed,
                bm25_search_fn=(bm25.search if bm25 and bm25.size > 0 else None),
                vector_search_fn=pgvector_search,
                generate_fn=safe_generate,
                retrieve_fn=orchestrator_retrieve,
                expand_query_fn=expand_query,
                assemble_context_fn=assemble_context,
                max_chunks=DEFAULT_MAX_CHUNKS,
                max_context_chars=TOKEN_BUDGET["MAX_KB_CONTEXT_CHARS"],
            )
        except Exception as _sr_exc:
            logger.warning("build_step_retriever failed (%s) — agents will use global context", _sr_exc)
            step_retriever_fn = None

        # Hotfix: derive session scope tokens from current query identifiers
        # so the Analyst's per-step retriever can anchor multi-step plans on
        # the same entities across steps instead of drifting.
        _session_scope: Optional[List[str]] = None
        try:
            from backend.retrieval.orchestrator import (
                _extract_identifiers as _hf_extract,
            )
            _session_scope = [
                cid for cid, _t in (_hf_extract(effective_query) or [])
            ]
        except Exception:
            _session_scope = None

        # Sprint 3A is disabled — session mode is no longer threaded into
        # the composer voice selector. The pipeline still accepts the
        # kwarg for backward compatibility; we pass None.
        _sprint3a_session_mode = None

        agent_result = run_agent_pipeline(
            query=effective_query,
            doc_context=doc_ctx,
            ranked_chunks=doc_ranked,
            source_names=doc_src,
            generate_fn=safe_generate,
            bedrock_client=bedrock,
            step_retriever_fn=step_retriever_fn,
            prior_messages=recent_msgs,
            stage=stage_result.stage,
            unresolved_count=stage_result.unresolved_count,
            pattern_context=pattern_context,
            session_scope=_session_scope,
            session_mode=_sprint3a_session_mode,
            # KB-Search composer voice activates only when
            # _effective_doc_kinds is a non-empty subset of {"kb","sop"}.
            # All other paths (no doc_kinds, ticket-only, mixed) leave the
            # composer on its default conversational voice.
            doc_kinds=_effective_doc_kinds,
        )
        raw_answer = agent_result.answer

        # If the agent answer is empty (pipeline hard-failed), fall back to hybrid
        if not (raw_answer or "").strip():
            logger.warning("Agent pipeline produced empty answer — falling back to hybrid")
            routing = route_and_generate(
                query=effective_query,
                doc_context=doc_ctx,
                ranked_chunks=doc_ranked,
                source_names=doc_src,
                retrieval_confidence=retrieval_confidence,
                recent_messages=recent_msgs,
                generate_fn=safe_generate,
                bedrock_client=bedrock,
                pattern_context=pattern_context,
                # KB-search system prompt addendum activates only when
                # _effective_doc_kinds is a non-empty subset of
                # {"kb","sop"}. All other call paths (no doc_kinds,
                # ticket-only, mixed) get the unchanged prompt.
                doc_kinds=_effective_doc_kinds,
            )
            raw_answer = routing.answer
    else:
        routing = route_and_generate(
            query=effective_query,
            doc_context=doc_ctx,
            ranked_chunks=doc_ranked,
            source_names=doc_src,
            retrieval_confidence=retrieval_confidence,
            recent_messages=recent_msgs,
            generate_fn=safe_generate,
            bedrock_client=bedrock,
            pattern_context=pattern_context,
            # See note above — doc_kinds drives the KB-search addendum,
            # nothing else is affected when it's None / contains
            # non-KB kinds.
            doc_kinds=_effective_doc_kinds,
        )
        raw_answer = routing.answer

    # Choose a model-used label for the validator: routing.model_used when hybrid ran,
    # else reflect the agent pipeline (agents internally use Sonnet + Haiku).
    _model_used_label = (
        routing.model_used if routing is not None
        else ("agents (sonnet+haiku)" if agent_result is not None else "unknown")
    )

    # Goal 2.3: bound retry that re-invokes hybrid generation with a stronger
    # extraction directive. Validator calls this once when it detects a false
    # refusal on deterministic retrieval modes.
    def _retry_generate(stronger_prompt: bool = False) -> str:
        if stronger_prompt:
            retry_query = (
                "The answer IS in the documents below. Read them carefully "
                "and extract it. Do not hedge or refuse.\n\n"
                f"{effective_query}"
            )
        else:
            retry_query = effective_query
        r = route_and_generate(
            query=retry_query,
            doc_context=doc_ctx,
            ranked_chunks=doc_ranked,
            source_names=doc_src,
            retrieval_confidence=retrieval_confidence,
            recent_messages=recent_msgs,
            generate_fn=safe_generate,
            bedrock_client=bedrock,
            pattern_context=pattern_context,
        )
        return r.answer

    validation = validate_answer(
        query=effective_query,
        answer=raw_answer,
        doc_context=doc_ctx,
        ranked_chunks=doc_ranked,
        source_names=doc_src,
        model_used=_model_used_label,
        retrieval_stats=retrieval.stats,
        retry_fn=_retry_generate,
    )
    answer = validation.answer
    confidence = validation.confidence

    # Brief 5 / Part 3 — Layer 3 output sanitizer (never blocks; only scrubs).
    output_sanitizer_issues: List[str] = []
    if settings.OUTPUT_SANITIZER_ENABLED:
        try:
            from backend.services.output_sanitizer import sanitize_output
            answer, output_sanitizer_issues = sanitize_output(answer, effective_query)
        except Exception as _san_exc:
            logger.warning("[sanitizer] wrapper raised (%s)", _san_exc)
            output_sanitizer_issues = []

    # ── Evidence check (metadata only; never mutates the answer) ──
    try:
        evidence = check_evidence(
            answer=answer,
            doc_context=doc_ctx,
            source_names=doc_src,
            confidence=confidence,
        )
    except Exception as _ev_exc:
        logger.warning("Evidence checker wrapper raised (%s)", _ev_exc)
        from backend.routing.evidence_checker import EvidenceResult as _ER
        evidence = _ER(weak=False, reasons=[f"wrapper_error: {_ev_exc}"])

    ms = int((time.perf_counter() - start) * 1000)

    sources_dict = {"docs": doc_src}
    save_message_to_session(
        session_id=session_id,
        role="assistant",
        content=answer,
        owner_id=owner_id,
        sources=sources_dict,
    )

    context_stats = {
        "active_file_count": len(active_file_ids),
        "doc_after_rerank": len(doc_ranked),
        "doc_context_chars": len(doc_ctx),
        **retrieval.stats,
        "model_used": _model_used_label,
        "model_reason": (routing.reason if routing is not None else "agent pipeline (agent-first auto)"),
        "complexity_score": complexity.score,
        "complexity_tier": complexity.tier,
        "generation_ms": (routing.generation_ms if routing is not None else 0),
        "hybrid_path_used": routing is not None,
        "step_retrieval_applied_count": (
            sum(1 for m in getattr(agent_result.steps[0], "step_retrieval_meta", []) if m.get("applied"))
            if (agent_result is not None and agent_result.steps) else 0
        ),
        "step_retrieval_total_steps": (
            len(getattr(agent_result.steps[0], "step_retrieval_meta", []))
            if (agent_result is not None and agent_result.steps) else 0
        ),
        "mode_requested": requested_mode,
        "mode_effective": effective_mode,
        "intent": intent_result.intent,
        "intent_matched": intent_result.matched,
        "intent_phrase": intent_result.matched_phrase,
        "intent_upgraded_mode": intent_upgrade,
        "agent_mode": agent_result.agent_mode if agent_result else False,
        "agent_reason": agent_reason,
        "agent_steps": len(agent_result.steps) if agent_result else 0,
        "agent_tokens": agent_result.total_tokens if agent_result else 0,
        "agent_ms": agent_result.total_ms if agent_result else 0,
        "validation_passed": validation.passed,
        "validation_confidence": round(validation.confidence, 3),
        "validation_modified": validation.was_modified,
        "validation_issues": len(validation.issues),
        "validation_ms": validation.validation_ms,
        "version_warning": bool(validation.version_warning),
        "trivial_short_circuit": False,
        "original_query": req.q,
        "effective_query": effective_query,
        "query_rewrite_applied": rewrite_applied,
        "query_rewrite_reason": rewrite_reason,
        "query_rewrite_confidence": round(rewrite_confidence, 3),
        "chunk_original_count": chunk_original_count,
        "chunk_limited_count": chunk_limited_count,
        "chunk_limit_applied": chunk_limit_applied,
        "chunk_max_allowed": DEFAULT_MAX_CHUNKS,
        "cache_hit": False,
        "cache_stored": False,
        "input_guard_flagged": False,
        "input_guard_reason": GUARD_REASON_OK,
        "stage": stage_result.stage,
        "stage_filter_applied": stage_result.filter_applied,
        "stage_original_chunks": stage_result.original_chunk_count,
        "stage_filtered_chunks": stage_result.filtered_chunk_count,
        "stage_enforced_mode": stage_result.enforced_mode,
        "stage_escalate_reason": stage_result.escalate_reason,
        "unresolved_count": stage_result.unresolved_count,
        "weak_evidence": evidence.weak,
        "weak_reasons": evidence.reasons,
        "evidence_answer_chars": evidence.answer_chars,
        "evidence_overlap_tokens": evidence.overlap_tokens,
        "evidence_source_count": evidence.source_count,
        "guardrail_blocked": False,
        "guardrail_pii_scrubbed": bool(_pii_scrubbed_query is not None),
        "guardrail_pii_matches": _pii_scrub_matches,
        "output_sanitizer_issues": output_sanitizer_issues,
    }

    # ── Sprint 2: surface pattern + context-break signals in context_stats ──
    try:
        if pattern_context:
            context_stats["pattern_active"] = bool(
                pattern_context.get("pattern_active", False)
            )
            context_stats["pattern_topic"] = pattern_context.get("pattern_topic") or ""
            context_stats["pattern_data"] = pattern_context.get("pattern_data") or {}
        else:
            context_stats["pattern_active"] = False
            context_stats["pattern_topic"] = ""
            context_stats["pattern_data"] = {}
    except Exception as _ps_exc:
        logger.warning("[pattern_analytics] stats surface failed: %s", _ps_exc)
        context_stats["pattern_active"] = False
        context_stats["pattern_topic"] = ""
        context_stats["pattern_data"] = {}

    if _context_break_hit is not None:
        context_stats["context_break"] = True
        context_stats["context_break_source"] = _context_break_hit.get("source", "")
        context_stats["context_break_category"] = _context_break_hit.get("category", "")
        context_stats["context_break_matched_phrase"] = (
            _context_break_hit.get("matched_phrase", "")
        )
        context_stats["context_break_active_mode"] = (
            _context_break_hit.get("active_mode") or ""
        )
        context_stats["context_break_active_sub_mode"] = (
            _context_break_hit.get("active_sub_mode") or ""
        )
    else:
        context_stats["context_break"] = False

    if _pattern_session_mode:
        context_stats["session_mode"] = _pattern_session_mode

    # ── Store in answer cache (fail-safe; never blocks response) ──
    # Cache-poisoning fix: do NOT cache an answer the validator replaced
    # with the canned safe-fallback (was_modified=True) or that failed
    # validation outright (validation.passed=False). Persisting either
    # serves the failure to every future user who asks a semantically
    # similar question, training them to thumbs-down (which routes
    # through run_kb_pivot_pipeline) to get a real answer. Only cache
    # answers the model produced AND the validator accepted.
    _cache_eligible = bool(
        getattr(validation, "passed", False)
        and not getattr(validation, "was_modified", False)
    )
    if _cache_eligible:
        try:
            stored = ANSWER_CACHE.put(
                query=effective_query,
                owner_id=owner_id,
                active_file_ids=active_file_ids,
                payload={
                    "answer": answer,
                    "sources": doc_src,
                    "confidence": confidence,
                    "context_stats": context_stats,
                },
            )
            context_stats["cache_stored"] = bool(stored)
        except Exception as _cache_exc:
            logger.warning("Answer cache put wrapper failed: %s", _cache_exc)
            context_stats["cache_stored"] = False
    else:
        logger.info(
            "[answer_cache] skip PUT (validation.passed=%s was_modified=%s) "
            "-- not caching safe-fallback / failed answers",
            getattr(validation, "passed", None),
            getattr(validation, "was_modified", None),
        )
        context_stats["cache_stored"] = False

    # ── Brief 5 / Part 1 — Semantic answer cache put ──
    # Layer 1 gating (confidence >= 0.75, grounding passed, zero fabrications)
    # is enforced inside semantic_cache.put. Fail-safe: any error logs and
    # skips without blocking the response.
    context_stats["semantic_cache_id"] = None
    context_stats["semantic_cache_stored"] = False
    # Cache-poisoning fix (matches the answer_cache gate above): also skip
    # the semantic cache when the validator substituted a safe fallback.
    # semantic_cache.put has its own confidence/grounding gate, but it
    # does not see was_modified — so without this guard a substituted
    # answer could still be persisted if its confidence happens to scrape
    # above the floor.
    if (
        settings.SEMANTIC_CACHE_ENABLED
        and validation.passed
        and not getattr(validation, "was_modified", False)
    ):
        try:
            from backend.services.semantic_cache import put as _sem_put
            _grounding_passed = bool(
                validation.grounding_detail.passed
            ) if validation.grounding_detail else True
            _grounding_score = float(
                validation.grounding_detail.grounding_score
            ) if validation.grounding_detail else float(confidence or 0.0)
            _fabrications = (
                len(validation.grounding_detail.fabrications)
                if (validation.grounding_detail and validation.grounding_detail.fabrications)
                else 0
            )
            # Map source names back to file_ids (for invalidate_by_source_file)
            try:
                _name_to_id = {
                    f.get("name"): f.get("id")
                    for f in list_active_files_all()
                    if f.get("id") and f.get("name")
                }
                _source_file_ids = [
                    _name_to_id[name]
                    for name in (doc_src or [])
                    if isinstance(name, str) and name in _name_to_id
                ]
            except Exception:
                _source_file_ids = []
            _sem_cache_id = _sem_put(
                query=effective_query,
                answer_text=answer,
                answer_metadata={
                    "sources": doc_src,
                    "confidence": confidence,
                    "model_used": _model_used_label,
                    "grounding_score": _grounding_score,
                },
                source_file_ids=_source_file_ids,
                active_file_ids=active_file_ids,
                confidence=float(confidence or 0.0),
                grounding_score=_grounding_score,
                grounding_passed=_grounding_passed,
                fabrications=_fabrications,
                model_used=_model_used_label,
                origin_owner_id=owner_id or "",
            )
            context_stats["semantic_cache_id"] = _sem_cache_id
            context_stats["semantic_cache_stored"] = bool(_sem_cache_id)
        except Exception as _sem_exc:
            logger.warning("Semantic cache put wrapper failed: %s", _sem_exc)

    # Sprint 3B — low-similarity banner. Populate confidence_band only
    # when we actually ran ticket-history retrieval (doc_ranked is the
    # ranked list of (id, text, meta, score) tuples). Cached/trivial
    # paths leave this None so the frontend omits the banner.
    _confidence_band: Optional[str] = None
    try:
        _top_score = float(doc_ranked[0][3]) if doc_ranked else 0.0
        _confidence_band = (
            "low" if _top_score < settings.LOW_SIMILARITY_THRESHOLD else "normal"
        )
        context_stats["confidence_band"] = _confidence_band
        context_stats["top_chunk_score"] = round(_top_score, 4)
        logger.info(
            "[confidence_band] top_score=%.4f threshold=%.2f band=%s",
            _top_score, settings.LOW_SIMILARITY_THRESHOLD, _confidence_band,
        )
    except Exception as _cb_exc:
        logger.warning("[confidence_band] compute failed: %s", _cb_exc)
        _confidence_band = None

    return AnswerResponse(
        answer=answer,
        sources=doc_src,
        confidence=confidence,
        processing_time_ms=ms,
        session_id=session_id,
        context_stats=context_stats,
        confidence_band=_confidence_band,
    )


class FeedbackSubmitRequest(BaseModel):
    session_id: Optional[str] = None
    message_index: Optional[int] = None
    feedback_type: str = Field(pattern="^(like|dislike)$")
    feedback_text: str = Field(default="", max_length=1200)
    question: Optional[str] = None
    answer: Optional[str] = None
    model_config = ConfigDict(extra="ignore")


class FeedbackStateRequest(BaseModel):
    session_id: str
    message_index: int
    feedback_type: str = Field(pattern="^(like|dislike|none)$")
    semantic_cache_id: Optional[str] = None
    # Sprint 3B — optional context carried by the frontend on 👎 so the
    # backend can run the KB pivot pipeline. Both fields are ignored
    # unless feedback_type == "dislike" AND session_mode == "troubleshooting".
    # Missing fields short-circuit the pivot.
    session_mode: Optional[str] = None
    original_query: Optional[str] = None
    model_config = ConfigDict(extra="ignore")


# Sprint 3B — KB pivot dedupe window. An in-process dict keyed by
# (session_id, message_index) → timestamp. Subsequent 👎s within
# _KB_PIVOT_DEDUPE_SECONDS of the first are skipped with kind="dedupe".
# Fine for single-instance; move to Redis when horizontally scaled.
_KB_PIVOT_DEDUPE: Dict[Tuple[str, int], float] = {}
_KB_PIVOT_DEDUPE_SECONDS: float = 60.0


def _kb_pivot_recent(session_id: str, message_index: int) -> bool:
    """Return True if a KB pivot already ran for this (session, msg) within
    the dedupe window. Also prunes stale entries opportunistically."""
    now = _time.time()
    key = (session_id, int(message_index))
    # Opportunistic prune of stale entries (cap O(n) work).
    for k, ts in list(_KB_PIVOT_DEDUPE.items())[:32]:
        if now - ts > _KB_PIVOT_DEDUPE_SECONDS:
            _KB_PIVOT_DEDUPE.pop(k, None)
    last = _KB_PIVOT_DEDUPE.get(key)
    if last is not None and (now - last) <= _KB_PIVOT_DEDUPE_SECONDS:
        return True
    _KB_PIVOT_DEDUPE[key] = now
    return False


@app.post("/feedback/state")
async def save_feedback_state(
    req: FeedbackStateRequest,
    user_id: Optional[str] = Depends(auth_dependency),
):
    owner = _normalize_owner_id(user_id)
    feedback_value = None if req.feedback_type == "none" else req.feedback_type
    ok = update_message_feedback(req.session_id, owner, req.message_index, feedback_value)
    if not ok:
        raise HTTPException(404, "Session not found")

    # Brief 5 / Part 1 — Layer 3 feedback-driven invalidation.
    # A dislike on a semantic-cache-served answer wipes that row so the next
    # user gets a fresh pipeline run and the answer is re-cached after validation.
    invalidated = False
    if req.feedback_type == "dislike" and req.semantic_cache_id:
        try:
            from backend.services.semantic_cache import invalidate as _sem_invalidate
            invalidated = _sem_invalidate(
                req.semantic_cache_id, reason="user_dislike"
            )
        except Exception as _inv_exc:
            logger.warning("[semantic_cache] invalidate wrapper failed: %s", _inv_exc)

    # Sprint 3B — 👎 KB/Runbook pivot. Only fires in troubleshooting mode
    # when the frontend sent `original_query`. Dedupes repeated 👎s on
    # the same message within a 60s window.
    pivot_payload: Optional[Dict[str, Any]] = None
    _should_pivot = (
        req.feedback_type == "dislike"
        and (req.session_mode or "").lower() == "troubleshooting"
        and bool((req.original_query or "").strip())
    )
    if _should_pivot:
        if _kb_pivot_recent(req.session_id, req.message_index):
            logger.info(
                "[kb_pivot] dedupe hit session=%s msg_idx=%s — skipping",
                req.session_id, req.message_index,
            )
            pivot_payload = {"kind": "dedupe"}
        else:
            try:
                from backend.agents.orchestrator import run_kb_pivot_pipeline
                _active_file_ids = _get_active_indexed_file_ids(user_id)
                _mode_snap = get_session_mode(req.session_id)

                pivot_result = run_kb_pivot_pipeline(
                    query=(req.original_query or "").strip(),
                    session_mode=_mode_snap,
                    generate_fn=safe_generate,
                    bedrock_client=bedrock,
                    embed_fn=safe_embed,
                    owner_id=owner or "",
                    allowed_file_ids=set(_active_file_ids or []),
                    bm25_search_fn=(
                        bm25.search if bm25 and bm25.size > 0 else None
                    ),
                    vector_search_fn=pgvector_search,
                )

                if pivot_result.kb_pivot_empty or not (pivot_result.answer or "").strip():
                    pivot_payload = {
                        "kind": "no_kb_match",
                        "message": (
                            "No matching KB/runbook content found for this issue. "
                            "Consider escalation to the appropriate team."
                        ),
                    }
                else:
                    pivot_payload = {
                        "kind": "kb_guidance",
                        "answer": pivot_result.answer,
                    }
            except Exception as _piv_exc:
                logger.warning("[kb_pivot] pipeline wrapper failed: %s", _piv_exc)
                pivot_payload = None

    response: Dict[str, Any] = {
        "status": "saved",
        "feedback_type": req.feedback_type,
        "semantic_cache_invalidated": invalidated,
    }
    if pivot_payload is not None:
        response["pivot"] = pivot_payload
    return response


@app.post("/feedback/submit")
async def submit_feedback(
    req: FeedbackSubmitRequest,
    background_tasks: BackgroundTasks,
    user_id: Optional[str] = Depends(auth_dependency),
):
    user_info = _resolve_user_display(user_id)
    display_name = user_info["name"]
    display_email = user_info["email"]

    is_like = req.feedback_type == "like"
    emoji = "\U0001f44d" if is_like else "\U0001f44e"
    label = "Positive" if is_like else "Negative"
    color = "#10b981" if is_like else "#dc2626"
    header_bg = "#4f46e5" if is_like else "#dc2626"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    subject = f"Acadia Doc IQ — {emoji} {label} Feedback from {display_name}"

    body_text = (
        f"Feedback Type: {label} ({emoji})\n"
        f"User: {display_name}\n"
        f"Email: {display_email}\n"
        f"Timestamp: {now}\n"
        f"Session: {req.session_id or 'N/A'}\n"
    )
    if req.question:
        body_text += f"Question: {req.question}\n"
    if req.answer:
        body_text += f"Answer Preview: {req.answer[:300]}\n"
    if req.feedback_text.strip():
        body_text += f"\nFeedback Message:\n{req.feedback_text}\n"

    feedback_html = ""
    if req.feedback_text.strip():
        feedback_html = f"""
            <div style="margin-top: 16px; padding: 16px; background: white; border: 1px solid #dee2e6; border-radius: 8px;">
                <p style="font-weight: bold; margin: 0 0 8px 0; color: {color};">Feedback Message:</p>
                <p style="margin: 0; white-space: pre-wrap;">{req.feedback_text}</p>
            </div>
        """

    question_row = f'<tr><td style="padding: 8px 0; font-weight: bold;">Question</td><td>{req.question}</td></tr>' if req.question else ""
    answer_row = f'<tr><td style="padding: 8px 0; font-weight: bold;">Answer</td><td>{(req.answer or "")[:500]}</td></tr>' if req.answer else ""

    body_html = f"""
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
        <div style="background: {header_bg}; color: white; padding: 20px; border-radius: 8px 8px 0 0;">
            <h2 style="margin: 0;">{emoji} {label} Feedback</h2>
        </div>
        <div style="background: #f8f9fb; padding: 20px; border: 1px solid #dee2e6; border-radius: 0 0 8px 8px;">
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 8px 0; font-weight: bold;">Feedback Type</td>
                    <td style="color: {color}; font-weight: bold; font-size: 16px;">{emoji} {label}</td>
                </tr>
                <tr><td style="padding: 8px 0; font-weight: bold;">Name</td><td>{display_name}</td></tr>
                <tr><td style="padding: 8px 0; font-weight: bold;">Email</td><td>{display_email}</td></tr>
                <tr><td style="padding: 8px 0; font-weight: bold;">Sent At</td><td>{now}</td></tr>
                {question_row}
                {answer_row}
            </table>
            {feedback_html}
        </div>
    </div>
    """

    background_tasks.add_task(send_feedback_email, subject, body_text, body_html)
    return {"status": "sent", "message": "Thank you for your feedback!"}


@app.exception_handler(HTTPException)
async def http_err(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "path": request.url.path,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


@app.exception_handler(Exception)
async def general_err(request, exc):
    logger.exception("Unhandled: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "path": request.url.path,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    
@app.get("/debug/file/{file_id}")
async def debug_file(file_id: str, user_id: Optional[str] = Depends(auth_dependency)):
    owner_id = _normalize_owner_id(user_id)

    from backend.db.connection import SessionLocal
    from sqlalchemy import text

    with SessionLocal() as db:
        doc = db.execute(
            text(
                """
                SELECT id::text, owner_id, name, status, current_version_id::text
                FROM documents
                WHERE id = :file_id
                """
            ),
            {"file_id": file_id},
        ).mappings().first()

        versions = db.execute(
            text(
                """
                SELECT id::text, document_id::text, is_active, fingerprint, uploaded_at
                FROM document_versions
                WHERE document_id = :file_id
                ORDER BY uploaded_at DESC
                """
            ),
            {"file_id": file_id},
        ).mappings().all()

        chunk_count = db.execute(
            text(
                """
                SELECT COUNT(*)
                FROM chunks
                WHERE document_id = :file_id
                """
            ),
            {"file_id": file_id},
        ).scalar_one()

        embedding_count = db.execute(
            text(
                """
                SELECT COUNT(*)
                FROM embeddings
                WHERE chunk_id IN (
                    SELECT id FROM chunks WHERE document_id = :file_id
                )
                """
            ),
            {"file_id": file_id},
        ).scalar_one()

    if not doc:
        raise HTTPException(404, "File not found")

    if doc["owner_id"] != owner_id:
        raise HTTPException(404, "File not found")

    return {
        "document": dict(doc),
        "versions": [dict(v) for v in versions],
        "chunk_count": int(chunk_count or 0),
        "embedding_count": int(embedding_count or 0),
    }