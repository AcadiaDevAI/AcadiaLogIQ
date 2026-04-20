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
    BackgroundTasks, Depends, FastAPI, File, Header,
    HTTPException, Query, Request, UploadFile, status,
)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from pydantic import BaseModel, ConfigDict, Field, field_validator

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
from backend.agents.orchestrator import should_escalate_to_agents, run_agent_pipeline
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


logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("acadia-log-iq")


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
    boto_cfg = BotoConfig(
        retries={"max_attempts": 10, "mode": "adaptive"},
        read_timeout=120, connect_timeout=30, tcp_keepalive=True,
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

    # ── Log auth configuration on startup ──
    clerk_on = is_clerk_enabled()
    logger.info("=== AUTH CONFIG ===")
    logger.info("  CLERK_ENABLED setting: %s", getattr(settings, 'CLERK_ENABLED', 'NOT SET'))
    logger.info("  CLERK_SECRET_KEY set: %s", bool(getattr(settings, 'CLERK_SECRET_KEY', '')))
    logger.info("  CLERK_PUBLISHABLE_KEY set: %s", bool(getattr(settings, 'CLERK_PUBLISHABLE_KEY', '')))
    logger.info("  is_clerk_enabled() = %s", clerk_on)
    if not clerk_on:
        logger.warning("  ⚠️  Clerk is NOT enabled — all requests will be 'anonymous'!")
        logger.warning("  ⚠️  Set CLERK_ENABLED=true, CLERK_SECRET_KEY, and CLERK_PUBLISHABLE_KEY in .env")
    logger.info("===================")

    bm25 = get_bm25_index()
    doc_count = rebuild_bm25_from_postgres()
    logger.info("BM25 ready from PostgreSQL: %d docs", doc_count)

    # Rebuild glossary from document content (learns abbreviations automatically)
    glossary_count = rebuild_glossary_from_postgres()
    logger.info("Glossary store ready: %d acronyms learned from documents", glossary_count)

    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Acadia's Log IQ API",
    description="AI log analysis — Hybrid Search + Re-ranking",
    version="2.3.0",
    lifespan=lifespan,
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Processing-Time"],
    max_age=600,
)


def verify_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> bool:
    if settings.API_KEY:
        if not x_api_key or x_api_key != settings.API_KEY:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return True


async def auth_dependency(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> Optional[str]:
    """
    Returns the authenticated user's Clerk ID, or raises 401.

    When Clerk is enabled:
      - Valid JWT → returns user_id (e.g., "user_2xABC123")
      - Missing/invalid JWT → clerk_auth_dependency raises 401
    When Clerk is disabled:
      - API key mode or open mode → returns None (becomes "anonymous")
    """
    if is_clerk_enabled():
        user_id = await clerk_auth_dependency(request)
        # Extra safety: if clerk_auth_dependency somehow returns None
        # without raising, reject anyway
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required. Please sign in.",
            )
        logger.debug("Auth: clerk user_id=%s", user_id)
        return user_id

    # Clerk not enabled — fall back to API key or open mode
    verify_api_key(x_api_key)
    logger.debug("Auth: Clerk disabled, using anonymous")
    return None


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


def safe_generate(prompt: str, max_tokens: int = None) -> str:
    if max_tokens is None:
        max_tokens = TOKEN_BUDGET["MAX_GENERATION_TOKENS"]
    try:
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
):
    owner_id = _normalize_owner_id(owner_id)
    update_ingestion_job(job_id, status="running")
    t_total_start = _time.perf_counter()

    try:
        local_path = storage.resolve_local_path(storage_uri)
        if not local_path or not local_path.exists():
            raise RuntimeError(f"Stored file path is not readable: {storage_uri}")

        job = get_ingestion_job(job_id)
        file_hash = job["file_hash"] if job else ""

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
        )

        t_parse_end = _time.perf_counter()
        logger.info(
            "[PERF] %s — Parse + metadata: %.1fs (%d chunks)",
            filename, t_parse_end - t_parse_start, len(processed.get("chunk_rows", []))
        )

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


@app.middleware("http")
async def log_requests(request: Request, call_next):
    import time

    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000.0
    response.headers["X-Processing-Time"] = f"{ms:.2f}ms"
    logger.info("%s %s -> %s (%.2fms)", request.method, request.url.path, response.status_code, ms)
    return response


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
        "services": {
            "vector_store": f"{chunk_count} chunks" if chunk_count >= 0 else "uninitialized",
            "bm25_index": f"{bm25.size} docs" if bm25 else "uninitialized",
            "glossary_store": f"{get_glossary_store().size} acronyms (learned from docs)",
            "bedrock": "available",
        },
        "search_mode": "hybrid (pgvector + BM25 + re-ranking + doc-aware query expansion)",
        "auth_mode": "clerk" if is_clerk_enabled() else ("api_key" if settings.API_KEY else "open"),
    }


# =========================================================
# Auth Debug — call this to diagnose config issues
# =========================================================
@app.get("/auth/debug")
async def auth_debug(request: Request):
    """
    Diagnostic endpoint — shows auth configuration and whether
    the current request has a valid JWT. No auth required.
    """
    clerk_enabled = is_clerk_enabled()
    has_bearer = bool(request.headers.get("Authorization", "").startswith("Bearer "))
    
    result = {
        "clerk_enabled": clerk_enabled,
        "clerk_enabled_setting": getattr(settings, 'CLERK_ENABLED', 'NOT SET'),
        "clerk_secret_key_set": bool(getattr(settings, 'CLERK_SECRET_KEY', '')),
        "clerk_publishable_key_set": bool(getattr(settings, 'CLERK_PUBLISHABLE_KEY', '')),
        "request_has_bearer_token": has_bearer,
        "auth_mode": "clerk" if clerk_enabled else ("api_key" if settings.API_KEY else "open"),
    }
    
    # If there's a Bearer token and Clerk is enabled, try to decode it
    if clerk_enabled and has_bearer:
        try:
            from backend.clerk_auth import extract_bearer_token, verify_clerk_token
            token = extract_bearer_token(request)
            if token:
                payload = verify_clerk_token(token)
                result["jwt_valid"] = True
                result["jwt_user_id"] = payload.get("sub")
                result["jwt_issuer"] = payload.get("iss")
            else:
                result["jwt_valid"] = False
                result["jwt_error"] = "No token extracted"
        except Exception as e:
            result["jwt_valid"] = False
            result["jwt_error"] = str(e)
    elif has_bearer and not clerk_enabled:
        result["warning"] = "Bearer token present but Clerk is NOT enabled — token is being IGNORED"
    
    return result


@app.get("/me")
async def get_current_user(request: Request, user_id: Optional[str] = Depends(auth_dependency)):
    if is_clerk_enabled() and user_id:
        payload = getattr(request.state, "clerk_payload", {})
        return {
            "authenticated": True,
            "user_id": user_id,
            "issuer": payload.get("iss"),
            "auth_mode": "clerk",
        }
    return {
        "authenticated": False,
        "user_id": None,
        "auth_mode": "api_key" if settings.API_KEY else "open",
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


@app.post("/upload", response_model=UploadResponse)
@limiter.limit("100/minute")
async def upload(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    file_type: str = Query(default="kb", pattern="^(kb)$"),
    user_id: Optional[str] = Depends(auth_dependency),
):
    ext = Path(file.filename).suffix[1:].lower() if file.filename else ""
    if not ext or ext not in settings.ALLOWED_FILE_TYPES:
        raise HTTPException(400, f"Type '{ext}' not allowed. Allowed: {settings.ALLOWED_FILE_TYPES}")

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
    )

    return UploadResponse(
        job_id=job_id,
        file_id=file_id,
        message="Uploaded. Processing started.",
        file_hash=file_hash,
    )


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
    #         "Query expansion: '%s' → acronyms=%s",
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
            "Query expansion: '%s' → acronyms=%s",
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

    retrieval = orchestrator_retrieve(
        query=expanded.expanded_text,
        query_embedding=q_emb,
        owner_id=owner_id,
        allowed_file_ids=active_file_ids,
        file_type="kb",
        generate_fn=safe_generate,
        bm25_search_fn=bm25.search if bm25 and bm25.size > 0 else None,
        vector_search_fn=pgvector_search,
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
    should_try_variants = (
        len(expanded.variants) > 1
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
            _clarify_res = _try_clarify(
                query=effective_query,
                ranked_chunks=doc_ranked,
                triage_confidence=_triage_conf,
                carried_identifiers=list(_carried_entities or []),
                session_id=session_id,
                recent_messages=recent_msgs_all,
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
    try:
        from backend.services.pattern_analytics import enrich_if_needed as _pattern_enrich
        from backend.db.queries import load_similar_tickets_for_topic as _load_similar

        pattern_context = _pattern_enrich(
            query=effective_query,
            retrieved_chunks=doc_ranked,
            session_mode=None,
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
            "[triage] overriding complexity tier: heuristic=%s → llm=%s (conf=%.2f)",
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
            "Intent '%s' upgrading auto → %s",
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

    # ── Store in answer cache (fail-safe; never blocks response) ──
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

    # ── Brief 5 / Part 1 — Semantic answer cache put ──
    # Layer 1 gating (confidence >= 0.75, grounding passed, zero fabrications)
    # is enforced inside semantic_cache.put. Fail-safe: any error logs and
    # skips without blocking the response.
    context_stats["semantic_cache_id"] = None
    context_stats["semantic_cache_stored"] = False
    if settings.SEMANTIC_CACHE_ENABLED and validation.passed:
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

    return AnswerResponse(
        answer=answer,
        sources=doc_src,
        confidence=confidence,
        processing_time_ms=ms,
        session_id=session_id,
        context_stats=context_stats,
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
    model_config = ConfigDict(extra="ignore")


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

    return {
        "status": "saved",
        "feedback_type": req.feedback_type,
        "semantic_cache_invalidated": invalidated,
    }


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