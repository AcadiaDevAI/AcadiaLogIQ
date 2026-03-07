import json
import logging
import hashlib
import os
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict, Set

import boto3
from botocore.config import Config as BotoConfig
from fastapi import (
    BackgroundTasks, Depends, FastAPI, File, Header,
    HTTPException, Query, Request, UploadFile, status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from pydantic import BaseModel, ConfigDict, Field, field_validator

from config import settings
from vector_store import get_collection, get_bm25_index, rebuild_bm25_from_chroma, reset_chroma_collection
from clerk_auth import clerk_auth_dependency, is_clerk_enabled


if sys.version_info < (3, 11):
    raise RuntimeError("Python 3.11+ required")


# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("acadia-log-iq")


# ============================================================================
# TOKEN BUDGET — MISTRAL 7B (32K CONTEXT)
# ============================================================================
CHARS_PER_TOKEN = 3

TOKEN_BUDGET = {
    "MODEL_MAX_TOKENS": 32_768,
    "MAX_LOG_CONTEXT_CHARS": 24_000,
    "MAX_KB_CONTEXT_CHARS": 24_000,
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


# ============================================================================
# TYPES
# ============================================================================
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


UPLOAD_JOBS: Dict[str, JobInfo] = {}
coll = None
bm25 = None


# ============================================================================
# IN-MEMORY STORES
# ============================================================================
CHAT_SESSIONS: Dict[str, dict] = {}
UPLOADED_FILES: Dict[str, dict] = {}


# ============================================================================
# BEDROCK CLIENT
# ============================================================================
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


# ============================================================================
# HELPERS
# ============================================================================
def _normalize_owner_id(user_id: Optional[str]) -> str:
    return user_id or "anonymous"


def _filter_files_for_owner(user_id: Optional[str]) -> List[dict]:
    owner_id = _normalize_owner_id(user_id)
    return [
        f for f in UPLOADED_FILES.values()
        if f.get("owner_id", "anonymous") == owner_id
    ]


def _get_active_indexed_file_ids(user_id: Optional[str]) -> Set[str]:
    owner_id = _normalize_owner_id(user_id)
    return {
        f["id"]
        for f in UPLOADED_FILES.values()
        if f.get("owner_id", "anonymous") == owner_id and f.get("status") == "indexed"
    }


def _delete_physical_uploads_for_file(filename: str, job_id: Optional[str] = None) -> int:
    deleted = 0
    try:
        upload_dir = settings.UPLOAD_DIR
        if not upload_dir.exists():
            return 0

        for f in upload_dir.iterdir():
            if not f.is_file():
                continue
            matched = False
            if job_id and f.name.startswith(f"{job_id}_"):
                matched = True
            elif filename and (f.name == filename or filename in f.name):
                matched = True

            if matched:
                try:
                    f.unlink(missing_ok=True)
                    deleted += 1
                except Exception as e:
                    logger.warning("Could not delete upload file %s: %s", f.name, e)
    except Exception as e:
        logger.warning("Physical upload cleanup failed for %s: %s", filename, e)
    return deleted


def _remove_file_references_from_sessions(filename: str, file_id: Optional[str] = None) -> int:
    updated = 0
    for session in CHAT_SESSIONS.values():
        changed = False
        for msg in session.get("messages", []):
            sources = msg.get("sources")
            if not isinstance(sources, dict):
                continue
            docs = sources.get("docs")
            if isinstance(docs, list):
                new_docs = [d for d in docs if d != filename]
                if new_docs != docs:
                    sources["docs"] = new_docs
                    changed = True
        if changed:
            session["updated_at"] = datetime.now(timezone.utc).isoformat()
            updated += 1
    return updated


# ============================================================================
# LIFESPAN
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global coll, bm25
    logger.info("Starting API...")
    coll = get_collection()
    logger.info("Chroma ready: %s", settings.COLLECTION_NAME)
    bm25 = get_bm25_index()
    doc_count = rebuild_bm25_from_chroma(coll)
    logger.info("BM25 ready: %d docs", doc_count)
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Acadia's Log IQ API",
    description="AI log analysis — Hybrid Search + Re-ranking",
    version="2.1.0",
    lifespan=lifespan,
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
        "*",
    ],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
    expose_headers=["X-Processing-Time"], max_age=600,
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
    if is_clerk_enabled():
        user_id = await clerk_auth_dependency(request)
        return user_id
    else:
        verify_api_key(x_api_key)
        return None


# ============================================================================
# PYDANTIC MODELS
# ============================================================================
class Question(BaseModel):
    q: str = Field(min_length=1, max_length=1000)
    session_id: Optional[str] = None
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


class AnswerResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float = Field(ge=0, le=1)
    processing_time_ms: Optional[int] = None
    context_stats: Optional[Dict] = None
    session_id: Optional[str] = None
    model_config = ConfigDict(extra="ignore")


class ChatMessage(BaseModel):
    role: str
    content: str
    sources: Optional[Dict] = None
    timestamp: str


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


# ============================================================================
# EMBEDDING
# ============================================================================
def safe_embed(text: str) -> Optional[List[float]]:
    if not text or not text.strip():
        return None
    try:
        body = json.dumps({"inputText": text[: settings.MAX_CHARS]}).encode("utf-8")
        resp = bedrock.invoke_model(
            modelId=settings.BEDROCK_EMBED_MODEL, body=body,
            accept="application/json", contentType="application/json",
        )
        payload = json.loads(resp["body"].read().decode("utf-8"))
        emb = payload.get("embedding")
        return emb if isinstance(emb, list) else None
    except Exception as e:
        logger.exception("Embed failed: %s", e)
        return None


# ============================================================================
# LLM GENERATION
# ============================================================================
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
                prompt = prompt[: max_prompt_chars - len(tail) - 80] + \
                    "\n\n[... context truncated ...]\n" + tail
            else:
                prompt = prompt[:max_prompt_chars] + "\n\n[... truncated ...]\n"

        logger.info("safe_generate: %d chars (~%d tok)", len(prompt), estimate_tokens(prompt))

        body = json.dumps({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "top_p": 0.9,
        }).encode("utf-8")

        resp = bedrock.invoke_model(
            modelId=settings.BEDROCK_LLM_MODEL, body=body,
            accept="application/json", contentType="application/json",
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


# ============================================================================
# HYBRID SEARCH
# ============================================================================
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
        cr = coll.query(
            query_embeddings=[query_embedding],
            n_results=HYBRID_CONFIG["VECTOR_CANDIDATES"],
            where={"file_type": file_type},
        )

        ids = (cr.get("ids") or [[]])[0]
        docs = (cr.get("documents") or [[]])[0]
        metas = (cr.get("metadatas") or [[]])[0]
        dists = (cr.get("distances") or [[]])[0]

        rank = 0
        for i, doc_id in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            meta_file_id = meta.get("file_id")
            meta_owner_id = meta.get("owner_id", "anonymous")

            if allowed_file_ids is not None and meta_file_id not in allowed_file_ids:
                continue
            if meta_owner_id != owner_id:
                continue

            rank += 1
            vector_results[doc_id] = {
                "text": docs[i] if i < len(docs) else "",
                "metadata": meta,
                "rank": rank,
                "similarity": max(0, 1 - (dists[i] if i < len(dists) else 1.0)),
            }
    except Exception as e:
        logger.warning("Vector search failed (%s): %s", file_type, e)

    bm25_results = {}
    try:
        if bm25 and bm25.size > 0:
            raw_bm25 = bm25.search(
                query,
                n_results=HYBRID_CONFIG["BM25_CANDIDATES"],
                file_type=file_type
            )
            rank = 0
            for doc_id, text, meta, score in raw_bm25:
                meta_file_id = meta.get("file_id")
                meta_owner_id = meta.get("owner_id", "anonymous")

                if allowed_file_ids is not None and meta_file_id not in allowed_file_ids:
                    continue
                if meta_owner_id != owner_id:
                    continue

                rank += 1
                bm25_results[doc_id] = {
                    "text": text,
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

    results = [
        (did, all_data[did]["text"], all_data[did]["metadata"], combined[did])
        for did in sorted_ids[:n_results]
    ]

    logger.info(
        "Hybrid(%s): vec=%d, bm25=%d, merged=%d, returned=%d, allowed_files=%d, owner=%s",
        file_type,
        len(vector_results),
        len(bm25_results),
        len(combined),
        len(results),
        len(allowed_file_ids or []),
        owner_id,
    )
    return results


# ============================================================================
# RE-RANKING
# ============================================================================
def rerank_chunks(
    query: str, chunks: List[Tuple[str, str, Dict, float]], top_k: int = 6,
) -> List[Tuple[str, str, Dict, float]]:
    if not chunks or len(chunks) <= 1:
        return chunks[:top_k]

    candidates = chunks[:min(len(chunks), 12)]

    previews = []
    for i, (_, text, meta, _) in enumerate(candidates):
        preview = text[:600].replace("\n", " ").strip()
        src = meta.get("source", "?")
        previews.append(f"[{i+1}] (source: {src}) {preview}")

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
        logger.info("Re-ranked %d -> top %d", len(candidates), top_k)
        return scored[:top_k]

    except Exception as e:
        logger.warning("Re-rank failed (%s), using hybrid scores", e)
        return candidates[:top_k]


# ============================================================================
# CONTEXT ASSEMBLY
# ============================================================================
def truncate_chunk(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    t = text[:max_chars]
    nl = t.rfind("\n")
    if nl > max_chars * 0.7:
        t = t[:nl]
    return t + "\n[... truncated ...]"


# def assemble_context(
#     ranked: List[Tuple[str, str, Dict, float]],
#     max_total_chars: int,
#     max_sources: int = 3,
# ) -> Tuple[str, List[str]]:
#     """
#     Build context from ranked chunks.

#     Only sources from included chunks are considered.
#     Source list is stricter:
#     - aggregate included score per source
#     - keep max 3
#     - if top source strongly dominates, return only top source
#     """
#     max_chunk = TOKEN_BUDGET["MAX_SINGLE_CHUNK_CHARS"]
#     if not ranked:
#         return "", []

#     parts = []
#     included_source_scores: Dict[str, float] = {}
#     total = 0

#     for _, text, meta, score in ranked:
#         if not text or not text.strip():
#             continue

#         chunk = truncate_chunk(text.strip(), max_chunk)
#         src = meta.get("source", "unknown")
#         entry = f"[Source: {src}]\n{chunk}"

#         if total + len(entry) > max_total_chars:
#             remaining = max_total_chars - total
#             if remaining > 300:
#                 parts.append(f"[Source: {src}]\n{truncate_chunk(chunk, remaining - 60)}")
#                 included_source_scores[src] = included_source_scores.get(src, 0.0) + score
#             break

#         parts.append(entry)
#         included_source_scores[src] = included_source_scores.get(src, 0.0) + score
#         total += len(entry)

#     if not included_source_scores:
#         return "\n\n".join(parts), []

#     ranked_sources = sorted(included_source_scores.items(), key=lambda x: x[1], reverse=True)
#     final_sources = []

#     if len(ranked_sources) == 1:
#         final_sources = [ranked_sources[0][0]]
#     else:
#         top_name, top_score = ranked_sources[0]
#         second_score = ranked_sources[1][1] if len(ranked_sources) > 1 else 0.0

#         # If top source clearly dominates, show only it.
#         if second_score <= 0 or top_score >= second_score * 2.0:
#             final_sources = [top_name]
#         else:
#             threshold = top_score * 0.60
#             final_sources = [src for src, score in ranked_sources if score >= threshold][:max_sources]
#             if not final_sources:
#                 final_sources = [top_name]

#     return "\n\n".join(parts), final_sources

def assemble_context(
    ranked: List[Tuple[str, str, Dict, float]],
    max_total_chars: int,
    max_sources: int = 5,
) -> Tuple[str, List[str]]:
    """
    Build context from ranked chunks. Returns (context_str, sources).

    Source selection logic:
    - Aggregate relevance scores per source document
    - Any source with aggregate score >= 40% of the top source is included
    - This naturally handles:
      * Question about 1 doc → only that doc (others score far below 40%)
      * Question about 2 docs → both docs (both score similarly)
      * Question about all docs → all relevant docs shown
    """
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

    # Sort by aggregate score
    ranked_sources = sorted(source_scores.items(), key=lambda x: x[1], reverse=True)
    top_score = ranked_sources[0][1]

    # Include any source scoring >= 40% of the top source
    threshold = top_score * 0.40
    final_sources = [src for src, score in ranked_sources if score >= threshold]

    logger.info(
        "Source selection: %s | threshold=%.6f | final=%s",
        {s: f"{sc:.6f}" for s, sc in ranked_sources},
        threshold,
        final_sources,
    )

    return "\n\n".join(parts), final_sources[:max_sources]


# ============================================================================
# FILE HASHING
# ============================================================================
def calculate_file_hash(fp: Path) -> str:
    sha = hashlib.sha256()
    with fp.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


# ============================================================================
# TEXT EXTRACTION
# ============================================================================
def extract_text_from_pdf(fp: Path) -> str:
    try:
        import fitz
        parts = []
        with fitz.open(str(fp)) as doc:
            for i, page in enumerate(doc):
                t = page.get_text("text")
                if t and t.strip():
                    parts.append(f"--- Page {i+1} ---\n{t}")
        if parts:
            logger.info("PDF: %d pages, %d chars from %s", len(parts), sum(len(p) for p in parts), fp.name)
            return "\n\n".join(parts)
        return ""
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
            parts.append(f"--- Table {ti+1} ---")
            parts.extend(rows)
    return "\n".join(parts)


def extract_text(fp: Path) -> str:
    ext = fp.suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(fp)
    if ext == ".docx":
        return extract_text_from_docx(fp)
    return fp.read_text(encoding="utf-8", errors="ignore")


# ============================================================================
# CHUNKING
# ============================================================================
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
        try:
            text = extract_text(fp)
            if not text or not text.strip():
                return
            logger.info("Extracted %d chars from %s", len(text), fp.name)
            n = 0
            for c in iter_text_chunks(text, lines_per=lines_per):
                n += 1
                yield c
            logger.info("%d chunks from %s", n, fp.name)
        except RuntimeError as e:
            logger.error("Extraction: %s", e)
            raise
        except Exception as e:
            logger.exception("Failed: %s", e)
        return
    try:
        text = fp.read_text(encoding="utf-8", errors="ignore")
        if text and text.strip():
            for c in iter_text_chunks(text, lines_per=lines_per):
                yield c
    except Exception as e:
        logger.exception("Read failed %s: %s", fp, e)


# ============================================================================
# INDEX JOB
# ============================================================================
async def index_file_job(
    job_id: str,
    fp: Path,
    filename: str,
    file_type: str,
    file_id: str,
    owner_id: Optional[str],
):
    owner_id = _normalize_owner_id(owner_id)

    try:
        job = UPLOAD_JOBS.get(job_id)
        if not job:
            return

        job["status"] = "running"

        if file_id in UPLOADED_FILES:
            UPLOADED_FILES[file_id]["status"] = "processing"

        total, ok = 0, 0
        b_emb, b_doc, b_meta, b_ids = [], [], [], []
        bm25_ids, bm25_docs, bm25_metas = [], [], []

        for chunk in iter_line_chunks(fp):
            if not chunk or not chunk.strip():
                continue

            total += 1
            emb = safe_embed(chunk)
            if not emb:
                continue

            cid = f"{file_id}:{job_id}:{ok}"
            meta = {
                "file_id": file_id,
                "owner_id": owner_id,
                "source": filename,
                "file_type": file_type,
                "job_id": job_id,
                "chunk_index": ok,
                "char_count": len(chunk),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            b_ids.append(cid)
            b_emb.append(emb)
            b_doc.append(chunk)
            b_meta.append(meta)

            bm25_ids.append(cid)
            bm25_docs.append(chunk)
            bm25_metas.append(meta)

            ok += 1
            if ok % 10 == 0:
                job["processed_chunks"] = ok

            if len(b_emb) >= settings.BATCH_SIZE:
                coll.add(ids=b_ids, embeddings=b_emb, metadatas=b_meta, documents=b_doc)
                b_ids, b_emb, b_meta, b_doc = [], [], [], []

        if b_emb:
            coll.add(ids=b_ids, embeddings=b_emb, metadatas=b_meta, documents=b_doc)

        if bm25 and bm25_ids:
            bm25.add_documents_batch(bm25_ids, bm25_docs, bm25_metas)
            logger.info("BM25 +%d (total: %d)", len(bm25_ids), bm25.size)

        try:
            fp.unlink(missing_ok=True)
        except Exception:
            pass

        job.update({
            "status": "done",
            "processed_chunks": ok,
            "total_chunks": total,
            "successful_chunks": ok,
            "completed_at": datetime.now(timezone.utc),
        })

        if file_id in UPLOADED_FILES:
            UPLOADED_FILES[file_id]["status"] = "indexed"

        logger.info("Job %s DONE: %d/%d from %s", job_id, ok, total, filename)

    except Exception as e:
        logger.exception("Job failed %s: %s", job_id, e)
        if job_id in UPLOAD_JOBS:
            UPLOAD_JOBS[job_id].update({
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.now(timezone.utc),
            })
        if file_id in UPLOADED_FILES:
            UPLOADED_FILES[file_id]["status"] = "failed"
        try:
            fp.unlink(missing_ok=True)
        except Exception:
            pass


# ============================================================================
# MIDDLEWARE
# ============================================================================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    import time
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000.0
    response.headers["X-Processing-Time"] = f"{ms:.2f}ms"
    logger.info("%s %s -> %s (%.2fms)", request.method, request.url.path, response.status_code, ms)
    return response


# ============================================================================
# ROUTES — HEALTH
# ============================================================================
@app.get("/health")
async def health_check():
    cc = 0
    try:
        if coll:
            cc = coll.count()
    except Exception:
        pass
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": settings.BEDROCK_LLM_MODEL,
        "services": {
            "vector_store": f"{cc} chunks" if coll else "uninitialized",
            "bm25_index": f"{bm25.size} docs" if bm25 else "uninitialized",
            "bedrock": "available",
        },
        "search_mode": "hybrid (vector + BM25 + re-ranking)",
        "auth_mode": "clerk" if is_clerk_enabled() else ("api_key" if settings.API_KEY else "open"),
        "token_budget": {
            "model_max": TOKEN_BUDGET["MODEL_MAX_TOKENS"],
            "max_log_chars": TOKEN_BUDGET["MAX_LOG_CONTEXT_CHARS"],
            "max_kb_chars": TOKEN_BUDGET["MAX_KB_CONTEXT_CHARS"],
        },
    }


# ============================================================================
# ROUTES — AUTH / USER
# ============================================================================
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


# ============================================================================
# ROUTES — RESET
# ============================================================================
@app.post("/reset")
async def reset_all(user_id: Optional[str] = Depends(auth_dependency)):
    global coll, bm25
    errors = []

    deleted_chunks = 0
    try:
        deleted_chunks = reset_chroma_collection()
        coll = get_collection()
        logger.info("RESET: ChromaDB cleared (%d chunks deleted)", deleted_chunks)
    except Exception as e:
        logger.exception("RESET: ChromaDB reset failed: %s", e)
        errors.append(f"ChromaDB: {e}")
        try:
            coll = get_collection()
        except Exception:
            pass

    try:
        if bm25:
            old_size = bm25.size
            bm25.clear()
            logger.info("RESET: BM25 cleared (%d docs)", old_size)
    except Exception as e:
        logger.exception("RESET: BM25 clear failed: %s", e)
        errors.append(f"BM25: {e}")

    deleted_files = 0
    try:
        upload_dir = settings.UPLOAD_DIR
        if upload_dir.exists():
            for f in upload_dir.iterdir():
                if f.is_file():
                    try:
                        f.unlink()
                        deleted_files += 1
                    except Exception as fe:
                        logger.warning("RESET: Could not delete %s: %s", f.name, fe)
            logger.info("RESET: Deleted %d files from %s", deleted_files, upload_dir)
    except Exception as e:
        logger.exception("RESET: Upload cleanup failed: %s", e)
        errors.append(f"Uploads: {e}")

    job_count = len(UPLOAD_JOBS)
    UPLOAD_JOBS.clear()
    UPLOADED_FILES.clear()
    CHAT_SESSIONS.clear()
    logger.info("RESET: Cleared %d job records", job_count)

    verify_count = 0
    try:
        if coll:
            verify_count = coll.count()
    except Exception:
        pass

    if errors:
        return {
            "status": "partial",
            "message": f"Reset completed with {len(errors)} error(s)",
            "errors": errors,
            "deleted_chunks": deleted_chunks,
            "deleted_files": deleted_files,
            "cleared_jobs": job_count,
            "chroma_count_after": verify_count,
            "bm25_count_after": bm25.size if bm25 else 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    return {
        "status": "success",
        "message": "All data deleted",
        "deleted_chunks": deleted_chunks,
        "deleted_files": deleted_files,
        "cleared_jobs": job_count,
        "chroma_count_after": verify_count,
        "bm25_count_after": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ============================================================================
# ROUTES — PURGE ORPHAN CHUNKS
# ============================================================================
@app.post("/purge_orphans")
async def purge_orphan_chunks(user_id: Optional[str] = Depends(auth_dependency)):
    """
    Delete all ChromaDB chunks that don't belong to any currently registered file.

    This cleans up ghost data from previously deleted/uploaded files whose
    chunks survived in the vector store (e.g., files uploaded before file_id
    tracking was added, or files deleted while the server was down).

    Safe to run anytime. Also rebuilds BM25 from the cleaned ChromaDB.
    """
    global bm25

    if not coll:
        raise HTTPException(503, "Vector store not ready")

    # Get all known file_ids from current registry
    known_file_ids = {f["id"] for f in UPLOADED_FILES.values()}

    # Scan all chunks in ChromaDB
    total_chunks = coll.count()
    if total_chunks == 0:
        return {"status": "clean", "total_chunks": 0, "orphans_deleted": 0}

    orphan_ids = []
    batch_size = 500
    for offset in range(0, total_chunks, batch_size):
        limit = min(batch_size, total_chunks - offset)
        results = coll.get(
            limit=limit, offset=offset,
            include=["metadatas"],
        )
        ids = results.get("ids", [])
        metas = results.get("metadatas", [])
        for i, chunk_id in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            chunk_file_id = meta.get("file_id")
            # If chunk has no file_id or its file_id is not in registry → orphan
            if not chunk_file_id or chunk_file_id not in known_file_ids:
                orphan_ids.append(chunk_id)

    # Delete orphans in batches
    deleted = 0
    for i in range(0, len(orphan_ids), 500):
        batch = orphan_ids[i:i+500]
        coll.delete(ids=batch)
        deleted += len(batch)

    # Rebuild BM25 from clean ChromaDB
    bm25_count = 0
    if bm25:
        from vector_store import rebuild_bm25_from_chroma
        bm25_count = rebuild_bm25_from_chroma(coll)

    logger.info("PURGE: Deleted %d orphan chunks out of %d total. BM25 rebuilt with %d docs.",
                deleted, total_chunks, bm25_count)

    return {
        "status": "purged",
        "total_chunks_before": total_chunks,
        "orphans_deleted": deleted,
        "chunks_remaining": coll.count(),
        "bm25_rebuilt": bm25_count,
        "known_files": len(known_file_ids),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ============================================================================
# ROUTES — UPLOAD
# ============================================================================
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
    file_id = uuid.uuid4().hex
    fp = settings.UPLOAD_DIR / f"{job_id}_{Path(file.filename).name}"

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(400, f"File {size_mb:.1f}MB exceeds {settings.MAX_FILE_SIZE_MB}MB limit")

    fp.write_bytes(content)
    fh = calculate_file_hash(fp)

    UPLOAD_JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "processed_chunks": 0,
        "total_chunks": 0,
        "successful_chunks": 0,
        "file": file.filename,
        "file_type": file_type,
        "file_size_mb": size_mb,
        "file_hash": fh,
        "created_at": datetime.now(timezone.utc),
        "owner_id": owner_id,
        "file_id": file_id,
    }

    UPLOADED_FILES[file_id] = {
        "id": file_id,
        "name": file.filename,
        "file_type": file_type,
        "size_mb": round(size_mb, 2),
        "status": "uploading",
        "job_id": job_id,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "owner_id": owner_id,
    }

    background_tasks.add_task(index_file_job, job_id, fp, file.filename, file_type, file_id, owner_id)

    logger.info("Upload: %s (%.1fMB, %s, owner=%s, file_id=%s)", file.filename, size_mb, file_type, owner_id, file_id)

    return UploadResponse(
        job_id=job_id,
        file_id=file_id,
        message="Uploaded. Processing started.",
        file_hash=fh,
    )


@app.get("/upload_status/{job_id}", response_model=JobStatus)
async def upload_status(job_id: str, user_id: Optional[str] = Depends(auth_dependency)):
    job = UPLOAD_JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    owner_id = _normalize_owner_id(user_id)
    if job.get("owner_id", "anonymous") != owner_id:
        raise HTTPException(404, "Job not found")

    return JobStatus(**job)


# ============================================================================
# ROUTES — FILES
# ============================================================================
@app.get("/files")
async def list_files(user_id: Optional[str] = Depends(auth_dependency)):
    files = sorted(
        _filter_files_for_owner(user_id),
        key=lambda x: x["uploaded_at"],
        reverse=True,
    )
    return {"files": files, "total": len(files)}


@app.delete("/files/{file_id}")
async def delete_file(file_id: str, user_id: Optional[str] = Depends(auth_dependency)):
    """
    Fully delete a file and all its indexed data:
    1. Remove all chunks from ChromaDB by file_id
    2. Remove all chunks from BM25
    3. Remove physical upload file if still present
    4. Remove file registry + job registry
    5. Remove old source references from chat sessions
    """
    if file_id not in UPLOADED_FILES:
        raise HTTPException(404, "File not found")

    file_info = UPLOADED_FILES[file_id]
    owner_id = _normalize_owner_id(user_id)
    if file_info.get("owner_id", "anonymous") != owner_id:
        raise HTTPException(404, "File not found")

    filename = file_info["name"]
    job_id = file_info.get("job_id")

    deleted_chunks = 0
    bm25_removed = 0
    deleted_files = 0
    sessions_updated = 0
    errors = []

    # 1. Delete from Chroma by exact file_id
    try:
        if coll:
            results = coll.get(where={"file_id": file_id}, include=[])
            chunk_ids = results.get("ids", []) or []
            if chunk_ids:
                coll.delete(ids=chunk_ids)
                deleted_chunks = len(chunk_ids)
            logger.info(
                "DELETE FILE: Removed %d chunks from ChromaDB for file_id=%s (%s)",
                deleted_chunks, file_id, filename
            )
    except Exception as e:
        logger.exception("DELETE FILE: Chroma cleanup failed for '%s': %s", filename, e)
        errors.append(f"ChromaDB: {e}")

    # 2. Delete from BM25
    try:
        if bm25:
            if hasattr(bm25, "remove_documents_by_file_id"):
                bm25_removed = bm25.remove_documents_by_file_id(file_id)
            elif hasattr(bm25, "remove_documents_by_source"):
                bm25_removed = bm25.remove_documents_by_source(filename)
            logger.info("DELETE FILE: Removed %d docs from BM25 for '%s'", bm25_removed, filename)
    except Exception as e:
        logger.exception("DELETE FILE: BM25 cleanup failed for '%s': %s", filename, e)
        errors.append(f"BM25: {e}")

    # 3. Delete physical file
    try:
        deleted_files = _delete_physical_uploads_for_file(filename, job_id)
        logger.info("DELETE FILE: Removed %d upload file(s) for '%s'", deleted_files, filename)
    except Exception as e:
        logger.warning("DELETE FILE: Disk cleanup failed: %s", e)

    # 4. Remove from registries
    try:
        del UPLOADED_FILES[file_id]
    except KeyError:
        pass

    if job_id and job_id in UPLOAD_JOBS:
        del UPLOAD_JOBS[job_id]

    # 5. Remove from session source references
    try:
        sessions_updated = _remove_file_references_from_sessions(filename, file_id)
    except Exception as e:
        logger.warning("DELETE FILE: Session cleanup failed: %s", e)

    if errors:
        return {
            "status": "partial",
            "file_id": file_id,
            "filename": filename,
            "deleted_chunks": deleted_chunks,
            "bm25_removed": bm25_removed,
            "deleted_files": deleted_files,
            "sessions_updated": sessions_updated,
            "errors": errors,
        }

    return {
        "status": "deleted",
        "file_id": file_id,
        "filename": filename,
        "deleted_chunks": deleted_chunks,
        "bm25_removed": bm25_removed,
        "deleted_files": deleted_files,
        "sessions_updated": sessions_updated,
    }


# ============================================================================
# ROUTES — CHAT HISTORY
# ============================================================================
@app.get("/chat/sessions")
async def list_sessions(user_id: Optional[str] = Depends(auth_dependency)):
    owner_id = _normalize_owner_id(user_id)
    sessions = sorted(
        [
            {
                "id": s["id"],
                "title": s["title"],
                "updated_at": s["updated_at"],
                "message_count": len(s["messages"]),
            }
            for s in CHAT_SESSIONS.values()
            if s.get("owner_id", "anonymous") == owner_id
        ],
        key=lambda x: x["updated_at"],
        reverse=True,
    )
    return {"sessions": sessions}


@app.get("/chat/sessions/{session_id}")
async def get_session(session_id: str, user_id: Optional[str] = Depends(auth_dependency)):
    session = CHAT_SESSIONS.get(session_id)
    owner_id = _normalize_owner_id(user_id)
    if not session or session.get("owner_id", "anonymous") != owner_id:
        raise HTTPException(404, "Session not found")
    return session


@app.delete("/chat/sessions/{session_id}")
async def delete_session(session_id: str, user_id: Optional[str] = Depends(auth_dependency)):
    session = CHAT_SESSIONS.get(session_id)
    owner_id = _normalize_owner_id(user_id)
    if not session or session.get("owner_id", "anonymous") != owner_id:
        raise HTTPException(404, "Session not found")
    del CHAT_SESSIONS[session_id]
    return {"status": "deleted", "session_id": session_id}


@app.delete("/chat/sessions")
async def delete_all_sessions(user_id: Optional[str] = Depends(auth_dependency)):
    owner_id = _normalize_owner_id(user_id)
    to_delete = [sid for sid, s in CHAT_SESSIONS.items() if s.get("owner_id", "anonymous") == owner_id]
    for sid in to_delete:
        del CHAT_SESSIONS[sid]
    return {"status": "cleared", "deleted_count": len(to_delete)}


def _save_message_to_session(
    session_id: Optional[str],
    role: str,
    content: str,
    sources: Optional[Dict] = None,
    owner_id: Optional[str] = None,
) -> str:
    now = datetime.now(timezone.utc).isoformat()
    owner_id = _normalize_owner_id(owner_id)

    if not session_id or session_id not in CHAT_SESSIONS:
        session_id = uuid.uuid4().hex
        title = content[:60] + "..." if len(content) > 60 else content
        CHAT_SESSIONS[session_id] = {
            "id": session_id,
            "title": title,
            "messages": [],
            "created_at": now,
            "updated_at": now,
            "owner_id": owner_id,
        }

    msg = {"role": role, "content": content, "timestamp": now}
    if sources:
        msg["sources"] = sources

    CHAT_SESSIONS[session_id]["messages"].append(msg)
    CHAT_SESSIONS[session_id]["updated_at"] = now
    return session_id


# ============================================================================
# ROUTES — ASK
# ============================================================================
@app.post("/ask", response_model=AnswerResponse)
@limiter.limit("30/minute")
async def ask(request: Request, req: Question, user_id: Optional[str] = Depends(auth_dependency)):
    """
    Searches only currently active indexed files for this user.
    Deleted files are excluded even if old chunks exist elsewhere.
    """
    import time
    start = time.perf_counter()

    if not coll:
        raise HTTPException(503, "Vector store not ready")

    owner_id = _normalize_owner_id(user_id)

    # Save user message
    session_id = _save_message_to_session(req.session_id, "user", req.q, owner_id=owner_id)

    active_file_ids = _get_active_indexed_file_ids(user_id)
    if not active_file_ids:
        answer = "No indexed files are currently available. Please upload a document first."
        _save_message_to_session(session_id, "assistant", answer, {"docs": []}, owner_id=owner_id)
        return AnswerResponse(
            answer=answer,
            sources=[],
            confidence=0.0,
            processing_time_ms=int((time.perf_counter() - start) * 1000),
            session_id=session_id,
            context_stats={
                "search_mode": "hybrid (vector + BM25 + re-ranking)",
                "active_file_count": 0,
                "doc_candidates": 0,
                "doc_after_rerank": 0,
            },
        )

    q_emb = safe_embed(req.q)
    if not q_emb:
        raise HTTPException(500, "Embedding failed")

    doc_cands = hybrid_search(
        req.q,
        q_emb,
        "kb",
        HYBRID_CONFIG["VECTOR_CANDIDATES"],
        allowed_file_ids=active_file_ids,
        owner_id=owner_id,
    )

    # Extra hard safety filter
    doc_cands = [
        item for item in doc_cands
        if item[2].get("file_id") in active_file_ids
        and item[2].get("owner_id", "anonymous") == owner_id
    ]

    doc_ranked = rerank_chunks(
        req.q,
        doc_cands,
        HYBRID_CONFIG["RERANK_TOP_K_LOG"] 
    )

    max_ctx_chars = TOKEN_BUDGET["MAX_LOG_CONTEXT_CHARS"] + TOKEN_BUDGET["MAX_KB_CONTEXT_CHARS"]
    doc_ctx, doc_src = assemble_context(doc_ranked, max_ctx_chars)

    if not doc_ctx:
        answer = "I could not find relevant information in the currently uploaded files."
        ms = int((time.perf_counter() - start) * 1000)
        _save_message_to_session(session_id, "assistant", answer, {"docs": []}, owner_id=owner_id)
        return AnswerResponse(
            answer=answer,
            sources=[],
            confidence=0.05,
            processing_time_ms=ms,
            session_id=session_id,
            context_stats={
                "search_mode": "hybrid (vector + BM25 + re-ranking)",
                "active_file_count": len(active_file_ids),
                "doc_candidates": len(doc_cands),
                "doc_after_rerank": len(doc_ranked),
                "doc_context_chars": 0,
            },
        )

    if len(doc_ctx) > MAX_TOTAL_PROMPT_CHARS - TOKEN_BUDGET["PROMPT_OVERHEAD_CHARS"]:
        doc_ctx = truncate_chunk(doc_ctx, MAX_TOTAL_PROMPT_CHARS - TOKEN_BUDGET["PROMPT_OVERHEAD_CHARS"])

    confidence = min(0.3 + len(doc_ranked) * 0.1, 1.0)

#     prompt = f"""You are an expert AI assistant analyzing uploaded documents.

# IMPORTANT RULES:
# - Use ONLY the document context below.
# - Ignore any deleted, missing, or previously uploaded files not present in the context.
# - If the answer is mainly from one document, mention only that document.
# - In the final answer, be precise and avoid mixing unrelated documents.
# - If the information is not present in the current uploaded files, clearly say so.

# DOCUMENTS:
# {doc_ctx}

# USER QUESTION: {req.q}

# Provide a concise, accurate answer based only on the current uploaded documents.

# ANSWER:"""

    prompt = f"""You are an expert AI assistant analyzing uploaded documents.

IMPORTANT RULES:
- Use ONLY the document context below.
- Ignore any deleted, missing, or previously uploaded files not present in the context.
- If the answer is mainly from one document, rely only on that document.
- Do NOT mix unrelated documents.
- If the information is not present in the current uploaded files, clearly say so.
- Every answer MUST be in bullet-point format.
- Keep bullets concise, factual, and directly based on the provided document context.
- Do not add a source section inside the answer. Sources are handled separately.

DOCUMENTS:
{doc_ctx}

USER QUESTION: {req.q}

Provide a concise, accurate answer based only on the current uploaded documents.

ANSWER:"""

    ptok = estimate_tokens(prompt)
    max_tok = TOKEN_BUDGET["MODEL_MAX_TOKENS"] - TOKEN_BUDGET["MAX_GENERATION_TOKENS"]
    logger.info(
        "ASK: q='%s' | active_files=%d | cands=%d | reranked=%d | ~%d tok (limit %d)",
        req.q[:50], len(active_file_ids), len(doc_cands), len(doc_ranked), ptok, max_tok
    )

    answer = safe_generate(prompt)
    ms = int((time.perf_counter() - start) * 1000)

    sources_dict = {"docs": doc_src}
    _save_message_to_session(session_id, "assistant", answer, sources_dict, owner_id=owner_id)

    return AnswerResponse(
        answer=answer,
        sources=doc_src,
        confidence=confidence,
        processing_time_ms=ms,
        session_id=session_id,
        context_stats={
            "search_mode": "hybrid (vector + BM25 + re-ranking)",
            "active_file_count": len(active_file_ids),
            "doc_candidates": len(doc_cands),
            "doc_after_rerank": len(doc_ranked),
            "doc_context_chars": len(doc_ctx),
            "prompt_chars": len(prompt),
            "prompt_tokens": ptok,
            "max_tokens": max_tok,
            "headroom": max_tok - ptok,
        },
    )


# ============================================================================
# ERROR HANDLERS
# ============================================================================
@app.exception_handler(HTTPException)
async def http_err(request, exc):
    return JSONResponse(exc.status_code, {
        "error": exc.detail,
        "path": request.url.path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.exception_handler(Exception)
async def general_err(request, exc):
    logger.exception("Unhandled: %s", exc)
    return JSONResponse(500, {
        "error": "Internal server error",
        "path": request.url.path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })