"""
Interactive Clarifier — runs AFTER retrieval but BEFORE agent pipeline or
full RAG generation. Presents the user with 3-4 clickable options when
retrieval surfaces multiple distinct candidates and no clear winner.

Differs from agents/clarifier.py in three ways:
  1. Runs earlier (after retrieval, before agent pipeline)
  2. Output is structured JSON the frontend renders as buttons, not prose
  3. Triggered by ambiguity in retrieval results, not by domain regexes

Fails safe: any error returns None so /ask proceeds as if clarifier disabled.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from backend.config import settings
from backend.services.bedrock_haiku import haiku_client, _extract_json_object
from backend.retrieval.orchestrator import _extract_identifiers

logger = logging.getLogger("acadia-log-iq")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ClarificationOption:
    """One clickable option shown to the user."""
    id: str              # "opt_1", "opt_2", ..., "opt_other"
    label: str           # Human-readable button text
    refined_query: str   # Query sent back to /ask when user clicks this option
    record_ref: Optional[str] = None  # Source record ID if the option maps to one


@dataclass
class ClarificationResult:
    """Outcome of a clarification attempt."""
    needs_clarification: bool = False
    options: List[ClarificationOption] = field(default_factory=list)
    context_summary: str = ""
    ambiguity_score: float = 0.0
    reason: str = ""
    skip_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Hard-skip patterns (dynamic — no hardcoded domain/ticket knowledge)
# ---------------------------------------------------------------------------
_BROAD_INTENT_RE = re.compile(
    r"\b(all|every|across|globally|overall|entire|total|each)\b",
    re.I,
)

# Per-session clarification counter (in-memory; OK for single-instance MVP,
# move to Redis/DB when scaling horizontally)
_SESSION_COUNTS: Dict[str, int] = {}
_SESSION_COUNT_TTL_SECONDS: int = 3600  # 1 hour
_SESSION_COUNT_TIMESTAMPS: Dict[str, float] = {}


def _get_session_clarification_count(session_id: Optional[str]) -> int:
    """Per-session clarification counter with soft TTL."""
    if not session_id:
        return 0
    now = time.time()
    ts = _SESSION_COUNT_TIMESTAMPS.get(session_id, 0)
    if now - ts > _SESSION_COUNT_TTL_SECONDS:
        _SESSION_COUNTS.pop(session_id, None)
        _SESSION_COUNT_TIMESTAMPS.pop(session_id, None)
        return 0
    return int(_SESSION_COUNTS.get(session_id, 0))


def _bump_session_clarification_count(session_id: Optional[str]) -> None:
    if not session_id:
        return
    _SESSION_COUNTS[session_id] = _get_session_clarification_count(session_id) + 1
    _SESSION_COUNT_TIMESTAMPS[session_id] = time.time()


def _unpack_chunk(chunk: Any):
    """
    Normalize a ranked chunk into (text, metadata_dict, score).

    Handles both the tuple shape the retrieval layer actually returns —
    (chunk_id, text, metadata, score) — and object-shaped chunks (metadata,
    score, text attributes) for future-proofing and unit tests.
    """
    try:
        if isinstance(chunk, (list, tuple)):
            # (id, text, metadata, score)
            if len(chunk) >= 4:
                _cid, text, meta, score = chunk[0], chunk[1], chunk[2], chunk[3]
            elif len(chunk) == 3:
                text, meta, score = chunk
            else:
                return "", {}, 0.0
            text = text or ""
            if not isinstance(meta, dict):
                meta = {}
            try:
                score = float(score or 0.0)
            except Exception:
                score = 0.0
            return text, meta, score
        # Object-shaped fallback
        text = getattr(chunk, "text", "") or ""
        meta = getattr(chunk, "metadata", None)
        if not isinstance(meta, dict):
            meta = {}
        try:
            score = float(getattr(chunk, "score", 0) or 0.0)
        except Exception:
            score = 0.0
        return text, meta, score
    except Exception:
        return "", {}, 0.0


def _record_key(chunk: Any) -> Optional[str]:
    """
    Extract the source-record identifier from a chunk's metadata.
    Schema-agnostic — checks multiple common field names.
    """
    try:
        _text, meta, _score = _unpack_chunk(chunk)
        mj = meta.get("metadata_json") if isinstance(meta, dict) else None
        if not isinstance(mj, dict):
            # The tuple path often flattens metadata_json into meta itself.
            mj = meta if isinstance(meta, dict) else {}
        return (
            mj.get("primary_id")
            or mj.get("incident_number")
            or mj.get("record_id")
            or meta.get("source_record_id")
            or meta.get("source_file_id")
            or meta.get("document_id")
        )
    except Exception:
        return None


def _record_summary(chunk: Any) -> Optional[Dict[str, Any]]:
    """Build a compact summary dict for Haiku prompt context."""
    try:
        text, meta, score = _unpack_chunk(chunk)
        mj = meta.get("metadata_json") if isinstance(meta, dict) else None
        if not isinstance(mj, dict):
            mj = meta if isinstance(meta, dict) else {}
        return {
            "id": _record_key(chunk),
            "summary": (mj.get("summary") or mj.get("title") or "")[:200],
            "customer": mj.get("customer_name") or mj.get("organization"),
            "component": mj.get("component") or mj.get("component_category") or mj.get("category"),
            "preview": (text or "")[:240],
            "score": float(score or 0.0),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Ambiguity scoring
# ---------------------------------------------------------------------------
def compute_ambiguity_score(
    query: str,
    ranked_chunks: Sequence[Any],
    triage_confidence: Optional[float],
    carried_identifiers: Optional[Sequence[str]] = None,
) -> float:
    """
    Heuristic ambiguity score in [0, 1]. Higher = more ambiguous.
    Pure function, no I/O, fast.
    """
    score = 0.0

    # Signal 1: no identifier in query or carried context
    has_carried = bool(carried_identifiers)
    has_query_id = bool(_extract_identifiers(query or ""))
    if not has_carried and not has_query_id:
        score += 0.25

    # Signal 2: triage low-confidence
    if triage_confidence is not None and triage_confidence < 0.75:
        score += 0.20

    # Signals 3 & 4: retrieval candidate diversity + score tightness
    top_k = list(ranked_chunks)[:5]
    if top_k:
        unique_records = {rk for rk in (_record_key(c) for c in top_k) if rk}
        n_unique = len(unique_records)
        if n_unique >= 3:
            score += 0.30
        elif n_unique == 2:
            score += 0.15

        scores = [_unpack_chunk(c)[2] for c in top_k]
        scores = [s for s in scores if s > 0]
        if len(scores) >= 2:
            spread = scores[0] - scores[-1]
            if spread < 0.05:
                score += 0.15

    # Signal 5: short/medium query length (often under-specified)
    q = (query or "").strip()
    if 10 <= len(q) <= 60:
        score += 0.10

    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Haiku prompt + generator
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You help disambiguate user queries for a document-grounded assistant.

The user asked an ambiguous question. Multiple distinct records in the
document set could be the intent. Your job is to generate SHORT, SPECIFIC
clarifying options the user can click on — plus one "Other" option for
free-text.

Return ONLY a JSON object with this exact shape — no prose, no markdown:
{
  "context_summary": "One short sentence describing why clarification helps.",
  "options": [
    {"id": "opt_1", "label": "...", "refined_query": "..."},
    {"id": "opt_2", "label": "...", "refined_query": "..."},
    {"id": "opt_3", "label": "...", "refined_query": "..."},
    {"id": "opt_other", "label": "Something else - let me clarify", "refined_query": ""}
  ]
}

Rules:
- Generate 2-4 content options PLUS the "opt_other" option (always include opt_other).
- Each `label` is what the user sees on the button: max 14 words, human-friendly,
  names a distinguishing fact. Do NOT put the raw record ID in the label —
  use customer name or a short distinguishing phrase instead.
- Each `refined_query` is what the system runs after the user clicks. It MUST:
  (a) be a self-contained rewrite of the user's original query,
  (b) INCLUDE the source record identifier from the candidate's `id` field
      VERBATIM if one is present (e.g., "INC-10036", "KB-4201", "CASE-7789"),
      as the first distinguishing token after the verb,
  (c) be answerable by the system without further clarification.
- If a candidate has an `id` value, the refined_query for that option MUST
  include that id verbatim. Missing the id in refined_query is a critical
  failure of the clarifier.
- Options must be genuinely distinct - do not offer two options that would
  retrieve the same record.
- Do NOT invent record IDs, customer names, or details not present in the
  candidate context.
- If the candidate context does not justify distinct options, still return
  opt_other only so the user can clarify freely.
- The "opt_other" refined_query is always an empty string.

Example — good option for a candidate with id="INC-10036", customer="Enterprise-338":
  {
    "id": "opt_4",
    "label": "Enterprise-338: Equipment down after failed change request",
    "refined_query": "How do I troubleshoot INC-10036 — the router failure for Enterprise-338 caused by a failed change request?"
  }

Note: label is user-facing (no raw ID), refined_query is system-facing (id present).

Output ONLY the JSON."""


def _validate_and_repair_options(
    options: List[ClarificationOption],
    candidate_summaries: List[Dict[str, Any]],
    original_query: str,
) -> List[ClarificationOption]:
    """
    Enforce: every non-opt_other refined_query must include at least one
    candidate `id` verbatim. If a refined_query is missing its id but the
    label points clearly to a known candidate, auto-repair by prepending the
    id. If repair isn't possible, drop the option.

    This is a safety net — the prompt says to do this, but LLM drift happens.
    """
    valid_ids = [
        str(s.get("id") or "").strip()
        for s in candidate_summaries
        if s.get("id")
    ]
    valid_ids = [i for i in valid_ids if i]

    repaired: List[ClarificationOption] = []
    for opt in options:
        if opt.id == "opt_other":
            repaired.append(opt)
            continue

        rq = opt.refined_query or ""
        included = [i for i in valid_ids if i and i in rq]
        if included:
            repaired.append(opt)
            continue

        repaired_rq = None
        label_lower = (opt.label or "").lower()
        for summary in candidate_summaries:
            sid = str(summary.get("id") or "").strip()
            if not sid:
                continue
            customer = str(summary.get("customer") or "").strip()
            summary_preview = str(summary.get("summary") or "").strip().lower()
            customer_match = customer and customer.lower() in label_lower
            preview_match = (
                summary_preview
                and len(summary_preview) > 10
                and any(w in label_lower for w in summary_preview.split()[:4])
            )
            if customer_match or preview_match:
                repaired_rq = f"{sid} - {rq}" if rq else f"Tell me about {sid}."
                logger.info(
                    "[interactive_clarifier] repaired refined_query for %s: "
                    "prepended id=%s",
                    opt.id, sid,
                )
                break

        if repaired_rq:
            repaired.append(ClarificationOption(
                id=opt.id,
                label=opt.label,
                refined_query=repaired_rq[:500],
                record_ref=None,
            ))
        else:
            logger.warning(
                "[interactive_clarifier] dropping option %s - no id in refined_query "
                "and no matching candidate for repair: label=%r rq=%r",
                opt.id, opt.label, rq,
            )

    return repaired


def _invoke_haiku(prompt: str, max_tokens: int, timeout_s: float) -> str:
    """Single Haiku call. Returns raw text; caller parses."""
    import concurrent.futures

    def _call() -> str:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "system": _SYSTEM_PROMPT,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        }
        resp = haiku_client.client.invoke_model(
            modelId=settings.INTERACTIVE_CLARIFIER_MODEL,
            body=json.dumps(body).encode("utf-8"),
            accept="application/json",
            contentType="application/json",
        )
        payload = json.loads(resp["body"].read().decode("utf-8"))
        content = payload.get("content", [])
        return "\n".join(
            item.get("text", "") for item in content if item.get("type") == "text"
        ).strip()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_call)
        return future.result(timeout=timeout_s)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def try_clarify(
    *,
    query: str,
    ranked_chunks: Sequence[Any],
    triage_confidence: Optional[float],
    carried_identifiers: Optional[Sequence[str]],
    session_id: Optional[str],
    recent_messages: Optional[List[Dict[str, Any]]],
    session_clarif_count: int = 0,
) -> ClarificationResult:
    """
    Main entry. Returns ClarificationResult.
      - If needs_clarification=True, caller should return clarification payload
        to frontend instead of continuing pipeline.
      - If needs_clarification=False, caller proceeds normally.

    Never raises. Any error -> skip_reason set, needs_clarification=False.

    Sprint 2.7 Bug C — `session_clarif_count` is a hard per-chain cap:
    when the hotfix flag is on and count >= 1, skip clarification regardless
    of ambiguity score. Prevents the "double loop" UX where a user picks an
    option, submits the refined query, and gets asked more questions.
    """
    result = ClarificationResult()

    try:
        if not getattr(settings, "INTERACTIVE_CLARIFIER_ENABLED", False):
            result.skip_reason = "disabled"
            return result

        # Sprint 2.7 Bug C — at most one clarification round per logical
        # question chain. Caller passes count=1 on the refined /ask that
        # followed an earlier clarification selection.
        if session_clarif_count >= 1:
            result.skip_reason = "chain_cap_reached"
            logger.info(
                "[interactive_clarifier] skip reason=chain_cap_reached "
                "session_clarif_count=%d",
                session_clarif_count,
            )
            return result

        # Hard-skip checks (cheap, no LLM)
        if query and _BROAD_INTENT_RE.search(query):
            result.skip_reason = "broad_intent"
            logger.info("[interactive_clarifier] skip reason=broad_intent")
            return result

        if _extract_identifiers(query or ""):
            result.skip_reason = "specific_identifier"
            logger.info("[interactive_clarifier] skip reason=specific_identifier")
            return result

        if _get_session_clarification_count(session_id) >= int(
            settings.INTERACTIVE_CLARIFIER_MAX_PER_SESSION
        ):
            result.skip_reason = "rate_limited"
            logger.info("[interactive_clarifier] skip reason=rate_limited")
            return result

        # Don't re-clarify if user is answering a prior clarification
        if recent_messages:
            last_assistant = next(
                (m for m in reversed(recent_messages) if m.get("role") == "assistant"),
                None,
            )
            if last_assistant:
                stats = last_assistant.get("context_stats") or {}
                if stats.get("clarification_presented"):
                    result.skip_reason = "answering_prior_clarification"
                    logger.info("[interactive_clarifier] skip reason=answering_prior_clarification")
                    return result

        # Compute ambiguity
        score = compute_ambiguity_score(
            query=query,
            ranked_chunks=ranked_chunks,
            triage_confidence=triage_confidence,
            carried_identifiers=carried_identifiers,
        )
        result.ambiguity_score = score
        threshold = float(settings.INTERACTIVE_CLARIFIER_AMBIGUITY_THRESHOLD)
        if score < threshold:
            result.skip_reason = "score_below_threshold"
            logger.info(
                "[interactive_clarifier] skip reason=below_threshold score=%.2f threshold=%.2f",
                score, threshold,
            )
            return result

        # We have candidates - build compact summaries for the Haiku prompt
        summaries: List[Dict[str, Any]] = []
        seen_keys: set = set()
        for chunk in list(ranked_chunks)[:8]:
            s = _record_summary(chunk)
            if not s:
                continue
            key = s.get("id") or (s.get("preview") or "")[:80]
            if key in seen_keys:
                continue
            seen_keys.add(key)
            summaries.append(s)
            if len(summaries) >= 6:
                break

        if not summaries:
            result.skip_reason = "no_candidates"
            logger.info("[interactive_clarifier] skip reason=no_candidates")
            return result

        # Build Haiku prompt
        candidates_block = json.dumps(summaries, ensure_ascii=False, indent=2)
        user_prompt = (
            f"User's query: {query}\n\n"
            f"Top retrieval candidates (distinct records or chunks):\n{candidates_block}\n\n"
            f"JSON:"
        )

        raw = _invoke_haiku(
            prompt=user_prompt,
            max_tokens=int(settings.INTERACTIVE_CLARIFIER_MAX_TOKENS),
            timeout_s=float(settings.INTERACTIVE_CLARIFIER_TIMEOUT_SECONDS),
        )

        if not raw:
            result.skip_reason = "empty_llm_response"
            return result

        cleaned = _extract_json_object(raw)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning("[interactive_clarifier] parse_error raw=%r err=%s", raw[:200], exc)
            result.skip_reason = "parse_error"
            return result

        raw_options = parsed.get("options") or []
        if not isinstance(raw_options, list) or not raw_options:
            result.skip_reason = "empty_options"
            return result

        options: List[ClarificationOption] = []
        for i, opt in enumerate(raw_options):
            if not isinstance(opt, dict):
                continue
            opt_id = str(opt.get("id") or f"opt_{i+1}").strip()[:32]
            label = str(opt.get("label") or "").strip()
            refined = str(opt.get("refined_query") or "").strip()
            if not label:
                continue
            if opt_id != "opt_other" and not refined:
                continue
            if opt_id == "opt_other" and refined:
                refined = ""
            options.append(ClarificationOption(
                id=opt_id,
                label=label[:180],
                refined_query=refined[:500],
                record_ref=None,
            ))

        # Validate and repair refined_queries against candidate ids
        options = _validate_and_repair_options(options, summaries, query)

        # Ensure opt_other exists so user always has an escape hatch
        if not any(o.id == "opt_other" for o in options):
            options.append(ClarificationOption(
                id="opt_other",
                label="Something else - let me clarify",
                refined_query="",
            ))

        # Min 2 real options + opt_other, or we just skip clarification
        if sum(1 for o in options if o.id != "opt_other") < 2:
            result.skip_reason = "insufficient_distinct_options"
            return result

        result.needs_clarification = True
        result.options = options
        result.context_summary = str(parsed.get("context_summary") or "").strip()[:300]
        result.reason = "triggered"

        _bump_session_clarification_count(session_id)

        logger.info(
            "[interactive_clarifier] triggered score=%.2f opts=%d session_clarif_count=%d",
            score, len(options), _get_session_clarification_count(session_id),
        )
        for o in options:
            if o.id == "opt_other":
                continue
            logger.info(
                "[interactive_clarifier]   %s: label=%r refined=%r",
                o.id, o.label[:60], o.refined_query[:120],
            )
        return result

    except Exception as exc:
        logger.warning("[interactive_clarifier] failure (%s: %s) - falling through",
                       type(exc).__name__, exc)
        result.skip_reason = f"exception_{type(exc).__name__}"
        result.needs_clarification = False
        return result
