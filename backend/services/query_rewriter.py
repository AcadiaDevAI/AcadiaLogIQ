"""
Same-session query rewriter (Brief 3). Given the user's raw turn and
the last N messages from the same session, produce a self-contained
version of the query that can be answered without looking at prior
context. Conservative bias: when in doubt, return the original.

Never raises. On disabled/timeout/parse/validate errors the sentinel
returns the original query with reason set accordingly.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from time import monotonic
from typing import Any, Dict, List, Optional, Tuple

from backend.config import settings
from backend.services.bedrock_haiku import haiku_client, _extract_json_object
from backend.services.token_usage import record_token_usage, extract_bedrock_usage

logger = logging.getLogger("acadia-log-iq")


_SYSTEM_PROMPT = """You rewrite user queries in a document-analysis chat so that each query is fully
self-contained and can be answered without any prior context.

You receive:
1. The last few turns of the conversation (oldest to newest).
2. The user's current query.

Your job: if the current query depends on prior context (pronouns, ellipsis, ordinals,
filter swaps, or implicit subject), produce a rewritten query that stands alone.
Otherwise, return the original query unchanged.

Return a JSON object with EXACTLY these keys:
- rewritten_query: the full, self-contained version of the query (or the original
  if no rewrite is needed)
- was_rewritten: true if you changed the query, false if you kept it
- reason: one of "self_contained", "pronoun_resolved", "ordinal_resolved",
  "ellipsis_expanded", "filter_swap", "no_history_available", "ambiguous_kept_original"
- confidence: 0.0-1.0 — how sure you are the rewrite is correct

Rules:
- If the current query already contains a specific identifier (like INC-10015,
  PROJ-123, a named customer, or an explicit subject), return it UNCHANGED.
- Resolve "it", "that one", "this", "them", "the ticket", "the incident" against the
  most recently discussed subject in the conversation.
- Resolve "the first one", "the second one", "the top one", "the highest" against
  lists or rankings in the previous assistant message.
- For ellipsis like "and the resolution?" or "same for Nebula-Corp?", expand into a
  full question that references the previously-discussed subject or swaps the filter.
- If there is NO prior conversation OR the reference is too vague to resolve
  confidently, KEEP the original query and set reason="ambiguous_kept_original".
- NEVER invent an identifier that wasn't in the conversation history.
- NEVER add information the user didn't say.
- Keep the rewrite concise — don't pad with summaries or explanations.

CRITICAL RULE — self-contained queries must NEVER be rewritten:

A query is self-contained if it has ALL of these:
- Its own complete subject (e.g., "tickets", "incidents", a specific identifier)
- Its own complete predicate/filter (e.g., "P1", "Nebula-Corp", "missed SLA", "root cause")
- Does NOT start with a fragment marker ("and", "what about", "same for", "also",
  "how about") AND does not consist solely of a pronoun reference.

If the current query is self-contained, return it UNCHANGED with
reason="self_contained" — even if the prior turn discussed a related topic.
Topic continuity ("still asking about tickets") is NOT the same as ellipsis
("user omitted the required subject"). Do NOT inject filters from prior turns
into a self-contained query.

Ellipsis expansion is ONLY appropriate when the current query is a fragment that
cannot stand alone. Examples:
- "and the resolution?"        → ellipsis (no subject)
- "same for Enterprise-859?"   → ellipsis (no subject/predicate)
- "what about the second one?" → ordinal (needs list from prior turn)
- "who resolved it?"           → pronoun (needs antecedent)

Examples of self-contained queries that MUST pass through unchanged:
- "How many P1 tickets?"                 → self_contained (subject=tickets, filter=P1)
- "How many tickets missed SLA?"         → self_contained
- "What caused INC-10015?"               → self_contained (has identifier)
- "Tell me about Nebula-Corp's tickets"  → self_contained
- "What is the highest quality score?"   → self_contained

Output ONLY the JSON object — no preamble, no markdown fencing, no commentary."""


_FEW_SHOTS = """History:
[user] Tell me about INC-10015
[assistant] INC-10015 was a P1 incident at Enterprise-792 where 50 barcode scanners...

Current query: what caused it?
JSON: {"rewritten_query": "What caused INC-10015?", "was_rewritten": true, "reason": "pronoun_resolved", "confidence": 0.95}

---

History:
[user] How many Nebula-Corp tickets?
[assistant] Nebula-Corp has 9 tickets: INC-10000, INC-10007, INC-10018, INC-10023, INC-10027, INC-10034, INC-10035, INC-10043, INC-10044.

Current query: what about the second one?
JSON: {"rewritten_query": "Tell me about INC-10007.", "was_rewritten": true, "reason": "ordinal_resolved", "confidence": 0.9}

---

History:
[user] How many P1 tickets missed SLA?
[assistant] 8 P1 tickets missed SLA.

Current query: and the resolution?
JSON: {"rewritten_query": "What were the resolutions for the 8 P1 tickets that missed SLA?", "was_rewritten": true, "reason": "ellipsis_expanded", "confidence": 0.88}

---

History:
[user] List all P1 tickets for Nebula-Corp.
[assistant] Nebula-Corp has 5 P1 tickets: INC-10000, INC-10007, INC-10027, INC-10035, INC-10043.

Current query: same for Enterprise-859?
JSON: {"rewritten_query": "List all P1 tickets for Enterprise-859.", "was_rewritten": true, "reason": "filter_swap", "confidence": 0.94}

---

History:
[user] What was the root cause of INC-10000?
[assistant] An outdated SBC license from Ribbon...

Current query: What was the root cause of INC-10015?
JSON: {"rewritten_query": "What was the root cause of INC-10015?", "was_rewritten": false, "reason": "self_contained", "confidence": 0.99}

---

History: (empty)

Current query: what about it?
JSON: {"rewritten_query": "what about it?", "was_rewritten": false, "reason": "no_history_available", "confidence": 0.95}

---

History:
[user] hello
[assistant] Hello! Ask me anything about your uploaded documents.

Current query: that thing we talked about
JSON: {"rewritten_query": "that thing we talked about", "was_rewritten": false, "reason": "ambiguous_kept_original", "confidence": 0.85}

---

History:
[user] How many Nebula-Corp tickets?
[assistant] Nebula-Corp has 9 tickets: INC-10000, INC-10007, INC-10018, INC-10023, INC-10027, INC-10034, INC-10035, INC-10043, INC-10044.

Current query: How many P1 tickets?
JSON: {"rewritten_query": "How many P1 tickets?", "was_rewritten": false, "reason": "self_contained", "confidence": 0.98}

---

History:
[user] How many Nebula-Corp tickets?
[assistant] Nebula-Corp has 9 tickets.

Current query: How many tickets missed SLA?
JSON: {"rewritten_query": "How many tickets missed SLA?", "was_rewritten": false, "reason": "self_contained", "confidence": 0.97}

---

History:
[user] Tell me about INC-10015
[assistant] INC-10015 was a P1 incident at Enterprise-792 where 50 barcode scanners...

Current query: What caused INC-10020?
JSON: {"rewritten_query": "What caused INC-10020?", "was_rewritten": false, "reason": "self_contained", "confidence": 0.99}"""


_ALLOWED_REASONS = {
    "self_contained",
    "pronoun_resolved",
    "ordinal_resolved",
    "ellipsis_expanded",
    "filter_swap",
    "no_history_available",
    "ambiguous_kept_original",
    "too_short",
    "disabled",
    "timeout",
    "parse_error",
    "invalid_output",
    "error",
    "hallucinated_identifier",
    "over_eager_rewrite_rejected",
}

_ASSISTANT_TRUNCATE_CHARS = 500
_MAX_REWRITE_CHARS = 500

# Rough identifier shape — any ALL-CAPS prefix + dash + digits.
_IDENTIFIER_SHAPE_RE = re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b")


@dataclass
class RewriteResult:
    rewritten_query: str
    was_rewritten: bool
    reason: str
    confidence: float
    raw_response: str
    original_query: str


# --- Brief 4 / Opt 1: rewriter micro-cache -------------------------------
_rewrite_cache_lock = threading.Lock()
_rewrite_cache: "OrderedDict[str, Tuple[RewriteResult, float]]" = OrderedDict()


def _cache_key(query: str, recent_messages: Optional[List[Dict[str, Any]]]) -> str:
    q = (query or "").strip().lower()
    last_assistant = ""
    if recent_messages:
        for msg in reversed(recent_messages):
            if isinstance(msg, dict) and str(msg.get("role", "")).lower() == "assistant":
                content = msg.get("content")
                if content:
                    last_assistant = str(content)[:500]
                    break
    h = hashlib.sha256(last_assistant.encode("utf-8")).hexdigest()[:16]
    return f"{q}||{h}"


def _cache_get(key: str) -> Optional[RewriteResult]:
    if not getattr(settings, "QUERY_REWRITER_CACHE_ENABLED", False):
        return None
    ttl = max(0, int(getattr(settings, "QUERY_REWRITER_CACHE_TTL_SECONDS", 60)))
    now = monotonic()
    with _rewrite_cache_lock:
        entry = _rewrite_cache.get(key)
        if not entry:
            return None
        result, ts = entry
        if now - ts > ttl:
            _rewrite_cache.pop(key, None)
            return None
        _rewrite_cache.move_to_end(key)
        return result


def _cache_put(key: str, result: RewriteResult) -> None:
    if not getattr(settings, "QUERY_REWRITER_CACHE_ENABLED", False):
        return
    cap = max(1, int(getattr(settings, "QUERY_REWRITER_CACHE_MAX_ENTRIES", 1000)))
    with _rewrite_cache_lock:
        _rewrite_cache[key] = (result, monotonic())
        _rewrite_cache.move_to_end(key)
        while len(_rewrite_cache) > cap:
            _rewrite_cache.popitem(last=False)


def cache_stats() -> Dict[str, int]:
    with _rewrite_cache_lock:
        return {"entries": len(_rewrite_cache)}


def cache_clear() -> None:
    with _rewrite_cache_lock:
        _rewrite_cache.clear()


def _sentinel(query: str, reason: str, raw: str = "", confidence: float = 0.0) -> RewriteResult:
    return RewriteResult(
        rewritten_query=query,
        was_rewritten=False,
        reason=reason,
        confidence=confidence,
        raw_response=raw,
        original_query=query,
    )


def _format_history(messages: List[Dict[str, Any]]) -> str:
    """
    Render up to QUERY_REWRITER_MAX_HISTORY_TURNS messages as a plain-text
    block, truncating assistant messages to _ASSISTANT_TRUNCATE_CHARS so
    long answers don't blow the token budget.
    """
    cap = max(1, int(getattr(settings, "QUERY_REWRITER_MAX_HISTORY_TURNS", 8)))
    trimmed = [m for m in (messages or []) if isinstance(m, dict)]
    # Keep the most recent `cap` messages.
    trimmed = trimmed[-cap:]

    lines: List[str] = []
    for msg in trimmed:
        role = str(msg.get("role", "")).lower()
        if role not in {"user", "assistant"}:
            continue
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        if role == "assistant" and len(content) > _ASSISTANT_TRUNCATE_CHARS:
            content = content[:_ASSISTANT_TRUNCATE_CHARS].rstrip() + " …"
        lines.append(f"[{role}] {content}")
    return "\n".join(lines)


def _strip_current_query_echo(
    messages: Optional[List[Dict[str, Any]]],
    current_query: str,
) -> List[Dict[str, Any]]:
    """
    If the caller already persisted the current user turn before fetching
    history, the most recent message will be an exact echo of the current
    query. Drop it so the rewriter only sees PRIOR turns.
    """
    if not messages:
        return []
    cur = (current_query or "").strip().lower()
    filtered = list(messages)
    if (
        filtered
        and isinstance(filtered[-1], dict)
        and str(filtered[-1].get("role", "")).lower() == "user"
        and str(filtered[-1].get("content") or "").strip().lower() == cur
    ):
        filtered.pop()
    return filtered


def _collect_identifiers(text: str) -> set:
    return {m.group(0).upper() for m in _IDENTIFIER_SHAPE_RE.finditer(text or "")}


_FRAGMENT_MARKERS = (
    "and ", "and the ", "what about", "same for", "also",
    "how about", "and for", "what of",
)
_PRONOUN_TOKENS = {"it", "that", "this", "them", "those", "these", "one"}
_PRIORITY_TOKEN_RE = re.compile(r"\bP[1-4]\b", re.IGNORECASE)
_SLA_TOKEN_RE = re.compile(r"\bSLA\b", re.IGNORECASE)


# Sprint 2.8.1 — true-ellipsis structural signals. The rewriter was
# classifying complete, standalone queries as ellipsis_expanded and
# injecting stale customer scope from prior turns. A real ellipsis
# must fail to stand alone — that means either a referential pronoun
# at the head, a sentence-continuation conjunction, or a tiny query
# with no named entity to anchor it.
_REFERENTIAL_PRONOUNS = frozenset({
    "it", "that", "those", "them", "these", "this", "one", "ones",
    "they", "both", "either", "neither",
})
_SENTENCE_CONJUNCTIONS = frozenset({
    "and", "but", "or", "also", "plus", "additionally",
})
_NAMED_ENTITY_KEYWORD_RE = re.compile(
    r"\b(score|priority|status|customer|sla|rework|bgp|mpls|ospf|bgp|"
    r"vpn|firewall|citrix|exadata|oracle|ticket|ticketid|incident)\b",
    re.I,
)
_NAMED_ENTITY_ID_RE = re.compile(r"\b[A-Z]+-[A-Z0-9]+-\d+\b|\b[A-Z]+-\d+\b")


def _is_true_ellipsis(query: str) -> bool:
    """Sprint 2.8.1 — return True only when the query genuinely depends
    on prior context to be answerable. False for any query that can
    stand alone (has named entity / filter keyword / identifier /
    sufficient length and no fragment marker)."""
    q = (query or "").strip()
    if not q:
        return False
    q_lower = q.lower()
    tokens = q_lower.split()
    if not tokens:
        return False

    head = tokens[0].strip(".,?!:;")

    # Signal 1 — starts with a referential pronoun.
    if head in _REFERENTIAL_PRONOUNS:
        return True
    # Signal 2 — starts with a sentence-continuation conjunction.
    if head in _SENTENCE_CONJUNCTIONS:
        return True
    # Signal 3 — very short query with no named-entity anchor.
    # Check against the original-cased query so we can see capital
    # words like "BGP" / "Aetheris".
    if len(tokens) < 5:
        has_named_entity = (
            any(c.isupper() for c in query)
            or _NAMED_ENTITY_ID_RE.search(query) is not None
            or _NAMED_ENTITY_KEYWORD_RE.search(query) is not None
        )
        if not has_named_entity:
            return True

    return False


def _should_expand_ellipsis(
    query: str,
    confidence: float,
    min_conf: float,
) -> bool:
    """Sprint 2.8.1 — gated ellipsis expansion.

    Flag-off: only the raised confidence threshold applies — this
    alone prevents the observed 0.85 over-eager rewrite while staying
    within the rewriter's existing contract.

    Flag-on: ALSO require that the query shows a real structural
    ellipsis signal. Complete, standalone queries no longer take the
    expansion path even if the LLM returned ellipsis_expanded with
    high confidence."""
    if confidence < min_conf:
        return False
    if not getattr(settings, "LOGIQ_REWRITER_STRICT_ELLIPSIS", False):
        return True
    return _is_true_ellipsis(query)


def _looks_like_fragment(text: str) -> bool:
    """A query is a fragment if it starts with an ellipsis marker or is a
    pure pronoun/ordinal reference that can't stand alone."""
    t = (text or "").strip().lower()
    if not t:
        return True
    for marker in _FRAGMENT_MARKERS:
        if t.startswith(marker):
            return True
    # Strip trailing punctuation and check for lone pronouns.
    stripped = re.sub(r"[^\w\s-]", "", t).strip()
    tokens = stripped.split()
    if len(tokens) <= 3 and tokens and tokens[0] in _PRONOUN_TOKENS:
        return True
    return False


def _rewrite_is_safe(original: str, rewritten: str) -> bool:
    """Defensive guard against over-eager rewrites. Reject a rewrite that
    appears to inject context-carried filters into an already self-contained
    query. Returns True only when the rewrite looks justified."""
    if not getattr(settings, "REWRITER_SAFETY_GUARD_ENABLED", True):
        return True
    orig = (original or "").strip()
    rewr = (rewritten or "").strip()
    if not rewr or rewr.lower() == orig.lower():
        return True
    # Only police rewrites of queries that already look self-contained —
    # i.e. not starting with a fragment marker and not a pure pronoun.
    if _looks_like_fragment(orig):
        return True
    orig_lower = orig.lower()
    rewr_lower = rewr.lower()
    # Guard 1: priority token injection. If rewrite contains P1..P4 and the
    # original does not, the LLM carried a priority filter from history.
    orig_prio = set(m.group(0).upper() for m in _PRIORITY_TOKEN_RE.finditer(orig))
    rewr_prio = set(m.group(0).upper() for m in _PRIORITY_TOKEN_RE.finditer(rewr))
    if rewr_prio - orig_prio:
        return False
    # Guard 2: SLA-filter injection.
    if _SLA_TOKEN_RE.search(rewr) and not _SLA_TOKEN_RE.search(orig):
        return False
    # Guard 3: customer/identifier injection. Any Alnum-hyphen token (e.g.
    # Nebula-Corp, Enterprise-859) that appears in the rewrite but not in
    # the original is a filter-injection signal.
    token_pattern = re.compile(r"\b[A-Z][A-Za-z0-9]*-[A-Za-z0-9]+\b")
    orig_tokens = {m.group(0).lower() for m in token_pattern.finditer(orig)}
    rewr_tokens = {m.group(0).lower() for m in token_pattern.finditer(rewr)}
    injected = rewr_tokens - orig_tokens
    # Allow injected identifiers that look like ticket IDs (prefix-digits
    # only) — those are covered by the separate hallucination guard which
    # checks against history. This guard catches textual customer names.
    id_shape = re.compile(r"^[A-Z][A-Z0-9]+-\d+$", re.IGNORECASE)
    injected_non_ids = {t for t in injected if not id_shape.match(t)}
    if injected_non_ids:
        return False
    return True


def _invoke_bedrock(prompt: str, max_tokens: int, temperature: float) -> str:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "system": _SYSTEM_PROMPT,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ],
    }
    response = haiku_client.client.invoke_model(
        modelId=settings.QUERY_REWRITER_MODEL,
        body=json.dumps(body).encode("utf-8"),
        accept="application/json",
        contentType="application/json",
    )
    payload = json.loads(response["body"].read().decode("utf-8"))
    record_token_usage("query_rewrite", settings.QUERY_REWRITER_MODEL, *extract_bedrock_usage(payload, response))
    content = payload.get("content", [])
    return "\n".join(
        item.get("text", "") for item in content if item.get("type") == "text"
    ).strip()


def rewrite_query(
    query: str,
    recent_messages: Optional[List[Dict[str, Any]]],
) -> RewriteResult:
    """
    Return a self-contained version of `query` using the last few turns.

    Never raises. On disabled/timeout/error returns RewriteResult with
    rewritten_query=query, was_rewritten=False, reason="<cause>".

    recent_messages shape: [{"role": "user"|"assistant", "content": "..."}],
    oldest first. If the caller already persisted the current user turn,
    this function strips the trailing echo automatically.
    """
    q = (query or "").strip()
    if not q:
        return _sentinel(query or "", "invalid_output")

    if not getattr(settings, "QUERY_REWRITER_ENABLED", False):
        logger.info("[rewriter] skipped (disabled) query=%r", q[:120])
        return _sentinel(q, "disabled")

    min_words = max(1, int(getattr(settings, "QUERY_REWRITER_MIN_QUERY_LEN_FOR_REWRITE", 2)))
    if len(q.split()) < min_words:
        logger.info("[rewriter] skipped (too_short) query=%r", q[:120])
        return _sentinel(q, "too_short", confidence=1.0)

    history = _strip_current_query_echo(recent_messages, q)
    if not history:
        logger.info("[rewriter] skipped (no_history_available) query=%r", q[:120])
        return _sentinel(q, "no_history_available", confidence=0.95)

    history_block = _format_history(history)
    if not history_block:
        logger.info("[rewriter] skipped (no_history_available) query=%r", q[:120])
        return _sentinel(q, "no_history_available", confidence=0.95)

    cache_key = _cache_key(q, history)
    cached = _cache_get(cache_key)
    if cached is not None:
        logger.info(
            "[rewriter] cache HIT key=%s... reason=%s confidence=%.2f",
            cache_key[:20], cached.reason, cached.confidence,
        )
        return cached

    prompt = (
        f"{_FEW_SHOTS}\n\n---\n\nHistory:\n{history_block}\n\nCurrent query: {q}\nJSON:"
    )

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                _invoke_bedrock,
                prompt,
                settings.QUERY_REWRITER_MAX_TOKENS,
                settings.QUERY_REWRITER_TEMPERATURE,
            )
            raw = future.result(timeout=settings.QUERY_REWRITER_TIMEOUT_SECONDS)
    except concurrent.futures.TimeoutError:
        logger.warning(
            "[rewriter] timeout after %.1fs query=%r",
            settings.QUERY_REWRITER_TIMEOUT_SECONDS, q[:120],
        )
        return _sentinel(q, "timeout")
    except Exception as exc:
        logger.warning("[rewriter] error (%s): %s", type(exc).__name__, exc)
        return _sentinel(q, "error", raw=f"<error: {exc}>")

    if not raw:
        logger.warning("[rewriter] error (empty response) query=%r", q[:120])
        return _sentinel(q, "error", raw="<empty>")

    cleaned = _extract_json_object(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning(
            "[rewriter] error (parse_error): %s raw=%r query=%r",
            exc, raw[:300], q[:120],
        )
        return _sentinel(q, "parse_error", raw=raw)

    rewritten = data.get("rewritten_query")
    was_rewritten = bool(data.get("was_rewritten"))
    reason = data.get("reason", "ambiguous_kept_original")
    if reason not in _ALLOWED_REASONS:
        reason = "ambiguous_kept_original"

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    if not isinstance(rewritten, str) or not rewritten.strip():
        logger.warning("[rewriter] error (invalid_output): missing rewritten_query raw=%r", raw[:300])
        return _sentinel(q, "invalid_output", raw=raw)

    rewritten = rewritten.strip()
    if len(rewritten) > _MAX_REWRITE_CHARS:
        logger.warning(
            "[rewriter] error (invalid_output): rewrite too long (%d chars)", len(rewritten),
        )
        return _sentinel(q, "invalid_output", raw=raw)

    # Guard: if was_rewritten is true but the text is identical (case-insensitive),
    # treat it as no-op rather than letting the LLM claim a phantom rewrite.
    if was_rewritten and rewritten.lower() == q.lower():
        was_rewritten = False
        reason = "self_contained"

    # Hallucination guard: any identifier in the rewrite that wasn't in the
    # original query OR in history is disallowed — drop the rewrite.
    if was_rewritten:
        rewrite_ids = _collect_identifiers(rewritten)
        allowed_ids = _collect_identifiers(q) | _collect_identifiers(history_block)
        phantom = rewrite_ids - allowed_ids
        if phantom:
            logger.warning(
                "[rewriter] dropping rewrite (hallucinated_identifier=%s) original=%r rewrite=%r",
                sorted(phantom), q[:120], rewritten[:120],
            )
            return _sentinel(q, "hallucinated_identifier", raw=raw, confidence=confidence)

    # Over-eagerness guard: reject rewrites that inject filters into an
    # already-self-contained query (e.g. "How many P1 tickets?" →
    # "How many P1 tickets does Nebula-Corp have?").
    if was_rewritten and not _rewrite_is_safe(q, rewritten):
        logger.warning(
            "[rewriter] dropping rewrite (over_eager_rewrite_rejected) original=%r rewrite=%r",
            q[:120], rewritten[:120],
        )
        return _sentinel(q, "over_eager_rewrite_rejected", raw=raw, confidence=confidence)

    # Sprint 2.8.1 — ellipsis_expanded gate. Observed incident: the
    # LLM returned reason=ellipsis_expanded at confidence=0.85 on a
    # complete standalone query (`present me the BGP related incidents
    # with Resolution_Quality_Score of 5`) and injected `for Aetheris
    # Corp` from session history. Raised threshold (0.85 → 0.92) and,
    # under LOGIQ_REWRITER_STRICT_ELLIPSIS, require a true-fragment
    # structural signal before honoring ellipsis_expanded.
    if was_rewritten and reason == "ellipsis_expanded":
        min_conf = float(
            getattr(settings, "REWRITER_ELLIPSIS_MIN_CONF", 0.92)
        )
        if not _should_expand_ellipsis(q, confidence, min_conf):
            logger.info(
                "[rewriter] dropping rewrite (ellipsis_not_true_fragment, conf=%.2f) "
                "original=%r rewrite=%r",
                confidence, q[:120], rewritten[:120],
            )
            return _sentinel(
                q, "over_eager_rewrite_rejected", raw=raw, confidence=confidence,
            )

    result = RewriteResult(
        rewritten_query=rewritten if was_rewritten else q,
        was_rewritten=was_rewritten,
        reason=reason,
        confidence=confidence,
        raw_response=raw,
        original_query=q,
    )

    if was_rewritten:
        logger.info(
            "[rewriter] rewrote (%s, conf=%.2f) original=%r → rewritten=%r",
            reason, confidence, q, rewritten,
        )
    else:
        logger.info(
            "[rewriter] kept original (%s, conf=%.2f) query=%r",
            reason, confidence, q,
        )
    _cache_put(cache_key, result)
    return result
