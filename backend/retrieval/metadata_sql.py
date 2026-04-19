"""
Metadata SQL Fast-Path — direct JSONB aggregation over `chunks.metadata_json`
for queries whose answer is a count or list over structured ticket fields.

Why this exists: full RAG is wasteful (and often wrong) for queries like
"How many Nebula-Corp tickets?" or "Which tickets missed SLA?". Those are
set-operations on known columns, not semantic search. A parameterized SQL
query returns the correct answer in milliseconds and with zero LLM spend.

This module is self-contained and import-safe. SQL is built exclusively with
`text()` + `bindparams()` — never string concatenation — to avoid injection.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from sqlalchemy import bindparam, text

from backend.config import settings
from backend.db.connection import SessionLocal

logger = logging.getLogger("acadia-log-iq")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class AggIntent:
    """
    Parsed aggregation intent.

    operation ∈ {
        "count",              # how many tickets match the filters
        "list",               # enumerate matching tickets
        "rank",               # order tickets by a numeric field
        "group_by_customer",  # tally tickets per customer and rank customers
    }
    """
    operation: str
    customer_name: Optional[str] = None
    priority: Optional[str] = None
    sla_target_met: Optional[bool] = None
    component: Optional[str] = None
    ranking_field: Optional[str] = None
    ranking_direction: Optional[Literal["highest", "lowest"]] = None
    # Bug 1 / Patch 4 — "more than N" queries use HAVING COUNT >= min_count.
    min_count: Optional[int] = None
    # Bug 3 — rework filter. None = no filter; True/False = must match.
    rework_detected: Optional[bool] = None
    # Bug 4B — numeric_field drives avg/sum/min/max operations.
    numeric_field: Optional[str] = None
    raw_query: str = ""


@dataclass
class AggResult:
    """Outcome of an aggregation run. Caller inspects `count` to decide
    whether to short-circuit or fall through to full RAG."""
    count: int = 0
    incident_numbers: List[str] = field(default_factory=list)
    prose_summary: str = ""
    filters_applied: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Detector patterns
# ---------------------------------------------------------------------------
# Aggregation verbs at the start of the query. Goal 3.1 broadens the phrasings
# we accept so natural variants ("number of incidents for X", "how many tickets
# missed SLA", "which tickets for X") hit the SQL fast-path instead of RAG.
_AGG_COUNT_RE = re.compile(
    r"^\s*(how\s+many|count|number\s+of)\s+(incidents?|tickets?|cases?|issues?)\b",
    re.I,
)
_AGG_LIST_RE = re.compile(
    r"^\s*(list|show)(\s+(all|me\s+all|me))?\s+(incidents?|tickets?|cases?|issues?)\b",
    re.I,
)
# "which tickets where/with/that/for ..." — list-style question.
_AGG_WHICH_TICKETS_RE = re.compile(
    r"^\s*which\s+(incidents?|tickets?|cases?|issues?)\b",
    re.I,
)

# Bug 2 (AGGREGATION_SQL_FIX_BRIEF) — customer-before-noun count phrasings,
# e.g. "How many Nebula-Corp tickets?". The current _AGG_COUNT_RE requires the
# noun ("tickets") directly after "how many", so these never matched.
_AGG_COUNT_CUSTOMER_BEFORE_RE = re.compile(
    r"^\s*(?:how\s+many|count\s+of|number\s+of)"
    r"\s+(?P<customer>[A-Za-z][A-Za-z0-9\-]+(?:\s+[A-Za-z0-9\-]+)?)"
    r"\s+(?:tickets?|incidents?|cases?|issues?)\b",
    re.I,
)
# Bug 2 — noun-first with explicit marker, e.g. "How many tickets for Nebula-Corp".
# Captures the customer directly so we don't depend on the separate _extract_customer
# chain for these phrasings.
_AGG_COUNT_CUSTOMER_AFTER_RE = re.compile(
    r"^\s*(?:how\s+many|count\s+of|number\s+of)"
    r"\s+(?:tickets?|incidents?|cases?|issues?)"
    r"\s+(?:for|by|filed\s+for|at|from)"
    r"\s+(?P<customer>[A-Za-z][A-Za-z0-9\-]+(?:\s+[A-Za-z0-9\-]+)?)\b",
    re.I,
)

# Ranking: "which ticket had the highest resolution quality score".
# The trailing capture is the field being ranked; we map it to a known column.
_RANKING_RE = re.compile(
    r"^\s*(which|what)\s+(ticket|incident|case)s?\s+"
    r"(had|has|have|got|scored)\s+(the\s+)?"
    r"(highest|lowest|best|worst|top|bottom)\s+"
    r"([A-Za-z][A-Za-z0-9 _\-]{2,}?)"
    r"(\s*[?.]|$)",
    re.I,
)

# Customer grouping: "which customer had the most incidents".
_GROUP_BY_CUSTOMER_RE = re.compile(
    r"^\s*which\s+(customer|client|enterprise|company)s?\s+"
    r"(had|has|have)\s+(the\s+)?(most|fewest|least|highest|lowest)\s+"
    r"(incidents?|tickets?|issues?|cases?)\b",
    re.I,
)

_CUSTOMER_NEBULA_RE = re.compile(r"\bNebula-Corp\b", re.I)
_CUSTOMER_ENTERPRISE_RE = re.compile(r"\bEnterprise-\d+\b", re.I)
# Generic "customer-X" / "customer X" / "for customer X" where X is a word.
_CUSTOMER_GENERIC_RE = re.compile(
    r"\bcustomer[\s-]+([A-Za-z][A-Za-z0-9_\-]{1,})\b",
    re.I,
)
# Hyphenated proper-noun fallback — catches Acme-Labs, Foo-Tech etc. when the
# user omits the word "customer". Safe because it requires an internal hyphen.
_CUSTOMER_HYPHEN_RE = re.compile(r"\b([A-Z][A-Za-z]+-[A-Za-z0-9][A-Za-z0-9_-]*)\b")

_PRIORITY_RE = re.compile(r"\b(P[1-5])\b|\b(critical|high|medium|low)\s+priority\b", re.I)

# Issue 2 — priority-aware parsing. The customer-before-noun regex used to
# greedily capture bare priority tokens ("How many P1 tickets?" → customer=P1).
# We extract any P[1-4] token up front, strip it from the query text handed to
# the customer regex, and reject any captured "customer" that's actually a
# priority code or other known non-customer token.
PRIORITY_PATTERN = re.compile(r"\b(P[1-4])\b", re.IGNORECASE)
NON_CUSTOMER_TOKENS = {
    "P1", "P2", "P3", "P4",
    "SLA", "SBC", "IP", "WIFI", "VPN", "DNS",
    "HIGH", "LOW", "MEDIUM", "OPEN", "CLOSED", "RESOLVED",
}


def _extract_priority_and_strip(query: str) -> Tuple[str, Optional[str]]:
    """Pull a P[1-4] token out of the query and return (stripped_query, priority).
    If the query has no priority token, priority is None and the query is
    returned unchanged. Whitespace collapsed after stripping."""
    q = query or ""
    if not getattr(settings, "REGEX_PRIORITY_AWARE_PARSING_ENABLED", True):
        return q, None
    m = PRIORITY_PATTERN.search(q)
    if not m:
        return q, None
    priority = m.group(1).upper()
    stripped = PRIORITY_PATTERN.sub(" ", q)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return stripped, priority

# SLA polarity — accept more natural phrasings on both sides.
_SLA_MISS_RE = re.compile(
    r"\b(missed|miss|breach(ed)?|violat(ed|ion)|failed|did\s+not\s+meet)\s+(their\s+)?sla\b",
    re.I,
)
_SLA_MET_RE = re.compile(
    r"\b(met|meeting|within|passed|satisfied)\s+(their\s+)?sla\b",
    re.I,
)

# Component category — conservative; only fires on explicit "component <word>".
_COMPONENT_RE = re.compile(r"\bcomponent\s+([A-Za-z][A-Za-z0-9_\-]{1,})\b", re.I)

# Rework polarity — FINAL_CLEANUP Bug 1 completeness. Regex-side extractor so
# "show me tickets where rework was detected" doesn't return the full universe
# when the LLM classifier is skipped. Matches the same shape as SLA polarity.
_REWORK_DETECTED_RE = re.compile(
    r"\b("
    r"rework(?:\s+was\s+detected|\s+detected|-detected)?"
    r"|reopened"
    r"|redone"
    r"|had\s+to\s+be\s+redone"
    r")\b",
    re.I,
)
_REWORK_CLEAN_RE = re.compile(
    r"\b(no\s+rework|without\s+rework|first[\s-]time|clean\s+(?:resolution|fix))\b",
    re.I,
)


# Map free-form ranking phrases to canonical metadata_json fields.
# Keep this small and explicit — new fields are easy to add, but matching must
# stay deterministic (longest-match-first so "resolution quality score" beats
# "quality score" which beats "resolution").
_RANKING_FIELD_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("resolution_quality_score", "resolution quality score"),
    ("resolution_quality_score", "quality score"),
    ("resolution_quality_score", "resolution score"),
    ("resolution_quality_score", "resolution quality"),
)

_HIGHEST_WORDS = {"highest", "best", "top"}
_LOWEST_WORDS = {"lowest", "worst", "bottom"}


# ---------------------------------------------------------------------------
# Bug 4A — content-filter rejection
# ---------------------------------------------------------------------------
# When a query mentions content-level concepts (equipment brands, change-request
# status, technical keywords that require scanning chunk text), the aggregation
# fast-path will silently return the wrong answer — either the universe of
# tickets or a generic list. These queries MUST fall through to hybrid
# retrieval + the agent pipeline so content is actually inspected.
_CONTENT_ONLY_KEYWORDS = [
    # Equipment / brand names — require a chunk-text scan to locate.
    re.compile(r"\bcisco\b", re.I),
    re.compile(r"\badtran\b", re.I),
    re.compile(r"\bnetgear\b", re.I),
    re.compile(r"\bjuniper\b", re.I),
    re.compile(r"\barista\b", re.I),
    re.compile(r"\bmotorola\b", re.I),
    re.compile(r"\bsymbol\b", re.I),
    # Change / process status terms that need content inspection.
    re.compile(r"\bchange\s+request\b", re.I),
    re.compile(r"\bfailed\s+change\b", re.I),
    re.compile(r"\bescalated\b", re.I),
    re.compile(r"\bvoided\b", re.I),
    re.compile(r"\breopened\s+from\b", re.I),
    # Technical concepts requiring content scan.
    re.compile(r"\bhardware\s+failure\b", re.I),
    re.compile(r"\bconfig(uration)?\s+error\b", re.I),
    re.compile(r"\bsoftware\s+bug\b", re.I),
    re.compile(r"\bauthentication\b", re.I),
    re.compile(r"\bdns\b", re.I),
    re.compile(r"\bvlan\b", re.I),
    re.compile(r"\bssid\b", re.I),
    re.compile(r"\bfirewall\b", re.I),
    re.compile(r"\bdhcp\b", re.I),
    # Content-semantic filter verbs.
    re.compile(r"\binvolved\b", re.I),
    re.compile(r"\bmentioned\b", re.I),
    re.compile(r"\bdiscussed\b", re.I),
    re.compile(r"\bcaused\s+by\b", re.I),
]


def _requires_content_retrieval(query: str) -> bool:
    """True if the query mentions content-level filters that aren't in the
    flat metadata_json columns. Such queries must go through RAG + agent
    pipeline, not the aggregation SQL fast-path."""
    if not getattr(settings, "AGG_CONTENT_FILTER_REJECT_ENABLED", True):
        return False
    q = query or ""
    for pat in _CONTENT_ONLY_KEYWORDS:
        if pat.search(q):
            return True
    return False


# Bug 4B — numeric aggregation operations and the columns they may target.
_NUMERIC_AGG_OPS = {"avg", "sum", "min", "max"}
_ALLOWED_NUMERIC_FIELDS_SQL = {
    "resolution_quality_score",
    "time_to_first_response_seconds",
}


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
def _match_ranking_field(field_phrase: str) -> Optional[str]:
    """Longest-match lookup for ranking field aliases."""
    phrase = field_phrase.strip().lower()
    for canonical, alias in _RANKING_FIELD_PATTERNS:
        if alias in phrase:
            return canonical
    return None


def _extract_customer(q: str) -> Optional[str]:
    """Pull the customer name out of the query using a priority chain."""
    if _CUSTOMER_NEBULA_RE.search(q):
        return "Nebula-Corp"
    m = _CUSTOMER_ENTERPRISE_RE.search(q)
    if m:
        return m.group(0)
    m = _CUSTOMER_GENERIC_RE.search(q)
    if m:
        return m.group(1)
    m = _CUSTOMER_HYPHEN_RE.search(q)
    if m:
        return m.group(1)
    return None


def detect_aggregation_intent(query: str) -> Optional[AggIntent]:
    """
    Return an AggIntent if the query is an aggregation over tickets,
    otherwise None. When returning None, emit a diagnostic log so operators
    can see which patterns were tried (Goal 3.2).
    """
    q = (query or "").strip()
    if not q:
        return None

    # ---- Try ranking first: "which ticket had the highest resolution quality score" ----
    m_rank = _RANKING_RE.match(q)
    if m_rank:
        direction_word = m_rank.group(5).lower()
        field_phrase = m_rank.group(6)
        canonical_field = _match_ranking_field(field_phrase)
        if canonical_field:
            direction = "highest" if direction_word in _HIGHEST_WORDS else "lowest"
            return AggIntent(
                operation="rank",
                ranking_field=canonical_field,
                ranking_direction=direction,
                raw_query=q,
            )

    # ---- Group-by-customer: "which customer had the most incidents" ----
    m_group = _GROUP_BY_CUSTOMER_RE.match(q)
    if m_group:
        direction_word = m_group.group(4).lower()
        # "most/highest" → rank desc; "fewest/least/lowest" → rank asc.
        direction = "highest" if direction_word in {"most", "highest"} else "lowest"
        return AggIntent(
            operation="group_by_customer",
            ranking_direction=direction,
            raw_query=q,
        )

    # ---- Count / list / which-tickets ----
    # Bug 2: check the customer-aware count patterns FIRST — they both imply
    # operation="count" and also hand us the customer_name directly. Pattern B
    # (noun-first with "for/by/…") is more specific, so try it before Pattern A
    # (customer-before-noun) to avoid Pattern A greedily swallowing the marker.
    # Issue 2: strip P[1-4] priority tokens from the query text we feed to the
    # customer-before-noun regex, so "How many P1 tickets?" no longer captures
    # "P1" as a customer. The stripped query is only used for customer-regex
    # matching — operation detection still uses the original `q`.
    q_for_customer, priority_token = _extract_priority_and_strip(q)
    # FINAL_CLEANUP Bug 1 — also strip rework-tokens from the customer-match
    # text so "rework-detected Nebula-Corp tickets" resolves to customer=
    # "Nebula-Corp" instead of "rework-detected Nebula-Corp".
    if getattr(settings, "REWORK_FILTER_ENABLED", True):
        q_for_customer = re.sub(
            r"\b(rework(?:-detected|\s+detected)?|reopened|redone)\s*",
            " ",
            q_for_customer,
            flags=re.I,
        )
        q_for_customer = re.sub(r"\s+", " ", q_for_customer).strip()
    customer_from_regex: Optional[str] = None
    m_count_after = _AGG_COUNT_CUSTOMER_AFTER_RE.search(q_for_customer)
    m_count_before = (
        None if m_count_after else _AGG_COUNT_CUSTOMER_BEFORE_RE.search(q_for_customer)
    )
    if m_count_after or m_count_before:
        captured = (m_count_after or m_count_before).group("customer")
        captured = (captured or "").rstrip(".,?!:;").strip() or None
        # Issue 2 — reject any capture that's actually a non-customer token
        # (priority, SLA, etc.). Compare case-insensitively.
        if captured and captured.upper() in NON_CUSTOMER_TOKENS:
            captured = None
        # Defensive — also reject bare P[1-4] matches that slipped through.
        if captured and PRIORITY_PATTERN.fullmatch(captured.strip()):
            captured = None
        if captured:
            operation = "count"
            customer_from_regex = captured
        elif _AGG_COUNT_RE.search(q) or _AGG_COUNT_RE.search(q_for_customer):
            # Customer capture was rejected, but the query is still a count.
            operation = "count"
        else:
            logger.info(
                "[aggregation] no intent match for query=%r "
                "(count=no, list=no, which=no, ranking=%s, group_by=%s)",
                q,
                "no" if not m_rank else "field-unmapped",
                "no" if not m_group else "yes-but-fell-through",
            )
            return None
    elif _AGG_COUNT_RE.search(q) or _AGG_COUNT_RE.search(q_for_customer):
        operation = "count"
    elif _AGG_LIST_RE.search(q) or _AGG_WHICH_TICKETS_RE.search(q):
        operation = "list"
    else:
        logger.info(
            "[aggregation] no intent match for query=%r "
            "(count=no, list=no, which=no, ranking=%s, group_by=%s)",
            q,
            "no" if not m_rank else "field-unmapped",
            "no" if not m_group else "yes-but-fell-through",
        )
        return None

    intent = AggIntent(operation=operation, raw_query=q)
    # Prefer the regex-captured customer (it came from the matching count pattern
    # itself, so it's definitionally the filter subject); fall back to the
    # priority chain in _extract_customer for list/which/unfiltered counts.
    intent.customer_name = customer_from_regex or _extract_customer(q)

    # Issue 2 — belt-and-braces: if _extract_customer returned a priority token
    # (shouldn't happen given the existing chain, but guard anyway), drop it.
    if intent.customer_name and intent.customer_name.upper() in NON_CUSTOMER_TOKENS:
        intent.customer_name = None
    if intent.customer_name and PRIORITY_PATTERN.fullmatch(intent.customer_name.strip()):
        intent.customer_name = None

    # Priority — prefer the token we already extracted (covers the stripped
    # case), fall back to the broader _PRIORITY_RE for "critical priority" etc.
    if priority_token:
        intent.priority = priority_token
    else:
        m = _PRIORITY_RE.search(q)
        if m:
            intent.priority = (m.group(1) or m.group(2) or "").upper()

    # SLA
    if _SLA_MISS_RE.search(q):
        intent.sla_target_met = False
    elif _SLA_MET_RE.search(q):
        intent.sla_target_met = True

    # Component
    m = _COMPONENT_RE.search(q)
    if m:
        intent.component = m.group(1)

    # FINAL_CLEANUP Bug 1 — rework polarity. Same shape as SLA: a "clean/no-rework"
    # phrase wins over a generic "rework" mention so "tickets with no rework" isn't
    # mis-flagged as rework_detected=True. Respects the same feature flag used by
    # _build_ticket_scope_clauses.
    if getattr(settings, "REWORK_FILTER_ENABLED", True):
        if _REWORK_CLEAN_RE.search(q):
            intent.rework_detected = False
        elif _REWORK_DETECTED_RE.search(q):
            intent.rework_detected = True

    # A bare "how many tickets?" with no filters is still a valid aggregation —
    # it's just "count all allowed tickets".
    return intent


# ---------------------------------------------------------------------------
# Two-tier detector (regex → LLM classifier)
# ---------------------------------------------------------------------------
# Classifier's "desc"/"asc" → AggIntent's "highest"/"lowest".
_CLASSIFIER_DIRECTION_MAP = {"desc": "highest", "asc": "lowest"}


def detect_aggregation_intent_v2(query: str) -> Optional[AggIntent]:
    """
    Two-tier aggregation intent detection:
      1. Regex fast-path (detect_aggregation_intent). If it matches, return.
      2. If regex misses AND classifier is enabled, call the LLM classifier.
         When is_aggregation=true with confidence ≥ threshold, build an
         AggIntent from the classifier result.
      3. Otherwise return None (fall through to RAG).

    Emits a [aggregation] routed via=regex|classifier line on success so
    production logs can attribute each hit to a tier.
    """
    q = (query or "").strip()
    if not q:
        return None

    # Bug 4A — content-level filters can't be satisfied by metadata SQL; reject
    # the fast-path up front so the query falls through to hybrid retrieval.
    if _requires_content_retrieval(q):
        logger.info(
            "[aggregation] rejecting fast-path: query requires content retrieval "
            "(keyword hit): %r", q,
        )
        return None

    # Tier 1 — regex
    regex_hit = detect_aggregation_intent(q)
    if regex_hit is not None:
        logger.info(
            "[aggregation] routed via=regex op=%s customer=%s",
            regex_hit.operation, regex_hit.customer_name,
        )
        return regex_hit

    # Tier 2 — classifier
    if not getattr(settings, "AGGREGATION_CLASSIFIER_ENABLED", False):
        return None

    # Lazy import — avoids pulling boto3 at module load time and keeps
    # metadata_sql.py importable in pure-SQL test contexts.
    from backend.services.aggregation_classifier import classify_aggregation_intent

    result = classify_aggregation_intent(q)
    if not result.is_aggregation:
        return None
    if result.confidence < settings.AGGREGATION_CLASSIFIER_MIN_CONFIDENCE:
        return None
    if result.operation is None:
        return None

    mapped_direction = _CLASSIFIER_DIRECTION_MAP.get(
        result.ranking_direction or "", None
    )
    # Bug 1 fix — classifier-routed group_by_customer used to default to ASC
    # when ranking_direction was missing, so "Break down tickets by customer"
    # led with Enterprise-125 (1 ticket) instead of Nebula-Corp (9). Real-world
    # group-by intent almost always wants the biggest buckets first, so default
    # to "highest" (DESC) when the operation is group_by_customer and the
    # classifier didn't return a direction.
    if (
        mapped_direction is None
        and result.operation == "group_by_customer"
        and getattr(settings, "GROUP_BY_DEFAULT_DESC_ENABLED", True)
    ):
        mapped_direction = "highest"

    intent = AggIntent(
        operation=result.operation,
        customer_name=result.customer_name,
        priority=result.priority,
        sla_target_met=result.sla_met,
        component=result.component,
        ranking_field=result.ranking_field,
        ranking_direction=mapped_direction,
        min_count=getattr(result, "min_count", None),
        rework_detected=getattr(result, "rework_detected", None),
        numeric_field=getattr(result, "numeric_field", None),
        raw_query=q,
    )
    # Bug 4B — if the operation is numeric (avg/sum/min/max) but no
    # whitelisted numeric_field was resolved, default to resolution_quality_score
    # when the query mentions "quality" / "score"; otherwise drop back to None
    # so the executor skips the op rather than returning a garbage answer.
    if (
        intent.operation in _NUMERIC_AGG_OPS
        and intent.numeric_field is None
        and getattr(settings, "NUMERIC_AGGREGATION_OPS_ENABLED", True)
    ):
        ql = q.lower()
        if "quality" in ql or "score" in ql:
            intent.numeric_field = "resolution_quality_score"
    logger.info(
        "[aggregation] routed via=classifier op=%s confidence=%.2f",
        intent.operation, result.confidence,
    )
    return intent


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------
def _build_prose_summary(intent: AggIntent, incident_numbers: List[str]) -> str:
    """Craft a short, LLM-free answer the caller can return verbatim."""
    count = len(incident_numbers)
    # Render filter subject — "Nebula-Corp" / "P1" / "SLA-missed" / generic.
    subject_parts: List[str] = []
    if intent.customer_name:
        subject_parts.append(intent.customer_name)
    if intent.priority:
        subject_parts.append(f"priority={intent.priority}")
    if intent.sla_target_met is True:
        subject_parts.append("SLA met")
    elif intent.sla_target_met is False:
        subject_parts.append("SLA missed")
    if intent.component:
        subject_parts.append(f"component={intent.component}")
    if intent.rework_detected is True:
        subject_parts.append("rework detected")
    elif intent.rework_detected is False:
        subject_parts.append("no rework")
    subject = " ".join(subject_parts) if subject_parts else "Tickets"

    if count == 0:
        return f"- No tickets match: {subject}."

    # Cap the id list in the prose to keep the line readable.
    shown = incident_numbers[:25]
    tail = "" if len(shown) == len(incident_numbers) else f", … (+{len(incident_numbers) - len(shown)} more)"
    ids_joined = ", ".join(shown) + tail
    if subject_parts:
        return f"- {subject} has {count} ticket{'s' if count != 1 else ''}: {ids_joined}."
    return f"- Found {count} ticket{'s' if count != 1 else ''}: {ids_joined}."


_RANKABLE_INT_FIELDS = {"resolution_quality_score"}


def _build_ticket_scope_clauses(
    *,
    owner_id: str,
    allowed_file_ids,
    filters: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """
    Build WHERE-fragment and bind-param dict shared by all three aggregation
    query builders. A clause is emitted ONLY when its filter value is not None
    — that's the whole point: the `:param IS NULL OR expr = :param` shape with
    a Python `None` binds fails under psycopg3 type inference and also trips
    SQLAlchemy when the same nullable param appears twice. Dropping the clause
    entirely means we never bind a None, so the error cannot occur.

    Recognized filter keys: customer_name, priority, sla_met, component.
    """
    clauses = [
        "d.owner_id = :owner_id",
        "LOWER(d.file_type) IN ('ticket', 'tickets', 'incident')",
        "d.status = 'active'",
        "dv.is_active = TRUE",
        "d.current_version_id = dv.id",
        "(c.metadata_json->>'incident_number') IS NOT NULL",
    ]
    params: Dict[str, Any] = {"owner_id": owner_id}

    if allowed_file_ids:
        clauses.append("d.id IN :allowed_ids")
        params["allowed_ids"] = tuple(str(x) for x in allowed_file_ids)

    cn = filters.get("customer_name")
    if cn is not None:
        clauses.append("LOWER(c.metadata_json->>'customer_name') = LOWER(:customer_name)")
        params["customer_name"] = cn

    pr = filters.get("priority")
    if pr is not None:
        clauses.append("UPPER(c.metadata_json->>'priority') = UPPER(:priority)")
        params["priority"] = pr

    sm = filters.get("sla_met")
    if sm is not None:
        clauses.append("(c.metadata_json->>'sla_target_met')::boolean = :sla_met")
        params["sla_met"] = bool(sm)

    comp = filters.get("component")
    if comp is not None:
        clauses.append("LOWER(c.metadata_json->>'component') = LOWER(:component)")
        params["component"] = comp

    # Bug 3 — rework filter. Reads from metadata_json->>'rework_detected',
    # which contextual_ingestion must populate (YES/NO → boolean) for the
    # filter to match any rows. When ingestion is behind, this clause simply
    # filters to zero tickets — no false positives.
    rw = filters.get("rework_detected")
    if rw is not None:
        clauses.append(
            "COALESCE(LOWER(c.metadata_json->>'rework_detected'), '') IN "
            "(CASE WHEN :rework THEN 'true' ELSE 'false' END, "
            " CASE WHEN :rework THEN 'yes' ELSE 'no' END, "
            " CASE WHEN :rework THEN '1' ELSE '0' END)"
        )
        params["rework"] = bool(rw)

    return " AND ".join(clauses), params


def _run_ranking(
    intent: AggIntent,
    owner_id: str,
    allowed_file_ids: Set[str],
) -> AggResult:
    """Execute a ranking query — top/bottom N tickets by a numeric field."""
    if intent.ranking_field not in _RANKABLE_INT_FIELDS:
        return AggResult()

    direction_sql = "DESC" if intent.ranking_direction == "highest" else "ASC"
    # Field name is whitelisted above, not user input — safe to interpolate.
    field = intent.ranking_field

    # Bug 2 — the WHERE clause must apply to the ENTIRE ranking result set,
    # not just the top row. Previously we passed filters={} here so "lowest
    # Nebula-Corp quality score" leaked other customers' tickets into the
    # "Others near" list. Feed the intent's metadata filters through so the
    # scope stays consistent across top-1 and rest.
    ranking_filters: Dict[str, Any] = {}
    if getattr(settings, "RANKING_CUSTOMER_SCOPE_FIX_ENABLED", True):
        if intent.customer_name is not None:
            ranking_filters["customer_name"] = intent.customer_name
        if intent.priority is not None:
            ranking_filters["priority"] = intent.priority
        if intent.sla_target_met is not None:
            ranking_filters["sla_met"] = intent.sla_target_met
        if intent.component is not None:
            ranking_filters["component"] = intent.component
        if intent.rework_detected is not None:
            ranking_filters["rework_detected"] = intent.rework_detected

    where_sql, params = _build_ticket_scope_clauses(
        owner_id=owner_id,
        allowed_file_ids=allowed_file_ids,
        filters=ranking_filters,
    )

    sql = text(f"""
        SELECT DISTINCT
            c.metadata_json->>'incident_number' AS incident_number,
            (c.metadata_json->>'{field}')::int  AS score,
            c.metadata_json->>'customer_name'   AS customer_name
        FROM chunks c
        JOIN documents d          ON d.id = c.document_id
        JOIN document_versions dv ON dv.id = c.document_version_id
        WHERE {where_sql}
          AND (c.metadata_json->>'{field}') IS NOT NULL
          AND (c.metadata_json->>'{field}') ~ '^[0-9]+$'
        ORDER BY score {direction_sql} NULLS LAST, incident_number
        LIMIT 5
    """).bindparams(bindparam("allowed_ids", expanding=True))

    rows: List[Dict[str, Any]] = []
    try:
        with SessionLocal() as db:
            rows = [dict(r) for r in db.execute(sql, params).mappings().all()]
    except Exception as exc:
        logger.warning("metadata_sql ranking failed: %s", exc)
        return AggResult()

    if not rows:
        logger.info("[aggregation] SQL fast-path: 0 results (ranking)")
        return AggResult()

    top = rows[0]
    rest = rows[1:]
    pretty_field = intent.ranking_field.replace("_", " ")
    extreme_word = "highest" if intent.ranking_direction == "highest" else "lowest"
    rest_parts = [f"{r['incident_number']} ({r['score']})" for r in rest]
    rest_phrase = f" Others near: {', '.join(rest_parts)}." if rest_parts else ""
    # Bug 2 — when a customer filter narrows the ranking scope, surface the
    # scope in the prose so users don't mistake the list for a global ranking.
    scope_prefix = ""
    if (
        getattr(settings, "RANKING_CUSTOMER_SCOPE_FIX_ENABLED", True)
        and intent.customer_name
    ):
        scope_prefix = f"Within {intent.customer_name}'s tickets, "
    prose = (
        f"- {scope_prefix}{top['incident_number']} has the {extreme_word} {pretty_field} "
        f"at {top['score']}.{rest_phrase}"
    )
    logger.info("[aggregation] SQL fast-path: %d results (ranking)", len(rows))
    return AggResult(
        count=len(rows),
        incident_numbers=[r["incident_number"] for r in rows],
        prose_summary=prose,
        filters_applied={
            "ranking_field": intent.ranking_field,
            "ranking_direction": intent.ranking_direction,
        },
    )


def _run_group_by_customer(
    intent: AggIntent,
    owner_id: str,
    allowed_file_ids: Set[str],
) -> AggResult:
    """Group tickets by customer_name and rank customers by ticket count."""
    direction_sql = "DESC" if intent.ranking_direction == "highest" else "ASC"

    # FINAL_CLEANUP Bug 2 — same structural bug the ranking path had: secondary
    # filters (priority, sla_met, rework_detected, component) were dropped when
    # building the GROUP BY scope. Thread them through _build_ticket_scope_clauses
    # so "Among P1 tickets, which customers had the most SLA misses?" narrows the
    # aggregate to P1∧sla_met=False instead of returning the global P1 distribution.
    group_filters: Dict[str, Any] = {}
    if getattr(settings, "GROUP_BY_FULL_FILTER_ENABLED", True):
        if intent.customer_name is not None:
            group_filters["customer_name"] = intent.customer_name
        if intent.priority is not None:
            group_filters["priority"] = intent.priority
        if intent.sla_target_met is not None:
            group_filters["sla_met"] = intent.sla_target_met
        if intent.component is not None:
            group_filters["component"] = intent.component
        if intent.rework_detected is not None:
            group_filters["rework_detected"] = intent.rework_detected

    where_sql, params = _build_ticket_scope_clauses(
        owner_id=owner_id,
        allowed_file_ids=allowed_file_ids,
        filters=group_filters,
    )

    # Bug 1 / Patch 4 — "more than N" queries carry min_count=N+1. We use
    # HAVING on the DISTINCT ticket count so the threshold matches the
    # number displayed to the user (one count per customer, not one per chunk).
    having_sql = ""
    if intent.min_count and intent.min_count > 0:
        having_sql = (
            f"HAVING COUNT(DISTINCT c.metadata_json->>'incident_number') "
            f">= {int(intent.min_count)}"
        )

    sql = text(f"""
        SELECT
            c.metadata_json->>'customer_name' AS customer,
            COUNT(DISTINCT c.metadata_json->>'incident_number') AS ticket_count
        FROM chunks c
        JOIN documents d          ON d.id = c.document_id
        JOIN document_versions dv ON dv.id = c.document_version_id
        WHERE {where_sql}
          AND (c.metadata_json->>'customer_name') IS NOT NULL
        GROUP BY customer
        {having_sql}
        ORDER BY ticket_count {direction_sql}, customer
        LIMIT 10
    """).bindparams(bindparam("allowed_ids", expanding=True))

    rows: List[Dict[str, Any]] = []
    try:
        with SessionLocal() as db:
            rows = [dict(r) for r in db.execute(sql, params).mappings().all()]
    except Exception as exc:
        logger.warning("metadata_sql group_by_customer failed: %s", exc)
        return AggResult()

    if not rows:
        logger.info("[aggregation] SQL fast-path: 0 results (group_by_customer)")
        return AggResult()

    top = rows[0]
    rest = rows[1:]
    extreme_word = "most" if intent.ranking_direction == "highest" else "fewest"
    rest_parts = [f"{r['customer']} ({r['ticket_count']})" for r in rest]
    rest_phrase = f" Next: {', '.join(rest_parts)}." if rest_parts else ""
    # FINAL_CLEANUP Bug 2 — echo the filter scope so "Nebula-Corp has the most (3)"
    # isn't mistaken for a global count when filters have narrowed the universe.
    scope_parts: List[str] = []
    if getattr(settings, "GROUP_BY_FULL_FILTER_ENABLED", True):
        if intent.priority:
            scope_parts.append(intent.priority)
        if intent.sla_target_met is True:
            scope_parts.append("SLA-met")
        elif intent.sla_target_met is False:
            scope_parts.append("SLA-missed")
        if intent.rework_detected is True:
            scope_parts.append("rework-detected")
        if intent.component:
            scope_parts.append(intent.component)
    scope_prefix = f"Among {' '.join(scope_parts)} tickets, " if scope_parts else ""
    prose = (
        f"- {scope_prefix}{top['customer']} has the {extreme_word}, with {top['ticket_count']} "
        f"ticket{'s' if top['ticket_count'] != 1 else ''}.{rest_phrase}"
    )
    logger.info(
        "[aggregation] SQL fast-path: %d results (group_by_customer)", len(rows)
    )
    return AggResult(
        count=int(top["ticket_count"]),
        incident_numbers=[r["customer"] for r in rows],
        prose_summary=prose,
        filters_applied={
            "operation": "group_by_customer",
            "ranking_direction": intent.ranking_direction,
        },
    )


def _run_numeric_aggregation(
    intent: AggIntent,
    owner_id: str,
    allowed_file_ids: Set[str],
) -> AggResult:
    """Bug 4B — avg/sum/min/max over a whitelisted numeric metadata_json field.

    Filters (customer, priority, sla_met, component, rework_detected) are
    pushed through `_build_ticket_scope_clauses` so "average quality score
    for Nebula-Corp" narrows to Nebula-Corp's tickets before the aggregate.
    """
    if not getattr(settings, "NUMERIC_AGGREGATION_OPS_ENABLED", True):
        return AggResult()
    op = (intent.operation or "").lower()
    if op not in _NUMERIC_AGG_OPS:
        return AggResult()
    field = intent.numeric_field or ""
    if field not in _ALLOWED_NUMERIC_FIELDS_SQL:
        return AggResult()

    op_sql = {"avg": "AVG", "sum": "SUM", "min": "MIN", "max": "MAX"}[op]

    filters: Dict[str, Any] = {}
    if intent.customer_name is not None:
        filters["customer_name"] = intent.customer_name
    if intent.priority is not None:
        filters["priority"] = intent.priority
    if intent.sla_target_met is not None:
        filters["sla_met"] = intent.sla_target_met
    if intent.component is not None:
        filters["component"] = intent.component
    if intent.rework_detected is not None:
        filters["rework_detected"] = intent.rework_detected

    where_sql, params = _build_ticket_scope_clauses(
        owner_id=owner_id,
        allowed_file_ids=allowed_file_ids,
        filters=filters,
    )

    # Per-ticket deduplication: chunks duplicate incident rows, so take the
    # numeric value once per incident via a subquery before applying the
    # aggregate. The inner `~ '^-?[0-9.]+$'` keeps non-numeric strings out.
    sql = text(f"""
        SELECT {op_sql}(val) AS result, COUNT(*) AS n_tickets
        FROM (
            SELECT DISTINCT ON (c.metadata_json->>'incident_number')
                (c.metadata_json->>'{field}')::float AS val
            FROM chunks c
            JOIN documents d          ON d.id = c.document_id
            JOIN document_versions dv ON dv.id = c.document_version_id
            WHERE {where_sql}
              AND (c.metadata_json->>'{field}') IS NOT NULL
              AND (c.metadata_json->>'{field}') ~ '^-?[0-9]+(\\.[0-9]+)?$'
            ORDER BY c.metadata_json->>'incident_number', c.id
        ) t
    """).bindparams(bindparam("allowed_ids", expanding=True))

    try:
        with SessionLocal() as db:
            row = db.execute(sql, params).mappings().first()
    except Exception as exc:
        logger.warning("metadata_sql numeric aggregation failed: %s", exc)
        return AggResult()

    if not row or row.get("result") is None:
        logger.info("[aggregation] SQL fast-path: 0 results (numeric %s)", op)
        return AggResult()

    result_val = float(row["result"])
    n_tickets = int(row.get("n_tickets") or 0)
    pretty_field = field.replace("_", " ")
    scope = ""
    if intent.customer_name:
        scope = f" for {intent.customer_name}"
    elif intent.priority:
        scope = f" for {intent.priority}"
    op_words = {"avg": "average", "sum": "total", "min": "minimum", "max": "maximum"}
    prose = (
        f"- The {op_words[op]} {pretty_field}{scope} is {result_val:.2f} "
        f"(across {n_tickets} ticket{'s' if n_tickets != 1 else ''})."
    )
    logger.info(
        "[aggregation] SQL fast-path: numeric op=%s field=%s result=%.2f n=%d",
        op, field, result_val, n_tickets,
    )
    return AggResult(
        count=n_tickets,
        incident_numbers=[],
        prose_summary=prose,
        filters_applied={
            "operation": op,
            "numeric_field": field,
            **{k: v for k, v in filters.items() if v is not None},
        },
    )


def _count_distinct_customers(
    owner_id: str,
    allowed_file_ids: Set[str],
) -> int:
    """Quick DISTINCT-customer count used in the total_count prose summary."""
    where_sql, params = _build_ticket_scope_clauses(
        owner_id=owner_id,
        allowed_file_ids=allowed_file_ids,
        filters={},
    )
    sql = text(f"""
        SELECT COUNT(DISTINCT c.metadata_json->>'customer_name') AS n
        FROM chunks c
        JOIN documents d          ON d.id = c.document_id
        JOIN document_versions dv ON dv.id = c.document_version_id
        WHERE {where_sql}
          AND (c.metadata_json->>'customer_name') IS NOT NULL
    """).bindparams(bindparam("allowed_ids", expanding=True))
    try:
        with SessionLocal() as db:
            row = db.execute(sql, params).mappings().first()
        return int(row["n"]) if row and row.get("n") is not None else 0
    except Exception as exc:
        logger.warning("metadata_sql _count_distinct_customers failed: %s", exc)
        return 0


def _build_total_count_prose(
    incident_numbers: List[str],
    owner_id: str,
    allowed_file_ids: Set[str],
) -> str:
    n = len(incident_numbers)
    if n == 0:
        return "- No tickets are currently indexed."
    customers = _count_distinct_customers(owner_id, allowed_file_ids)
    if customers > 0:
        return (
            f"- There are {n} ticket{'s' if n != 1 else ''} across {customers} "
            f"customer{'s' if customers != 1 else ''} in the uploaded documents."
        )
    return f"- There are {n} ticket{'s' if n != 1 else ''} in the uploaded documents."


def run_aggregation(
    intent: AggIntent,
    owner_id: str,
    allowed_file_ids: Set[str],
) -> AggResult:
    """
    Execute the aggregation against `chunks.metadata_json`. Returns an
    AggResult; caller may treat `count == 0` as "fall through to full RAG".

    SQL is fully parameterized — no string interpolation of user input.
    """
    if not allowed_file_ids:
        return AggResult()

    # Goal 3.3: specialized operations bypass the filter-based count/list SQL.
    if intent.operation == "rank":
        return _run_ranking(intent, owner_id, allowed_file_ids)
    if intent.operation == "group_by_customer":
        return _run_group_by_customer(intent, owner_id, allowed_file_ids)
    # Bug 4B — numeric aggregation (avg/sum/min/max) has its own executor.
    if intent.operation in _NUMERIC_AGG_OPS:
        return _run_numeric_aggregation(intent, owner_id, allowed_file_ids)

    # Brief 2: total_count is a filter-less count — force filters to None so
    # _build_ticket_scope_clauses drops every optional clause. The existing
    # count SQL then returns the universe of in-scope tickets.
    is_total_count = intent.operation == "total_count"
    if is_total_count:
        filters = {
            "customer_name": None,
            "priority": None,
            "sla_met": None,
            "component": None,
            "rework_detected": None,
        }
    else:
        filters = {
            "customer_name": intent.customer_name,
            "priority": intent.priority,
            "sla_met": intent.sla_target_met,
            "component": intent.component,
            # Bug 3 — include rework_detected so SLA+rework compound queries
            # don't silently drop the rework half of the filter.
            "rework_detected": intent.rework_detected,
        }

    # Per-chunk rows may duplicate `incident_number` across multiple chunks of
    # the same ticket — DISTINCT in SQL so the count is per ticket, not per
    # chunk. Optional filters are NOT in the SQL body; they're injected by
    # _build_ticket_scope_clauses only when non-None, so None never reaches the
    # driver (the definitive fix for the earlier AmbiguousParameter + SQLAlchemy
    # InvalidRequestError combo).
    where_sql, params = _build_ticket_scope_clauses(
        owner_id=owner_id,
        allowed_file_ids=allowed_file_ids,
        filters=filters,
    )

    sql = text(f"""
        SELECT DISTINCT
            c.metadata_json->>'incident_number' AS incident_number
        FROM chunks c
        JOIN documents d          ON d.id = c.document_id
        JOIN document_versions dv ON dv.id = c.document_version_id
        WHERE {where_sql}
        ORDER BY incident_number
    """).bindparams(bindparam("allowed_ids", expanding=True))

    incident_numbers: List[str] = []
    try:
        with SessionLocal() as db:
            rows = db.execute(sql, params).mappings().all()
        incident_numbers = [r["incident_number"] for r in rows if r.get("incident_number")]
    except Exception as exc:
        # Fail soft — caller falls through to full RAG if this raises.
        logger.warning("metadata_sql aggregation failed: %s", exc)
        return AggResult()

    logger.info("[aggregation] SQL fast-path: %d results", len(incident_numbers))

    if is_total_count:
        prose = _build_total_count_prose(incident_numbers, owner_id, allowed_file_ids)
        filters_applied = {"operation": "total_count"}
    else:
        prose = _build_prose_summary(intent, incident_numbers)
        filters_applied = {
            k: v for k, v in (
                ("customer_name", intent.customer_name),
                ("priority", intent.priority),
                ("sla_target_met", intent.sla_target_met),
                ("component", intent.component),
                ("rework_detected", intent.rework_detected),
            ) if v is not None
        }

    return AggResult(
        count=len(incident_numbers),
        incident_numbers=incident_numbers,
        prose_summary=prose,
        filters_applied=filters_applied,
    )
