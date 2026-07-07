"""
Tier 2 aggregation-intent classifier — LLM fallback for the regex
fast-path in metadata_sql.py. A single Haiku call returns a strict
JSON object declaring whether the query is an aggregation and, if so,
which operation + filters to apply.

Contract: never raises. Any error (timeout, Bedrock fault, JSON parse
failure, missing keys) degrades to a sentinel ClassifierResult with
is_aggregation=False and confidence=0.0 so the caller transparently
falls through to standard retrieval.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from backend.config import settings
from backend.services.bedrock_haiku import haiku_client, _extract_json_object
from backend.services.token_usage import record_token_usage, extract_bedrock_usage

logger = logging.getLogger("acadia-log-iq")


_SYSTEM_PROMPT = """You are an aggregation-intent classifier for a ticket analysis system. Given a user
query, decide whether it is asking for aggregate information across multiple records
(count, list, rank, group) versus asking about a specific record by ID or asking a
general non-aggregate question.

Return a JSON object with ONLY these keys:
- is_aggregation: boolean
- operation: "count" | "list" | "rank" | "group_by_customer" | "total_count" | "avg" | "sum" | "min" | "max" | null
- customer_name: the customer mentioned in the query (as written), or null
- priority: "P1" | "P2" | "P3" | "P4" | null
- sla_met: true (met SLA) | false (missed SLA) | null
- component: a component name mentioned (wifi, telephony, vdi, etc.), or null
- ranking_field: "resolution_quality_score" | null  (null if no ranking)
- ranking_direction: "desc" (highest/most/top) | "asc" (lowest/worst/bottom) | null
- min_count: integer (N+1 for "more than N") | null
- rework_detected: true (rework/reopened/redone) | false (clean/first-time) | null
- numeric_field: "resolution_quality_score" | "time_to_first_response_seconds" | null
- numeric_filter_field: "resolution_quality_score" | "time_to_first_response_seconds" | null   # Sprint 2.6
- numeric_filter_value: integer | null                                                         # Sprint 2.6
- confidence: a float 0.0-1.0 indicating how confident you are in the classification

Rules for operation:
- "count" = "how many X" with a filter (customer, priority, SLA, component).
  Example: "how many Nebula-Corp tickets" → count + customer_name=Nebula-Corp.
- "list" = "show me all X" or "list the X" with or without a filter.
  Example: "list all P1 tickets" → list + priority=P1.
- "rank" = "which one scored highest/lowest", "top N", "bottom N", "best/worst".
  Example: "top 5 by quality" → rank + ranking_direction=desc.
- "group_by_customer" = "which customer had the most", "break down by customer",
  "by customer", "rank customers", "which customers have more than N tickets".
  For plain "break down" / "group" / "rank customers" default ranking_direction
  to "desc" (most-first). Use "asc" only when the query explicitly says
  "fewest"/"lowest"/"least".
- "total_count" = "how many tickets total", "how many records do you have",
  "total number" — a count question with NO customer/priority/SLA/component filter.
- "avg" = query asks for the average/mean of a numeric column across tickets.
  Example: "What's the average quality score for Nebula-Corp tickets?" →
  avg + customer_name=Nebula-Corp + numeric_field=resolution_quality_score.
- "sum" = query asks for the total sum of a numeric column.
- "min" / "max" = query asks for the lowest/highest numeric VALUE itself
  (distinct from "rank", which returns the ticket IDs at the extreme).

Rules for rework_detected:
- Set rework_detected=true when the query contains "rework", "redone",
  "reopened", "retry", or "second attempt" with a positive framing.
- Set rework_detected=false when the query says "no rework", "clean",
  "first-time resolution".
- Leave rework_detected=null if rework is not mentioned.

Rules for filter inference (CRITICAL — avoid hallucinated filters):
- Only set `customer_name` if the query text explicitly contains a customer name
  (e.g., "Nebula-Corp", "Enterprise-859"). Do NOT infer a customer from implicit
  context.
- Only set `priority` if the query text contains "P1", "P2", "P3", "P4", or an
  explicit priority word like "critical/high/medium/low priority".
- Only set `sla_met` if the query text mentions "SLA" with met/missed/breach/hit.
- NEVER carry filters from prior conversation history — the query text you were
  given is the ONLY source of filter information. If the user's query is
  "How many P1 tickets?", the scope is GLOBAL — return customer_name=null.
- Any filter you return MUST be textually present in the query. If not, omit it.

If the query is asking about a SPECIFIC ticket by ID (e.g. "what is INC-10005"),
is_aggregation MUST be false.

If the query is a greeting, a general "what can you do" question, or asks for a
comparison of two named tickets, is_aggregation MUST be false.

Output ONLY the JSON object — no preamble, no markdown code fence, no commentary."""


_FEW_SHOTS = """Query: "How many Nebula-Corp tickets?"
JSON: {"is_aggregation": true, "operation": "count", "customer_name": "Nebula-Corp", "priority": null, "sla_met": null, "component": null, "ranking_field": null, "ranking_direction": null, "confidence": 0.98}

Query: "List all P1 tickets that missed SLA"
JSON: {"is_aggregation": true, "operation": "list", "customer_name": null, "priority": "P1", "sla_met": false, "component": null, "ranking_field": null, "ranking_direction": null, "confidence": 0.95}

Query: "Which customer had the most incidents?"
JSON: {"is_aggregation": true, "operation": "group_by_customer", "customer_name": null, "priority": null, "sla_met": null, "component": null, "ranking_field": null, "ranking_direction": "desc", "confidence": 0.97}

Query: "Break down tickets by customer"
JSON: {"is_aggregation": true, "operation": "group_by_customer", "customer_name": null, "priority": null, "sla_met": null, "component": null, "ranking_field": null, "ranking_direction": "desc", "confidence": 0.96}

Query: "Group tickets by customer"
JSON: {"is_aggregation": true, "operation": "group_by_customer", "customer_name": null, "priority": null, "sla_met": null, "component": null, "ranking_field": null, "ranking_direction": "desc", "confidence": 0.96}

Query: "Rank customers by ticket count"
JSON: {"is_aggregation": true, "operation": "group_by_customer", "customer_name": null, "priority": null, "sla_met": null, "component": null, "ranking_field": null, "ranking_direction": "desc", "confidence": 0.95}

Query: "Which customers have the fewest tickets?"
JSON: {"is_aggregation": true, "operation": "group_by_customer", "customer_name": null, "priority": null, "sla_met": null, "component": null, "ranking_field": null, "ranking_direction": "asc", "confidence": 0.95}

Query: "Which customers have more than 2 tickets?"
JSON: {"is_aggregation": true, "operation": "group_by_customer", "customer_name": null, "priority": null, "sla_met": null, "component": null, "ranking_field": null, "ranking_direction": "desc", "min_count": 3, "confidence": 0.94}

Query: "How many P1 tickets?"
JSON: {"is_aggregation": true, "operation": "count", "customer_name": null, "priority": "P1", "sla_met": null, "component": null, "ranking_field": null, "ranking_direction": null, "confidence": 0.97}

Query: "How many tickets missed SLA?"
JSON: {"is_aggregation": true, "operation": "count", "customer_name": null, "priority": null, "sla_met": false, "component": null, "ranking_field": null, "ranking_direction": null, "confidence": 0.96}

Query: "Which ticket had the highest quality score?"
JSON: {"is_aggregation": true, "operation": "rank", "customer_name": null, "priority": null, "sla_met": null, "component": null, "ranking_field": "resolution_quality_score", "ranking_direction": "desc", "confidence": 0.93}

Query: "how can you help me with incident tickets? how many tickets do you have now?"
JSON: {"is_aggregation": true, "operation": "total_count", "customer_name": null, "priority": null, "sla_met": null, "component": null, "ranking_field": null, "ranking_direction": null, "confidence": 0.9}

Query: "What was the root cause of INC-10015?"
JSON: {"is_aggregation": false, "operation": null, "customer_name": null, "priority": null, "sla_met": null, "component": null, "ranking_field": null, "ranking_direction": null, "confidence": 0.99}

Query: "Compare INC-10005 and INC-10006"
JSON: {"is_aggregation": false, "operation": null, "customer_name": null, "priority": null, "sla_met": null, "component": null, "ranking_field": null, "ranking_direction": null, "confidence": 0.97}

Query: "hello"
JSON: {"is_aggregation": false, "operation": null, "customer_name": null, "priority": null, "sla_met": null, "component": null, "ranking_field": null, "ranking_direction": null, "confidence": 0.99}

Query: "Which SLA-missed tickets had rework detected?"
JSON: {"is_aggregation": true, "operation": "list", "customer_name": null, "priority": null, "sla_met": false, "component": null, "ranking_field": null, "ranking_direction": null, "rework_detected": true, "confidence": 0.94}

Query: "Show me tickets that were reopened"
JSON: {"is_aggregation": true, "operation": "list", "customer_name": null, "priority": null, "sla_met": null, "component": null, "ranking_field": null, "ranking_direction": null, "rework_detected": true, "confidence": 0.92}

Query: "Tickets resolved first-time correctly"
JSON: {"is_aggregation": true, "operation": "list", "customer_name": null, "priority": null, "sla_met": null, "component": null, "ranking_field": null, "ranking_direction": null, "rework_detected": false, "confidence": 0.9}

Query: "Rework detected Nebula-Corp tickets"
JSON: {"is_aggregation": true, "operation": "list", "customer_name": "Nebula-Corp", "priority": null, "sla_met": null, "component": null, "ranking_field": null, "ranking_direction": null, "rework_detected": true, "confidence": 0.94}

Query: "What's the average quality score for Nebula-Corp tickets?"
JSON: {"is_aggregation": true, "operation": "avg", "customer_name": "Nebula-Corp", "priority": null, "sla_met": null, "component": null, "ranking_field": null, "ranking_direction": null, "numeric_field": "resolution_quality_score", "confidence": 0.95}

Query: "What's the total time to first response across all P1 tickets?"
JSON: {"is_aggregation": true, "operation": "sum", "customer_name": null, "priority": "P1", "sla_met": null, "component": null, "ranking_field": null, "ranking_direction": null, "numeric_field": "time_to_first_response_seconds", "confidence": 0.92}

Query: "List tickets where Resolution_Quality_Score equals 5"
JSON: {"is_aggregation": true, "operation": "list", "customer_name": null, "priority": null, "sla_met": null, "component": null, "ranking_field": null, "ranking_direction": null, "numeric_field": null, "numeric_filter_field": "resolution_quality_score", "numeric_filter_value": 5, "confidence": 0.95}

Query: "Show me Aetheris Corp tickets with score of 5"
JSON: {"is_aggregation": true, "operation": "list", "customer_name": "Aetheris Corp", "priority": null, "sla_met": null, "component": null, "ranking_field": null, "ranking_direction": null, "numeric_field": null, "numeric_filter_field": "resolution_quality_score", "numeric_filter_value": 5, "confidence": 0.93}"""


_ALLOWED_OPERATIONS = {
    "count", "list", "rank", "group_by_customer", "total_count",
    # Bug 4B — numeric aggregation operations.
    "avg", "sum", "min", "max",
}
_ALLOWED_PRIORITIES = {"P1", "P2", "P3", "P4"}
_ALLOWED_DIRECTIONS = {"desc", "asc"}
# Bug 4B — whitelisted numeric_field values; anything else is rejected.
_ALLOWED_NUMERIC_FIELDS = {
    "resolution_quality_score",
    "time_to_first_response_seconds",
}


@dataclass
class ClassifierResult:
    is_aggregation: bool
    operation: Optional[str]
    customer_name: Optional[str]
    priority: Optional[str]
    sla_met: Optional[bool]
    component: Optional[str]
    ranking_field: Optional[str]
    ranking_direction: Optional[str]
    confidence: float
    raw_response: str
    # Bug 1 / Patch 4 — "more than N" queries carry min_count for HAVING.
    min_count: Optional[int] = None
    # Bug 3 — rework filter (true/false/None).
    rework_detected: Optional[bool] = None
    # Bug 4B — numeric_field names the column avg/sum/min/max run over.
    numeric_field: Optional[str] = None
    # Sprint 2.6 — equality filter on a whitelisted numeric field.
    numeric_filter_field: Optional[str] = None
    numeric_filter_value: Optional[int] = None


def _sentinel(raw_response: str = "") -> ClassifierResult:
    return ClassifierResult(
        is_aggregation=False,
        operation=None,
        customer_name=None,
        priority=None,
        sla_met=None,
        component=None,
        ranking_field=None,
        ranking_direction=None,
        confidence=0.0,
        raw_response=raw_response,
        min_count=None,
        rework_detected=None,
        numeric_field=None,
        # Sprint 2.6
        numeric_filter_field=None,
        numeric_filter_value=None,
    )


def _invoke_bedrock(prompt: str, max_tokens: int, temperature: float) -> str:
    """Blocking Bedrock Messages API call — returns the raw text body."""
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
        modelId=settings.AGGREGATION_CLASSIFIER_MODEL,
        body=json.dumps(body).encode("utf-8"),
        accept="application/json",
        contentType="application/json",
    )
    payload = json.loads(response["body"].read().decode("utf-8"))
    record_token_usage("aggregation", settings.AGGREGATION_CLASSIFIER_MODEL, *extract_bedrock_usage(payload, response))
    content = payload.get("content", [])
    return "\n".join(
        item.get("text", "") for item in content if item.get("type") == "text"
    ).strip()


def _parse_payload(data: Dict[str, Any], raw: str) -> ClassifierResult:
    """Coerce Haiku's JSON into a validated ClassifierResult."""
    is_aggregation = bool(data.get("is_aggregation"))

    operation = data.get("operation")
    if operation not in _ALLOWED_OPERATIONS:
        operation = None

    priority = data.get("priority")
    if isinstance(priority, str):
        priority_norm = priority.strip().upper()
        priority = priority_norm if priority_norm in _ALLOWED_PRIORITIES else None
    else:
        priority = None

    sla_met = data.get("sla_met")
    if sla_met is not None and not isinstance(sla_met, bool):
        sla_met = None

    ranking_direction = data.get("ranking_direction")
    if isinstance(ranking_direction, str):
        ranking_direction = ranking_direction.strip().lower()
        if ranking_direction not in _ALLOWED_DIRECTIONS:
            ranking_direction = None
    else:
        ranking_direction = None

    customer_name = data.get("customer_name")
    if isinstance(customer_name, str):
        customer_name = customer_name.strip() or None
    else:
        customer_name = None

    component = data.get("component")
    if isinstance(component, str):
        component = component.strip() or None
    else:
        component = None

    ranking_field = data.get("ranking_field")
    if isinstance(ranking_field, str):
        ranking_field = ranking_field.strip() or None
    else:
        ranking_field = None

    confidence_raw = data.get("confidence", 0.0)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    # Bug 1 / Patch 4 — parse min_count (for "more than N" group-by).
    min_count_raw = data.get("min_count")
    try:
        min_count = int(min_count_raw) if min_count_raw is not None else None
        if min_count is not None and min_count <= 0:
            min_count = None
    except (TypeError, ValueError):
        min_count = None

    # Bug 3 — rework_detected (true/false/None only).
    rework_detected = data.get("rework_detected")
    if rework_detected is not None and not isinstance(rework_detected, bool):
        rework_detected = None

    # Bug 4B — numeric_field must be in the whitelist.
    numeric_field = data.get("numeric_field")
    if isinstance(numeric_field, str):
        numeric_field = numeric_field.strip() or None
        if numeric_field not in _ALLOWED_NUMERIC_FIELDS:
            numeric_field = None
    else:
        numeric_field = None

    # Sprint 2.6 — numeric equality filter.
    nff = data.get("numeric_filter_field")
    nfv = data.get("numeric_filter_value")
    if isinstance(nff, str):
        nff = nff.strip() or None
        if nff not in _ALLOWED_NUMERIC_FIELDS:
            nff = None
            nfv = None
    else:
        nff = None
    if nfv is not None:
        try:
            nfv = int(nfv)
        except (ValueError, TypeError):
            nff = None
            nfv = None
    # If either is None, drop both — the filter only makes sense as a pair.
    if nff is None or nfv is None:
        nff = None
        nfv = None

    return ClassifierResult(
        is_aggregation=is_aggregation,
        operation=operation,
        customer_name=customer_name,
        priority=priority,
        sla_met=sla_met,
        component=component,
        ranking_field=ranking_field,
        ranking_direction=ranking_direction,
        confidence=confidence,
        raw_response=raw,
        min_count=min_count,
        rework_detected=rework_detected,
        numeric_field=numeric_field,
        numeric_filter_field=nff,   # Sprint 2.6
        numeric_filter_value=nfv,   # Sprint 2.6
    )


# ---------------------------------------------------------------------------
# Bug 3 — Defensive filter guard
# ---------------------------------------------------------------------------
# The classifier is stateless by contract (it only sees the query string, not
# conversation history). The Haiku model can still hallucinate filters from
# priors — e.g. returning customer_name="Nebula-Corp" for "How many P1
# tickets?" when a recent turn discussed Nebula-Corp. This post-hoc guard
# strips any filter that is not textually supported in the query.
#
# Do NOT confuse this with a semantic check — we only require the filter
# TOKEN to appear somewhere in the raw query. That's cheap, deterministic,
# and sufficient to catch history-bleed hallucinations.


def _validate_filters_against_query(
    result: ClassifierResult, query: str
) -> ClassifierResult:
    """
    Strip classifier filters that aren't textually supported by the query.
    Protects against LLM hallucination or context bleed from prior turns.
    Called after _parse_payload but before returning to caller.
    """
    if not getattr(settings, "AGG_CLASSIFIER_STATELESS_ENABLED", True):
        return result

    q_lower = (query or "").lower()

    if result.customer_name:
        customer_token = result.customer_name.lower()
        # Accept the exact string OR an alphanumeric-only form (protects
        # against hyphens: "Nebula-Corp" vs "nebula corp" in the query).
        alnum_token = "".join(ch for ch in customer_token if ch.isalnum())
        alnum_q = "".join(ch for ch in q_lower if ch.isalnum())
        if customer_token not in q_lower and (
            not alnum_token or alnum_token not in alnum_q
        ):
            logger.warning(
                "[agg_classifier] dropping hallucinated customer filter: %r "
                "not in query %r",
                result.customer_name, query,
            )
            result.customer_name = None

    if result.priority:
        priority_token = result.priority.lower()
        if priority_token not in q_lower:
            logger.warning(
                "[agg_classifier] dropping hallucinated priority filter: %r "
                "not in query %r",
                result.priority, query,
            )
            result.priority = None

    if result.sla_met is not None and "sla" not in q_lower:
        logger.warning(
            "[agg_classifier] dropping hallucinated sla_met filter "
            "(no 'sla' in query %r)",
            query,
        )
        result.sla_met = None

    if result.component:
        comp_token = result.component.lower()
        if comp_token not in q_lower:
            logger.warning(
                "[agg_classifier] dropping hallucinated component filter: %r "
                "not in query %r",
                result.component, query,
            )
            result.component = None

    # Bug 3 — drop a rework_detected filter the query doesn't support.
    if result.rework_detected is not None:
        if (
            "rework" not in q_lower
            and "reopened" not in q_lower
            and "redone" not in q_lower
            and "retry" not in q_lower
            and "first-time" not in q_lower
            and "clean" not in q_lower
        ):
            logger.warning(
                "[agg_classifier] dropping hallucinated rework filter "
                "(no rework/reopened/redone in query %r)",
                query,
            )
            result.rework_detected = None

    return result


def classify_aggregation_intent(query: str) -> ClassifierResult:
    """
    Use Haiku to classify whether a query is an aggregation question and
    extract operation + filters. Never raises — on any error returns a
    sentinel ClassifierResult(is_aggregation=False, confidence=0.0, ...).
    """
    if not query or not query.strip():
        return _sentinel("<empty query>")

    prompt = f"{_FEW_SHOTS}\n\nQuery: \"{query.strip()}\"\nJSON:"

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                _invoke_bedrock,
                prompt,
                settings.AGGREGATION_CLASSIFIER_MAX_TOKENS,
                settings.AGGREGATION_CLASSIFIER_TEMPERATURE,
            )
            raw = future.result(
                timeout=settings.AGGREGATION_CLASSIFIER_TIMEOUT_SECONDS
            )
    except concurrent.futures.TimeoutError:
        logger.warning(
            "[agg_classifier] timeout after %.1fs query=%r",
            settings.AGGREGATION_CLASSIFIER_TIMEOUT_SECONDS, query[:120],
        )
        return _sentinel("<timeout>")
    except Exception as exc:
        logger.warning(
            "[agg_classifier] bedrock error (%s) query=%r", exc, query[:120],
        )
        return _sentinel(f"<error: {exc}>")

    if not raw:
        logger.warning("[agg_classifier] empty response query=%r", query[:120])
        return _sentinel("<empty response>")

    cleaned = _extract_json_object(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning(
            "[agg_classifier] JSON parse failed (%s) raw=%r query=%r",
            exc, raw[:300], query[:120],
        )
        return _sentinel(f"<json-parse-error: {exc}>")

    result = _parse_payload(data, raw)

    # Bug 3 — strip any filter that is not textually present in the query.
    # Runs before the early-return branches so non-aggregation sentinels are
    # untouched (they have no filters to strip) and aggregation results are
    # cleaned before the caller ever builds an AggIntent from them.
    result = _validate_filters_against_query(result, query)

    if not result.is_aggregation:
        logger.info(
            "[agg_classifier] non-aggregation (confidence=%.2f) query=%r",
            result.confidence, query[:120],
        )
        return result

    if result.confidence < settings.AGGREGATION_CLASSIFIER_MIN_CONFIDENCE:
        logger.info(
            "[agg_classifier] below threshold (confidence=%.2f, min=%.2f) — "
            "falling through to RAG",
            result.confidence, settings.AGGREGATION_CLASSIFIER_MIN_CONFIDENCE,
        )
        return result

    logger.info(
        "[agg_classifier] op=%s customer=%s priority=%s sla_met=%s "
        "min_count=%s rework_detected=%s numeric_field=%s "
        "numeric_filter_field=%s numeric_filter_value=%s "
        "confidence=%.2f query=%r",
        result.operation, result.customer_name, result.priority,
        result.sla_met, result.min_count, result.rework_detected,
        result.numeric_field, result.numeric_filter_field,
        result.numeric_filter_value, result.confidence, query[:120],
    )
    return result
