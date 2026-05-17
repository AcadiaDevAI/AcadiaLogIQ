"""
Pattern Analytics — selective, dynamic cross-ticket statistics engine (Layer 3).

Computes "occurred X times", "top 3 actions", "most successful resolution",
"X% effectiveness" ONLY for queries that benefit from pattern insights.

Design:
- Rule-based classifier (90% of decisions, zero LLM cost)
- Skips trivial/identifier/pure-aggregation queries
- Mode-aware (force-enables in LogIQ Troubleshooting Mode)
- Cache-integrated (topic-level TTL cache)
- Confidence-gated (below threshold → transparency message, not misleading stats)
- Recency-weighted (last N days weighted for trend display)
- All behavior gated behind PATTERN_ANALYTICS_ENABLED for instant rollback
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.config import settings
from backend.services.pattern_analytics_cache import (
    get_cached_pattern,
    store_pattern,
)

logger = logging.getLogger("acadia-log-iq")


# ─────────────────────────────────────────────────────────────
# Keyword signals (rule-based layer, zero LLM cost)
# ─────────────────────────────────────────────────────────────

PATTERN_SIGNAL_KEYWORDS: Dict[str, List[str]] = {
    "frequency": [
        "how often", "how frequently", "how many times",
        "frequency", "rate of", "how common",
    ],
    "common": [
        "common", "typical", "usually", "most", "popular",
        "frequent", "regular", "standard", "recurring",
    ],
    "success": [
        "most successful", "what usually resolves", "what works",
        "best resolution", "effective", "success rate", "resolution rate",
    ],
    "pattern": [
        "pattern", "trend", "patterns of", "historical pattern",
        "across tickets", "across incidents", "over time",
    ],
    "historical": [
        "in the past", "historically", "past incidents",
        "previous cases", "prior tickets",
    ],
    "troubleshooting": [
        "how do i troubleshoot", "how to fix", "how to resolve",
        "troubleshoot", "diagnose", "investigate",
    ],
}

# Queries that should NEVER trigger pattern analytics
# (even if they match trigger keywords — specificity wins).
_IDENTIFIER_REGEX = re.compile(
    r"\b(INC|CASE|TKT|REQ|CHG)[-_ ]?\d+\b",
    re.IGNORECASE,
)

_AGGREGATION_ONLY_PHRASES = [
    "how many", "count of", "total number", "list all",
    "show me all", "break down", "group by",
]


def _has_identifier(query: str) -> bool:
    return bool(_IDENTIFIER_REGEX.search(query or ""))


def _is_pure_aggregation(query: str) -> bool:
    q_lower = (query or "").lower()
    return any(sig in q_lower for sig in _AGGREGATION_ONLY_PHRASES)


def _matches_pattern_signals(query: str) -> Tuple[bool, List[str]]:
    q_lower = (query or "").lower()
    matched: List[str] = []
    for category, keywords in PATTERN_SIGNAL_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            matched.append(category)
    return (bool(matched), matched)


def _extract_ticket_id_from_chunk(chunk: Any) -> Optional[str]:
    """Schema-agnostic: pull a primary identifier from a chunk-like object."""
    if chunk is None:
        return None

    metadata: Dict[str, Any] = {}
    # dict form
    if isinstance(chunk, dict):
        metadata = chunk.get("metadata") or chunk.get("metadata_json") or {}
    else:
        # object / tuple form
        meta = getattr(chunk, "metadata", None) or getattr(chunk, "metadata_json", None)
        if isinstance(meta, dict):
            metadata = meta
        elif isinstance(chunk, tuple) and len(chunk) >= 3 and isinstance(chunk[2], dict):
            metadata = chunk[2]

    if not isinstance(metadata, dict):
        return None

    # Some chunk rows nest metadata under metadata_json again.
    inner = metadata.get("metadata_json")
    if isinstance(inner, dict):
        metadata = {**metadata, **inner}

    for field in ("primary_id", "incident_number", "record_id",
                  "source_record_id", "ticket_id"):
        value = metadata.get(field)
        if value:
            return str(value).strip()
    return None


# ─────────────────────────────────────────────────────────────
# Classifier — should pattern analytics run for this query?
# ─────────────────────────────────────────────────────────────

def should_run_pattern_analytics(
    query: str,
    retrieved_chunks: List[Any],
    session_mode: Optional[str] = None,
    is_clarifier_refined: bool = False,
) -> Tuple[bool, str]:
    """
    Decide whether pattern analytics should run.

    Returns (should_run, reason). Fast path, no LLM calls:
      1. Feature flag off → False
      2. Specific identifier → False (single-ticket focus)
      3. Pure aggregation → False (SQL fast-path handles)
      4. Troubleshooting mode → True (Layer 4 mandate)
      5. Post-clarifier refined query → True
      6. Explicit pattern keywords → True
      7. Retrieval spans ≥4 distinct tickets → True
      8. Otherwise → False (conservative default)
    """
    if not getattr(settings, "PATTERN_ANALYTICS_ENABLED", False):
        return (False, "disabled")

    if _has_identifier(query):
        return (False, "has_identifier")

    if _is_pure_aggregation(query):
        return (False, "pure_aggregation")

    if (
        session_mode == "troubleshooting"
        and getattr(
            settings,
            "PATTERN_ANALYTICS_FORCE_ENABLE_IN_TROUBLESHOOTING_MODE",
            True,
        )
    ):
        return (True, "troubleshooting_mode")

    if is_clarifier_refined:
        return (True, "clarifier_refined")

    matched, categories = _matches_pattern_signals(query)
    if matched:
        return (True, f"keyword_match:{','.join(categories)}")

    if retrieved_chunks:
        unique_tickets = set()
        for chunk in retrieved_chunks:
            tid = _extract_ticket_id_from_chunk(chunk)
            if tid:
                unique_tickets.add(tid)
        if len(unique_tickets) >= 4:
            return (True, f"diverse_retrieval:{len(unique_tickets)}_tickets")

    return (False, "no_signals")


# ─────────────────────────────────────────────────────────────
# Topic extraction — normalize query → topic key
# ─────────────────────────────────────────────────────────────

_DOMAIN_NOUNS: List[str] = [
    "router", "switch", "firewall", "wireless", "wifi", "wan", "lan",
    "vpn", "dhcp", "dns", "bgp", "ospf", "mpls", "sd-wan", "gpon",
    "sbc", "voip", "telephony", "sip", "trunk",
    "server", "storage", "backup", "authentication", "ssl", "tls",
    "scanner", "access point", "controller", "circuit",
    "outage", "failure", "slowness", "latency", "packet loss",
]


def extract_topic(query: str, retrieved_chunks: List[Any]) -> Optional[str]:
    """Extract a normalized topic key from the query (with chunk fallback)."""
    q_lower = (query or "").lower()
    matches = [noun for noun in _DOMAIN_NOUNS if noun in q_lower]
    if matches:
        # Hotfix: emit the most-specific single noun so patterns aren't
        # fragmented across every permutation of co-occurring terms
        # (e.g. 'router_vpn_wan' vs 'vpn_wan' losing shared history).
        return sorted(matches, key=len, reverse=True)[0]

    if retrieved_chunks:
        sections: List[str] = []
        for chunk in retrieved_chunks[:5]:
            section = None
            if isinstance(chunk, dict):
                section = chunk.get("operational_section") or (
                    (chunk.get("metadata") or {}).get("operational_section")
                )
            else:
                section = getattr(chunk, "operational_section", None)
            if section and section not in sections:
                sections.append(str(section))
        if sections:
            return "_".join(sections[:2])

    return None


# ─────────────────────────────────────────────────────────────
# Pattern computation — extract stats from matched tickets
# ─────────────────────────────────────────────────────────────

def _count_recent_tickets(
    dates: List[Any],
    recency_days: int,
) -> int:
    """Count tickets opened within the recency window."""
    if not dates:
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=recency_days)
    count = 0
    for d in dates:
        if d is None:
            continue
        try:
            dt = d if d.tzinfo else d.replace(tzinfo=timezone.utc)
        except AttributeError:
            continue
        if dt >= cutoff:
            count += 1
    return count


def _first_clause(text: str, min_len: int = 10, max_len: int = 150) -> Optional[str]:
    """Return the first sentence of `text`, trimmed, if within length bounds."""
    if not text:
        return None
    first = re.split(r"[.;]", text)[0].strip()
    if min_len < len(first) < max_len:
        return first
    return None


def _compute_confidence_score(
    total_count: int,
    top_actions: List[Dict[str, Any]],
    success_rate: float,
) -> float:
    """
    Confidence in pattern reliability:
      - volume: data sufficiency (caps at 10 tickets)
      - concentration: top-action dominance
      - decisiveness: success-rate distance from 50/50 coin flip
    """
    volume = min(total_count / 10.0, 1.0)
    concentration = (
        top_actions[0]["count"] / total_count
        if top_actions and total_count > 0
        else 0.0
    )
    decisiveness = abs(success_rate - 0.5) * 2
    confidence = (volume * 0.4) + (concentration * 0.4) + (decisiveness * 0.2)
    return round(confidence, 2)


def compute_pattern_analytics(
    query: str,
    topic: str,
    retrieved_chunks: List[Any],
    similar_tickets: List[Dict[str, Any]],
    organization_id: str = "default",
) -> Optional[Dict[str, Any]]:
    """
    Compute pattern statistics from matched tickets.

    similar_tickets rows expected to have any of:
        ticket_id, opened_date (datetime|None), resolution_text,
        sla_met (bool), quality_score (int), resolution_groups (list)

    Returns pattern_data dict, a transparency-message dict (below confidence),
    or None (insufficient data / cache miss with no tickets loaded).
    """
    min_tickets = getattr(settings, "PATTERN_ANALYTICS_MIN_SIMILAR_TICKETS", 3)
    confidence_threshold = getattr(settings, "PATTERN_ANALYTICS_CONFIDENCE_THRESHOLD", 0.7)

    cached = get_cached_pattern(organization_id, topic)
    if cached:
        logger.info("[pattern_analytics] using cached stats for topic=%r", (topic or "")[:40])
        return cached

    if len(similar_tickets or []) < min_tickets:
        logger.info(
            "[pattern_analytics] insufficient data: %d tickets < %d threshold",
            len(similar_tickets or []), min_tickets,
        )
        return None

    total_count = len(similar_tickets)

    # Occurrence timeframe
    dates = [t.get("opened_date") for t in similar_tickets if t.get("opened_date")]
    if dates:
        earliest = min(dates)
        latest = max(dates)
        try:
            months_span = max(1, (latest - earliest).days // 30)
        except Exception:
            months_span = 1
    else:
        months_span = 1

    recency_days = getattr(settings, "PATTERN_ANALYTICS_RECENCY_WINDOW_DAYS", 30)
    recent_count = _count_recent_tickets(dates, recency_days)

    # Top-N actions (from resolution_text first clause)
    top_n = getattr(settings, "PATTERN_ANALYTICS_TOP_N_ACTIONS", 3)
    action_counter: Counter = Counter()
    for ticket in similar_tickets:
        clause = _first_clause(ticket.get("resolution_text") or "")
        if clause:
            action_counter[clause] += 1

    top_actions = [
        {"action": action, "count": count}
        for action, count in action_counter.most_common(top_n)
    ]

    # Successful actions — weighted by sla_met + quality_score>=4
    successful_actions: Counter = Counter()
    for ticket in similar_tickets:
        clause = _first_clause(ticket.get("resolution_text") or "")
        if not clause:
            continue
        is_successful = (
            bool(ticket.get("sla_met"))
            and int(ticket.get("quality_score") or 0) >= 4
        )
        if is_successful:
            successful_actions[clause] += 1

    most_successful = (
        successful_actions.most_common(1)[0][0] if successful_actions else None
    )
    success_count = sum(successful_actions.values())
    success_rate = success_count / total_count if total_count > 0 else 0.0

    confidence = _compute_confidence_score(
        total_count=total_count,
        top_actions=top_actions,
        success_rate=success_rate,
    )

    if confidence < confidence_threshold:
        logger.info(
            "[pattern_analytics] confidence %.2f below threshold %.2f — transparency msg",
            confidence, confidence_threshold,
        )
        return {
            "confidence_score": confidence,
            "total_count": total_count,
            "insufficient_confidence": True,
            "message": "Limited historical data for reliable patterns.",
        }

    pattern_data: Dict[str, Any] = {
        "occurrence_count": total_count,
        "timeframe_months": months_span,
        "top_actions": top_actions,
        "most_successful": most_successful,
        "success_rate": success_rate,
        "success_count": success_count,
        "total_count": total_count,
        "recent_count_30d": recent_count,
        "confidence_score": confidence,
        "matched_ticket_ids": [
            t.get("ticket_id") for t in similar_tickets if t.get("ticket_id")
        ][:20],
    }

    store_pattern(organization_id, topic, pattern_data)

    logger.info(
        "[pattern_analytics] computed: count=%d top_actions=%d success=%.0f%% confidence=%.2f",
        total_count, len(top_actions), success_rate * 100, confidence,
    )
    return pattern_data


# ─────────────────────────────────────────────────────────────
# Response formatting — render pattern block for prompt
# ─────────────────────────────────────────────────────────────

def format_pattern_block(pattern_data: Optional[Dict[str, Any]]) -> str:
    """Format stats into a text block injected into the generator prompt."""
    if not pattern_data:
        return ""

    if pattern_data.get("insufficient_confidence"):
        return pattern_data.get("message", "")

    lines: List[str] = ["\n[HISTORICAL PATTERN DATA]"]

    count = int(pattern_data.get("occurrence_count", 0) or 0)
    months = int(pattern_data.get("timeframe_months", 0) or 0)
    lines.append(f"This type of issue occurred {count} times in the past {months} months.")

    top_actions = pattern_data.get("top_actions") or []
    if top_actions:
        lines.append(f"\nTop {len(top_actions)} actions taken historically:")
        for i, a in enumerate(top_actions, 1):
            lines.append(f"  {i}. {a['action']} ({a['count']} occurrences)")

    most_successful = pattern_data.get("most_successful")
    if most_successful:
        success_rate = float(pattern_data.get("success_rate", 0.0) or 0.0)
        success_count = int(pattern_data.get("success_count", 0) or 0)
        total = int(pattern_data.get("total_count", 0) or 0)
        lines.append(f"\nMost successful resolution: {most_successful}")
        lines.append(
            f"Resolution effectiveness: {success_count} of {total} cases ({success_rate * 100:.0f}%)"
        )

    recent = int(pattern_data.get("recent_count_30d", 0) or 0)
    if recent > 0:
        lines.append(f"\nRecent trend: {recent} similar incidents in last 30 days")

    lines.append("[/HISTORICAL PATTERN DATA]\n")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Main entry point — called from /ask handler
# ─────────────────────────────────────────────────────────────

def enrich_if_needed(
    query: str,
    retrieved_chunks: List[Any],
    session_mode: Optional[str] = None,
    is_clarifier_refined: bool = False,
    organization_id: str = "default",
    similar_tickets_loader: Optional[Callable[[str, str], List[Dict[str, Any]]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Main entry point. Returns a dict with pattern_block / pattern_data / topic / reason,
    or None when pattern analytics should not run for this query.

    similar_tickets_loader(topic, organization_id) -> list of ticket dicts.
    Injected by caller to avoid circular imports.
    """
    should_run, reason = should_run_pattern_analytics(
        query=query,
        retrieved_chunks=retrieved_chunks,
        session_mode=session_mode,
        is_clarifier_refined=is_clarifier_refined,
    )

    if not should_run:
        logger.info("[pattern_analytics] skip reason=%s", reason)
        return None

    logger.info("[pattern_analytics] triggered reason=%s", reason)

    topic = extract_topic(query, retrieved_chunks)
    if not topic:
        logger.info("[pattern_analytics] no topic extractable from query")
        return None

    logger.info("[pattern_analytics] topic=%r", topic)

    if not similar_tickets_loader:
        logger.warning("[pattern_analytics] no similar_tickets_loader provided")
        return None

    try:
        similar_tickets = similar_tickets_loader(topic, organization_id) or []
    except Exception as exc:
        logger.warning("[pattern_analytics] similar_tickets_loader failed: %s", exc)
        return None

    pattern_data = compute_pattern_analytics(
        query=query,
        topic=topic,
        retrieved_chunks=retrieved_chunks,
        similar_tickets=similar_tickets,
        organization_id=organization_id,
    )

    if not pattern_data:
        return None

    pattern_block = format_pattern_block(pattern_data)

    return {
        "pattern_block": pattern_block,
        "pattern_data": pattern_data,
        "topic": topic,
        "reason": reason,
    }
