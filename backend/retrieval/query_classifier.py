"""
Query Intent Classifier — decides search strategy per query.
Classifies queries as 'keyword', 'semantic', or 'mixed' based on
surface-level heuristics (fast, no LLM call needed for most queries).
Falls back to Haiku classification for ambiguous cases if enabled.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from backend.config import settings

logger = logging.getLogger("acadia-log-iq")


# ---------------------------------------------------------------------------
# Dataclass returned by the classifier
# ---------------------------------------------------------------------------
@dataclass
class QueryIntent:
    """
    Holds the classification result for a single user query.

    Fields:
        strategy   — 'keyword' | 'semantic' | 'mixed'
        keyword_score  — 0.0-1.0, how keyword-like the query is
        semantic_score — 0.0-1.0, how semantic/conceptual the query is
        extracted_terms — exact tokens suitable for keyword/FTS search
        metadata_hints  — vendor/product/domain hints extracted from the query
        reason          — human-readable explanation of the classification
    """
    strategy: str = "mixed"
    keyword_score: float = 0.5
    semantic_score: float = 0.5
    extracted_terms: List[str] = field(default_factory=list)
    metadata_hints: dict = field(default_factory=dict)
    reason: str = ""


# ---------------------------------------------------------------------------
# Pattern libraries for heuristic classification
# ---------------------------------------------------------------------------

# Patterns that strongly indicate keyword/exact-match search intent
_KEYWORD_PATTERNS = [
    # Error codes: HTTP 500, ERR-1234, ORA-00942, 0x8007000E
    re.compile(r"\b(?:ERR|ERROR|ORA|SQL|HTTP|HResult|0x)[- ]?\d{2,}", re.IGNORECASE),
    # Status codes: 200, 404, 500, 503
    re.compile(r"\b[1-5]\d{2}\b"),
    # CLI commands: kubectl, docker, systemctl, grep, aws s3
    re.compile(r"\b(?:kubectl|docker|systemctl|journalctl|grep|awk|sed|curl|wget|aws\s+\w+|az\s+\w+|gcloud)\b", re.IGNORECASE),
    # IP addresses: 10.0.0.1, 192.168.1.1
    re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    # File paths: /var/log/syslog, C:\Windows\System32
    re.compile(r"(?:/[\w.-]+){2,}|[A-Z]:\\[\w\\.-]+", re.IGNORECASE),
    # Config keys: max_connections, retry.backoff.ms
    re.compile(r"\b\w+[._]\w+[._]\w+\b"),
    # Log-style patterns: [WARN], [ERROR], FATAL
    re.compile(r"\[(?:WARN|ERROR|INFO|DEBUG|FATAL|CRITICAL)\]", re.IGNORECASE),
    # Version strings: v2.1.3, 3.5.0-rc1
    re.compile(r"\bv?\d+\.\d+\.\d+(?:-\w+)?\b"),
    # UUIDs
    re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE),
    # Quoted exact phrases
    re.compile(r'"[^"]{3,}"'),
]

# Patterns that indicate semantic/conceptual search intent
_SEMANTIC_PATTERNS = [
    re.compile(r"\b(?:how|why|what|explain|describe|summarize|overview|compare|difference)\b", re.IGNORECASE),
    re.compile(r"\b(?:best practice|recommendation|guideline|procedure|steps? to|process for)\b", re.IGNORECASE),
    re.compile(r"\b(?:troubleshoot|diagnose|resolve|fix|root cause|impact)\b", re.IGNORECASE),
    re.compile(r"\b(?:when should|what happens|is it possible)\b", re.IGNORECASE),
]

# Vendor/product/domain terms to extract as metadata hints
_VENDOR_TERMS = re.compile(
    r"\b(aws|azure|gcp|cisco|palo alto|fortinet|vmware|oracle|microsoft|linux|windows|"
    r"kubernetes|k8s|docker|nginx|apache|redis|postgres|mysql|mongodb|elasticsearch|kafka|"
    r"splunk|datadog|grafana|prometheus|terraform|ansible|jenkins|github)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Core classification logic
# ---------------------------------------------------------------------------

def classify_query(query: str) -> QueryIntent:
    """
    Classify a user query into a retrieval strategy.

    1. Count keyword-pattern hits and semantic-pattern hits
    2. Extract exact terms for keyword search
    3. Extract vendor/product metadata hints
    4. Decide strategy based on score thresholds from config
    """
    query = (query or "").strip()
    if not query:
        return QueryIntent(strategy="mixed", reason="empty query")

    # --- Step 1: Score keyword signals ---
    keyword_hits = 0
    extracted_terms: List[str] = []

    for pattern in _KEYWORD_PATTERNS:
        matches = pattern.findall(query)
        if matches:
            keyword_hits += len(matches)
            # Add matched terms for keyword search
            for m in matches:
                term = m.strip().strip('"')
                if len(term) >= 2:
                    extracted_terms.append(term)

    # --- Step 2: Score semantic signals ---
    semantic_hits = 0
    for pattern in _SEMANTIC_PATTERNS:
        if pattern.search(query):
            semantic_hits += 1

    # --- Step 3: Extract metadata hints (vendor, product, domain) ---
    metadata_hints = {}
    vendor_matches = _VENDOR_TERMS.findall(query)
    if vendor_matches:
        metadata_hints["vendors"] = list(set(m.lower() for m in vendor_matches))

    # --- Step 4: Normalize scores to 0.0-1.0 ---
    # More keyword hits → higher keyword score
    keyword_score = min(1.0, keyword_hits * 0.25)
    # More semantic hits → higher semantic score
    semantic_score = min(1.0, semantic_hits * 0.30)

    # Short queries with no clear signals default to mixed
    words = query.split()
    if len(words) <= 3 and keyword_hits == 0 and semantic_hits == 0:
        keyword_score = 0.4
        semantic_score = 0.4

    # --- Step 5: Decide strategy ---
    strategy = "mixed"
    reason = ""

    if keyword_score >= settings.KEYWORD_QUERY_THRESHOLD and keyword_score > semantic_score:
        strategy = "keyword"
        reason = f"keyword patterns detected (score={keyword_score:.2f}): {extracted_terms[:5]}"
    elif semantic_score >= settings.SEMANTIC_QUERY_THRESHOLD and semantic_score > keyword_score:
        strategy = "semantic"
        reason = f"semantic patterns detected (score={semantic_score:.2f})"
    else:
        strategy = "mixed"
        reason = f"mixed signals (kw={keyword_score:.2f}, sem={semantic_score:.2f})"

    # Also extract important non-stopword tokens for keyword search fallback
    if not extracted_terms:
        stop_words = {
            "the", "a", "an", "is", "are", "to", "for", "of", "in", "on", "how",
            "what", "when", "where", "why", "do", "does", "can", "i", "me", "my",
            "you", "your", "please", "tell", "about", "and", "or", "not", "this",
            "that", "with", "from", "it", "be", "was", "were", "been", "have", "has",
        }
        extracted_terms = [
            w for w in re.findall(r"\w+", query.lower())
            if len(w) > 2 and w not in stop_words
        ]

    logger.debug("Query classified: strategy=%s reason=%s terms=%s", strategy, reason, extracted_terms[:8])

    return QueryIntent(
        strategy=strategy,
        keyword_score=keyword_score,
        semantic_score=semantic_score,
        extracted_terms=extracted_terms[:20],
        metadata_hints=metadata_hints,
        reason=reason,
    )
