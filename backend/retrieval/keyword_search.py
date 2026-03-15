"""
Keyword Search — PostgreSQL full-text search for exact technical terms.
Uses ts_vector/ts_query for ranked keyword matching, plus ILIKE fallback
for error codes, commands, and IDs that FTS tokenizers may mangle.
Only searches ACTIVE document versions by default.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set

from sqlalchemy import text, bindparam

from backend.config import settings
from backend.db.connection import SessionLocal

logger = logging.getLogger("acadia-log-iq")


# ---------------------------------------------------------------------------
# Helper: build a ts_query string from raw search terms
# ---------------------------------------------------------------------------
def _build_tsquery(terms: List[str]) -> str:
    """
    Build a PostgreSQL tsquery from a list of extracted terms.
    Joins terms with OR (|) so any term match contributes.
    Exact multi-word phrases get wrapped with <-> (phrase match).
    """
    if not terms:
        return ""

    parts = []
    for term in terms:
        # Clean term: remove characters that break tsquery syntax
        cleaned = re.sub(r"[^a-zA-Z0-9._\-/]", " ", term).strip()
        if not cleaned:
            continue

        words = cleaned.split()
        if len(words) > 1:
            # Multi-word phrase → use proximity operator
            phrase = " <-> ".join(w for w in words if w)
            if phrase:
                parts.append(f"({phrase})")
        else:
            parts.append(cleaned)

    return " | ".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Main full-text search function
# ---------------------------------------------------------------------------
def fulltext_search(
    *,
    terms: List[str],
    n_results: int = 15,
    allowed_file_ids: Optional[Set[str]] = None,
    owner_id: str = "anonymous",
    include_old_versions: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """
    Search chunks using PostgreSQL full-text search (tsvector + tsquery).

    How it works:
    1. Builds a tsquery from the extracted terms
    2. Runs ts_rank against chunk content for ranked matches
    3. Falls back to ILIKE for terms that may not tokenize well
       (error codes, IP addresses, dotted identifiers)
    4. Only searches active document versions unless overridden

    Returns list of hit dicts with same shape as pgvector_search output.
    """
    if not terms:
        return []

    include_old = settings.INCLUDE_OLD_VERSIONS if include_old_versions is None else include_old_versions

    # --- Build the tsquery string ---
    tsquery_str = _build_tsquery(terms)
    if not tsquery_str:
        return []

    # --- Construct SQL with ts_rank ---
    # We search both content and contextualized_content via a combined tsvector
    sql = """
        SELECT
            c.id,
            c.content,
            c.contextualized_content,
            c.summary,
            c.section_heading,
            c.chunk_type,
            c.labels_json,
            c.metadata_json,
            d.id::text       AS file_id,
            d.owner_id,
            d.name           AS source,
            d.file_type,
            ts_rank(
                to_tsvector('english', COALESCE(c.contextualized_content, '') || ' ' || COALESCE(c.content, '')),
                to_tsquery('english', :tsquery)
            ) AS rank
        FROM chunks c
        JOIN documents d          ON d.id = c.document_id
        JOIN document_versions dv ON dv.id = c.document_version_id
        WHERE
            -- Full-text match
            to_tsvector('english', COALESCE(c.contextualized_content, '') || ' ' || COALESCE(c.content, ''))
            @@ to_tsquery('english', :tsquery)
            -- Owner filter
            AND d.owner_id = :owner_id
    """

    # Active-only filter (default: exclude superseded docs)
    if not include_old:
        sql += """
            AND d.status = 'active'
            AND dv.is_active = TRUE
            AND d.current_version_id = dv.id
        """

    params: Dict[str, Any] = {
        "tsquery": tsquery_str,
        "owner_id": owner_id,
        "min_rank": settings.FTS_MIN_RANK,
        "limit": n_results,
    }

    # File-ID whitelist filter
    if allowed_file_ids:
        sql += " AND d.id IN :allowed_ids"
        params["allowed_ids"] = tuple(allowed_file_ids)

    sql += """
        AND ts_rank(
            to_tsvector('english', COALESCE(c.contextualized_content, '') || ' ' || COALESCE(c.content, '')),
            to_tsquery('english', :tsquery)
        ) >= :min_rank
        ORDER BY rank DESC
        LIMIT :limit
    """

    hits: List[Dict[str, Any]] = []

    try:
        with SessionLocal() as db:
            stmt = text(sql)
            if allowed_file_ids:
                stmt = stmt.bindparams(bindparam("allowed_ids", expanding=True))
            rows = db.execute(stmt, params).mappings().all()

        for row in rows:
            hits.append({
                "id": row["id"],
                "text": row["contextualized_content"] or row["content"],
                "rank": float(row["rank"]),
                "search_type": "fulltext",
                "metadata": {
                    "file_id": row["file_id"],
                    "owner_id": row["owner_id"],
                    "source": row["source"],
                    "file_type": row["file_type"],
                    "section_heading": row["section_heading"],
                    "chunk_type": row["chunk_type"],
                    "summary": row["summary"],
                    "labels_json": row["labels_json"] or {},
                    "metadata_json": row["metadata_json"] or {},
                },
            })
    except Exception as exc:
        logger.warning("Full-text search failed: %s", exc)

    # --- ILIKE fallback for terms that FTS tokenizer may miss ---
    # Error codes (ORA-00942), IPs (10.0.0.1), dotted keys (retry.backoff.ms)
    ilike_terms = [t for t in terms if re.search(r"[.\-:/\\]", t) or re.match(r"^0x", t, re.IGNORECASE)]

    if ilike_terms and len(hits) < n_results:
        ilike_hits = _ilike_fallback(
            terms=ilike_terms,
            n_results=n_results - len(hits),
            allowed_file_ids=allowed_file_ids,
            owner_id=owner_id,
            include_old=include_old,
            exclude_ids={h["id"] for h in hits},
        )
        hits.extend(ilike_hits)

    return hits[:n_results]


# ---------------------------------------------------------------------------
# ILIKE fallback for structured tokens that FTS can't handle well
# ---------------------------------------------------------------------------
def _ilike_fallback(
    *,
    terms: List[str],
    n_results: int = 10,
    allowed_file_ids: Optional[Set[str]] = None,
    owner_id: str = "anonymous",
    include_old: bool = False,
    exclude_ids: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Fallback search using ILIKE for terms that PostgreSQL FTS
    may not tokenize correctly (error codes, IPs, file paths).
    Each matched term contributes +1 to a simple hit-count score.
    """
    if not terms:
        return []

    # Build ILIKE conditions: match any term in content or contextualized_content
    ilike_conditions = []
    params: Dict[str, Any] = {"owner_id": owner_id, "limit": n_results}

    for i, term in enumerate(terms[:8]):  # cap at 8 terms to keep query sane
        param_name = f"term_{i}"
        params[param_name] = f"%{term}%"
        ilike_conditions.append(
            f"(COALESCE(c.contextualized_content, '') ILIKE :{param_name} "
            f"OR c.content ILIKE :{param_name})"
        )

    if not ilike_conditions:
        return []

    where_clause = " OR ".join(ilike_conditions)

    sql = f"""
        SELECT
            c.id,
            c.content,
            c.contextualized_content,
            c.summary,
            c.section_heading,
            c.chunk_type,
            c.labels_json,
            c.metadata_json,
            d.id::text       AS file_id,
            d.owner_id,
            d.name           AS source,
            d.file_type
        FROM chunks c
        JOIN documents d          ON d.id = c.document_id
        JOIN document_versions dv ON dv.id = c.document_version_id
        WHERE ({where_clause})
            AND d.owner_id = :owner_id
    """

    if not include_old:
        sql += """
            AND d.status = 'active'
            AND dv.is_active = TRUE
            AND d.current_version_id = dv.id
        """

    if allowed_file_ids:
        sql += " AND d.id IN :allowed_ids"
        params["allowed_ids"] = tuple(allowed_file_ids)

    if exclude_ids:
        sql += " AND c.id NOT IN :exclude_ids"
        params["exclude_ids"] = tuple(exclude_ids)

    sql += " LIMIT :limit"

    hits = []
    try:
        with SessionLocal() as db:
            stmt = text(sql)
            if allowed_file_ids:
                stmt = stmt.bindparams(bindparam("allowed_ids", expanding=True))
            if exclude_ids:
                stmt = stmt.bindparams(bindparam("exclude_ids", expanding=True))
            rows = db.execute(stmt, params).mappings().all()

        for row in rows:
            # Score = count how many terms appear in this chunk
            combined = (row["contextualized_content"] or "") + " " + (row["content"] or "")
            combined_lower = combined.lower()
            match_count = sum(1 for t in terms if t.lower() in combined_lower)

            hits.append({
                "id": row["id"],
                "text": row["contextualized_content"] or row["content"],
                "rank": float(match_count),
                "search_type": "ilike_fallback",
                "metadata": {
                    "file_id": row["file_id"],
                    "owner_id": row["owner_id"],
                    "source": row["source"],
                    "file_type": row["file_type"],
                    "section_heading": row["section_heading"],
                    "chunk_type": row["chunk_type"],
                    "summary": row["summary"],
                    "labels_json": row["labels_json"] or {},
                    "metadata_json": row["metadata_json"] or {},
                },
            })

        # Sort by match count descending
        hits.sort(key=lambda h: h["rank"], reverse=True)

    except Exception as exc:
        logger.warning("ILIKE fallback search failed: %s", exc)

    return hits[:n_results]


# ---------------------------------------------------------------------------
# Metadata-filtered search
# ---------------------------------------------------------------------------
def metadata_filter_search(
    *,
    metadata_hints: Dict[str, Any],
    n_results: int = 10,
    allowed_file_ids: Optional[Set[str]] = None,
    owner_id: str = "anonymous",
) -> List[Dict[str, Any]]:
    """
    Find chunks whose document metadata matches extracted hints
    (vendor, product, domain). Supplements other search channels
    when the query mentions a specific product or vendor.

    Uses the metadata_json JSONB column on chunks for filtering.
    """
    vendors = metadata_hints.get("vendors", [])
    if not vendors:
        return []

    # Build JSONB conditions for vendor/product/domain matching
    conditions = []
    params: Dict[str, Any] = {"owner_id": owner_id, "limit": n_results}

    for i, vendor in enumerate(vendors[:4]):
        p = f"vendor_{i}"
        params[p] = f"%{vendor}%"
        conditions.append(
            f"(LOWER(c.metadata_json->>'vendor') LIKE :{p} "
            f"OR LOWER(c.metadata_json->>'product') LIKE :{p} "
            f"OR LOWER(c.metadata_json->>'domain') LIKE :{p})"
        )

    where = " OR ".join(conditions)

    sql = f"""
        SELECT
            c.id,
            c.content,
            c.contextualized_content,
            c.summary,
            c.section_heading,
            c.chunk_type,
            c.labels_json,
            c.metadata_json,
            d.id::text       AS file_id,
            d.owner_id,
            d.name           AS source,
            d.file_type
        FROM chunks c
        JOIN documents d          ON d.id = c.document_id
        JOIN document_versions dv ON dv.id = c.document_version_id
        WHERE ({where})
            AND d.owner_id = :owner_id
            AND d.status = 'active'
            AND dv.is_active = TRUE
            AND d.current_version_id = dv.id
    """

    if allowed_file_ids:
        sql += " AND d.id IN :allowed_ids"
        params["allowed_ids"] = tuple(allowed_file_ids)

    sql += " LIMIT :limit"

    hits = []
    try:
        with SessionLocal() as db:
            stmt = text(sql)
            if allowed_file_ids:
                stmt = stmt.bindparams(bindparam("allowed_ids", expanding=True))
            rows = db.execute(stmt, params).mappings().all()

        for row in rows:
            hits.append({
                "id": row["id"],
                "text": row["contextualized_content"] or row["content"],
                "rank": 1.0,  # metadata matches are all equally ranked
                "search_type": "metadata_filter",
                "metadata": {
                    "file_id": row["file_id"],
                    "owner_id": row["owner_id"],
                    "source": row["source"],
                    "file_type": row["file_type"],
                    "section_heading": row["section_heading"],
                    "chunk_type": row["chunk_type"],
                    "summary": row["summary"],
                    "labels_json": row["labels_json"] or {},
                    "metadata_json": row["metadata_json"] or {},
                },
            })
    except Exception as exc:
        logger.warning("Metadata filter search failed: %s", exc)

    return hits[:n_results]
