"""
CSV column-aware parser.

Each row becomes one ParsedChunk. Columns are auto-detected and mapped to
standard metadata fields (primary_id, customer, priority, etc.) via
organization schema mapping if available, else via heuristic detection.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.config import settings
from backend.ingestion.structured_parser import ParsedChunk

logger = logging.getLogger("acadia-log-iq")


# Heuristic column name patterns for auto-mapping (case-insensitive, normalized)
COLUMN_HEURISTICS: Dict[str, List[str]] = {
    "primary_id": [
        "incident_number", "ticket_id", "ticket_number", "incident_id",
        "case_ref", "case_id", "record_id", "id", "ref",
    ],
    "customer": [
        "customer", "customer_name", "customer_id", "account",
        "account_name", "client", "org", "organization",
    ],
    "priority": [
        "priority", "severity", "urgency", "p_level", "impact",
    ],
    "component": [
        "component", "category", "service", "product", "technology",
        "tech_domain", "platform",
    ],
    "status": [
        "status", "state", "resolution_state", "ticket_status",
    ],
    "opened_date": [
        "opened", "open_date", "created", "created_at", "reported_at",
        "timestamp", "date_opened",
    ],
    "resolved_date": [
        "resolved", "resolution_date", "closed", "closed_at", "date_closed",
    ],
    "summary": [
        "summary", "description", "short_description", "title", "subject",
    ],
    "resolution": [
        "resolution", "resolution_notes", "close_notes", "resolution_text",
        "solution",
    ],
}


def _normalize_column_name(name: str) -> str:
    """Lowercase and strip special chars for matching."""
    return (name or "").strip().lower().replace(" ", "_").replace("-", "_")


def _infer_column_mapping(
    headers: List[str],
    org_schema: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Map CSV column headers to canonical field names.

    If org_schema is provided (from organization_schemas table), use it.
    Otherwise fall back to heuristic matching via COLUMN_HEURISTICS.

    Returns: {canonical_field: actual_column_name}
    Example: {"primary_id": "Ticket_ID", "customer": "Customer_Name"}
    """
    mapping: Dict[str, str] = {}
    normalized_headers = {_normalize_column_name(h): h for h in headers}

    # Priority 1: organization-provided schema mapping
    if org_schema:
        for canonical, actual_col in org_schema.items():
            if actual_col in headers:
                mapping[canonical] = actual_col

    # Priority 2: heuristic matching for unmapped canonicals
    for canonical, candidates in COLUMN_HEURISTICS.items():
        if canonical in mapping:
            continue
        for candidate in candidates:
            if candidate in normalized_headers:
                mapping[canonical] = normalized_headers[candidate]
                break

    return mapping


def _iter_rows(local_path: Path, max_rows: Optional[int]):
    """Yield (headers, row_dicts) using pandas when available, stdlib csv otherwise."""
    try:
        import pandas as pd  # optional dep — fallback if unavailable
        df = pd.read_csv(local_path, nrows=max_rows, dtype=str, keep_default_na=False)
        headers = list(df.columns)
        rows = df.to_dict(orient="records")
        return headers, rows
    except Exception as exc:
        logger.warning(
            "[csv_parser] pandas unavailable or failed on %s: %s — using stdlib csv",
            local_path, exc,
        )
        with open(local_path, encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            headers = list(reader.fieldnames or [])
            rows = []
            for i, row in enumerate(reader):
                if max_rows is not None and i >= max_rows:
                    break
                rows.append({k: (v if v is not None else "") for k, v in row.items()})
        return headers, rows


def parse_csv(
    local_path: Path,
    org_schema: Optional[Dict[str, str]] = None,
    max_rows: Optional[int] = None,
) -> List[ParsedChunk]:
    """
    Parse CSV into ParsedChunk list — one chunk per row.

    Each chunk carries:
    - text: human-readable rendering of the row
    - section_heading: primary_id value if mappable
    - operational_section: "record"
    - chunk_type: "csv_row"
    - metadata: original row dict + canonical field mappings

    Args:
        local_path: path to CSV file
        org_schema: optional {canonical_field: column_name} from org config
        max_rows: optional cap; defaults to settings.CSV_MAX_ROWS_PER_UPLOAD

    Returns: List of ParsedChunks (one per non-empty row)
    """
    if not getattr(settings, "CSV_COLUMN_AWARE_PARSING_ENABLED", True):
        # Fall back to flat text treatment (legacy behavior)
        logger.info("[csv_parser] column-aware parsing disabled — flat text fallback")
        text = local_path.read_text(encoding="utf-8", errors="replace")
        return [ParsedChunk(
            chunk_index=0,
            text=text[:50000],
            chunk_type="text",
            section_heading="CSV Data",
            operational_section="document",
            page_number=None,
            source_order=0,
            token_estimate=max(1, len(text) // 4),
            metadata={},
        )]

    cap = max_rows if max_rows is not None else getattr(settings, "CSV_MAX_ROWS_PER_UPLOAD", 50000)
    headers, rows = _iter_rows(local_path, cap)

    if not rows:
        logger.warning("[csv_parser] empty CSV: %s", local_path)
        return []

    column_mapping = _infer_column_mapping(headers, org_schema)

    logger.info(
        "[csv_parser] file=%s rows=%d cols=%d mapping=%s",
        local_path.name, len(rows), len(headers),
        {k: v for k, v in column_mapping.items()},
    )

    chunks: List[ParsedChunk] = []
    for idx, row in enumerate(rows):
        # Build human-readable text representation
        text_lines: List[str] = []
        for header in headers:
            value = str(row.get(header, "") or "").strip()
            if value:
                text_lines.append(f"{header}: {value}")
        text = "\n".join(text_lines)

        if not text.strip():
            continue

        # Extract primary_id for section_heading if available
        primary_id_col = column_mapping.get("primary_id")
        primary_id_value = (
            str(row.get(primary_id_col, "") or "").strip()
            if primary_id_col else ""
        ) or f"row_{idx}"

        # Build metadata with canonical field extractions
        metadata: Dict[str, Any] = {
            "raw_row": {h: str(row.get(h, "") or "") for h in headers},
            "column_mapping": column_mapping,
        }
        for canonical, actual_col in column_mapping.items():
            val = str(row.get(actual_col, "") or "").strip()
            if val:
                metadata[canonical] = val

        chunks.append(ParsedChunk(
            chunk_index=idx,
            text=text,
            chunk_type="csv_row",
            section_heading=primary_id_value,
            operational_section="record",
            page_number=None,
            source_order=idx,
            token_estimate=max(1, len(text) // 4),
            metadata=metadata,
        ))

    logger.info("[csv_parser] parsed %d chunks from %s", len(chunks), local_path.name)
    return chunks
