"""
Organization schema inference and persistence.

On upload, loads existing schema for (organization_id, source_type). If none
exists, infers a mapping from file content, persists it with a confidence
score, and returns it. Admin UI can later review/confirm/edit.

Feature flag: settings.DYNAMIC_SCHEMA_INFERENCE_ENABLED
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from backend.config import settings
from backend.db.connection import engine

logger = logging.getLogger("acadia-log-iq")


# ---------------------------------------------------------------------------
# Read path
# ---------------------------------------------------------------------------
def get_org_schema(
    organization_id: str,
    source_type: str,
) -> Optional[Dict[str, Any]]:
    """Fetch persisted schema for (org, source_type), or None if absent."""
    if not getattr(settings, "DYNAMIC_SCHEMA_INFERENCE_ENABLED", True):
        return None

    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT schema_mapping, identifier_pattern, confidence_score
                    FROM organization_schemas
                    WHERE organization_id = :oid AND source_type = :stype
                    ORDER BY confidence_score DESC
                    LIMIT 1
                    """
                ),
                {"oid": organization_id, "stype": source_type},
            ).mappings().first()
    except Exception as exc:
        # Table may not exist yet on fresh installs — treat as "no schema".
        logger.warning(
            "[schema_inference] get_org_schema failed (org=%s type=%s): %s",
            organization_id, source_type, exc,
        )
        return None

    if not row:
        return None

    mapping = row["schema_mapping"]
    # Some drivers return JSONB as a str; normalize to dict.
    if isinstance(mapping, str):
        try:
            mapping = json.loads(mapping)
        except Exception:
            mapping = {}

    return {
        "schema_mapping": mapping or {},
        "identifier_pattern": row["identifier_pattern"],
        "confidence_score": float(row["confidence_score"] or 0.0),
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def infer_schema_from_sample(
    source_type: str,
    sample_data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Infer a schema from the first few records/rows of a dataset.

    Returns:
        {
            "schema_mapping": {"primary_id": "Ticket_ID", ...},
            "identifier_pattern": "[A-Z]{2,4}-\\d+" or None,
            "confidence_score": 0.0-1.0,
        }
    """
    from backend.ingestion.csv_parser import _infer_column_mapping, COLUMN_HEURISTICS

    if not sample_data:
        return {"schema_mapping": {}, "identifier_pattern": None, "confidence_score": 0.0}

    # Take union of all keys across the first 10 sampled records.
    all_keys: List[str] = []
    seen = set()
    for record in sample_data[:10]:
        if not isinstance(record, dict):
            continue
        for k in record.keys():
            if k not in seen:
                seen.add(k)
                all_keys.append(k)

    mapping = _infer_column_mapping(all_keys)

    # Confidence: fraction of canonical fields successfully mapped.
    canonical_fields_total = max(1, len(COLUMN_HEURISTICS))
    confidence = len(mapping) / canonical_fields_total

    # Detect identifier pattern from primary_id column values.
    identifier_pattern = None
    primary_id_col = mapping.get("primary_id")
    if primary_id_col:
        sample_ids: List[str] = []
        for r in sample_data[:20]:
            if not isinstance(r, dict):
                continue
            v = r.get(primary_id_col)
            if v is None:
                continue
            s = str(v).strip()
            if s:
                sample_ids.append(s)
        identifier_pattern = _detect_identifier_pattern(sample_ids)

    logger.info(
        "[schema_inference] inferred: mapping=%s pattern=%s confidence=%.2f",
        mapping, identifier_pattern, confidence,
    )
    return {
        "schema_mapping": mapping,
        "identifier_pattern": identifier_pattern,
        "confidence_score": confidence,
    }


def _detect_identifier_pattern(sample_ids: List[str]) -> Optional[str]:
    """Detect a repeating ID shape from sample values; return a regex or None."""
    if not sample_ids:
        return None

    patterns = [
        (re.compile(r"^[A-Z]{2,4}-\d+$"), r"[A-Z]{2,4}-\d+"),
        (re.compile(r"^[A-Z]+\d+$"),      r"[A-Z]+\d+"),
        (re.compile(r"^\d+$"),            r"\d+"),
        (re.compile(r"^[a-f0-9-]{36}$"),  r"[a-f0-9-]{36}"),  # UUID
    ]

    for test_re, extract_regex in patterns:
        matches = sum(1 for sid in sample_ids if test_re.match(sid))
        if matches >= len(sample_ids) * 0.8:
            return extract_regex
    return None


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------
def persist_org_schema(
    organization_id: str,
    source_type: str,
    schema_mapping: Dict[str, str],
    identifier_pattern: Optional[str],
    confidence_score: float,
    confirmed_by: Optional[str] = None,
) -> None:
    """Insert or update organization schema row."""
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO organization_schemas
                        (organization_id, source_type, file_pattern, schema_mapping,
                         identifier_pattern, confidence_score, confirmed_by)
                    VALUES (:oid, :stype, '', CAST(:mapping AS JSONB),
                            :pattern, :conf, :confirmed_by)
                    ON CONFLICT (organization_id, source_type, file_pattern)
                    DO UPDATE SET
                        schema_mapping = EXCLUDED.schema_mapping,
                        identifier_pattern = EXCLUDED.identifier_pattern,
                        confidence_score = EXCLUDED.confidence_score,
                        confirmed_by = EXCLUDED.confirmed_by,
                        updated_at = NOW()
                    """
                ),
                {
                    "oid": organization_id,
                    "stype": source_type,
                    "mapping": json.dumps(schema_mapping),
                    "pattern": identifier_pattern,
                    "conf": float(confidence_score),
                    "confirmed_by": confirmed_by,
                },
            )
    except Exception as exc:
        logger.warning(
            "[schema_inference] persist failed (org=%s type=%s): %s",
            organization_id, source_type, exc,
        )
        return

    logger.info(
        "[schema_inference] persisted schema for org=%s source=%s confidence=%.2f",
        organization_id, source_type, confidence_score,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def get_or_infer_schema(
    organization_id: str,
    source_type: str,
    sample_data: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Main entry point: return schema for (org, source_type).

    If a persisted row exists, return it. Otherwise, if sample_data is given,
    infer a schema, persist it (unless confidence is 0), and return it.
    """
    existing = get_org_schema(organization_id, source_type)
    if existing and existing.get("schema_mapping"):
        return existing

    if not sample_data:
        return None

    inferred = infer_schema_from_sample(source_type, sample_data)
    if inferred["schema_mapping"] and inferred["confidence_score"] > 0:
        persist_org_schema(
            organization_id=organization_id,
            source_type=source_type,
            schema_mapping=inferred["schema_mapping"],
            identifier_pattern=inferred["identifier_pattern"],
            confidence_score=inferred["confidence_score"],
        )

    return inferred


def get_or_infer_schema_from_csv(
    organization_id: str,
    local_path: Path,
    sample_rows: int = 10,
) -> Optional[Dict[str, Any]]:
    """
    Convenience wrapper for the CSV dispatch path: samples the first N rows,
    then delegates to get_or_infer_schema with source_type='csv'.
    """
    existing = get_org_schema(organization_id, "csv")
    if existing and existing.get("schema_mapping"):
        return existing

    sample: List[Dict[str, Any]] = []
    try:
        import csv as _csv
        with open(local_path, encoding="utf-8", errors="replace", newline="") as f:
            reader = _csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= sample_rows:
                    break
                sample.append({k: (v if v is not None else "") for k, v in row.items()})
    except Exception as exc:
        logger.warning(
            "[schema_inference] CSV sample read failed (%s): %s", local_path, exc,
        )
        return None

    return get_or_infer_schema(
        organization_id=organization_id,
        source_type="csv",
        sample_data=sample,
    )
