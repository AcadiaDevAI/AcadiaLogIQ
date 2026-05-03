#!/usr/bin/env python3
"""Sprint 11 — One-shot backfill of rich-parent keys onto chunks.metadata_json.

The slim builder at `_ingest_gold_ticket_json` (and its Sprint 4 rich pass-
through) wrote 225 ticket chunks before Sprint 11 added the four parents the
journey readers need (`Executive_Sharable_RCA`, `Incident_Summary`,
`Forensic_Performance_Audit`, `Key_Contributors`), plus the two bonus parents
(`QA_Auditor_Feedback`, `ITIL_5_Why`).

This script re-reads the original upload bytes from
`document_versions.storage_uri`, parses the source JSON, and merges the
rich parents into each chunk's `metadata_json` via JSONB `||` concat. The
vector / embedding / chunk text are not touched — only metadata grows.

Usage:
  python -m backend.scripts.backfill_rich_schema --dry-run
  python -m backend.scripts.backfill_rich_schema
  python -m backend.scripts.backfill_rich_schema --document-id <uuid>
  python -m backend.scripts.backfill_rich_schema --rate-limit 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

from sqlalchemy import text

from backend.db.connection import SessionLocal

logger = logging.getLogger("backfill_rich_schema")

RICH_PARENT_KEYS = (
    "Executive_Sharable_RCA",
    "Incident_Summary",
    "Forensic_Performance_Audit",
    "Key_Contributors",
    "QA_Auditor_Feedback",
    "ITIL_5_Why",
)


def _resolve_local_uri(storage_uri: str) -> Path:
    """`local://<path>` → absolute Path. Raises ValueError on non-local URIs."""
    if not storage_uri or not storage_uri.startswith("local://"):
        raise ValueError(f"Backfill only supports local:// URIs, got: {storage_uri!r}")
    raw = storage_uri[len("local://"):]
    return Path(unquote(raw)).resolve()


def _load_source_tickets(path: Path) -> List[Dict[str, Any]]:
    """Returns list of ticket dicts, or [] if file unreadable / unrecognized
    shape. Tolerates BOMs and `{"tickets": [...]}` wrappers per §4.6."""
    try:
        text_data = path.read_text(encoding="utf-8-sig")
        data = json.loads(text_data)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("[backfill] could not load source file %s: %s", path, exc)
        return []
    if isinstance(data, list):
        return [t for t in data if isinstance(t, dict)]
    if isinstance(data, dict) and isinstance(data.get("tickets"), list):
        return [t for t in data["tickets"] if isinstance(t, dict)]
    logger.warning("[backfill] unrecognized JSON shape in %s — skipping", path)
    return []


def _build_lookup(tickets: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """{Incident_Number: <entire source ticket dict>}.

    Sprint 11 — Full-fidelity backfill. The prior version cherry-picked
    RICH_PARENT_KEYS only (the six parents the journey readers happened
    to need at audit time). The new version mirrors the ingest change at
    contextual_ingestion_service.py: pass the entire source ticket.
    Postgres JSONB `||` shallow-merges the dict on top of each chunk's
    existing slim metadata; existing flat fields stay intact (no key
    collisions because slim flats are snake_case and source parents are
    TitleCase) and any future schema field is auto-propagated.
    """
    lookup: Dict[str, Dict[str, Any]] = {}
    for ticket in tickets:
        meta = ticket.get("Metadata")
        incident_number = (
            meta.get("Incident_Number") if isinstance(meta, dict) else None
        )
        if not incident_number:
            continue
        # dict(ticket) — shallow copy is enough; we don't mutate it.
        lookup[str(incident_number)] = dict(ticket)
    return lookup


def _candidate_documents(db, document_id: Optional[str]) -> List[Dict[str, Any]]:
    """Return documents whose chunks may need backfilling. Filters at SQL
    level so already-rich docs are skipped entirely (per §4.2 step 2)."""
    base = """
        SELECT d.id AS document_id, dv.storage_uri AS storage_uri
        FROM documents d
        JOIN document_versions dv ON dv.id = d.current_version_id
        WHERE EXISTS (
            SELECT 1 FROM chunks c
            WHERE c.document_id = d.id
              AND c.metadata_json->>'doc_kind' = 'ticket'
              AND NOT (c.metadata_json ? 'Executive_Sharable_RCA')
        )
    """
    params: Dict[str, Any] = {}
    if document_id:
        base += " AND d.id = :document_id"
        params["document_id"] = document_id
    rows = db.execute(text(base), params).mappings().all()
    return [dict(r) for r in rows]


def _chunks_for_document(db, document_id: str) -> List[Dict[str, Any]]:
    rows = db.execute(
        text(
            """
            SELECT id, metadata_json
            FROM chunks
            WHERE document_id = :document_id
              AND metadata_json->>'doc_kind' = 'ticket'
            """
        ),
        {"document_id": document_id},
    ).mappings().all()
    out: List[Dict[str, Any]] = []
    for row in rows:
        meta = row["metadata_json"]
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                meta = {}
        out.append({"id": row["id"], "metadata_json": meta or {}})
    return out


def _rate_limited_sleep(rate_limit: Optional[float], last_write_ts: float) -> float:
    """Token-bucket-lite — keep writes under `rate_limit` per second.
    Returns the new last_write_ts."""
    now = time.monotonic()
    if rate_limit and rate_limit > 0:
        min_interval = 1.0 / rate_limit
        elapsed = now - last_write_ts
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
            now = time.monotonic()
    return now


def backfill(
    *,
    dry_run: bool = False,
    document_id: Optional[str] = None,
    rate_limit: Optional[float] = None,
) -> Dict[str, int]:
    """Run the backfill. Returns a summary dict for the caller / tests."""
    totals = {
        "docs_processed": 0,
        "chunks_updated": 0,
        "chunks_skipped_already_rich": 0,
        "errors": 0,
    }
    last_write_ts = 0.0

    with SessionLocal() as db:
        documents = _candidate_documents(db, document_id)
        if not documents:
            logger.info("[backfill] no candidate documents found — nothing to do")
            return totals

        for doc in documents:
            doc_id = str(doc["document_id"])
            storage_uri = doc.get("storage_uri")
            try:
                source_path = _resolve_local_uri(storage_uri)
            except ValueError as exc:
                logger.warning("[backfill] doc=%s — %s; skipping", doc_id, exc)
                totals["errors"] += 1
                continue

            if not source_path.exists():
                logger.warning(
                    "[backfill] doc=%s — source file %s not found; skipping",
                    doc_id,
                    source_path,
                )
                totals["errors"] += 1
                continue

            tickets = _load_source_tickets(source_path)
            lookup = _build_lookup(tickets)
            chunks = _chunks_for_document(db, doc_id)

            updated_for_doc = 0
            skipped_for_doc = 0

            for chunk in chunks:
                meta = chunk["metadata_json"] or {}
                if "Executive_Sharable_RCA" in meta:
                    skipped_for_doc += 1
                    continue
                incident_number = (
                    (meta.get("Metadata") or {}).get("Incident_Number")
                    if isinstance(meta.get("Metadata"), dict)
                    else None
                ) or meta.get("incident_number")
                if not incident_number:
                    continue
                rich = lookup.get(str(incident_number))
                if not rich:
                    continue

                if dry_run:
                    logger.info(
                        "[backfill] DRY-RUN doc=%s chunk=%s incident=%s would_add=%s",
                        doc_id,
                        chunk["id"],
                        incident_number,
                        sorted(rich.keys()),
                    )
                    updated_for_doc += 1
                    continue

                last_write_ts = _rate_limited_sleep(rate_limit, last_write_ts)
                # JSONB || merge — RHS wins on key collision; existing slim
                # keys (Metadata, Symptom_Solution_Mapping, …) untouched.
                # Belt-and-suspenders WHERE clause re-checks idempotency at
                # the row level so a partial backfill is safe to resume.
                result = db.execute(
                    text(
                        """
                        UPDATE chunks
                        SET metadata_json = metadata_json || CAST(:rich AS JSONB)
                        WHERE id = :chunk_id
                          AND NOT (metadata_json ? 'Executive_Sharable_RCA')
                        """
                    ),
                    {"chunk_id": chunk["id"], "rich": json.dumps(rich)},
                )
                if result.rowcount:
                    updated_for_doc += result.rowcount
                else:
                    skipped_for_doc += 1

            if not dry_run:
                db.commit()

            totals["docs_processed"] += 1
            totals["chunks_updated"] += updated_for_doc
            totals["chunks_skipped_already_rich"] += skipped_for_doc

            logger.info(
                "[backfill] doc=%s tickets_in_source=%d chunks_updated=%d skipped=%d",
                doc_id,
                len(tickets),
                updated_for_doc,
                skipped_for_doc,
            )

    logger.info(
        "[backfill] SUMMARY docs=%d chunks_updated=%d skipped=%d errors=%d%s",
        totals["docs_processed"],
        totals["chunks_updated"],
        totals["chunks_skipped_already_rich"],
        totals["errors"],
        " (dry-run)" if dry_run else "",
    )
    return totals


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill rich parent keys into chunks.metadata_json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log planned updates but do not write to the DB.",
    )
    parser.add_argument(
        "--document-id",
        default=None,
        help="Restrict backfill to a single document (UUID).",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=None,
        help="Maximum SQL UPDATE writes per second (omit for no throttle).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = _parse_args(argv)
    totals = backfill(
        dry_run=args.dry_run,
        document_id=args.document_id,
        rate_limit=args.rate_limit,
    )
    return 0 if totals["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
