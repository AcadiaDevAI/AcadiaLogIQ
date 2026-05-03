"""Sprint 11 — one-shot read-only verification (no writes).

Runs the §5 SQL and prints the row. Intended to be run before/after the
backfill to confirm progress. Safe to delete after Sprint 11 closes.
"""
from __future__ import annotations

from sqlalchemy import text

from backend.db.connection import SessionLocal


SQL = """
SELECT
  COUNT(*) AS total_ticket_chunks,
  COUNT(*) FILTER (WHERE metadata_json ? 'Executive_Sharable_RCA') AS has_exec_rca,
  COUNT(*) FILTER (WHERE metadata_json ? 'Incident_Summary') AS has_incident_summary,
  COUNT(*) FILTER (WHERE metadata_json ? 'Forensic_Performance_Audit') AS has_forensic,
  COUNT(*) FILTER (WHERE metadata_json ? 'Key_Contributors') AS has_key_contributors,
  COUNT(*) FILTER (WHERE NOT (metadata_json ? 'Executive_Sharable_RCA')) AS missing_count
FROM chunks
WHERE metadata_json->>'doc_kind' = 'ticket'
"""


def main() -> int:
    with SessionLocal() as db:
        row = db.execute(text(SQL)).mappings().first()
    print(dict(row) if row else "no rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
