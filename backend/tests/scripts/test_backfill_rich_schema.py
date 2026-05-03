"""Sprint 11 — Backfill script unit tests.

These run fully offline. The DB layer is faked: a `_FakeSession` mimics the
SQLAlchemy `Session` interface that `backfill_rich_schema.backfill` uses,
returning a controlled set of candidate documents and chunk rows, and
recording every UPDATE so the tests can assert on writes (or the absence
thereof for --dry-run / idempotency runs).

Run with: py -m unittest backend.tests.scripts.test_backfill_rich_schema
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import quote

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _gold_ticket(incident="INC-1", *, drop=()):
    """Build one source ticket dict. `drop` lets a test omit a parent."""
    base = {
        "Metadata": {"Incident_Number": incident, "customer_name": "Acme"},
        "Executive_Sharable_RCA": {"Resolution_Quality_Score": 4},
        "Incident_Summary": {"INCIDENT": f"summary for {incident}"},
        "Forensic_Performance_Audit": [{"Critical_Intervention": "x"}],
        "Key_Contributors": {"Key_Impact_Players": [{"name": "A"}]},
        "QA_Auditor_Feedback": {"Rework_Detected": False},
        "ITIL_5_Why": [{"why": 1}],
    }
    for key in drop:
        base.pop(key, None)
    return base


def _slim_chunk_meta(incident="INC-1"):
    """Mimic the slim shape produced by the pre-Sprint-11 ingestion path."""
    return {
        "primary_id": incident,
        "doc_kind": "ticket",
        "incident_number": incident,
        "Metadata": {"Incident_Number": incident, "customer_name": "Acme"},
        "Symptom_Solution_Mapping": {"sym": "x"},
    }


def _write_source(tmp: Path, tickets):
    p = tmp / "file1_txt.txt"
    p.write_text(json.dumps(tickets), encoding="utf-8")
    return p


def _local_uri(path: Path) -> str:
    return f"local://{quote(str(path))}"


class _FakeResult:
    """Stand-in for sqlalchemy Result/CursorResult."""

    def __init__(self, rows=None, rowcount=0):
        self._rows = rows or []
        self.rowcount = rowcount

    def mappings(self):
        return self

    def all(self):
        return list(self._rows)


class _FakeSession:
    """Minimal sqlalchemy Session shim for the backfill script.

    Routes execute() based on the SQL fragment so the test controls what
    each query returns, and records every UPDATE so assertions can inspect
    write-side effects (rowcount, args)."""

    def __init__(self, *, candidate_docs, chunks_by_doc):
        self._candidate_docs = candidate_docs
        self._chunks_by_doc = chunks_by_doc
        self.updates = []
        self.commits = 0

    def execute(self, statement, params=None):
        sql = " ".join(str(statement).lower().split())
        params = params or {}
        if "update chunks" in sql:
            self.updates.append(params)
            chunk_id = params["chunk_id"]
            for _doc_id, rows in self._chunks_by_doc.items():
                for row in rows:
                    if str(row["id"]) == str(chunk_id):
                        meta = row["metadata_json"] or {}
                        if "Executive_Sharable_RCA" in meta:
                            return _FakeResult(rowcount=0)
                        merged = dict(meta)
                        merged.update(json.loads(params["rich"]))
                        row["metadata_json"] = merged
                        return _FakeResult(rowcount=1)
            return _FakeResult(rowcount=0)
        if "from documents" in sql:
            doc_id = params.get("document_id")
            rows = self._candidate_docs
            if doc_id:
                rows = [r for r in rows if str(r["document_id"]) == str(doc_id)]
            return _FakeResult(rows=rows)
        if "from chunks" in sql:
            doc_id = params["document_id"]
            return _FakeResult(rows=self._chunks_by_doc.get(str(doc_id), []))
        return _FakeResult()

    def commit(self):
        self.commits += 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class BackfillScriptTests(unittest.TestCase):
    """§4.7 — seven tests covering idempotency, dry-run, error handling,
    and per-document logging contract."""

    def setUp(self):
        self.tmp_obj = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmp_obj.name)
        self.addCleanup(self.tmp_obj.cleanup)

    def _patch_session(self, candidate_docs, chunks_by_doc):
        from backend.scripts import backfill_rich_schema as bf

        fake = _FakeSession(
            candidate_docs=candidate_docs, chunks_by_doc=chunks_by_doc
        )
        patcher = mock.patch.object(bf, "SessionLocal", lambda: fake)
        patcher.start()
        self.addCleanup(patcher.stop)
        return fake

    def test_skips_chunk_already_having_executive_sharable_rca(self):
        from backend.scripts import backfill_rich_schema as bf

        source = _write_source(self.tmp, [_gold_ticket("INC-1")])
        already_rich = _slim_chunk_meta("INC-1")
        already_rich["Executive_Sharable_RCA"] = {"existing": True}
        chunks = [{"id": "c1", "metadata_json": already_rich}]
        fake = self._patch_session(
            candidate_docs=[{"document_id": "d1", "storage_uri": _local_uri(source)}],
            chunks_by_doc={"d1": chunks},
        )

        totals = bf.backfill()
        self.assertEqual(len(fake.updates), 0)
        self.assertEqual(totals["chunks_updated"], 0)
        self.assertEqual(totals["chunks_skipped_already_rich"], 1)
        # Pre-existing Executive_Sharable_RCA must not be overwritten.
        self.assertEqual(chunks[0]["metadata_json"]["Executive_Sharable_RCA"], {"existing": True})

    def test_merges_rich_parents_into_chunk_metadata(self):
        from backend.scripts import backfill_rich_schema as bf

        source = _write_source(self.tmp, [_gold_ticket("INC-1")])
        chunks = [{"id": "c1", "metadata_json": _slim_chunk_meta("INC-1")}]
        fake = self._patch_session(
            candidate_docs=[{"document_id": "d1", "storage_uri": _local_uri(source)}],
            chunks_by_doc={"d1": chunks},
        )

        bf.backfill()
        self.assertEqual(len(fake.updates), 1)
        meta = chunks[0]["metadata_json"]
        for key in (
            "Executive_Sharable_RCA",
            "Incident_Summary",
            "Forensic_Performance_Audit",
            "Key_Contributors",
            "QA_Auditor_Feedback",
            "ITIL_5_Why",
        ):
            self.assertIn(key, meta)
        # Existing slim keys preserved.
        self.assertEqual(meta["doc_kind"], "ticket")
        self.assertEqual(meta["primary_id"], "INC-1")
        self.assertIn("Symptom_Solution_Mapping", meta)

    def test_handles_missing_parent_in_source(self):
        from backend.scripts import backfill_rich_schema as bf

        source = _write_source(
            self.tmp, [_gold_ticket("INC-1", drop=("Forensic_Performance_Audit",))]
        )
        chunks = [{"id": "c1", "metadata_json": _slim_chunk_meta("INC-1")}]
        self._patch_session(
            candidate_docs=[{"document_id": "d1", "storage_uri": _local_uri(source)}],
            chunks_by_doc={"d1": chunks},
        )

        bf.backfill()
        meta = chunks[0]["metadata_json"]
        self.assertNotIn("Forensic_Performance_Audit", meta)
        # Other parents still merged.
        self.assertIn("Executive_Sharable_RCA", meta)
        self.assertIn("Key_Contributors", meta)

    def test_dry_run_makes_no_db_writes(self):
        from backend.scripts import backfill_rich_schema as bf

        source = _write_source(self.tmp, [_gold_ticket("INC-1")])
        chunks = [{"id": "c1", "metadata_json": _slim_chunk_meta("INC-1")}]
        fake = self._patch_session(
            candidate_docs=[{"document_id": "d1", "storage_uri": _local_uri(source)}],
            chunks_by_doc={"d1": chunks},
        )

        with self.assertLogs("backfill_rich_schema", level="INFO") as captured:
            totals = bf.backfill(dry_run=True)
        self.assertEqual(len(fake.updates), 0)
        self.assertEqual(fake.commits, 0)
        self.assertEqual(totals["chunks_updated"], 1)
        self.assertNotIn("Executive_Sharable_RCA", chunks[0]["metadata_json"])
        joined = "\n".join(captured.output)
        self.assertIn("DRY-RUN", joined)

    def test_unreadable_storage_uri_skips_doc_logs_warning(self):
        from backend.scripts import backfill_rich_schema as bf

        missing = self.tmp / "does-not-exist.txt"
        fake = self._patch_session(
            candidate_docs=[{"document_id": "d1", "storage_uri": _local_uri(missing)}],
            chunks_by_doc={"d1": [{"id": "c1", "metadata_json": _slim_chunk_meta()}]},
        )

        with self.assertLogs("backfill_rich_schema", level="WARNING") as captured:
            totals = bf.backfill()
        self.assertEqual(len(fake.updates), 0)
        self.assertEqual(totals["errors"], 1)
        self.assertEqual(totals["chunks_updated"], 0)
        joined = "\n".join(captured.output)
        self.assertIn("not found", joined)

    def test_idempotent_on_rerun(self):
        from backend.scripts import backfill_rich_schema as bf

        source = _write_source(self.tmp, [_gold_ticket("INC-1")])
        chunks = [{"id": "c1", "metadata_json": _slim_chunk_meta("INC-1")}]
        fake = self._patch_session(
            candidate_docs=[{"document_id": "d1", "storage_uri": _local_uri(source)}],
            chunks_by_doc={"d1": chunks},
        )

        bf.backfill()
        first_update_count = len(fake.updates)
        self.assertGreater(first_update_count, 0)

        # Second pass — the FakeSession's candidate-doc query still returns
        # the doc (the candidate filter is SQL-level and is mocked), but
        # the per-chunk re-check inside the loop must still skip the
        # already-rich chunk. That's the belt-and-suspenders idempotency.
        totals_second = bf.backfill()
        self.assertEqual(len(fake.updates), first_update_count)
        self.assertEqual(totals_second["chunks_updated"], 0)
        self.assertEqual(totals_second["chunks_skipped_already_rich"], 1)

    def test_per_document_summary_logged(self):
        from backend.scripts import backfill_rich_schema as bf

        source = _write_source(
            self.tmp, [_gold_ticket("INC-1"), _gold_ticket("INC-2")]
        )
        chunks = [
            {"id": "c1", "metadata_json": _slim_chunk_meta("INC-1")},
            {"id": "c2", "metadata_json": _slim_chunk_meta("INC-2")},
        ]
        self._patch_session(
            candidate_docs=[{"document_id": "d1", "storage_uri": _local_uri(source)}],
            chunks_by_doc={"d1": chunks},
        )

        with self.assertLogs("backfill_rich_schema", level="INFO") as captured:
            bf.backfill()
        joined = "\n".join(captured.output)
        self.assertIn("doc=d1", joined)
        self.assertIn("tickets_in_source=2", joined)
        self.assertIn("chunks_updated=2", joined)
        self.assertIn("skipped=0", joined)


if __name__ == "__main__":
    unittest.main()
