"""Tests for the tabular ingestion fast-path metadata synthesizer.

Verifies that row chunks (csv_row / xlsx_row) get a complete, LLM-free
metadata dict synthesized directly from the parser's structured fields —
no column data lost, same shape as the LLM / fallback path.

Run with: py -m unittest backend.tests.services.test_tabular_fastpath
"""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/test")
os.environ.setdefault("AWS_SECRETS_DISABLED", "true")

from backend.ingestion.structured_parser import ParsedChunk  # noqa: E402
from backend.services.contextual_ingestion_service import (  # noqa: E402
    _TABULAR_CHUNK_TYPES,
    _fallback_chunk_metadata,
    _tabular_chunk_metadata,
)


def _row_chunk():
    return ParsedChunk(
        chunk_index=3,
        text="Ticket_ID: INC-1\nCustomer: Acme\nSummary: Login fails",
        chunk_type="xlsx_row",
        section_heading="INC-1",
        operational_section="record",
        page_number=None,
        source_order=3,
        token_estimate=12,
        metadata={
            "sheet": "Tickets",
            "raw_row": {"Ticket_ID": "INC-1", "Customer": "Acme", "Summary": "Login fails"},
            "column_mapping": {"primary_id": "Ticket_ID", "customer": "Customer", "summary": "Summary"},
            "primary_id": "INC-1",
            "customer": "Acme",
            "summary": "Login fails",
        },
    )


class TestTabularFastPath(unittest.TestCase):
    def test_chunk_types_registered(self):
        self.assertIn("xlsx_row", _TABULAR_CHUNK_TYPES)
        self.assertIn("csv_row", _TABULAR_CHUNK_TYPES)
        self.assertIn("xlsx_context", _TABULAR_CHUNK_TYPES)

    def test_metadata_shape_matches_fallback(self):
        meta = _tabular_chunk_metadata(_row_chunk(), "tickets.xlsx", "kb")
        for key in _fallback_chunk_metadata(_row_chunk(), "tickets.xlsx", "kb"):
            self.assertIn(key, meta)

    def test_preserves_columns_and_summary_without_llm(self):
        meta = _tabular_chunk_metadata(_row_chunk(), "tickets.xlsx", "kb")
        self.assertEqual(meta["keywords"], ["Ticket_ID", "Customer", "Summary"])
        self.assertEqual(meta["summary"], "Login fails")
        self.assertIn({"name": "Acme", "type": "customer"}, meta["entities"])
        self.assertEqual(meta["section"], "INC-1")

    def test_handles_chunk_with_no_parser_metadata(self):
        bare = ParsedChunk(
            chunk_index=0,
            text="a: 1",
            chunk_type="csv_row",
            section_heading="row_0",
            operational_section="record",
            page_number=None,
            source_order=0,
            token_estimate=1,
            metadata={},
        )
        meta = _tabular_chunk_metadata(bare, "x.csv", "kb")
        self.assertEqual(meta["section"], "row_0")
        self.assertIsNone(meta["summary"])


if __name__ == "__main__":
    unittest.main()
