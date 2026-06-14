"""
Ingestion Report — Per-document chunking visibility.

Why
---
Today the only chunking signal in the logs is "1109 chunks" — we have no
idea how many were heading-bounded vs. size-split, how many pages tripped
the extraction guard, or how much overlap was actually applied.

Without those numbers we cannot tell whether the new overlap / guard /
LLM-fallback knobs are doing anything useful, and we cannot decide when
to flip OCR_SHADOW_MODE off.

This module gives us ONE structured log line at the end of every ingest
that captures everything you need to answer "did chunking work well on
this doc?".

Usage
-----
    report = IngestionReport.new(doc_label="TelcoDCN_pdf.pdf",
                                 source_kind="pdf")
    ... mutate counters during parse / chunk ...
    report.emit()  # one JSON log line at the end
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional

logger = logging.getLogger("acadia-log-iq")


@dataclass
class IngestionReport:
    """
    Aggregates per-document ingestion stats. All fields default to 0 / None
    so partial reports still serialize cleanly when a stage is skipped.

    Fields are deliberately flat (no nested dicts) so the JSON log line is
    easy to query in CloudWatch Logs Insights.
    """

    # Identity
    doc_label: str
    source_kind: str                        # "pdf" | "docx" | "plain"

    # Extraction stage
    pages_total: int = 0                    # only meaningful for PDFs
    chars_extracted: int = 0
    pages_ocr_candidate: int = 0            # extraction_guard flagged
    pages_ocr_skipped_blank: int = 0        # low text + no images
    pages_ocr_invoked: int = 0              # live Textract calls made
    pages_with_tables: int = 0              # pages where fitz.find_tables hit
    ocr_reasons: Dict[str, int] = field(default_factory=dict)

    # Heading detection stage
    heading_count: int = 0
    heading_strategy: str = "unknown"       # "word_styles" | "content_patterns"
                                            # | "llm_fallback" | "none"

    # Chunking stage
    chunk_count: int = 0
    heading_bounded_chunks: int = 0         # split at a detected heading
    size_split_chunks: int = 0              # split because > CHUNK_MAX_CHARS
    table_chunks: int = 0
    overlap_chars_added: int = 0
    chunk_chars_min: Optional[int] = None
    chunk_chars_max: Optional[int] = None
    chunk_chars_avg: Optional[int] = None

    # Safety
    limits_passed: bool = True

    # ------------------------------------------------------------------
    # Mutators — kept tiny so callers stay readable.
    # ------------------------------------------------------------------
    @classmethod
    def new(cls, *, doc_label: str, source_kind: str) -> "IngestionReport":
        """Create a fresh report. Prefer this over the bare constructor."""
        return cls(doc_label=doc_label, source_kind=source_kind)

    def record_ocr_decision(self, *, needs_ocr: bool, reason: str) -> None:
        """Called once per page by parse_pdf — accumulates guard stats."""
        self.ocr_reasons[reason] = self.ocr_reasons.get(reason, 0) + 1
        if needs_ocr:
            self.pages_ocr_candidate += 1
        elif reason == "low-text-no-images":
            self.pages_ocr_skipped_blank += 1

    def finalize_chunk_stats(self, chunk_texts: list) -> None:
        """
        Compute min/max/avg chunk-char stats from the finished chunks.
        Call this once at the end of build_chunks().
        """
        if not chunk_texts:
            return
        lengths = [len(t) for t in chunk_texts]
        self.chunk_count = len(lengths)
        self.chunk_chars_min = min(lengths)
        self.chunk_chars_max = max(lengths)
        self.chunk_chars_avg = sum(lengths) // len(lengths)

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------
    def emit(self) -> None:
        """
        Log the report as a single structured INFO line. The JSON-style
        body is wrapped in '[ingestion_report]' for easy grep / log
        insights filtering.
        """
        # asdict() handles the nested ocr_reasons dict cleanly.
        payload = asdict(self)
        logger.info("[ingestion_report] %s", payload)
