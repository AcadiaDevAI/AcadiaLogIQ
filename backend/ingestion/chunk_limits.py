"""
Chunk Limits — Pre-flight safety caps for document ingestion.

Why
---
Without caps a single malformed or accidentally-huge upload can:
  * Cost dozens of dollars in Bedrock embeddings and Haiku metadata calls.
  * Saturate the connection pool (we already see
    "Connection pool is full" warnings during 1109-chunk ingests).
  * Block other users by holding the in-process worker for hours.

This module fails LOUD and EARLY rather than silently truncating. The
caller (parse_file / build_chunks) raises `DocumentTooLargeError`, which
the ingestion service can surface to the user with a clear remediation
("split this document into sections").

The limits are intentionally generous (1000 pages, 5000 chunks) — they
exist to catch pathological uploads, not to constrain normal docs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from backend.config import settings


class DocumentTooLargeError(RuntimeError):
    """Raised when a document exceeds an ingestion safety cap."""


@dataclass(frozen=True)
class LimitCheck:
    """Result of a limits check — what was measured vs. allowed."""

    name: str       # which cap was checked ("pages", "chars", "chunks")
    measured: int
    limit: int
    passed: bool


def check_page_count(page_count: int, limit: Optional[int] = None) -> LimitCheck:
    """
    Enforce MAX_DOC_PAGES on a PDF before any per-page work happens.
    Cheap and runs before fitz iterates pages.
    """
    cap = limit if limit is not None else settings.MAX_DOC_PAGES
    return LimitCheck(
        name="pages",
        measured=page_count,
        limit=cap,
        passed=page_count <= cap,
    )


def check_char_total(char_total: int, limit: Optional[int] = None) -> LimitCheck:
    """
    Enforce MAX_DOC_CHARS on the post-extract text total. Catches huge
    plaintext logs / dumps that fit under the page cap but still produce
    massive chunk counts.
    """
    cap = limit if limit is not None else settings.MAX_DOC_CHARS
    return LimitCheck(
        name="chars",
        measured=char_total,
        limit=cap,
        passed=char_total <= cap,
    )


def check_chunk_count(chunk_count: int, limit: Optional[int] = None) -> LimitCheck:
    """
    Enforce MAX_DOC_CHUNKS after build_chunks(). Last line of defence
    against runaway fan-out into embed + Haiku calls.
    """
    cap = limit if limit is not None else settings.MAX_DOC_CHUNKS
    return LimitCheck(
        name="chunks",
        measured=chunk_count,
        limit=cap,
        passed=chunk_count <= cap,
    )


def enforce(check: LimitCheck, *, doc_label: str) -> None:
    """
    Raise DocumentTooLargeError if the check failed.

    The error message embeds the doc label, measurement and limit so
    operators can act on it directly from the log line.
    """
    if check.passed:
        return
    raise DocumentTooLargeError(
        f"Document '{doc_label}' exceeds {check.name} cap: "
        f"measured={check.measured} > limit={check.limit}. "
        f"Split the document into sections or raise the cap in settings."
    )


def enforce_many(checks: List[LimitCheck], *, doc_label: str) -> None:
    """Convenience wrapper to run multiple checks and report the first failure."""
    for check in checks:
        enforce(check, doc_label=doc_label)
