"""
PDF Table Extractor — Recovers tabular structure from PDFs.

Why
---
`fitz.Page.get_text("text")` flattens tables into space-aligned text.
That works ok for prose but destroys table semantics — an escalation
matrix's "Severity → Owner → SLA" rows become run-on paragraphs that
the chunker may sever mid-row. The DOCX path already handles tables
explicitly (via python-docx); this module gives PDFs the same treatment.

How
---
fitz ships `Page.find_tables()` (PyMuPDF ≥ 1.23) which detects tables
using line / cell heuristics. We use the `lines_strict` strategy by
default because it reduces false positives in code blocks and aligned
non-table text.

Each detected table is rendered as a pipe-delimited representation that
matches the DOCX table block format used elsewhere in the pipeline:

    Header 1 | Header 2 | Header 3
    Cell A1  | Cell A2  | Cell A3

This means downstream code (`build_chunks`, table atomicity, chunk type
classification) treats PDF tables and DOCX tables identically.

Safety
------
* The function is wrapped in a broad try/except. If fitz changes its
  table API, or a particular page makes the detector crash, we return
  no table blocks for that page — the text path still runs, so
  ingestion never fails because of table extraction.
* Empty tables (no header, no rows) are skipped silently.
* Tables that don't add information vs. the plain-text path can still
  be useful for retrieval because the column structure improves
  embedding signal on row-oriented queries. We accept some duplication.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

from backend.config import settings
from backend.ingestion.structured_parser import ParsedBlock

if TYPE_CHECKING:
    import fitz  # noqa: F401

logger = logging.getLogger("acadia-log-iq")


def extract_tables_from_page(
    page: "fitz.Page",
    *,
    page_number: int,
    current_heading: Optional[str] = None,
    starting_source_order: int = 0,
) -> List[ParsedBlock]:
    """
    Detect and render tables on a single PDF page.

    Parameters
    ----------
    page : fitz.Page
        The page to scan.
    page_number : int
        1-indexed page number (matches the rest of the pipeline).
    current_heading : str, optional
        Heading context to attach to the table blocks. The caller
        typically passes the most recent heading discovered on the page;
        if None, the chunker assigns one later.
    starting_source_order : int
        Where in the per-page source_order sequence to place these
        blocks. Caller increments after each ParsedBlock is appended.

    Returns
    -------
    List[ParsedBlock]
        Zero or more table blocks. Empty list on any extraction failure.
    """
    if not settings.ENABLE_PDF_TABLE_EXTRACTION:
        return []

    blocks: List[ParsedBlock] = []
    order = starting_source_order

    try:
        # fitz 1.23+: Page.find_tables(strategy=...) returns a TableFinder
        # whose `.tables` attribute is the list. Strategy strings vary by
        # version; "lines_strict" is the conservative choice. We fall
        # back to default strategy if the named one is rejected.
        try:
            table_finder = page.find_tables(strategy=settings.PDF_TABLE_STRATEGY)
        except (TypeError, ValueError):
            # Older or newer fitz might not accept this strategy; try default.
            table_finder = page.find_tables()

        tables = getattr(table_finder, "tables", None) or []
    except Exception as exc:
        logger.debug(
            "[pdf_table_extractor] find_tables failed on page=%d: %s",
            page_number, exc,
        )
        return []

    for table in tables:
        try:
            rendered = _render_table_as_pipes(table)
        except Exception as exc:
            logger.debug(
                "[pdf_table_extractor] failed to render table on page=%d: %s",
                page_number, exc,
            )
            continue

        if not rendered:
            continue

        blocks.append(
            ParsedBlock(
                text=rendered,
                block_type="table",
                heading=current_heading,
                page_number=page_number,
                source_order=order,
                metadata={"extracted_via": "fitz.find_tables"},
            )
        )
        order += 1

    return blocks


def _render_table_as_pipes(table) -> str:
    """
    Convert a fitz Table object to a pipe-delimited string.

    Matches the DOCX table format produced by `parse_docx`:
        cell1 | cell2 | cell3
        cell4 | cell5 | cell6

    Empty cells become a single space so column count stays consistent.
    Trims rows that are entirely empty (common in tables that span page
    boundaries with phantom trailing rows).
    """
    # fitz Table exposes either .extract() returning list-of-rows, or
    # .rows iterable. Prefer .extract() because it's the documented API.
    rows = []
    try:
        rows = table.extract() or []
    except Exception:
        rows = []

    if not rows:
        return ""

    rendered_rows: List[str] = []
    for raw_row in rows:
        cells = [_clean_cell(c) for c in raw_row]
        # Skip rows that have no content at all.
        if not any(cell for cell in cells):
            continue
        rendered_rows.append(" | ".join(cell or " " for cell in cells))

    return "\n".join(rendered_rows).strip()


def _clean_cell(cell) -> str:
    """Normalize whitespace and None within a cell."""
    if cell is None:
        return ""
    text = str(cell).strip()
    # Collapse internal newlines so pipe-rows stay single-line.
    return " ".join(text.split())
