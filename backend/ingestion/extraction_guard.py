"""
Extraction Guard — Per-page quality check for PDF text extraction.

Purpose
-------
`fitz.Page.get_text("text")` silently returns an empty string for scanned
(image-only) PDF pages. Today such pages are skipped without a trace and
the corpus develops invisible holes. This module catches that case BEFORE
chunking runs.

The guard runs in two complementary modes:

  * SHADOW MODE (default, settings.OCR_SHADOW_MODE = True)
      Logs every page that *would* be routed to OCR with the reason,
      but does NOT call Textract. Lets us calibrate the trigger on real
      corpus data before incurring spend. This is the "step 1" of the
      hybrid fitz → Textract-on-demand plan.

  * LIVE MODE (settings.OCR_SHADOW_MODE = False) — wiring point only.
      The actual Textract call is intentionally NOT implemented here yet;
      a later iteration adds backend/ingestion/textract_client.py and
      replaces the `_invoke_ocr_backend` stub.

Design notes
------------
* This module is intentionally `fitz`-free in its signatures — it takes
  primitives so it can be unit-tested without opening a PDF.
* The decision is per-page, not per-document. A 200-page manual with 5
  image pages should keep 195 fitz-parsed pages and only route those 5.
* Reasons are short, machine-parseable strings (kebab-case) so they can
  be aggregated in the IngestionReport and alerted on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from backend.config import settings

logger = logging.getLogger("acadia-log-iq")


# Reasons returned by the guard. Kept as constants so callers / dashboards
# can switch on a stable vocabulary instead of free-text strings.
REASON_OK = "ok"
REASON_LOW_TEXT_WITH_IMAGES = "low-text-with-images"
REASON_LOW_TEXT_NO_IMAGES = "low-text-no-images"   # likely blank/divider — do NOT OCR
REASON_GARBLED_TEXT = "garbled-text"                # fitz returned text but it's unreadable


def _printable_ratio(text: str) -> float:
    """
    Compute the share of characters that are printable ASCII or common
    whitespace. PDFs with broken CID-mapped fonts often extract as long
    runs of replacement characters or private-use codepoints — char count
    looks fine, but the content is junk.

    Returns 1.0 for empty / whitespace-only input so the low-text guard
    handles that case instead.
    """
    if not text:
        return 1.0
    stripped = text.strip()
    if not stripped:
        return 1.0
    printable_count = 0
    for ch in stripped:
        code = ord(ch)
        # Standard ASCII printable range + tab/newline/cr.
        if 0x20 <= code <= 0x7E or ch in ("\t", "\n", "\r"):
            printable_count += 1
            continue
        # Allow extended Latin-1 letters (accented chars in vendor docs)
        # and common Unicode punctuation. Block private-use area and
        # replacement chars (where CID-font breakage lands).
        if 0xA0 <= code <= 0x024F:
            printable_count += 1
            continue
        if code in (0x2013, 0x2014, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2026):
            printable_count += 1
    return printable_count / len(stripped)


@dataclass(frozen=True)
class PageExtractionDecision:
    """
    Result of inspecting one page's extracted text.

    Attributes
    ----------
    page_number : int
        1-indexed page number (matches the rest of the pipeline).
    char_count : int
        Length of the extracted text (after .strip()).
    has_images : bool
        True if fitz reports drawing/image objects on the page.
    needs_ocr : bool
        True only when the page looks like a scanned page (low text + images).
        Pure blank/divider pages return False even though char_count is low.
    reason : str
        One of the REASON_* constants. Useful for aggregation.
    """

    page_number: int
    char_count: int
    has_images: bool
    needs_ocr: bool
    reason: str


def evaluate_page(
    page_number: int,
    extracted_text: str,
    has_images: bool,
    min_chars_threshold: Optional[int] = None,
) -> PageExtractionDecision:
    """
    Decide whether a page should be routed to OCR.

    A page is flagged when BOTH conditions hold:
      1. Stripped text length < min_chars_threshold
      2. The page contains at least one image object

    Pure blank pages (no text, no images) are explicitly NOT flagged —
    they are legitimate (divider pages, section separators) and OCR'ing
    them wastes money without adding value.

    Parameters
    ----------
    page_number : int
        1-indexed page number, used in the decision and log output.
    extracted_text : str
        Raw text from fitz.Page.get_text("text").
    has_images : bool
        Whether fitz reports any image objects on the page (caller passes
        `bool(page.get_images(full=False))`). Kept as a primitive so this
        function stays unit-testable.
    min_chars_threshold : int, optional
        Override for settings.EXTRACTION_GUARD_MIN_CHARS_PER_PAGE.

    Returns
    -------
    PageExtractionDecision
    """
    threshold = (
        min_chars_threshold
        if min_chars_threshold is not None
        else settings.EXTRACTION_GUARD_MIN_CHARS_PER_PAGE
    )
    char_count = len(extracted_text.strip())

    if char_count >= threshold:
        # Char count is healthy — but is the content actually readable?
        # Broken CID fonts produce long runs of garbage that still pass
        # the length check. Check printable ratio and flag for OCR if
        # it's clearly corrupt. Only runs above the length threshold so
        # we don't double-flag low-text pages.
        if settings.ENABLE_GARBLED_TEXT_GUARD:
            ratio = _printable_ratio(extracted_text)
            if ratio < settings.GARBLED_TEXT_PRINTABLE_THRESHOLD:
                return PageExtractionDecision(
                    page_number=page_number,
                    char_count=char_count,
                    has_images=has_images,
                    needs_ocr=True,
                    reason=REASON_GARBLED_TEXT,
                )

        return PageExtractionDecision(
            page_number=page_number,
            char_count=char_count,
            has_images=has_images,
            needs_ocr=False,
            reason=REASON_OK,
        )

    if has_images:
        return PageExtractionDecision(
            page_number=page_number,
            char_count=char_count,
            has_images=True,
            needs_ocr=True,
            reason=REASON_LOW_TEXT_WITH_IMAGES,
        )

    # Low text, no images — most likely a legitimate blank/divider page.
    # Don't OCR; this prevents accidental spend on dividers.
    return PageExtractionDecision(
        page_number=page_number,
        char_count=char_count,
        has_images=False,
        needs_ocr=False,
        reason=REASON_LOW_TEXT_NO_IMAGES,
    )


def log_decision(decision: PageExtractionDecision, *, doc_label: str) -> None:
    """
    Emit a structured log line for a single decision.

    In shadow mode (default), this is the ONLY visible side effect of the
    guard — the corpus is still ingested via fitz-only output. The line is
    INFO when OCR is needed (so it stands out) and DEBUG otherwise.
    """
    if not settings.EXTRACTION_GUARD_ENABLED:
        return

    if decision.needs_ocr:
        logger.info(
            "[extraction_guard] doc=%s page=%d chars=%d has_images=True "
            "reason=%s mode=%s",
            doc_label,
            decision.page_number,
            decision.char_count,
            decision.reason,
            "shadow" if settings.OCR_SHADOW_MODE else "live",
        )
    else:
        logger.debug(
            "[extraction_guard] doc=%s page=%d chars=%d reason=%s",
            doc_label,
            decision.page_number,
            decision.char_count,
            decision.reason,
        )


def should_invoke_live_ocr(decision: PageExtractionDecision) -> bool:
    """
    Decide whether the caller should perform a live OCR call for this page.

    Replaces the old `maybe_ocr_page` stub. The caller (parse_pdf) renders
    the page to a PNG via fitz and invokes
    `backend.ingestion.textract_client.ocr_fitz_page` itself — we don't do
    it here because extraction_guard is intentionally fitz-free so it
    stays unit-testable without opening a PDF.

    Returns True only when:
      * The guard flagged this page as needing OCR (low-text-with-images
        OR garbled-text), AND
      * OCR_SHADOW_MODE is OFF (caller has explicitly opted into live
        Textract spend).
    """
    if not decision.needs_ocr:
        return False
    if settings.OCR_SHADOW_MODE:
        return False
    return True


# Backwards-compatible alias. The previous iteration imported
# `maybe_ocr_page` from this module; we keep the name working so any
# external caller (or in-flight branch) doesn't break. New code should
# use `should_invoke_live_ocr` for clarity.
def maybe_ocr_page(decision: PageExtractionDecision) -> Optional[str]:
    """Deprecated alias — see `should_invoke_live_ocr`."""
    return None if not should_invoke_live_ocr(decision) else None
