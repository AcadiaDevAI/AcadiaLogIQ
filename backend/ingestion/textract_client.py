"""
Textract Client — Live OCR backend for the extraction guard.

Purpose
-------
When `extraction_guard` flags a PDF page as scanned (or garbled), and the
caller has opted out of shadow mode (`OCR_SHADOW_MODE=False`), this
module renders the page to a PNG via fitz and runs AWS Textract's
`DetectDocumentText` API to recover the page's text.

Design
------
* **`DetectDocumentText` only, not `AnalyzeDocument`.** The cheap OCR
  endpoint is ~$1.50 per 1,000 pages. `AnalyzeDocument` with TABLES is
  10× the price and we do NOT need it on the OCR path — the PDF table
  extractor already runs on every page via fitz.find_tables(). Reserve
  AnalyzeDocument for a future "tables-on-scanned-pages" tier if needed.

* **Synchronous, per-page.** We send one page at a time as PNG bytes.
  Latency is ~1–3s per page, which is acceptable because we only call
  it on flagged pages (typically a tiny fraction of any document).
  The async StartDocumentAnalysis path is more complex and not needed
  at our current scale.

* **Degrades gracefully.** Any boto3 error returns `""` rather than
  raising. Worst case the page contributes no text, which is exactly
  what shadow mode does anyway. We never block ingestion on Textract.

* **Caller enforces the per-doc cap.** This module doesn't know how
  many pages have already been OCR'd. The caller (parse_pdf) keeps the
  counter and stops invoking once `MAX_OCR_PAGES_PER_DOC` is hit.

* **No bundled IAM setup.** Assumes the same boto3 default credential
  chain that already works for Bedrock and S3 in this codebase. The
  caller's IAM role/user needs `textract:DetectDocumentText`.

Safety
------
* All Textract activity is gated upstream by `OCR_SHADOW_MODE`. Importing
  this module does NOT make any AWS calls; the boto3 client is built
  lazily on first use.
* Renders happen at `TEXTRACT_RENDER_DPI_SCALE` (default 2.0× of base
  PDF DPI) — high enough for good OCR accuracy, low enough that PNG
  payloads stay under Textract's 5MB sync limit on typical pages.
"""

from __future__ import annotations

import logging
from threading import Lock
from typing import TYPE_CHECKING, Optional

from backend.config import settings

if TYPE_CHECKING:
    import fitz  # noqa: F401

logger = logging.getLogger("acadia-log-iq")

# Textract's sync API rejects payloads above 5MB. PNGs from a 2× render
# of a typical letter-size page are well under this (~300–800 KB), but
# we keep the constant explicit so the size check is obvious.
_TEXTRACT_SYNC_PAYLOAD_LIMIT_BYTES = 5 * 1024 * 1024


class _LazyTextractClient:
    """
    Wraps boto3 client construction so importing this module is free —
    no AWS calls happen until the first OCR request. Thread-safe so the
    parallel embedding workers don't race to build two clients.
    """

    def __init__(self):
        self._client = None
        self._lock = Lock()

    def get(self):
        if self._client is None:
            with self._lock:
                if self._client is None:
                    import boto3
                    self._client = boto3.client(
                        "textract", region_name=settings.AWS_REGION
                    )
                    logger.info(
                        "[textract] client initialized region=%s",
                        settings.AWS_REGION,
                    )
        return self._client


_lazy_client = _LazyTextractClient()


def detect_text_from_png(png_bytes: bytes) -> str:
    """
    Send PNG bytes to Textract's DetectDocumentText and return joined
    line text. Returns "" on any failure — never raises, so ingestion
    is not blocked by a transient Textract outage.

    Notes
    -----
    Textract's `Blocks` response includes `PAGE`, `LINE`, `WORD`. We
    keep only `LINE` blocks and join with newlines; that preserves the
    document's vertical structure well enough for downstream chunking,
    without losing word boundaries the way concatenated WORD blocks
    would.
    """
    if not png_bytes:
        return ""
    if len(png_bytes) > _TEXTRACT_SYNC_PAYLOAD_LIMIT_BYTES:
        logger.warning(
            "[textract] payload %d bytes exceeds %d sync limit — skipping",
            len(png_bytes), _TEXTRACT_SYNC_PAYLOAD_LIMIT_BYTES,
        )
        return ""

    try:
        client = _lazy_client.get()
        response = client.detect_document_text(Document={"Bytes": png_bytes})
    except Exception as exc:
        # ClientError, EndpointConnectionError, NoCredentialsError, etc.
        # We log and return empty rather than raising; the page just
        # contributes no text, same as shadow mode would.
        logger.warning("[textract] detect_document_text failed: %s", exc)
        return ""

    lines = [
        block.get("Text", "")
        for block in response.get("Blocks", [])
        if block.get("BlockType") == "LINE"
    ]
    text = "\n".join(t for t in lines if t).strip()
    return text


def ocr_fitz_page(page: "fitz.Page") -> str:
    """
    Convenience wrapper: render a fitz Page to PNG at the configured
    scale and run it through Textract. Intended to be called from
    parse_pdf after `should_invoke_live_ocr(decision)` returns True.

    Returns "" on render or OCR failure, never raises.
    """
    try:
        import fitz  # local import keeps module-load fitz-free
        matrix = fitz.Matrix(
            settings.TEXTRACT_RENDER_DPI_SCALE,
            settings.TEXTRACT_RENDER_DPI_SCALE,
        )
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        png_bytes = pix.tobytes("png")
    except Exception as exc:
        logger.warning(
            "[textract] failed to render page %s to PNG: %s",
            getattr(page, "number", "?"), exc,
        )
        return ""

    return detect_text_from_png(png_bytes)


def is_available() -> Optional[bool]:
    """
    Lightweight health check — returns True if the boto3 client can be
    constructed (credentials and region resolve cleanly). Useful for a
    startup probe; not called by the ingestion path itself.

    Returns None if boto3 isn't installed; True/False otherwise.
    """
    try:
        _lazy_client.get()
        return True
    except ImportError:
        return None
    except Exception as exc:
        logger.warning("[textract] unavailable: %s", exc)
        return False
