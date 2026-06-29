"""
File-Type Detector — Resolves a file's true format from CONTENT first,
then extension, and maps the result to a doc_kind ("ticket" or "kb").

Why content-first
-----------------
File extensions are unreliable: a user can rename `tickets.json` to
`.pdf`, the browser can mis-tag uploads as `.bin`, S3 keys can lose the
extension during slug normalization. Trusting only the extension means
a renamed JSON ticket dump lands in the KB pool, or a real PDF gets
ingested as JSON and fails parse.

Magic-byte sniffing + a small JSON-parse probe answers "what is this
file?" in a few hundred microseconds. Extension stays in the loop as
the **tiebreaker for ambiguous text formats** (CSV vs MD vs plain TXT —
all printable, no magic bytes).

Mapping policy
--------------
The detector returns one of these `detected` formats:

    pdf | docx | doc | json | csv | tsv | txt | md | unknown

Mapped to doc_kind:

    json, csv, tsv          → "ticket"
    pdf, docx, doc, txt, md → "kb"
    unknown                 → "kb"   (safer default — KB queries are
                                       global; ticket queries are
                                       per-Incident_Number scoped, so
                                       a misclassified KB doc as ticket
                                       would silently disappear from
                                       per-ticket retrieval)

If the operator explicitly supplied a doc_kind on the upload request,
that ALWAYS wins (the upload route is responsible for honoring it
before calling this detector).

Failure semantics
-----------------
Never raises. On read error, returns ``source="default"`` with
``detected="unknown"`` and ``doc_kind="kb"`` so the upload still
proceeds — corrupted files fail later at parse time with a clear error.
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("acadia-log-iq")


# How many bytes we read for content sniffing. 8 KB is enough to:
#   * Spot PDF / DOC / DOCX magic bytes (first 8 bytes)
#   * Parse a small JSON object / array
#   * Sample a handful of CSV rows for column-count heuristic
# More than this just bloats the I/O on every upload.
_SNIFF_BYTES = 8192


# Extension → detected format. Used for the tiebreaker on ambiguous
# printable-text files (where no magic bytes exist).
_EXT_TO_FORMAT = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "doc",
    ".json": "json",
    ".csv": "csv",
    ".tsv": "tsv",
    ".xlsx": "xlsx",
    ".xls": "xls",
    ".txt": "txt",
    ".md": "md",
    ".log": "txt",
}


# Detected format → doc_kind. Anything not in this map defaults to
# "kb" (see module-level docstring for the rationale).
#
# ONLY JSON maps to ticket history. CSV/TSV/Excel are tabular reference
# data (contacts, inventories, mappings) far more often than ticket
# exports, so they default to "kb". A user with a CSV/Excel ticket export
# can still pick doc_kind=ticket explicitly on the upload.
_FORMAT_TO_DOC_KIND = {
    "pdf":  "kb",
    "docx": "kb",
    "doc":  "kb",
    "txt":  "kb",
    "md":   "kb",
    "json": "ticket",
    "csv":  "kb",
    "tsv":  "kb",
    "xlsx": "kb",
    "xls":  "kb",
}


@dataclass(frozen=True)
class DetectionResult:
    """
    Output of `detect_file_kind`. Fields are intentionally flat so
    they serialize cleanly into the ingestion report / log lines.

    Attributes
    ----------
    detected : str
        The format identified — see module docstring for the vocabulary.
    doc_kind : str
        "ticket" or "kb" (or another VALID_DOC_KINDS value if the caller
        overrides). What the ingestion service writes to the documents
        table.
    source : str
        How the decision was reached:
          "content"   — magic bytes / JSON probe was conclusive
          "extension" — content was ambiguous; extension broke the tie
          "default"   — neither helped; fell back to "kb"
    agreement : bool
        True if both content and extension pointed to the same format.
        False is worth a log line — flags renamed-extension uploads.
    """

    detected: str
    doc_kind: str
    source: str
    agreement: bool


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def detect_file_kind(local_path: Path) -> DetectionResult:
    """
    Inspect a file at `local_path` and return its DetectionResult.

    Strategy:
      1. Read the first _SNIFF_BYTES bytes.
      2. Try strong content signals (magic bytes, JSON parse).
      3. If content is ambiguous, use the extension.
      4. If neither helps, default to "kb" / "unknown".

    The function never raises. A missing or unreadable file still
    produces a DetectionResult with `source="default"`.
    """
    extension = local_path.suffix.lower()
    ext_format = _EXT_TO_FORMAT.get(extension)  # None if extension unknown

    # ----- Read sniff buffer -----
    try:
        with open(local_path, "rb") as f:
            head = f.read(_SNIFF_BYTES)
    except OSError as exc:
        logger.warning(
            "[file_type_detector] could not read %s for sniffing: %s — "
            "falling back to extension-only",
            local_path, exc,
        )
        head = b""

    # ----- Run content sniffers in order of confidence -----
    content_format = _sniff_content(head, local_path)

    # ----- Combine signals -----
    if content_format and content_format != "unknown":
        # Strong content signal wins.
        agreement = (ext_format == content_format)
        if not agreement and ext_format:
            logger.warning(
                "[file_type_detector] extension/content mismatch on %s: "
                "ext=%s content=%s — trusting content",
                local_path.name, ext_format, content_format,
            )
        return _result(content_format, source="content", agreement=agreement)

    # Content was ambiguous (printable text with no magic).
    if ext_format:
        return _result(ext_format, source="extension", agreement=True)

    # Nothing usable — safe default.
    logger.info(
        "[file_type_detector] could not classify %s — defaulting to 'kb'",
        local_path.name,
    )
    return _result("unknown", source="default", agreement=False)


# ---------------------------------------------------------------------------
# Content sniffers
# ---------------------------------------------------------------------------
def _sniff_content(head: bytes, local_path: Path) -> Optional[str]:
    """
    Inspect the first chunk of bytes and return a detected format
    name, or None if the content is ambiguous (e.g. plain printable
    text with no distinguishing structure).
    """
    if not head:
        return None

    # PDF — strong magic bytes. Even if someone renamed it.
    if head.startswith(b"%PDF-"):
        return "pdf"

    # Old-format Word (.doc): OLE compound document magic.
    if head.startswith(b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"):
        return "doc"

    # ZIP container — could be DOCX, XLSX, PPTX, or just a ZIP. To
    # disambiguate DOCX we open the archive and look for the document
    # XML part. This requires reading the whole file (ZIP central
    # directory lives at the end), so we re-open it.
    if head.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(local_path) as z:
                names = set(z.namelist())
            if "word/document.xml" in names:
                return "docx"
        except (zipfile.BadZipFile, OSError):
            # Truncated upload or non-Office ZIP — fall through. The
            # extension may still tell us it's a docx (in which case
            # extension wins and parse fails later with a clear error).
            pass
        # Not a recognised Office archive; let extension decide.
        return None

    # JSON — accept if first non-whitespace char is { or [ AND a
    # truncated parse succeeds. The truncated-parse trick: even for
    # streams larger than _SNIFF_BYTES, a valid JSON file has a
    # well-formed opening that parses up to the point we cut. We use
    # `_first_json_object_length` to find a clean breakpoint.
    stripped = head.lstrip()
    if stripped[:1] in (b"{", b"["):
        if _is_probable_json(stripped):
            return "json"

    # CSV — heuristic: at least 3 consecutive non-empty lines with the
    # SAME comma count, count >= 2 (so we don't flag random prose as
    # CSV just because it has commas). Tabs are detected as TSV.
    sniffed_csv = _sniff_delimited(head)
    if sniffed_csv:
        return sniffed_csv

    # Could be plain text / markdown / something we don't have a strong
    # signal for. Return None → caller falls back to extension.
    return None


def _is_probable_json(stripped: bytes) -> bool:
    """
    Try to parse `stripped` (assumed to begin with `{` or `[`) as JSON.
    On large files our 8 KB window won't contain the whole document,
    so the parse will fail with "Unterminated...". We treat that as
    a positive signal IF the prefix that DID parse looks JSON-y.

    Belt-and-braces: also require the byte stream to be valid UTF-8 /
    ASCII; binary files that happen to start with 0x7B (`{`) won't
    pass this guard.
    """
    try:
        text = stripped.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return False
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError as exc:
        # If the parser made it past the first ~200 chars before the
        # cut, we trust this is JSON. "Expecting value: line 1 col 1"
        # would mean the open-brace was a coincidence and we reject.
        return bool(exc.pos and exc.pos > 200)


def _sniff_delimited(head: bytes) -> Optional[str]:
    """
    Heuristic CSV/TSV detection. Returns "csv", "tsv", or None.

    The signal: at least N non-empty lines (default 3) sharing the
    same delimiter count, with delimiter count >= 2. Mostly avoids
    false positives on prose (which has variable comma counts per
    line and no consistent column structure).
    """
    try:
        text = head.decode("utf-8", errors="replace")
    except UnicodeDecodeError:
        return None
    lines = [ln for ln in text.splitlines() if ln.strip()][:10]
    if len(lines) < 3:
        return None

    for delim, name in ((",", "csv"), ("\t", "tsv")):
        counts = [ln.count(delim) for ln in lines[:5]]
        if len(counts) >= 3 and counts[0] >= 2 and len(set(counts)) == 1:
            return name
    return None


# ---------------------------------------------------------------------------
# Result helper
# ---------------------------------------------------------------------------
def _result(detected: str, *, source: str, agreement: bool) -> DetectionResult:
    doc_kind = _FORMAT_TO_DOC_KIND.get(detected, "kb")
    return DetectionResult(
        detected=detected,
        doc_kind=doc_kind,
        source=source,
        agreement=agreement,
    )
