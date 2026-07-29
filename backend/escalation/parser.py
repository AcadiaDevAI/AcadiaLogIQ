"""PDF -> section-tagged chunks for the Escalation Procedures KB.

Detection strategy
------------------
Page-by-page text via PyMuPDF. For each page we inspect the first
handful of non-empty lines; if one of them matches a section anchor
(case-insensitive, punctuation-tolerant, length-capped to look like a
heading), that page is treated as the section start. Each section
spans from its detected start page to the page before the next
section's start (or to end-of-document for the last one).

The output is a list of chunks (~700-char windows with ~80-char
overlap) carrying ``section``/``page`` metadata so retrieval at query
time can filter strictly by section without any cross-talk.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz

from .sections import GENERAL_SECTION, SECTION_ANCHORS, SECTION_IDS


logger = logging.getLogger("acadia-log-iq")


@dataclass
class EscalationChunk:
    section: str
    page: int
    text: str


_PUNCT_TRIM = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[a-z0-9&]+")
_HEADING_MAX_CHARS = 120


def _normalize_line(line: str) -> str:
    return _PUNCT_TRIM.sub("", line.strip().lower())


def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def _line_contains_anchor(words: List[str], anchor_words: List[str]) -> bool:
    if not anchor_words or len(anchor_words) > len(words):
        return False
    span = len(anchor_words)
    for i in range(len(words) - span + 1):
        if words[i : i + span] == anchor_words:
            return True
    return False


def _detect_section_starts(page_texts: List[str]) -> Dict[str, int]:
    """Return ``{section_id: first_page_index}`` for every section found.

    Two-pass detection so the parser survives both formatting variation
    and a Table-of-Contents page that lists every section side by side:

    Pass 1 — scan every page for heading-like lines (length <=
    ``_HEADING_MAX_CHARS``) and record which sections appear on each
    page, where a hit means the anchor's tokens appear contiguously
    anywhere in the line (so "Third-Party Vendor Dispatch" still maps
    to the ``vendor_dispatch`` anchor).

    Pass 2 — for each section, prefer the first page where ONLY that
    section was hit (a real section divider), and fall back to the
    earliest hit otherwise. This pushes past the TOC page (which
    matches many anchors simultaneously) and lands on the real
    section divider page.
    """
    section_hits_per_page: List[Dict[str, int]] = []
    matches: Dict[str, List[int]] = {sid: [] for sid in SECTION_IDS}

    for raw_text in page_texts:
        page_hits: Dict[str, int] = {}
        if not raw_text:
            section_hits_per_page.append(page_hits)
            continue

        lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
        for line in lines:
            if len(line) > _HEADING_MAX_CHARS:
                continue
            normalized = _normalize_line(line)
            if not normalized:
                continue
            line_words = _tokenize(normalized)
            if not line_words:
                continue

            for section_id in SECTION_IDS:
                if section_id in page_hits:
                    continue
                for anchor in SECTION_ANCHORS[section_id]:
                    anchor_words = _tokenize(anchor)
                    if _line_contains_anchor(line_words, anchor_words):
                        page_hits[section_id] = 1
                        break

        section_hits_per_page.append(page_hits)

    for page_idx, page_hits in enumerate(section_hits_per_page):
        for section_id in page_hits:
            matches[section_id].append(page_idx)

    starts: Dict[str, int] = {}
    for section_id in SECTION_IDS:
        candidates = matches[section_id]
        if not candidates:
            continue
        # First preference: a page where ONLY this section was detected
        # (skips TOC, which fires many anchors on one page).
        for page_idx in candidates:
            if len(section_hits_per_page[page_idx]) == 1:
                starts[section_id] = page_idx
                break
        else:
            starts[section_id] = candidates[0]

    return starts


def _section_spans(
    starts: Dict[str, int], total_pages: int
) -> List[Tuple[str, int, int]]:
    """Convert ``{section: start_page}`` into ``[(section, start, end_exclusive), ...]``.

    Sections are sorted in document order; each section ends where the
    next one begins (or at ``total_pages`` for the final section).
    """
    ordered = sorted(starts.items(), key=lambda kv: kv[1])
    spans: List[Tuple[str, int, int]] = []
    for i, (section_id, start_page) in enumerate(ordered):
        end_page = ordered[i + 1][1] if i + 1 < len(ordered) else total_pages
        if end_page > start_page:
            spans.append((section_id, start_page, end_page))
    return spans


def _chunk_text(
    text: str, *, chunk_chars: int = 700, overlap_chars: int = 80
) -> List[str]:
    cleaned = _WHITESPACE.sub(" ", text).strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_chars:
        return [cleaned]

    out: List[str] = []
    step = max(1, chunk_chars - overlap_chars)
    for start in range(0, len(cleaned), step):
        piece = cleaned[start : start + chunk_chars].strip()
        if piece:
            out.append(piece)
        if start + chunk_chars >= len(cleaned):
            break
    return out


def parse_json(json_bytes: bytes) -> Tuple[List[EscalationChunk], Dict[str, Dict[str, int]]]:
    """Parse a JSON escalation document into chunks under GENERAL_SECTION.

    US Pharma uploads JSON escalation matrices (any structure) rather than a
    vendor-sectioned PDF. We render the JSON to indented text (lossless) and
    window it into overlapping chunks so ``/escalation/ask`` can retrieve
    across the whole document — there are no per-vendor sections to detect.

    Returns the same ``(chunks, section_summary)`` shape as :func:`parse_pdf`.
    """
    if not json_bytes:
        raise ValueError("Empty JSON payload")
    try:
        # utf-8-sig strips a leading BOM (Windows editors add one), which
        # otherwise makes json.loads fail with "Expecting value".
        data = json.loads(json_bytes.decode("utf-8-sig"))
    except Exception as exc:
        raise ValueError(f"Invalid JSON: {exc}")

    # Indented dump keeps keys next to values so a chunk stays readable.
    text = json.dumps(data, indent=2, ensure_ascii=False)
    pieces = _chunk_text(text, chunk_chars=1200, overlap_chars=150)
    if not pieces:
        raise ValueError("No content extracted from the JSON.")

    chunks = [
        EscalationChunk(section=GENERAL_SECTION, page=idx + 1, text=piece)
        for idx, piece in enumerate(pieces)
    ]
    summary: Dict[str, Dict[str, int]] = {
        GENERAL_SECTION: {"start_page": 1, "end_page": len(chunks), "chunks": len(chunks)}
    }
    return chunks, summary


def parse_pdf(pdf_bytes: bytes) -> Tuple[List[EscalationChunk], Dict[str, Dict[str, int]]]:
    """Parse the consolidated Escalation PDF into section-tagged chunks.

    Returns
    -------
    chunks
        Flat list of :class:`EscalationChunk` ready to embed + persist.
    section_summary
        ``{section_id: {"start_page": int, "end_page": int, "chunks": int}}``
        (1-indexed page numbers, end inclusive) for the upload response.
    """
    if not pdf_bytes:
        raise ValueError("Empty PDF payload")

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page_texts: List[str] = []
        for page in doc:
            try:
                page_texts.append(page.get_text("text") or "")
            except Exception:
                page_texts.append("")
        total_pages = len(page_texts)
    finally:
        doc.close()

    starts = _detect_section_starts(page_texts)
    if not starts:
        # No Acadia vendor-section headings (Cisco / Verizon / AT&T / Vendor
        # Dispatch) — e.g. a US Pharma vendor-escalation PDF or any other
        # unstructured escalation doc. Fall back to whole-document ingestion
        # under the catch-all "general" section (searched across the whole KB
        # at query time) instead of rejecting the upload with a 400.
        fallback_chunks = [
            EscalationChunk(section=GENERAL_SECTION, page=idx + 1, text=piece)
            for idx, page_text in enumerate(page_texts)
            for piece in _chunk_text(page_text)
        ]
        if not fallback_chunks:
            raise ValueError("No text could be extracted from the PDF.")
        summary = {
            GENERAL_SECTION: {
                "start_page": 1,
                "end_page": total_pages,
                "chunks": len(fallback_chunks),
            }
        }
        return fallback_chunks, summary

    chunks: List[EscalationChunk] = []
    summary: Dict[str, Dict[str, int]] = {}

    for section_id, start_page, end_page in _section_spans(starts, total_pages):
        section_chunk_count = 0
        for page_idx in range(start_page, end_page):
            for piece in _chunk_text(page_texts[page_idx]):
                chunks.append(
                    EscalationChunk(
                        section=section_id,
                        page=page_idx + 1,
                        text=piece,
                    )
                )
                section_chunk_count += 1
        summary[section_id] = {
            "start_page": start_page + 1,
            "end_page": end_page,
            "chunks": section_chunk_count,
        }

    return chunks, summary
