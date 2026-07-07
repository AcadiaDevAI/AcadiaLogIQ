"""
Structured Parser — Smart Adaptive Chunking.
Tries 3 detection strategies in order:
  1. Word heading styles (Heading 1/2/3) — fastest, free
  2. Content-based patterns (Scenario A:, Troubleshooting Runbook:) — fast, free
  3. LLM-based section discovery via Haiku — slowest, ~$0.002/doc, works on anything
Only falls back to LLM when strategies 1+2 find zero headings.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import fitz
from docx import Document

from backend.config import settings
from backend.metadata.structure_config import match_operational_section

# New-in-this-iteration: modular chunking enhancements.
# Each module is self-contained and can be disabled by flipping its
# corresponding settings flag — the original code path is preserved.
from backend.ingestion.extraction_guard import (
    evaluate_page,
    log_decision,
    should_invoke_live_ocr,
)
from backend.ingestion.chunk_overlap import apply_overlap, build_overlap_prefix
from backend.ingestion.chunk_limits import (
    check_chunk_count,
    check_char_total,
    check_page_count,
    enforce,
)
from backend.ingestion.chunk_metrics import IngestionReport

logger = logging.getLogger("acadia-log-iq")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ParsedBlock:
    text: str
    block_type: str
    heading: Optional[str] = None
    page_number: Optional[int] = None
    source_order: int = 0
    metadata: Dict = field(default_factory=dict)


@dataclass
class ParsedChunk:
    chunk_index: int
    text: str
    chunk_type: str
    section_heading: Optional[str]
    operational_section: Optional[str]
    page_number: Optional[int]
    source_order: int
    token_estimate: int
    metadata: Dict


# ---------------------------------------------------------------------------
# Regex patterns for classification
# ---------------------------------------------------------------------------
_HEADING_RE = re.compile(r"^\s*(#{1,6}\s+.+|[A-Z][A-Z0-9 /:_\-\(\)]{3,})\s*$")
_BULLET_RE = re.compile(r"^\s*([-*•]|\d+\.)\s+")
_CODE_RE = re.compile(
    r"^\s*(\$|>|kubectl |aws |curl |SELECT |INSERT |UPDATE |DELETE "
    r"|GET |POST |apiVersion:|kind:|FROM |WHERE )"
)
_TABLE_HINT_RE = re.compile(r"\s{2,}|\|")

# --- Content-based heading patterns (Strategy 2) ---
# Detects structural headings from text even when Word styles are 'Normal'.
_CONTENT_HEADING_PATTERNS = [
    # Runbook/document titles
    re.compile(r"^Troubleshooting Runbook\s*:", re.I),
    # Scenario headers: "Scenario A: ...", "Scenario 1: ..."
    re.compile(r"^Scenario\s+[A-Z0-9]+\s*:", re.I),
    # Numbered sections: "1. Overview", "3.2 Configuration"
    re.compile(r"^\d+(?:\.\d+)*\s+[A-Z]", re.I),
    # Chapter/Section/Part headers
    re.compile(r"^(?:Chapter|Section|Part)\s+\d+", re.I),
    # Step-based headers: "Step 1:", "Phase 1:"
    re.compile(r"^(?:Step|Phase|Stage)\s+\d+\s*:", re.I),
    # Common doc section titles (standalone lines)
    re.compile(
        r"^(?:Overview|Introduction|Prerequisites|Procedure|Conclusion|Appendix|"
        r"Escalation Matrix|Escalation Criteria|References|Glossary|"
        r"Executive Summary|Background|Scope|Objectives)\s*:?\s*$", re.I
    ),
    # Section dividers (dashes, equals, asterisks)
    re.compile(r"^[-=*]{5,}$"),
    # Problem / Solution / Workaround style (KB articles)
    re.compile(r"^(?:Problem|Solution|Workaround|Root Cause|Resolution|Impact)\s*:?\s*$", re.I),
]

# --- Sub-section labels (stay WITHIN parent chunk, not split boundaries) ---
_SUB_SECTION_PATTERNS = [
    re.compile(
        r"^(?:Alert Signatures|Severity|Incident Summary|Probable Causes|"
        r"Corrective Actions|Diagnostic Steps|Resolution Steps|"
        r"Validation Steps|Validation / Post.Check|Post-Check|"
        r"Escalation Criteria|Commands / Tools|Commands|"
        r"Expected Results|Root Cause|Workaround|"
        r"Affected Systems|Impact Assessment|"
        r"Pre-Conditions|Post-Conditions|Notes|Warning)\s*:?\s*$", re.I
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def normalize_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def _is_content_heading(text: str) -> bool:
    """Check if a line is a structural heading based on its content."""
    stripped = text.strip()
    return any(p.match(stripped) for p in _CONTENT_HEADING_PATTERNS)


def _is_sub_section_label(text: str) -> bool:
    """Check if a line is a sub-section label within a scenario."""
    stripped = text.strip()
    return any(p.match(stripped) for p in _SUB_SECTION_PATTERNS)


def _is_section_divider(text: str) -> bool:
    """Check if a line is a visual section divider."""
    return bool(re.match(r"^[-=*]{5,}$", text.strip()))


def classify_line(line: str) -> str:
    text = line.strip()
    if not text:
        return "blank"
    if _is_section_divider(text):
        return "divider"
    if _is_content_heading(text):
        return "heading"
    if _HEADING_RE.match(text):
        return "heading"
    if _BULLET_RE.match(text):
        return "bullet"
    if settings.ENABLE_CODE_BLOCK_DETECTION and _CODE_RE.match(text):
        return "code"
    if settings.ENABLE_TABLE_PARSING and (_TABLE_HINT_RE.search(text) and len(text.split()) >= 3):
        if "|" in text or "  " in text:
            return "table"
    return "paragraph"


# ---------------------------------------------------------------------------
# LLM-based section discovery (Strategy 3 — fallback)
# ---------------------------------------------------------------------------
def _llm_discover_sections(full_text: str) -> List[str]:
    """
    Use Claude Haiku to identify section boundaries in unstructured text.
    Sends the first ~8000 chars and asks Haiku to return a JSON array
    of section heading strings found in the text.
    Cost: ~$0.002 per document. Only called when strategies 1+2 find nothing.

    Returns a list of heading strings that appear verbatim in the text.
    """
    # Import here to avoid circular imports at module load time
    from backend.services.bedrock_haiku import haiku_client

    # Send a preview — enough to find the document's structural pattern
    preview = full_text[:settings.LLM_CHUNK_FALLBACK_PREVIEW_CHARS]

    system = (
        "You identify section boundaries in documents. "
        "Return strict JSON only. No markdown fences. No commentary."
    )

    prompt = f"""Analyze this document text and identify ALL section headings / titles
that represent logical boundaries between different topics or procedures.

Return a JSON array of the EXACT heading strings as they appear in the text.
Include document titles, scenario names, chapter headers, procedure names, etc.
Do NOT include sub-labels like "Probable Causes:" or "Severity:" — only major sections.

Example output: ["Introduction", "Scenario 1: Network Failure", "Appendix A"]

Document text:
{preview}

Section headings (JSON array only):"""

    try:
        result = haiku_client.invoke_json(
            system=system,
            prompt=prompt,
            max_tokens=1024,
            context="structured_parse",
        )

        # The result might be a dict with a key, or a raw list
        if isinstance(result, list):
            headings = [str(h).strip() for h in result if str(h).strip()]
        elif isinstance(result, dict):
            # Try common keys
            for key in ("headings", "sections", "section_headings", "results"):
                if key in result and isinstance(result[key], list):
                    headings = [str(h).strip() for h in result[key] if str(h).strip()]
                    break
            else:
                headings = []
        else:
            headings = []

        # Validate: only keep headings that actually appear in the text
        validated = []
        text_lower = full_text.lower()
        for h in headings:
            if h.lower() in text_lower and len(h) > 3:
                validated.append(h)

        logger.info(
            "LLM section discovery: found %d headings from %d candidates (preview=%d chars)",
            len(validated), len(headings), len(preview),
        )
        return validated

    except Exception as exc:
        logger.warning("LLM section discovery failed: %s", exc)
        return []


def _apply_llm_headings_to_blocks(
    blocks: List[ParsedBlock],
    llm_headings: List[str],
) -> List[ParsedBlock]:
    """
    Post-process blocks: upgrade any paragraph block whose text matches
    an LLM-discovered heading to block_type='heading'.
    """
    heading_set = {h.lower().strip() for h in llm_headings}

    updated = []
    current_heading = None

    for block in blocks:
        text_lower = block.text.strip().lower()

        # Check if this block's text matches an LLM-discovered heading
        if block.block_type == "paragraph" and text_lower in heading_set:
            block = ParsedBlock(
                text=block.text.strip(),
                block_type="heading",
                heading=block.text.strip(),
                page_number=block.page_number,
                source_order=block.source_order,
                metadata=block.metadata,
            )
            current_heading = block.text.strip()
        elif block.block_type == "heading":
            current_heading = block.heading or block.text
        else:
            # Update the heading context for non-heading blocks
            if current_heading and not block.heading:
                block.heading = current_heading

        updated.append(block)

    return updated


# ---------------------------------------------------------------------------
# Block extraction from raw text (for PDFs and plain text)
# ---------------------------------------------------------------------------
def extract_blocks_from_text(
    text: str,
    page_number: Optional[int] = None,
) -> List[ParsedBlock]:
    blocks: List[ParsedBlock] = []
    lines = normalize_text(text).splitlines()
    current_heading: Optional[str] = None
    order = 0
    buffer: List[str] = []
    buffer_type: Optional[str] = None

    def flush():
        nonlocal buffer, buffer_type, order
        if not buffer:
            return
        blocks.append(
            ParsedBlock(
                text="\n".join(buffer).strip(),
                block_type=buffer_type or "paragraph",
                heading=current_heading,
                page_number=page_number,
                source_order=order,
            )
        )
        order += 1
        buffer = []
        buffer_type = None

    for line in lines:
        line_type = classify_line(line)

        if line_type == "blank":
            flush()
            continue

        if line_type == "divider":
            flush()
            continue

        if line_type == "heading":
            flush()
            current_heading = line.strip().lstrip("#").strip()
            blocks.append(
                ParsedBlock(
                    text=current_heading,
                    block_type="heading",
                    heading=current_heading,
                    page_number=page_number,
                    source_order=order,
                )
            )
            order += 1
            continue

        if buffer_type and buffer_type != line_type:
            flush()

        buffer_type = line_type
        buffer.append(line)

    flush()
    return blocks


# ---------------------------------------------------------------------------
# File-type-specific parsers
# ---------------------------------------------------------------------------
def parse_pdf(
    path: Path,
    *,
    report: Optional[IngestionReport] = None,
) -> List[ParsedBlock]:
    """
    Parse a PDF into ParsedBlocks using fitz for text extraction.

    New in this iteration
    ---------------------
    * Pre-flight page-count cap (MAX_DOC_PAGES) — prevents pathological
      uploads from monopolizing the worker.
    * Per-page extraction guard — flags suspected scanned pages. In shadow
      mode (default) the decision is only logged; in live mode (later
      iteration) the page would be routed to Textract.
    * Per-page stats recorded onto the optional IngestionReport so the
      caller can emit ONE structured line at end-of-ingest.

    Core path (fitz.get_text + extract_blocks_from_text) is unchanged.
    """
    blocks: List[ParsedBlock] = []
    doc_label = path.name

    # Per-doc OCR counter — enforces settings.MAX_OCR_PAGES_PER_DOC so a
    # 1000-page scanned upload cannot route every page to Textract. When
    # we hit the cap we keep flagging in the log (so the guard's signal
    # stays visible) but stop making live calls.
    ocr_pages_invoked = 0

    with fitz.open(path) as doc:
        # Safety cap: fail loud before we iterate hundreds of pages.
        enforce(check_page_count(doc.page_count), doc_label=doc_label)
        if report is not None:
            report.pages_total = doc.page_count

        for page_idx, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""

            # --- Extraction guard (per-page) ------------------------------
            # Cheap check: did fitz actually get useful text out of this
            # page? If not AND the page has images, flag for OCR (shadow
            # mode just logs). Pure-blank pages are silently skipped.
            if settings.EXTRACTION_GUARD_ENABLED:
                has_images = bool(page.get_images(full=False))
                decision = evaluate_page(
                    page_number=page_idx,
                    extracted_text=text,
                    has_images=has_images,
                )
                log_decision(decision, doc_label=doc_label)
                if report is not None:
                    report.record_ocr_decision(
                        needs_ocr=decision.needs_ocr,
                        reason=decision.reason,
                    )

                # Live OCR path — only fires when the operator has
                # explicitly flipped OCR_SHADOW_MODE off AND we're
                # under the per-doc cap. Failure (network, IAM, etc.)
                # returns "" and we keep the original empty text;
                # ingestion never blocks on OCR.
                if should_invoke_live_ocr(decision):
                    if ocr_pages_invoked < settings.MAX_OCR_PAGES_PER_DOC:
                        # Lazy import: avoids importing boto3 at module
                        # load time when OCR is not used.
                        from backend.ingestion.textract_client import ocr_fitz_page
                        ocr_text = ocr_fitz_page(page)
                        if ocr_text:
                            text = ocr_text
                            ocr_pages_invoked += 1
                            if report is not None:
                                report.pages_ocr_invoked = (
                                    getattr(report, "pages_ocr_invoked", 0) + 1
                                )
                    else:
                        logger.warning(
                            "[textract] doc=%s page=%d skipped — per-doc OCR "
                            "cap reached (limit=%d)",
                            doc_label, page_idx,
                            settings.MAX_OCR_PAGES_PER_DOC,
                        )

            # --- Text extraction -----------------------------------------
            if text.strip():
                blocks.extend(extract_blocks_from_text(text, page_number=page_idx))

            # --- PDF table extraction (post-text) ------------------------
            # Runs AFTER text extraction so tables appear as additional
            # structured blocks alongside the prose. Match the DOCX path
            # which iterates paragraphs AND tables separately. Disabled
            # silently if ENABLE_PDF_TABLE_EXTRACTION is False.
            if settings.ENABLE_PDF_TABLE_EXTRACTION:
                # Lazy import keeps PDF parsing optional-table-extractor.
                from backend.ingestion.pdf_table_extractor import (
                    extract_tables_from_page,
                )
                # The most recent heading on the page (if any) gives
                # extracted tables some context. If no heading was found
                # in the page's prose, the chunker will inherit context
                # from the prior page's heading.
                most_recent_heading = next(
                    (b.heading for b in reversed(blocks)
                     if b.page_number == page_idx and b.heading),
                    None,
                )
                table_blocks = extract_tables_from_page(
                    page,
                    page_number=page_idx,
                    current_heading=most_recent_heading,
                    starting_source_order=len(blocks),
                )
                blocks.extend(table_blocks)
                if report is not None and table_blocks:
                    # Track pages that contributed at least one table so
                    # we can see how often the extractor fires.
                    report.pages_with_tables = (
                        getattr(report, "pages_with_tables", 0) + 1
                    )

    if report is not None:
        report.chars_extracted = sum(len(b.text) for b in blocks)

    return blocks


def _iter_docx_block_items(doc):
    """
    Yield ``(kind, item)`` tuples from a python-docx Document in **true
    document order**, where ``kind`` is ``"paragraph"`` or ``"table"`` and
    ``item`` is the corresponding python-docx object.

    Why this exists
    ---------------
    python-docx exposes ``doc.paragraphs`` and ``doc.tables`` as two
    independent iterators. The default ``parse_docx`` pattern (paragraph
    loop, then table loop) loses ALL positional information about where
    each table sits relative to the headings. The downstream effect:
    every table gets tagged with ``current_heading`` = the LAST heading
    in the whole document, not the heading the table actually belongs
    to. A table in section 5 ends up labelled "Appendix" or whatever the
    final section in the file happens to be, and KB retrieval can't
    surface it for queries about section 5's topic.

    By walking ``doc.element.body.iterchildren()`` ourselves, we get
    paragraphs and tables interleaved in document order — the way a
    reader sees them. The caller maintains its ``current_heading`` state
    across the single unified loop, so each table inherits the heading
    that's directly above it.

    Only top-level paragraphs and tables are yielded; nested elements
    (tables inside table cells, headers/footers) are intentionally
    skipped — same scope as the previous behaviour, just in the right
    order.
    """
    from docx.text.paragraph import Paragraph
    from docx.table import Table

    body = doc.element.body
    for child in body.iterchildren():
        tag = child.tag
        # tag is namespaced like '{http://...}p' or '{http://...}tbl'.
        # Suffix-match keeps us decoupled from the exact namespace URI.
        if tag.endswith("}p"):
            yield "paragraph", Paragraph(child, doc)
        elif tag.endswith("}tbl"):
            yield "table", Table(child, doc)


def parse_docx(path: Path) -> List[ParsedBlock]:
    """
    Parse a .docx file into structured blocks.

    Strategy order:
    1. Word heading styles (Heading 1/2/3) — check para.style
    2. Content-based patterns — regex on text content
    3. If both find zero headings → LLM fallback (Haiku discovers sections)

    Paragraphs AND tables are walked in true document order via
    `_iter_docx_block_items`, so each table is tagged with the heading
    directly above it. The previous two-loop implementation labeled
    every table with the LAST heading in the document (e.g. the
    appendix), which broke retrieval for table-content queries like
    "show me the Rapid Routing Matrix".
    """
    doc = Document(str(path))
    blocks: List[ParsedBlock] = []
    order = 0
    current_heading = None
    heading_count = 0  # Track how many headings we find

    for kind, item in _iter_docx_block_items(doc):
        # ─── Paragraph branch ────────────────────────────────────────
        if kind == "paragraph":
            text = item.text.strip()
            if not text:
                continue

            style = (item.style.name or "").lower() if item.style else ""

            # --- Detect block type ---
            # Priority 1: Word heading styles
            if "heading" in style or "title" in style:
                block_type = "heading"
                current_heading = text
                heading_count += 1
            # Priority 2: Content-based heading detection
            elif _is_content_heading(text):
                block_type = "heading"
                current_heading = text
                heading_count += 1
            # Priority 3: Section dividers
            elif _is_section_divider(text):
                continue
            # Priority 4: Sub-section labels (stay within chunk)
            elif _is_sub_section_label(text):
                block_type = "sub_heading"
            # Priority 5: Regular content
            elif text.startswith(("-", "*", "•")):
                block_type = "bullet"
            elif settings.ENABLE_CODE_BLOCK_DETECTION and _CODE_RE.match(text):
                block_type = "code"
            else:
                block_type = "paragraph"

            blocks.append(
                ParsedBlock(
                    text=text,
                    block_type=block_type,
                    heading=current_heading,
                    page_number=None,
                    source_order=order,
                )
            )
            order += 1
            continue

        # ─── Table branch ─────────────────────────────────────────────
        # `current_heading` here is the heading immediately above this
        # table in document order — exactly what was wrong in the
        # previous two-loop implementation. Build a pipe-delimited
        # rendering matching the existing format so downstream
        # build_chunks() treats it the same as before.
        if kind == "table" and settings.ENABLE_TABLE_PARSING:
            rows = []
            for row in item.rows:
                cells = [cell.text.strip() for cell in row.cells]
                rows.append(" | ".join(cells))
            if rows:
                blocks.append(
                    ParsedBlock(
                        text="\n".join(rows).strip(),
                        block_type="table",
                        heading=current_heading,
                        page_number=None,
                        source_order=order,
                        metadata={"under_heading": current_heading or ""},
                    )
                )
                order += 1

    # ===================================================================
    # Strategy 3: LLM fallback if no headings were found
    # This handles documents with all-Normal styles AND no recognizable
    # content patterns (vendor manuals, free-form reports, etc.)
    # ===================================================================
    if heading_count == 0 and len(blocks) > 5 and settings.ENABLE_LLM_CHUNK_FALLBACK:
        logger.info(
            "No headings detected (styles or patterns) in %d blocks — "
            "falling back to LLM section discovery",
            len(blocks),
        )
        # Reconstruct full text for LLM analysis
        full_text = "\n".join(b.text for b in blocks if b.text.strip())
        llm_headings = _llm_discover_sections(full_text)

        if llm_headings:
            logger.info("LLM found %d section headings, re-tagging blocks", len(llm_headings))
            blocks = _apply_llm_headings_to_blocks(blocks, llm_headings)
            heading_count = sum(1 for b in blocks if b.block_type == "heading")
            logger.info("After LLM re-tagging: %d heading blocks", heading_count)
        else:
            logger.warning("LLM fallback found no headings — chunking will use char limits only")

    logger.info(
        "parse_docx complete: %d blocks, %d headings (strategy: %s)",
        len(blocks), heading_count,
        "word_styles" if heading_count > 0 else "llm_fallback",
    )

    return blocks


def parse_plain(path: Path) -> List[ParsedBlock]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    blocks = extract_blocks_from_text(text)

    # Check if we found any headings
    heading_count = sum(1 for b in blocks if b.block_type == "heading")

    # LLM fallback for plain text files with no detected headings
    if heading_count == 0 and len(blocks) > 5 and settings.ENABLE_LLM_CHUNK_FALLBACK:
        logger.info("No headings in plain text (%d blocks) — trying LLM discovery", len(blocks))
        full_text = "\n".join(b.text for b in blocks if b.text.strip())
        llm_headings = _llm_discover_sections(full_text)
        if llm_headings:
            blocks = _apply_llm_headings_to_blocks(blocks, llm_headings)

    return blocks


def parse_file(path: Path) -> List[ParsedBlock]:
    """
    Top-level parser entry point. Dispatches to PDF/DOCX/plain-text parsers,
    enforces ingestion safety caps, and emits ONE structured ingestion
    report at the end so we can observe the chunking pipeline end-to-end.

    JSON / CSV are NOT handled here — they short-circuit upstream in
    contextual_ingestion_service.py with their own structured parsers
    (one chunk per record). Prose-style chunking is wrong for record
    data, so we keep that separation explicit.

    The IngestionReport is created here and threaded into parse_pdf so
    the per-page extraction-guard stats land in the same log line as the
    chunk stats. build_chunks() later mutates the same report.
    """
    suffix = path.suffix.lower()
    source_kind = (
        "pdf" if suffix == ".pdf"
        else "docx" if suffix == ".docx"
        else "plain"
    )

    report = IngestionReport.new(doc_label=path.name, source_kind=source_kind)

    try:
        if suffix == ".pdf":
            blocks = parse_pdf(path, report=report)

            # LLM fallback (Strategy 3) for PDFs whose extracted text has
            # no detectable headings. This was already wired before this
            # iteration; we just thread the heading strategy into the
            # report so it's visible end-to-end.
            heading_count = sum(1 for b in blocks if b.block_type == "heading")
            strategy = "content_patterns" if heading_count > 0 else "none"
            if heading_count == 0 and len(blocks) > 5 and settings.ENABLE_LLM_CHUNK_FALLBACK:
                logger.info("No headings in PDF (%d blocks) — trying LLM discovery", len(blocks))
                full_text = "\n".join(b.text for b in blocks if b.text.strip())
                llm_headings = _llm_discover_sections(full_text)
                if llm_headings:
                    blocks = _apply_llm_headings_to_blocks(blocks, llm_headings)
                    heading_count = sum(1 for b in blocks if b.block_type == "heading")
                    strategy = "llm_fallback"
            report.heading_count = heading_count
            report.heading_strategy = strategy

        elif suffix == ".docx":
            blocks = parse_docx(path)
            report.heading_count = sum(1 for b in blocks if b.block_type == "heading")
            report.heading_strategy = (
                "word_styles_or_patterns" if report.heading_count > 0 else "llm_fallback"
            )

        else:
            blocks = parse_plain(path)
            report.heading_count = sum(1 for b in blocks if b.block_type == "heading")
            report.heading_strategy = (
                "content_patterns" if report.heading_count > 0 else "none"
            )

        # Char-total safety cap — catches plaintext logs / dumps that
        # slip past the page cap (which only applies to PDFs).
        char_total = sum(len(b.text) for b in blocks)
        report.chars_extracted = max(report.chars_extracted, char_total)
        enforce(check_char_total(char_total), doc_label=path.name)

    except Exception:
        # Make sure a failed parse still emits its partial report — that
        # report often contains the diagnostic clue (e.g. all pages were
        # flagged as low-text-no-images, meaning the PDF is corrupted).
        report.limits_passed = False
        report.emit()
        raise

    # Attach the report to the returned list via a side channel so
    # build_chunks() can keep mutating it. We use a module-level dict
    # keyed by id(blocks) to avoid changing the ParsedBlock signature.
    _REPORT_REGISTRY[id(blocks)] = report
    return blocks


# Module-level registry: lets build_chunks() find the IngestionReport
# created in parse_file() without changing the public List[ParsedBlock]
# return type. The registry self-cleans on chunk-build completion to
# avoid memory growth across many ingests.
_REPORT_REGISTRY: dict = {}


# ---------------------------------------------------------------------------
# Chunk type classification
# ---------------------------------------------------------------------------
def choose_chunk_type(block_types: List[str], heading: Optional[str]) -> str:
    """Classify a chunk's operational type from its blocks and heading."""
    if heading:
        matched = match_operational_section(heading)
        if matched:
            return matched.chunk_type

    heading_lower = (heading or "").lower()
    if "scenario" in heading_lower:
        if any(kw in heading_lower for kw in [
            "troubleshoot", "diagnostic", "failure", "issue", "down",
            "error", "fault", "loss", "degrad",
        ]):
            return "diagnostic_chunk"
        return "general_chunk"
    if "escalation" in heading_lower:
        return "escalation_chunk"

    if "code" in block_types:
        return "command_chunk"
    if "table" in block_types:
        return "validation_chunk"
    return "general_chunk"


# ---------------------------------------------------------------------------
# Scenario-aware chunk builder
# ---------------------------------------------------------------------------
def build_chunks(blocks: List[ParsedBlock]) -> List[ParsedChunk]:
    """
    Build chunks from parsed blocks with scenario-aware boundaries.

    Works with all 3 heading detection strategies:
    - Word styles → headings already tagged
    - Content patterns → headings already tagged
    - LLM fallback → headings re-tagged by _apply_llm_headings_to_blocks

    Each heading starts a new chunk. Sub-section labels stay within their
    parent chunk. Only splits mid-section if content exceeds CHUNK_MAX_CHARS.

    New in this iteration
    ---------------------
    * Split reason is tracked per flush — heading-bounded vs. size-bounded.
      Reported via the IngestionReport so we can see whether overlap is
      doing anything useful on this corpus.
    * Overlap (CHUNK_OVERLAP_CHARS) is applied ONLY on size-based mid-
      section splits, snapped to a sentence / paragraph boundary by
      backend.ingestion.chunk_overlap. Heading-bounded chunks are
      unchanged — they already carry the heading as semantic context.
    * Tables are treated atomically: a size-split is never triggered
      while a table block is in the current group, preventing escalation
      matrices and similar from being severed mid-row.
    * Final chunk count is bounded by MAX_DOC_CHUNKS — fails loud rather
      than silently fanning out into thousands of embed/Haiku calls.
    """
    # Locate the IngestionReport created by parse_file (if any). The
    # registry indirection keeps the public signature of build_chunks
    # unchanged for callers that bypass parse_file.
    report: Optional[IngestionReport] = _REPORT_REGISTRY.pop(id(blocks), None)

    chunks: List[ParsedChunk] = []
    current_group: List[ParsedBlock] = []
    current_heading: Optional[str] = None
    current_operational_section: Optional[str] = None
    char_count = 0
    chunk_index = 0

    # Split-reason tracking. The CURRENT chunk's flush reason is
    # determined at the moment flush() is called; we use a mutable
    # one-element list so the closure can mutate without `nonlocal`.
    # "heading" means a heading boundary triggered the flush;
    # "size" means CHUNK_MAX_CHARS was exceeded.
    pending_flush_reason: List[str] = ["heading"]

    def flush():
        nonlocal current_group, char_count, chunk_index
        if not current_group:
            return

        text_parts = []
        for block in current_group:
            if block.block_type == "heading":
                continue
            text_parts.append(block.text)

        text = "\n\n".join(text_parts).strip()
        if not text:
            current_group = []
            char_count = 0
            return

        # Prepend heading to chunk text for better retrieval.
        if current_heading:
            text = f"[{current_heading}]\n\n{text}"

        # --- Overlap (size-split chunks only) ---------------------------
        # If the PREVIOUS chunk also belongs to the same section (i.e.
        # we just size-split it) AND overlap is enabled, prepend the
        # boundary-snapped tail of the previous chunk. This stops the
        # last sentence of chunk N from being orphaned when retrieval
        # picks N+1.
        reason = pending_flush_reason[0]
        added_overlap = 0
        if reason == "size" and chunks and settings.CHUNK_OVERLAP_CHARS > 0:
            prev_text = chunks[-1].text
            new_text = apply_overlap(prev_text, text)
            added_overlap = max(0, len(new_text) - len(text))
            text = new_text

        block_types = [b.block_type for b in current_group]
        chunks.append(
            ParsedChunk(
                chunk_index=chunk_index,
                text=text,
                chunk_type=choose_chunk_type(block_types, current_heading),
                section_heading=current_heading,
                operational_section=current_operational_section,
                page_number=current_group[0].page_number,
                source_order=current_group[0].source_order,
                token_estimate=estimate_tokens(text),
                metadata={
                    "block_types": block_types,
                    "page_numbers": [
                        b.page_number for b in current_group if b.page_number is not None
                    ],
                    # Provenance: how this chunk got its boundary. Helps
                    # downstream debugging ("why is this chunk so long?").
                    "split_reason": reason,
                    "overlap_chars": added_overlap,
                },
            )
        )

        # Report mutation — accumulate stats for the end-of-ingest log line.
        if report is not None:
            if reason == "heading":
                report.heading_bounded_chunks += 1
            elif reason == "size":
                report.size_split_chunks += 1
            if "table" in block_types:
                report.table_chunks += 1
            report.overlap_chars_added += added_overlap

        chunk_index += 1
        current_group = []
        char_count = 0
        # Reset for the next flush. The next caller will overwrite this
        # before they call flush() again.
        pending_flush_reason[0] = "heading"

    for block in blocks:
        # Heading blocks: start a new chunk
        if block.block_type == "heading":
            pending_flush_reason[0] = "heading"
            flush()
            current_heading = block.heading or block.text
            match = match_operational_section(current_heading or "")
            current_operational_section = match.canonical_name if match else None
            current_group.append(block)
            continue

        # Sub-heading blocks: stay within current chunk
        if block.block_type == "sub_heading":
            current_group.append(block)
            char_count += len(block.text) + 2
            continue

        # Regular content blocks
        projected = char_count + len(block.text)
        max_chunk_chars = max(settings.CHUNK_MAX_CHARS, 6000)

        # Table atomicity: do not size-split while a table block is in
        # the current group. Tables (escalation matrices, parameter
        # tables) lose all meaning when cut mid-row.
        has_pending_table = any(b.block_type == "table" for b in current_group)

        if (
            current_group
            and projected > max_chunk_chars
            and char_count >= settings.CHUNK_MIN_CHARS
            and not has_pending_table
        ):
            pending_flush_reason[0] = "size"
            flush()
            if current_heading:
                current_group.append(
                    ParsedBlock(
                        text=current_heading,
                        block_type="heading",
                        heading=current_heading,
                        page_number=block.page_number,
                        source_order=block.source_order,
                    )
                )

        current_group.append(block)
        char_count += len(block.text) + 2

    # Final flush — reason is whatever was last set. If we never tripped
    # the size threshold, this is heading-bounded by definition.
    flush()

    # --- Safety cap + report emission ----------------------------------
    if report is not None:
        report.finalize_chunk_stats([c.text for c in chunks])

    cap_check = check_chunk_count(len(chunks))
    if report is not None:
        report.limits_passed = cap_check.passed
        report.emit()
    enforce(cap_check, doc_label=(report.doc_label if report else "unknown"))

    return chunks
