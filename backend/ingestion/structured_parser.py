from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import fitz
from docx import Document

from backend.config import settings
from backend.metadata.structure_config import match_operational_section


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


_HEADING_RE = re.compile(r"^\s*(#{1,6}\s+.+|[A-Z][A-Z0-9 /:_\-\(\)]{3,})\s*$")
_BULLET_RE = re.compile(r"^\s*([-*•]|\d+\.)\s+")
_CODE_RE = re.compile(r"^\s*(\$|>|kubectl |aws |curl |SELECT |INSERT |UPDATE |DELETE |GET |POST |apiVersion:|kind:|FROM |WHERE )")
_TABLE_HINT_RE = re.compile(r"\s{2,}|\|")


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def normalize_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def classify_line(line: str) -> str:
    text = line.strip()
    if not text:
        return "blank"
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


def extract_blocks_from_text(text: str, page_number: Optional[int] = None) -> List[ParsedBlock]:
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


def parse_pdf(path: Path) -> List[ParsedBlock]:
    blocks: List[ParsedBlock] = []
    with fitz.open(path) as doc:
        for page_idx, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            if text.strip():
                blocks.extend(extract_blocks_from_text(text, page_number=page_idx))
    return blocks


def parse_docx(path: Path) -> List[ParsedBlock]:
    doc = Document(str(path))
    blocks: List[ParsedBlock] = []
    order = 0
    current_heading = None

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        style = (para.style.name or "").lower() if para.style else ""
        block_type = "paragraph"

        if "heading" in style:
            block_type = "heading"
            current_heading = text
        elif text.startswith(("-", "*", "•")):
            block_type = "bullet"
        elif settings.ENABLE_CODE_BLOCK_DETECTION and _CODE_RE.match(text):
            block_type = "code"

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

    if settings.ENABLE_TABLE_PARSING:
        for table in doc.tables:
            rows = []
            for row in table.rows:
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
                    )
                )
                order += 1

    return blocks


def parse_plain(path: Path) -> List[ParsedBlock]:
    return extract_blocks_from_text(path.read_text(encoding="utf-8", errors="ignore"))


def parse_file(path: Path) -> List[ParsedBlock]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path)
    if suffix == ".docx":
        return parse_docx(path)
    return parse_plain(path)


def choose_chunk_type(block_types: List[str], heading: Optional[str]) -> str:
    if heading:
        matched = match_operational_section(heading)
        if matched:
            return matched.chunk_type
    if "code" in block_types:
        return "command_chunk"
    if "table" in block_types:
        return "validation_chunk"
    return "general_chunk"


def build_chunks(blocks: List[ParsedBlock]) -> List[ParsedChunk]:
    chunks: List[ParsedChunk] = []
    current_group: List[ParsedBlock] = []
    current_heading: Optional[str] = None
    current_operational_section: Optional[str] = None
    char_count = 0
    chunk_index = 0

    def flush():
        nonlocal current_group, char_count, chunk_index
        if not current_group:
            return

        text = "\n\n".join(
            block.text for block in current_group if block.block_type != "heading"
        ).strip()

        if not text:
            current_group = []
            char_count = 0
            return

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
                },
            )
        )
        chunk_index += 1
        current_group = []
        char_count = 0

    for block in blocks:
        if block.block_type == "heading":
            flush()
            current_heading = block.heading or block.text
            match = match_operational_section(current_heading or "")
            current_operational_section = match.canonical_name if match else None
            current_group.append(block)
            continue

        projected = char_count + len(block.text)
        if current_group and projected > settings.CHUNK_MAX_CHARS and char_count >= settings.CHUNK_MIN_CHARS:
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

        if block.block_type in {"code", "table"} and char_count >= settings.CHUNK_MIN_CHARS:
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

    flush()
    return chunks