"""Dedicated Excel (.xlsx / .xls) parser — lossless tabular ingestion.

Why a separate module from csv_parser
--------------------------------------
Real-world workbooks are messier than CSVs:
  * a title/banner row above the real header,
  * the data living on a sheet that isn't the "active" one,
  * multiple sheets, each with its own table,
  * numbers / dates / booleans (not just strings),
  * ragged rows where some rows are wider than the header.

This parser is built so **no data is lost**:
  * EVERY sheet is read (openpyxl for .xlsx, xlrd for legacy .xls).
  * The header row is auto-detected (the first header-like row in the
    first few rows); rows above it (titles) are preserved as a context
    chunk.
  * Full used width is read — cells in columns past the header get a
    synthetic ``col_N`` name rather than being dropped.
  * Numbers, dates and booleans are stringified (no value lost).
  * One chunk per non-empty data row; only truly empty rows are skipped.

Output is List[ParsedChunk] — the same contract the CSV parser and the
rest of contextual_ingestion_service expect, so downstream embedding /
chunk insertion is unchanged. Works for ANY uploaded workbook — no
hardcoded schema, sheet names, or column layout.
"""
from __future__ import annotations

import datetime as _dt
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.config import settings
from backend.ingestion.structured_parser import ParsedChunk
# Reuse the CSV parser's column-mapping heuristics so Excel and CSV map
# the same canonical fields (primary_id, customer, priority, ...).
from backend.ingestion.csv_parser import _infer_column_mapping

logger = logging.getLogger("acadia-log-iq")

# How many leading rows to scan when guessing which row is the header.
_HEADER_SCAN_ROWS = 15


# ── Cell stringification (no value lost) ────────────────────────────
def _cell_to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, float):
        # Render whole floats as ints ("3" not "3.0"); keep real decimals.
        return str(int(value)) if value.is_integer() else repr(value)
    if isinstance(value, (_dt.datetime, _dt.date)):
        return value.isoformat()
    return str(value).strip()


def _looks_numeric(s: str) -> bool:
    t = s.strip().replace(",", "")
    if not t:
        return False
    try:
        float(t)
        return True
    except ValueError:
        return False


# ── Workbook readers → list[(sheet_name, raw_rows)] ─────────────────
def _read_xlsx_sheets(local_path: Path) -> List[Tuple[str, List[List[Any]]]]:
    from openpyxl import load_workbook

    wb = load_workbook(filename=str(local_path), read_only=True, data_only=True)
    try:
        sheets: List[Tuple[str, List[List[Any]]]] = []
        for ws in wb.worksheets:
            rows = [list(r) for r in ws.iter_rows(values_only=True)]
            sheets.append((ws.title, rows))
        return sheets
    finally:
        wb.close()


def _read_xls_sheets(local_path: Path) -> List[Tuple[str, List[List[Any]]]]:
    import xlrd  # legacy .xls only

    book = xlrd.open_workbook(str(local_path))
    datemode = book.datemode
    sheets: List[Tuple[str, List[List[Any]]]] = []
    for sh in book.sheets():
        rows: List[List[Any]] = []
        for ri in range(sh.nrows):
            row: List[Any] = []
            for ci in range(sh.ncols):
                cell = sh.cell(ri, ci)
                if cell.ctype == xlrd.XL_CELL_DATE:
                    try:
                        row.append(xlrd.xldate_as_datetime(cell.value, datemode))
                    except Exception:
                        row.append(cell.value)
                elif cell.ctype == xlrd.XL_CELL_BOOLEAN:
                    row.append(bool(cell.value))
                elif cell.ctype == xlrd.XL_CELL_EMPTY:
                    row.append(None)
                else:
                    row.append(cell.value)
            rows.append(row)
        sheets.append((sh.name, rows))
    return sheets


# ── Matrix normalisation + header detection ─────────────────────────
def _normalize_matrix(raw_rows: List[List[Any]]) -> Tuple[List[List[str]], int]:
    """Stringify all cells and pad to a rectangular matrix. Trailing
    columns that are empty in EVERY row (openpyxl padding) are trimmed,
    but any column holding data anywhere is preserved."""
    str_rows = [[_cell_to_str(c) for c in r] for r in raw_rows]
    if not str_rows:
        return [], 0

    width = max(len(r) for r in str_rows)
    for r in str_rows:
        if len(r) < width:
            r.extend([""] * (width - len(r)))

    # Last column index that has any non-empty value in any row.
    last_col = -1
    for r in str_rows:
        for i in range(width - 1, last_col, -1):
            if r[i].strip():
                last_col = i
                break
    width = last_col + 1
    if width <= 0:
        return [], 0
    str_rows = [r[:width] for r in str_rows]
    return str_rows, width


def _is_header_like(row: List[str]) -> bool:
    """A header row has >=2 non-empty cells, of which the majority are
    non-numeric text labels (headers are words like 'Customer', not 42)."""
    cells = [c for c in row if c.strip()]
    if len(cells) < 2:
        return False
    non_numeric = sum(1 for c in cells if not _looks_numeric(c) and len(c) <= 80)
    return non_numeric >= max(2, (len(cells) + 1) // 2)


def _detect_header(matrix: List[List[str]]) -> int:
    """Return the index of the FIRST header-like row within the first
    _HEADER_SCAN_ROWS, or -1 if none qualifies (data with no header row).

    We take the FIRST qualifying row rather than the highest-scoring one:
    the header precedes its data, and a wide all-text *data* row can
    otherwise outscore a narrower genuine header.
    """
    for i in range(min(len(matrix), _HEADER_SCAN_ROWS)):
        if _is_header_like(matrix[i]):
            return i
    return -1


def _finalize_headers(header_row: List[str], width: int) -> List[str]:
    """Produce `width` unique, non-empty column names. Empty header cells
    become col_N; duplicates get a numeric suffix."""
    out: List[str] = []
    seen: Dict[str, int] = {}
    for i in range(width):
        raw = header_row[i].strip() if i < len(header_row) else ""
        name = raw or f"col_{i + 1}"
        key = name.lower()
        if key in seen:
            seen[key] += 1
            name = f"{name}_{seen[key]}"
        else:
            seen[key] = 0
        out.append(name)
    return out


# ── Public entry point ──────────────────────────────────────────────
def parse_excel(
    local_path: Path,
    org_schema: Optional[Dict[str, str]] = None,
    max_rows: Optional[int] = None,
) -> List[ParsedChunk]:
    """Parse an .xlsx/.xls workbook into ParsedChunks — one per data row
    across ALL sheets, plus a context chunk per sheet for any pre-header
    (title) rows. Lossless: every non-empty cell ends up in some chunk.
    """
    suffix = local_path.suffix.lower()
    if suffix == ".xls":
        sheets = _read_xls_sheets(local_path)
    else:
        sheets = _read_xlsx_sheets(local_path)

    cap = max_rows if max_rows is not None else getattr(settings, "CSV_MAX_ROWS_PER_UPLOAD", 50000)

    chunks: List[ParsedChunk] = []
    source_order = 0
    rows_emitted = 0
    truncated = False

    for sheet_name, raw_rows in sheets:
        if truncated:
            break
        matrix, width = _normalize_matrix(raw_rows)
        if width == 0:
            continue

        header_idx = _detect_header(matrix)
        if header_idx >= 0:
            headers = _finalize_headers(matrix[header_idx], width)
            pre_rows = matrix[:header_idx]
            data_rows = matrix[header_idx + 1:]
        else:
            headers = _finalize_headers([], width)  # col_1..col_N
            pre_rows = []
            data_rows = matrix

        column_mapping = _infer_column_mapping(headers, org_schema)

        # Preserve any non-empty pre-header rows (titles/banners) so their
        # content isn't lost — one context chunk per sheet.
        context_lines = []
        for row in pre_rows:
            joined = " | ".join(c for c in row if c.strip())
            if joined:
                context_lines.append(joined)
        if context_lines:
            ctext = f"[Sheet: {sheet_name}]\n" + "\n".join(context_lines)
            chunks.append(ParsedChunk(
                chunk_index=len(chunks),
                text=ctext,
                chunk_type="xlsx_context",
                section_heading=f"{sheet_name} — header",
                operational_section="document",
                page_number=None,
                source_order=source_order,
                token_estimate=max(1, len(ctext) // 4),
                metadata={"sheet": sheet_name, "kind": "pre_header_context"},
            ))
            source_order += 1

        primary_col = column_mapping.get("primary_id")

        for row_no, row in enumerate(data_rows):
            if rows_emitted >= cap:
                truncated = True
                break

            row_map = {headers[i]: (row[i] if i < len(row) else "") for i in range(width)}
            text_lines = [
                f"{headers[i]}: {row[i]}"
                for i in range(width)
                if i < len(row) and row[i].strip()
            ]
            text = "\n".join(text_lines)
            if not text.strip():
                continue  # truly empty row — nothing to lose

            primary_val = (
                (row_map.get(primary_col, "").strip() if primary_col else "")
                or f"{sheet_name}_row_{row_no + 1}"
            )

            metadata: Dict[str, Any] = {
                "sheet": sheet_name,
                "raw_row": {h: row_map.get(h, "") for h in headers},
                "column_mapping": column_mapping,
            }
            for canonical, actual_col in column_mapping.items():
                val = str(row_map.get(actual_col, "") or "").strip()
                if val:
                    metadata[canonical] = val

            chunks.append(ParsedChunk(
                chunk_index=len(chunks),
                text=text,
                chunk_type="xlsx_row",
                section_heading=primary_val,
                operational_section="record",
                page_number=None,
                source_order=source_order,
                token_estimate=max(1, len(text) // 4),
                metadata=metadata,
            ))
            source_order += 1
            rows_emitted += 1

    if truncated:
        logger.warning(
            "[xlsx_parser] row cap %d reached for %s — remaining rows NOT ingested",
            cap, local_path.name,
        )
    logger.info(
        "[xlsx_parser] file=%s sheets=%d data_rows=%d chunks=%d",
        local_path.name, len(sheets), rows_emitted, len(chunks),
    )
    return chunks
