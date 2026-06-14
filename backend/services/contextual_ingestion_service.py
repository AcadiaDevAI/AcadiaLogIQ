from __future__ import annotations

import hashlib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.config import settings
from backend.ingestion.prompt_templates import (
    CHUNK_METADATA_SYSTEM,
    VERSION_DECISION_SYSTEM,
    build_chunk_metadata_prompt,
    build_version_decision_prompt,
)
from backend.ingestion.structured_parser import ParsedChunk, build_chunks, parse_file
from backend.metadata.structure_config import match_operational_section
from backend.services.bedrock_haiku import haiku_client
from backend.retrieval.query_expansion import extract_glossary_from_text

logger = logging.getLogger("acadia-log-iq")


def calculate_sha256_bytes(content: bytes) -> str:
    sha = hashlib.sha256()
    sha.update(content)
    return sha.hexdigest()


def normalize_filename(name: str) -> str:
    value = (name or "").strip().lower()
    value = Path(value).stem
    value = re.sub(r"\bv(?:ersion)?[\s._-]*\d+(?:\.\d+)?\b", "", value)
    value = re.sub(r"\b(final|draft|copy|rev|revision)\b", "", value)
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return re.sub(r"-{2,}", "-", value).strip("-")


def parse_version_rank(version: Optional[str]) -> float:
    if not version:
        return 0.0
    match = re.search(r"(\d+(?:\.\d+)?)", str(version))
    if not match:
        return 0.0
    try:
        return float(match.group(1))
    except Exception:
        return 0.0


def infer_document_type(filename: str, title: Optional[str]) -> str:
    text = f"{filename} {title or ''}".lower()
    if "runbook" in text:
        return "Runbook"
    if "sop" in text or "standard operating procedure" in text:
        return "SOP"
    if "kb" in text or "knowledge base" in text:
        return "KB"
    if "vendor" in text:
        return "Vendor doc"
    return "Unknown"


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _safe_str(value: Any, max_len: int = 300) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = re.sub(r"\s+", " ", text)
    return text[:max_len]


def _safe_list_of_str(values: Any, *, max_items: int, item_max_len: int = 80) -> List[str]:
    if not isinstance(values, list):
        return []
    cleaned: List[str] = []
    seen = set()
    for value in values:
        text = _safe_str(value, max_len=item_max_len)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
        if len(cleaned) >= max_items:
            break
    return cleaned


def _safe_entities(values: Any, *, max_items: int = 5) -> List[Dict[str, str]]:
    if not isinstance(values, list):
        return []
    cleaned: List[Dict[str, str]] = []
    seen = set()
    for item in values:
        if not isinstance(item, dict):
            continue
        text = _safe_str(item.get("text"), max_len=100)
        label = _safe_str(item.get("label"), max_len=40)
        if not text or not label:
            continue
        key = (text.lower(), label.upper())
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"text": text, "label": label.upper()})
        if len(cleaned) >= max_items:
            break
    return cleaned


def _safe_date(value: Any) -> Optional[str]:
    """
    Normalize a date string from Haiku into a valid ISO date (YYYY-MM-DD).
    Handles partial dates like '2024-10', '2024', or already-valid dates.
    Returns None if the value cannot be parsed.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    # Full ISO date: YYYY-MM-DD
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        try:
            datetime.strptime(text, "%Y-%m-%d")
            return text
        except ValueError:
            return None

    # Partial: YYYY-MM -> pad to first of month
    if re.fullmatch(r"\d{4}-\d{2}", text):
        try:
            datetime.strptime(text + "-01", "%Y-%m-%d")
            return text + "-01"
        except ValueError:
            return None

    # Partial: YYYY -> pad to Jan 1
    if re.fullmatch(r"\d{4}", text):
        return text + "-01-01"

    # Full ISO datetime -> extract just the date
    match = re.match(r"(\d{4}-\d{2}-\d{2})", text)
    if match:
        try:
            datetime.strptime(match.group(1), "%Y-%m-%d")
            return match.group(1)
        except ValueError:
            return None

    return None


def _fallback_chunk_metadata(
    chunk: ParsedChunk,
    document_name: str,
    source_type: str,
) -> Dict[str, Any]:
    heading = chunk.section_heading
    rule = match_operational_section(heading or "")
    return {
        "section": heading,
        "chunk_type": rule.chunk_type if rule else chunk.chunk_type,
        "document_type": infer_document_type(document_name, heading),
        "vendor": None,
        "product": None,
        "domain": None,
        "version": None,
        "date": None,
        "tags": [],
        "entities": [],
        "keywords": [],
        "summary": None,
        "purpose_description": None,
        "operational_context": rule.canonical_name if rule else chunk.operational_section,
        "title": document_name,
        "source_type": source_type,
        "document_date": None,
        "effective_date": None,
        "created_date": None,
    }


def _normalize_chunk_metadata(
    item: Dict[str, Any],
    document_meta: Dict[str, Any],
    chunk: ParsedChunk,
    document_name: str,
    source_type: str,
) -> Dict[str, Any]:
    fallback = _fallback_chunk_metadata(chunk, document_name, source_type)
    return {
        "section": _safe_str(item.get("section")) or fallback["section"],
        "chunk_type": _safe_str(item.get("chunk_type"), max_len=60) or fallback["chunk_type"],
        "document_type": _safe_str(item.get("document_type"), max_len=40)
        or _safe_str(document_meta.get("document_type"), max_len=40)
        or fallback["document_type"],
        "vendor": _safe_str(item.get("vendor"), max_len=80) or _safe_str(document_meta.get("vendor"), max_len=80),
        "product": _safe_str(item.get("product"), max_len=80) or _safe_str(document_meta.get("product"), max_len=80),
        "domain": _safe_str(item.get("domain"), max_len=80) or _safe_str(document_meta.get("domain"), max_len=80),
        "version": _safe_str(item.get("version"), max_len=40) or _safe_str(document_meta.get("version"), max_len=40),
        "date": _safe_date(item.get("date")) or _safe_date(document_meta.get("document_date")),
        "tags": _safe_list_of_str(item.get("tags"), max_items=6),
        "entities": _safe_entities(item.get("entities"), max_items=5),
        "keywords": _safe_list_of_str(item.get("keywords"), max_items=8),
        "summary": _safe_str(item.get("summary"), max_len=settings.MAX_CONTEXT_SUMMARY_CHARS),
        "purpose_description": _safe_str(item.get("purpose_description"), max_len=160),
        "operational_context": _safe_str(item.get("operational_context"), max_len=120)
        or fallback["operational_context"],
        "title": _safe_str(document_meta.get("title"), max_len=160) or document_name,
        "source_type": source_type,
        "document_date": _safe_date(document_meta.get("document_date")),
        "effective_date": _safe_date(document_meta.get("effective_date")),
        "created_date": _safe_date(document_meta.get("created_date")),
    }



def _estimate_output_tokens_per_chunk(chunk: ParsedChunk) -> int:
    """Rough per-chunk JSON output size (tokens).

    Haiku's chunk-metadata JSON scales with text length (summary,
    operational_context, entities, keywords). The envelope below was
    calibrated against observed outputs on verbose PDFs.
    """
    text_chars = len(chunk.text or "")
    # ~0.08 output tokens per input char + 180-token base envelope
    estimate = int(text_chars * 0.08) + 180
    return max(200, min(estimate, 800))


def _compute_adaptive_batch_size(chunks: List[ParsedChunk]) -> int:
    """Size Haiku batches so their aggregate output fits under the
    adaptive token budget. Keeps batches inside [min, max] clamps.
    """
    if not settings.ADAPTIVE_INGESTION_BATCHING_ENABLED or not chunks:
        return settings.CHUNK_BATCH_SIZE

    target = max(1, int(settings.ADAPTIVE_TARGET_OUTPUT_TOKENS))
    avg_per_chunk = sum(_estimate_output_tokens_per_chunk(c) for c in chunks) // max(1, len(chunks))
    avg_per_chunk = max(1, avg_per_chunk)

    raw = target // avg_per_chunk
    clamped = max(settings.ADAPTIVE_BATCH_SIZE_MIN, min(raw, settings.ADAPTIVE_BATCH_SIZE_MAX))
    return clamped


def _compute_adaptive_max_tokens(chunks: List[ParsedChunk]) -> int:
    """Size the Haiku max_tokens ceiling to fit this batch's estimated
    output plus a safety buffer, clamped to [min, max].
    """
    if not settings.ADAPTIVE_INGESTION_BATCHING_ENABLED or not chunks:
        return settings.HAIKU_MAX_TOKENS

    estimated = sum(_estimate_output_tokens_per_chunk(c) for c in chunks)
    buffered = int(estimated * float(settings.ADAPTIVE_MAX_TOKENS_BUFFER))
    clamped = max(settings.ADAPTIVE_MAX_TOKENS_MIN, min(buffered, settings.ADAPTIVE_MAX_TOKENS_MAX))
    return clamped


def _extract_chunk_metadata_once(
    *,
    document_name: str,
    source_type: str,
    chunks: List[ParsedChunk],
) -> Optional[Dict[int, Dict[str, Any]]]:
    payload = []
    for chunk in chunks:
        payload.append(
            {
                "chunk_index": chunk.chunk_index,
                "section_heading": chunk.section_heading,
                "operational_section": chunk.operational_section,
                "chunk_type_hint": chunk.chunk_type,
                "text": chunk.text[: min(settings.MAX_METADATA_INPUT_CHARS, 2200)],
            }
        )

    prompt = build_chunk_metadata_prompt(
        document_name=document_name,
        source_type=source_type,
        chunk_batch_json=_safe_json(payload),
    )
    adaptive_max_tokens = _compute_adaptive_max_tokens(chunks)
    result = haiku_client.invoke_json(
        system=CHUNK_METADATA_SYSTEM,
        prompt=prompt,
        max_tokens=adaptive_max_tokens,
    )

    if not result or "chunks" not in result or not isinstance(result.get("chunks"), list):
        return None

    indexed: Dict[int, Dict[str, Any]] = {}
    document_meta = result.get("document", {}) if isinstance(result.get("document"), dict) else {}
    chunk_lookup = {chunk.chunk_index: chunk for chunk in chunks}

    for item in result.get("chunks", []):
        if not isinstance(item, dict):
            continue
        try:
            chunk_index = int(item.get("chunk_index", -1))
        except Exception:
            continue
        chunk = chunk_lookup.get(chunk_index)
        if chunk is None:
            continue
        indexed[chunk_index] = _normalize_chunk_metadata(item, document_meta, chunk, document_name, source_type)

    if not indexed:
        return None

    return indexed


def batch_extract_chunk_metadata(
    *,
    document_name: str,
    source_type: str,
    chunks: List[ParsedChunk],
) -> Dict[int, Dict[str, Any]]:
    if not settings.ENABLE_METADATA_EXTRACTION:
        return {
            chunk.chunk_index: _fallback_chunk_metadata(chunk, document_name, source_type)
            for chunk in chunks
        }

    result = _extract_chunk_metadata_once(
        document_name=document_name,
        source_type=source_type,
        chunks=chunks,
    )

    # Self-correcting halved-batch retry: if Haiku either returned nothing
    # or filled in fewer than half the chunks (typical symptom of JSON
    # truncation), split the batch in half and retry each half before
    # falling back to per-chunk. This recovers the majority of
    # "Haiku returned invalid JSON" cases without losing throughput.
    def _coverage_ok(res: Optional[Dict[int, Dict[str, Any]]]) -> bool:
        if res is None:
            return False
        threshold = max(1, len(chunks) // 2)
        return len(res) >= threshold

    if not _coverage_ok(result) and len(chunks) > 1:
        logger.info(
            "[adaptive_batch] retry halved doc=%s chunks=%d got=%d",
            document_name,
            len(chunks),
            0 if result is None else len(result),
        )
        mid = len(chunks) // 2
        halves = [chunks[:mid], chunks[mid:]]
        merged_half: Dict[int, Dict[str, Any]] = {}
        for half in halves:
            if not half:
                continue
            half_result = _extract_chunk_metadata_once(
                document_name=document_name,
                source_type=source_type,
                chunks=half,
            )
            if half_result:
                merged_half.update(half_result)
        if merged_half:
            result = merged_half

    if result is not None and _coverage_ok(result):
        for chunk in chunks:
            if chunk.chunk_index not in result:
                result[chunk.chunk_index] = _fallback_chunk_metadata(chunk, document_name, source_type)
        return result

    if len(chunks) > 1:
        merged: Dict[int, Dict[str, Any]] = {}
        if result:
            merged.update(result)
        for chunk in chunks:
            if chunk.chunk_index in merged:
                continue
            one = _extract_chunk_metadata_once(
                document_name=document_name,
                source_type=source_type,
                chunks=[chunk],
            )
            if one and chunk.chunk_index in one:
                merged[chunk.chunk_index] = one[chunk.chunk_index]
            else:
                merged[chunk.chunk_index] = _fallback_chunk_metadata(chunk, document_name, source_type)
        return merged

    return {
        chunk.chunk_index: _fallback_chunk_metadata(chunk, document_name, source_type)
        for chunk in chunks
    }


def decide_version(
    *,
    filename: str,
    owner_id: str,
    preliminary_doc_metadata: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not settings.ENABLE_VERSION_DETECTION or not candidates:
        return {
            "decision": "new_document",
            "matched_document_id": None,
            "reason": "no candidates or version detection disabled",
            "confidence": 0.9,
            "normalized_name": normalize_filename(filename),
            "version_family_key": normalize_filename(preliminary_doc_metadata.get("title") or filename),
            "version_label": preliminary_doc_metadata.get("version"),
            "version_rank": parse_version_rank(preliminary_doc_metadata.get("version")),
            "document_date": preliminary_doc_metadata.get("document_date"),
            "effective_date": preliminary_doc_metadata.get("effective_date"),
            "created_date": preliminary_doc_metadata.get("created_date"),
        }

    incoming = {
        "filename": filename,
        "owner_id": owner_id,
        **preliminary_doc_metadata,
    }

    prompt = build_version_decision_prompt(
        incoming_json=_safe_json(incoming),
        candidates_json=_safe_json(candidates),
    )
    result = haiku_client.invoke_json(system=VERSION_DECISION_SYSTEM, prompt=prompt)

    if not result:
        return {
            "decision": "new_document",
            "matched_document_id": None,
            "reason": "fallback: no model decision",
            "confidence": 0.5,
            "normalized_name": normalize_filename(filename),
            "version_family_key": normalize_filename(preliminary_doc_metadata.get("title") or filename),
            "version_label": preliminary_doc_metadata.get("version"),
            "version_rank": parse_version_rank(preliminary_doc_metadata.get("version")),
            "document_date": preliminary_doc_metadata.get("document_date"),
            "effective_date": preliminary_doc_metadata.get("effective_date"),
            "created_date": preliminary_doc_metadata.get("created_date"),
        }

    result["normalized_name"] = result.get("normalized_name") or normalize_filename(filename)
    result["version_family_key"] = result.get("version_family_key") or normalize_filename(
        preliminary_doc_metadata.get("title") or filename
    )
    result["version_rank"] = float(result.get("version_rank") or 0.0)
    return result


# ---------------------------------------------------------------------------
# Fix 1 — Gold-ticket JSON fast-path
# ---------------------------------------------------------------------------
# Why: When the uploaded file is our structured gold-ticket JSON, every field
# we'd ask Haiku to guess (incident number, customer, priority, component,
# SLA, resolution quality) is already present in the source. Routing to a
# deterministic path skips N Haiku calls per upload and produces strictly
# correct metadata the retrieval layer (Fix 2 / Fix 6) can rely on.

def _is_gold_ticket_json(file_bytes: bytes) -> bool:
    """Return True only when `file_bytes` is our gold-ticket list format.

    Why these two markers: Metadata.Incident_Number is load-bearing for the
    ticket-ID exact-match path (Fix 2); Executive_Sharable_RCA OR ITIL_5_Why
    distinguishes gold tickets from arbitrary list-of-dict JSON we must not
    misroute. Any parse or shape mismatch → False (falls back to normal flow).
    """
    try:
        data = json.loads(file_bytes.decode("utf-8", errors="replace"))
    except Exception:
        return False
    if not isinstance(data, list) or not data:
        return False
    first = data[0]
    if not isinstance(first, dict):
        return False
    metadata = first.get("Metadata")
    if not isinstance(metadata, dict) or not metadata.get("Incident_Number"):
        return False
    return bool(first.get("Executive_Sharable_RCA")) or bool(first.get("ITIL_5_Why"))


# Sprint 2.9 — JSON structure diagnosis.
# Goal: distinguish
#   (a) "file is JSON and parses" → let downstream schemas handle it
#   (b) "file LOOKS like JSON but is malformed" → REJECT with detail
#   (c) "file is not JSON at all" → skip, proceed to PDF/DOCX/text path
# This is called BEFORE STRUCTURED_SCHEMAS iteration in process_document().

_JSON_LEADING_BYTES = ("{", "[")


def _looks_like_json(file_bytes: bytes) -> bool:
    """True if the file's first non-whitespace byte is { or [.

    Why: only signal used to decide whether JSON validation applies —
    file extension is NOT consulted, so a .txt containing JSON and a
    .json containing JSON are treated identically, and a PDF is never
    considered.
    """
    if not file_bytes:
        return False
    sample = file_bytes[:512].lstrip(b" \t\r\n\xef\xbb\xbf")
    if not sample:
        return False
    return sample[:1].decode("ascii", errors="replace") in _JSON_LEADING_BYTES


def _diagnose_json_structure(file_bytes: bytes) -> Dict[str, Any]:
    """Sprint 2.9 — structural diagnosis for JSON-looking files.

    Returns one of:
      {"kind": "not_json"}              → first byte ≠ {/[, proceed normal path
      {"kind": "valid_json", "data": parsed_obj}  → valid JSON of any shape
      {"kind": "malformed_json", "line": N, "col": M, "reason": msg}
                                        → REJECT ingestion
    """
    if not _looks_like_json(file_bytes):
        return {"kind": "not_json"}

    try:
        text = file_bytes.decode("utf-8", errors="replace")
    except Exception as exc:
        return {
            "kind": "malformed_json",
            "line": 1,
            "col": 1,
            "reason": f"Unable to decode file as UTF-8: {exc}",
        }

    try:
        data = json.loads(text)
        return {"kind": "valid_json", "data": data}
    except json.JSONDecodeError as exc:
        line = getattr(exc, "lineno", 1) or 1
        col = getattr(exc, "colno", 1) or 1
        msg = getattr(exc, "msg", "unknown parse error") or "unknown parse error"

        lower_msg = msg.lower()
        hint = ""
        if "expecting ',' delimiter" in lower_msg or "expecting value" in lower_msg:
            hint = (
                " — looks like missing ',' between objects. "
                "If this file contains multiple tickets, wrap them in "
                "a top-level array: [ {...}, {...}, ... ]"
            )
        elif "extra data" in lower_msg:
            hint = (
                " — trailing data after the first JSON value. "
                "Multiple top-level objects concatenated? Wrap them in a "
                "top-level array: [ {...}, {...}, ... ]"
            )
        elif "expecting property name" in lower_msg:
            hint = " — missing or malformed key in object near this position."
        elif "unterminated" in lower_msg:
            hint = " — unterminated string (missing closing quote)."

        return {
            "kind": "malformed_json",
            "line": int(line),
            "col": int(col),
            "reason": f"{msg}{hint}",
        }


def _is_empty(value: Any) -> bool:
    """True if the value carries no signal (None, blank string, empty dict/list)."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (dict, list, tuple, set)):
        return len(value) == 0
    return False


def _as_text(value: Any) -> str:
    """Coerce scalar/str to a trimmed string. Complex types fall back to JSON."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, default=str)


def _render_gold_ticket_body(ticket: Dict[str, Any]) -> str:
    """Flatten a ticket dict into LABELED PROSE for embedding/BM25/LLM context.

    Why this shape (not raw json.dumps per section): the LLM needs clear
    structural cues — "ROOT CAUSE:" is a much stronger signal than a nested
    JSON key buried in a 30k-char dump. Empty sections (None / "" / [] / {})
    are skipped so the chunk stays lean and the embedding isn't diluted.
    """
    incident_summary = ticket.get("Incident_Summary") if isinstance(ticket.get("Incident_Summary"), dict) else {}
    exec_rca = ticket.get("Executive_Sharable_RCA") if isinstance(ticket.get("Executive_Sharable_RCA"), dict) else {}
    itil = ticket.get("ITIL_5_Why") if isinstance(ticket.get("ITIL_5_Why"), dict) else {}
    sop = ticket.get("Operational_SOP") if isinstance(ticket.get("Operational_SOP"), dict) else {}
    qa = ticket.get("QA_Auditor_Feedback") if isinstance(ticket.get("QA_Auditor_Feedback"), dict) else {}
    forensic = ticket.get("Forensic_Performance_Audit") if isinstance(ticket.get("Forensic_Performance_Audit"), list) else []
    metadata = ticket.get("Metadata") if isinstance(ticket.get("Metadata"), dict) else {}

    blocks: List[str] = []

    def _append_block(header: str, body: str) -> None:
        body = body.strip() if body else ""
        if not body:
            return
        blocks.append(f"{header}\n{body}")

    def _append_kv(line: str) -> None:
        if line and line.strip():
            blocks.append(line.strip())

    # INCIDENT SUMMARY ← Incident_Summary.INCIDENT
    _append_block("INCIDENT SUMMARY:", _as_text(incident_summary.get("INCIDENT")))

    # ROOT CAUSE ← Executive_Sharable_RCA.Root_Cause_Technical_High_Level
    _append_block("ROOT CAUSE:", _as_text(exec_rca.get("Root_Cause_Technical_High_Level")))

    # EXECUTIVE SUMMARY ← Executive_Sharable_RCA.Executive_Summary
    _append_block("EXECUTIVE SUMMARY:", _as_text(exec_rca.get("Executive_Summary")))

    # RESOLUTION STEPS ← Executive_Sharable_RCA.Resolution_Steps (list → bullets)
    steps = exec_rca.get("Resolution_Steps")
    if isinstance(steps, list):
        bullets = [f"- {_as_text(s)}" for s in steps if not _is_empty(s)]
        if bullets:
            _append_block("RESOLUTION STEPS:", "\n".join(bullets))

    # RESOLUTION DETAIL ← Incident_Summary.RESOLUTION
    # (This is where the Ribbon/SBC/license narrative lives in INC-10000.)
    _append_block("RESOLUTION DETAIL:", _as_text(incident_summary.get("RESOLUTION")))

    # ITIL 5-WHY ROOT CAUSE ← ITIL_5_Why.Root_Cause
    _append_block("ITIL 5-WHY ROOT CAUSE:", _as_text(itil.get("Root_Cause")))

    # ITIL 5-WHY CHAIN ← paired Q1/A1..Qn/An. Skip pairs where either side is empty.
    chain_lines: List[str] = []
    for i in range(1, 10):  # accommodate future schemas beyond 5 whys
        q = itil.get(f"Q{i}")
        a = itil.get(f"A{i}")
        if _is_empty(q) and _is_empty(a):
            continue
        if not _is_empty(q):
            chain_lines.append(f"Q{i}: {_as_text(q)}")
        if not _is_empty(a):
            chain_lines.append(f"A{i}: {_as_text(a)}")
    if chain_lines:
        _append_block("ITIL 5-WHY CHAIN:", "\n".join(chain_lines))

    # TROUBLESHOOTING ← Incident_Summary.TROUBLESHOOTING
    _append_block("TROUBLESHOOTING:", _as_text(incident_summary.get("TROUBLESHOOTING")))

    # SOP DOMAIN + SOP EXECUTION STEPS
    sop_domain = _as_text(sop.get("domain"))
    if sop_domain:
        _append_kv(f"SOP DOMAIN: {sop_domain}")
    exec_steps = sop.get("execution_steps")
    if isinstance(exec_steps, list):
        bullets: List[str] = []
        for step in exec_steps:
            if isinstance(step, dict):
                action = _as_text(step.get("action"))
                if action:
                    bullets.append(f"- {action}")
            elif not _is_empty(step):
                bullets.append(f"- {_as_text(step)}")
        if bullets:
            _append_block("SOP EXECUTION STEPS:", "\n".join(bullets))

    # QA AUDITOR GAPS + rework + process improvement
    _append_block("QA AUDITOR GAPS:", _as_text(qa.get("Gaps_Identified")))
    rework = qa.get("Rework_Detected")
    if not _is_empty(rework):
        _append_kv(f"QA REWORK DETECTED: {_as_text(rework)}")
    proc_improve = qa.get("Process_Improvement_Action")
    if not _is_empty(proc_improve):
        _append_kv(f"QA PROCESS IMPROVEMENT: {_as_text(proc_improve)}")

    # FORENSIC AUDIT CRITICAL INTERVENTIONS — pull only the one load-bearing
    # field per contributor; full dump is too noisy.
    if isinstance(forensic, list):
        interventions: List[str] = []
        for item in forensic:
            if not isinstance(item, dict):
                continue
            ci = _as_text(item.get("Critical_Intervention"))
            if ci:
                interventions.append(f"- {ci}")
        if interventions:
            _append_block("FORENSIC AUDIT CRITICAL INTERVENTIONS:", "\n".join(interventions))

    # RESOLUTION GROUPS / SLA / quality score — flat key:value lines at the tail
    rgroups = metadata.get("Resolution_Groups")
    if isinstance(rgroups, list):
        cleaned = [_as_text(g) for g in rgroups if not _is_empty(g)]
        if cleaned:
            _append_kv("RESOLUTION GROUPS: " + ", ".join(cleaned))
    elif not _is_empty(rgroups):
        _append_kv(f"RESOLUTION GROUPS: {_as_text(rgroups)}")

    sla = exec_rca.get("SLA_Target_Met")
    if sla is not None:
        _append_kv(f"SLA TARGET MET: {_as_text(sla)}")
    rq_score = exec_rca.get("Resolution_Quality_Score")
    if rq_score is not None:
        _append_kv(f"RESOLUTION QUALITY SCORE: {_as_text(rq_score)}")

    # Blank line between blocks keeps the prose structurally legible for the LLM.
    return "\n\n".join(blocks).rstrip()


def _ingest_gold_ticket_json(
    *,
    file_bytes: bytes,
    filename: str,
    file_type: str,
    owner_id: str,
    fingerprint: str,
    exact_duplicate_lookup,
    version_candidate_lookup,
) -> Dict[str, Any]:
    """One-chunk-per-ticket ingestion — no Haiku metadata calls.

    Returns the same dict shape as the generic path so api.py Phase 2/3
    (embedding, BM25, insert) works unchanged.
    """
    # Duplicate check still applies — the fingerprint matches whatever was
    # uploaded before regardless of format.
    exact = None
    if settings.ENABLE_DUPLICATE_CHECK:
        exact = exact_duplicate_lookup(owner_id=owner_id, fingerprint=fingerprint)
    if exact:
        return {
            "status": "exact_duplicate",
            "document_metadata": {},
            "version_decision": {
                "decision": "exact_duplicate",
                "matched_document_id": exact["document_id"],
                "reason": "same fingerprint already exists",
                "confidence": 1.0,
                "normalized_name": exact.get("normalized_name") or normalize_filename(filename),
                "version_family_key": exact.get("version_family_key") or normalize_filename(filename),
                "version_label": exact.get("version_label"),
                "version_rank": float(exact.get("version_rank") or 0.0),
                "document_date": exact.get("document_date"),
                "effective_date": exact.get("effective_date"),
                "created_date": exact.get("created_date"),
            },
            "chunk_rows": [],
        }

    try:
        tickets = json.loads(file_bytes.decode("utf-8", errors="replace"))
    except Exception as exc:
        # Detector already parsed once, so this should not happen. Fail safe
        # by raising so the caller sees a real error rather than silent drop.
        raise RuntimeError(f"Gold-ticket JSON re-parse failed: {exc}")

    enriched_rows: List[Dict[str, Any]] = []
    latest_resolved: Optional[str] = None

    for idx, ticket in enumerate(tickets):
        if not isinstance(ticket, dict):
            continue
        metadata = ticket.get("Metadata") if isinstance(ticket.get("Metadata"), dict) else {}
        exec_rca = ticket.get("Executive_Sharable_RCA") if isinstance(ticket.get("Executive_Sharable_RCA"), dict) else {}
        qa = ticket.get("QA_Auditor_Feedback") if isinstance(ticket.get("QA_Auditor_Feedback"), dict) else {}

        incident_number = _safe_str(metadata.get("Incident_Number"), max_len=80)
        customer_name = _safe_str(metadata.get("customer_name"), max_len=160)
        priority = _safe_str(metadata.get("priority"), max_len=40)
        component_category = _safe_str(metadata.get("component_category"), max_len=80)

        # Header line — deterministic BM25 / exact-match target. Missing fields
        # render as '?' rather than being dropped so the layout stays uniform.
        header = (
            f"TICKET: {incident_number or '?'} | "
            f"CUSTOMER: {customer_name or '?'} | "
            f"PRIORITY: {priority or '?'} | "
            f"COMPONENT: {component_category or '?'}"
        )
        body = _render_gold_ticket_body(ticket)
        content = f"{header}\n\n{body}" if body else header

        resolved_date = _safe_date(metadata.get("resolved_date"))
        if resolved_date and (latest_resolved is None or resolved_date > latest_resolved):
            latest_resolved = resolved_date

        section_heading = f"Ticket {incident_number}" if incident_number else f"Ticket #{idx + 1}"

        row_metadata_json = {
            # Universal identifier fields — schema-agnostic retrieval (Goal 1/2).
            # `primary_id` is what identifier_exact_search matches against;
            # `id_type` lets the orchestrator log the schema family and future
            # callers route by type. `incident_number` is preserved as a ticket-
            # specific alias so legacy readers keep working.
            "primary_id": incident_number,
            "id_type": "ticket_number",
            # Ticket-native fields — these are what Fix 2 and Fix 6 read.
            "doc_kind": "ticket",
            "incident_number": incident_number,
            "customer_name": customer_name,
            "priority": priority,
            "component_category": component_category,
            "ticket_status": _safe_str(metadata.get("ticket_status"), max_len=40),
            "resolved_date": resolved_date,
            "resolution_groups": metadata.get("Resolution_Groups"),
            "sla_target_met": exec_rca.get("SLA_Target_Met"),
            "resolution_quality_score": exec_rca.get("Resolution_Quality_Score"),
            # FINAL_CLEANUP Bug 1 — rework filter SQL reads metadata_json->>'rework_detected'.
            # Populate from QA_Auditor_Feedback so `_build_ticket_scope_clauses` rework clause
            # actually matches rows. bool() coerces Python-native False so the JSON-encoded
            # value is 'false' (not 'null') — the SQL clause accepts true/false/yes/no/1/0.
            "rework_detected": (
                bool(qa.get("Rework_Detected", False))
                if getattr(settings, "INGEST_REWORK_METADATA_ENABLED", True)
                else None
            ),
            "llm_enrichment_status": ticket.get("llm_enrichment_status"),
            # Generic fields downstream code still reads.
            "title": filename,
            "source_type": file_type,
            "document_type": "Ticket",
            "vendor": None,
            "product": None,
            "domain": None,
            "version": None,
            "document_date": resolved_date,
            "effective_date": None,
            "created_date": None,
            "purpose_description": None,
            "operational_context": "ticket",
        }

        # Sprint 4 — Fingerprint-First Expert Copilot. Copy the rich
        # gold-ticket sections into metadata_json so the GIN indexes on
        # Fingerprints / Domain_Type / full jsonb_path_ops can match
        # them at retrieval time. These keys are passed through verbatim
        # — the Expert Copilot composer reads the raw JSON structure
        # directly, so we must not flatten or rename them.
        #
        # The ingestion is schema-tolerant: missing sections are simply
        # skipped (no KeyError, no placeholder).
        # Sprint 11 — Full-fidelity ingest. Replaces the prior cherry-pick
        # (Sprint 4 + initial Sprint 11) which retained 6 then 12 specific
        # parents. The single .update(ticket) below copies every top-level
        # source key so any current or future journey reader can walk any
        # path without an ingestion change. The slim flat fields above
        # (snake_case: incident_number, priority, customer_name, doc_kind,
        # …) coexist with the rich TitleCase keys (Metadata,
        # Executive_Sharable_RCA, Forensic_Performance_Audit, …) — zero
        # collisions, so the slim "SQL filter" view and the rich "render"
        # view live side-by-side in the same JSONB.
        row_metadata_json.update(ticket)

        # ── Old cherry-pick, retained as commented history (Sprint 4
        #    + Sprint 11 first pass). The "we forgot to add field X
        #    to the keep list" failure mode (Stage 2 Technical_Snapshot,
        #    the four parents the journey readers needed) is closed
        #    permanently by .update() above. Do not re-introduce.
        #
        # if isinstance(ticket.get("Metadata"), dict):
        #     row_metadata_json["Metadata"] = ticket["Metadata"]
        # if isinstance(ticket.get("Symptom_Solution_Mapping"), dict):
        #     row_metadata_json["Symptom_Solution_Mapping"] = ticket["Symptom_Solution_Mapping"]
        # if isinstance(ticket.get("Operational_SOP"), dict):
        #     row_metadata_json["Operational_SOP"] = ticket["Operational_SOP"]
        # if isinstance(ticket.get("Knowledge_Base"), (list, dict)):
        #     row_metadata_json["Knowledge_Base"] = ticket["Knowledge_Base"]
        # if isinstance(ticket.get("remediation_payload"), dict):
        #     row_metadata_json["remediation_payload"] = ticket["remediation_payload"]
        # if isinstance(ticket.get("Header"), str):
        #     row_metadata_json["Header"] = ticket["Header"]
        # if ticket.get("Executive_Sharable_RCA") is not None:
        #     row_metadata_json["Executive_Sharable_RCA"] = ticket["Executive_Sharable_RCA"]
        # if ticket.get("Incident_Summary") is not None:
        #     row_metadata_json["Incident_Summary"] = ticket["Incident_Summary"]
        # if ticket.get("Forensic_Performance_Audit") is not None:
        #     row_metadata_json["Forensic_Performance_Audit"] = ticket["Forensic_Performance_Audit"]
        # if ticket.get("Key_Contributors") is not None:
        #     row_metadata_json["Key_Contributors"] = ticket["Key_Contributors"]
        # if ticket.get("QA_Auditor_Feedback") is not None:
        #     row_metadata_json["QA_Auditor_Feedback"] = ticket["QA_Auditor_Feedback"]
        # if ticket.get("ITIL_5_Why") is not None:
        #     row_metadata_json["ITIL_5_Why"] = ticket["ITIL_5_Why"]

        enriched_rows.append(
            {
                "chunk_index": idx,
                "content": content,
                "contextualized_content": content,
                "summary": None,
                "section_heading": section_heading,
                "chunk_type": "ticket",
                "page_number": None,
                "token_estimate": max(1, len(content) // 4),
                "source_order": idx,
                "labels_json": {
                    "tags": [],
                    "entities": [],
                    "keywords": [],
                    "operational_context": "ticket",
                },
                "metadata_json": row_metadata_json,
            }
        )

    logger.info(
        "[ingest] structured fast-path: %d records, schema=GoldTicketSchema",
        len(enriched_rows),
    )

    # The upload pipeline sets file_type from the upload classifier (usually
    # 'kb'). For gold-ticket JSON the document IS a ticket collection, so
    # override here — this is what downstream aggregation scope filters
    # ('ticket', 'tickets', 'incident') key off of.
    if file_type != "ticket":
        logger.info(
            "[ingest] gold-ticket fast-path: document file_type set to 'ticket' (was '%s')",
            file_type,
        )

    doc_metadata = {
        "title": filename,
        "document_type": "Ticket",
        "file_type": "ticket",
        "vendor": None,
        "product": None,
        "domain": None,
        "version_label": None,
        "document_date": latest_resolved,
        "effective_date": None,
        "created_date": None,
        "section_count": len(enriched_rows),
        "chunk_count": len(enriched_rows),
        "metadata_version": "gold-ticket-v1",
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "metadata_json": {
            "title": filename,
            "document_type": "Ticket",
            "vendor": None,
            "product": None,
            "domain": None,
            "version": None,
            "document_date": latest_resolved,
            "effective_date": None,
            "created_date": None,
            "glossary": {},
            "doc_kind": "ticket",
        },
    }

    candidates = version_candidate_lookup(
        owner_id=owner_id,
        normalized_name=normalize_filename(filename),
        title=filename,
    )
    version_decision = decide_version(
        filename=filename,
        owner_id=owner_id,
        preliminary_doc_metadata={
            "title": filename,
            "version": None,
            "document_date": latest_resolved,
            "effective_date": None,
            "created_date": None,
        },
        candidates=candidates,
    )

    return {
        "status": "ready",
        "document_metadata": doc_metadata,
        "version_decision": version_decision,
        "chunk_rows": enriched_rows,
    }


# ---------------------------------------------------------------------------
# Goal 1 — Schema-agnostic structured ingestion
# ---------------------------------------------------------------------------
# A schema is any structured document shape we can deterministically unpack
# into one-chunk-per-record with a known primary_id. Today we ship two:
#   - GoldTicketSchema: the original gold-ticket JSON (INC-\d+).
#   - GenericArraySchema: list-of-dicts (or {records|items|data|tickets: [...]})
#     where records carry an ID-like field (id, uuid, key, issue_key, ...).
# Adding a new schema = adding a new class to STRUCTURED_SCHEMAS. No caller
# changes: process_document iterates the registry and falls through to the
# legacy parse + chunk + Haiku pipeline if nothing detects.

_GENERIC_ID_FIELDS = (
    "id",
    "uuid",
    "primary_id",
    "key",
    "number",
    "incident_number",
    "ticket_number",
    "ticket_id",
    "case_number",
    "case_id",
    "issue_key",
    "kb_id",
    "article_id",
    "doc_id",
    "record_id",
)

# Permissive fallback: any bare key that looks identifier-shaped. Prevents
# the candidate list from becoming a schema-catalog bottleneck.
_GENERIC_ID_FIELD_RE = re.compile(r"^[A-Za-z]*(?:id|key|number|no)$", re.IGNORECASE)


def _find_generic_id_field(record: Dict[str, Any]) -> Optional[tuple]:
    """Pick the best (field_name, value) identifier from a record.

    Returns (id_type, primary_id) or None. Priority: known candidates in
    the order listed (stable preference — id > uuid > issue_key ...) before
    falling back to the regex heuristic.
    """
    if not isinstance(record, dict):
        return None
    lowered = {str(k).lower(): (k, v) for k, v in record.items() if isinstance(k, str)}
    for candidate in _GENERIC_ID_FIELDS:
        if candidate in lowered:
            _, raw_value = lowered[candidate]
            value = _safe_str(raw_value, max_len=120)
            if value:
                return candidate, value
    # Regex fallback — first matching key wins.
    for raw_key, raw_value in record.items():
        if not isinstance(raw_key, str):
            continue
        if _GENERIC_ID_FIELD_RE.fullmatch(raw_key):
            value = _safe_str(raw_value, max_len=120)
            if value:
                return raw_key.lower(), value
    return None


def _render_generic_record_body(record: Dict[str, Any]) -> str:
    """Flatten an arbitrary record dict into labeled prose.

    Same shape as _render_gold_ticket_body: KEY: value lines + list-bullet
    and nested-dict blocks, so the LLM gets strong structural cues and BM25
    gets literal field names.
    """
    blocks: List[str] = []
    for key, value in record.items():
        if _is_empty(value):
            continue
        label = str(key).upper().replace("_", " ")
        if isinstance(value, list):
            bullets: List[str] = []
            for item in value:
                if _is_empty(item):
                    continue
                if isinstance(item, (str, int, float, bool)):
                    bullets.append(f"- {_as_text(item)}")
                elif isinstance(item, dict):
                    inner_parts = [
                        f"{ik}: {_as_text(iv)}"
                        for ik, iv in item.items()
                        if not _is_empty(iv)
                    ]
                    if inner_parts:
                        bullets.append("- " + "; ".join(inner_parts))
                else:
                    bullets.append(f"- {_as_text(item)}")
            if bullets:
                blocks.append(f"{label}:\n" + "\n".join(bullets))
        elif isinstance(value, dict):
            inner_lines = [
                f"  {ik}: {_as_text(iv)}"
                for ik, iv in value.items()
                if not _is_empty(iv)
            ]
            if inner_lines:
                blocks.append(f"{label}:\n" + "\n".join(inner_lines))
        else:
            blocks.append(f"{label}: {_as_text(value)}")
    return "\n\n".join(blocks).rstrip()


def _extract_generic_array(data: Any) -> List[Dict[str, Any]]:
    """Extract a list-of-dicts from a root array or a wrapper object with a
    known records-style key. Returns [] when no list can be found.
    """
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    if isinstance(data, dict):
        for wrapper in ("records", "items", "data", "tickets", "entries", "results"):
            val = data.get(wrapper)
            if isinstance(val, list):
                return [d for d in val if isinstance(d, dict)]
    return []


def _ingest_generic_array(
    *,
    file_bytes: bytes,
    filename: str,
    file_type: str,
    owner_id: str,
    fingerprint: str,
    exact_duplicate_lookup,
    version_candidate_lookup,
) -> Dict[str, Any]:
    """One-chunk-per-record ingestion for schema-less list-of-dicts JSON.

    Mirrors _ingest_gold_ticket_json's return shape so the upload pipeline
    (embedding → BM25 → insert) works unchanged.
    """
    exact = None
    if settings.ENABLE_DUPLICATE_CHECK:
        exact = exact_duplicate_lookup(owner_id=owner_id, fingerprint=fingerprint)
    if exact:
        return {
            "status": "exact_duplicate",
            "document_metadata": {},
            "version_decision": {
                "decision": "exact_duplicate",
                "matched_document_id": exact["document_id"],
                "reason": "same fingerprint already exists",
                "confidence": 1.0,
                "normalized_name": exact.get("normalized_name") or normalize_filename(filename),
                "version_family_key": exact.get("version_family_key") or normalize_filename(filename),
                "version_label": exact.get("version_label"),
                "version_rank": float(exact.get("version_rank") or 0.0),
                "document_date": exact.get("document_date"),
                "effective_date": exact.get("effective_date"),
                "created_date": exact.get("created_date"),
            },
            "chunk_rows": [],
        }

    try:
        data = json.loads(file_bytes.decode("utf-8", errors="replace"))
    except Exception as exc:
        raise RuntimeError(f"Generic-array JSON re-parse failed: {exc}")

    records = _extract_generic_array(data)
    enriched_rows: List[Dict[str, Any]] = []

    for idx, record in enumerate(records):
        id_pair = _find_generic_id_field(record)
        if id_pair is None:
            # Records without an identifier are unroutable for exact match.
            # Skip rather than emit a chunk with primary_id=None, which would
            # break the SQL short-circuit's assumptions downstream.
            continue
        id_type, primary_id = id_pair

        header = f"RECORD: {primary_id} | TYPE: {id_type}"
        body = _render_generic_record_body(record)
        content = f"{header}\n\n{body}" if body else header
        section_heading = f"Record {primary_id}"

        row_metadata_json = {
            # Universal identifier fields — the whole point of this schema.
            "primary_id": primary_id,
            "id_type": id_type,
            "doc_kind": "generic_record",
            # Generic descriptor fields expected by downstream code.
            "title": filename,
            "source_type": file_type,
            "document_type": "Structured Record",
            "vendor": None,
            "product": None,
            "domain": None,
            "version": None,
            "document_date": None,
            "effective_date": None,
            "created_date": None,
            "purpose_description": None,
            "operational_context": "structured_record",
        }

        enriched_rows.append(
            {
                "chunk_index": idx,
                "content": content,
                "contextualized_content": content,
                "summary": None,
                "section_heading": section_heading,
                "chunk_type": "record",
                "page_number": None,
                "token_estimate": max(1, len(content) // 4),
                "source_order": idx,
                "labels_json": {
                    "tags": [],
                    "entities": [],
                    "keywords": [],
                    "operational_context": "structured_record",
                },
                "metadata_json": row_metadata_json,
            }
        )

    logger.info(
        "[ingest] structured fast-path: %d records, schema=GenericArraySchema",
        len(enriched_rows),
    )

    doc_metadata = {
        "title": filename,
        "document_type": "Structured Records",
        "file_type": "structured_record",
        "vendor": None,
        "product": None,
        "domain": None,
        "version_label": None,
        "document_date": None,
        "effective_date": None,
        "created_date": None,
        "section_count": len(enriched_rows),
        "chunk_count": len(enriched_rows),
        "metadata_version": "generic-array-v1",
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "metadata_json": {
            "title": filename,
            "document_type": "Structured Records",
            "vendor": None,
            "product": None,
            "domain": None,
            "version": None,
            "document_date": None,
            "effective_date": None,
            "created_date": None,
            "glossary": {},
            "doc_kind": "generic_record",
        },
    }

    candidates = version_candidate_lookup(
        owner_id=owner_id,
        normalized_name=normalize_filename(filename),
        title=filename,
    )
    version_decision = decide_version(
        filename=filename,
        owner_id=owner_id,
        preliminary_doc_metadata={
            "title": filename,
            "version": None,
            "document_date": None,
            "effective_date": None,
            "created_date": None,
        },
        candidates=candidates,
    )

    return {
        "status": "ready",
        "document_metadata": doc_metadata,
        "version_decision": version_decision,
        "chunk_rows": enriched_rows,
    }


# ---------------------------------------------------------------------------
# Sprint 3-PREP-C — Contact directory (customer + vendor) ingestion.
# Routes JSON files produced by etl_excel_contacts.py (or the validator
# schema in general) to one-chunk-per-contact-record. Every chunk's
# content is a flat labeled-prose "contact card" so embedding + BM25 +
# reranker can score it as a single unit — the exact opposite of long
# narrative docs where we merge adjacent paragraphs.
# ---------------------------------------------------------------------------

_VALID_CONTACT_KINDS = ("contact_customer", "contact_vendor")


def _render_contact_body(rec: Dict[str, Any]) -> str:
    """Flatten a canonical contact record to labeled prose for embedding.

    Keys match the canonical schema in docs/CONTACT_SCHEMA.md. Missing
    fields are rendered as empty strings so the flattened card has a
    stable shape regardless of which optional blocks the ETL filled.
    """
    org = rec.get("organization") or {}
    team = rec.get("team") or {}
    person = rec.get("person") or {}
    escalation = rec.get("escalation") or {}
    lines = [
        f"Contact: {person.get('name') or ''} ({person.get('role') or ''})",
        f"Organization: {org.get('name') or ''} ({org.get('type') or ''})",
        f"Team: {team.get('name') or ''} — Escalation Level {team.get('escalation_level')}",
        f"Hours: {team.get('hours') or ''}",
        f"Phone: {person.get('phone_primary') or ''}",
        f"Email: {person.get('email') or ''}",
    ]
    if person.get("phone_secondary"):
        lines.append(f"Phone (secondary): {person['phone_secondary']}")
    if person.get("pager"):
        lines.append(f"Pager: {person['pager']}")
    triggers = escalation.get("triggers") or []
    if triggers:
        lines.append("Escalation triggers: " + ", ".join(str(t) for t in triggers))
    if escalation.get("after_hours_path"):
        lines.append(f"After-hours path: {escalation['after_hours_path']}")
    vendor_details = rec.get("vendor_details") or {}
    if vendor_details:
        if vendor_details.get("product_lines"):
            lines.append(
                "Product lines: " + ", ".join(str(p) for p in vendor_details["product_lines"])
            )
        if vendor_details.get("support_portal_url"):
            lines.append(f"Support portal: {vendor_details['support_portal_url']}")
        if vendor_details.get("tac_phone"):
            lines.append(f"TAC phone: {vendor_details['tac_phone']}")
        if vendor_details.get("sla_tier"):
            lines.append(f"SLA tier: {vendor_details['sla_tier']}")
        if vendor_details.get("account_manager"):
            lines.append(f"Account manager: {vendor_details['account_manager']}")
    if rec.get("notes"):
        lines.append(f"Notes: {rec['notes']}")
    return "\n".join(lines)


def _is_contact_json(file_bytes: bytes) -> bool:
    """Return True if the payload is a list whose first record declares
    kind ∈ {'contact_customer', 'contact_vendor'} and has a person block
    OR a contact_id. Keeps detect() strict enough that arbitrary JSON
    arrays with 'kind' fields (e.g. Kubernetes manifests) don't collide."""
    try:
        data = json.loads(file_bytes.decode("utf-8", errors="replace"))
    except Exception:
        return False
    if not isinstance(data, list) or not data:
        return False
    first = data[0]
    if not isinstance(first, dict):
        return False
    if first.get("kind") not in _VALID_CONTACT_KINDS:
        return False
    return bool(first.get("person") or first.get("contact_id"))


def _ingest_contact_array(
    *,
    file_bytes: bytes,
    filename: str,
    file_type: str,
    owner_id: str,
    fingerprint: str,
    exact_duplicate_lookup,
    version_candidate_lookup,
) -> Dict[str, Any]:
    """One-chunk-per-contact-record ingestion.

    Mirrors _ingest_generic_array's return shape. Each chunk's
    `metadata_json.primary_id` is the contact_id, so the SQL
    short-circuit can route exact-identifier queries straight to the
    chunk without going through retrieval.
    """
    exact = None
    if settings.ENABLE_DUPLICATE_CHECK:
        exact = exact_duplicate_lookup(owner_id=owner_id, fingerprint=fingerprint)
    if exact:
        return {
            "status": "exact_duplicate",
            "document_metadata": {},
            "version_decision": {
                "decision": "exact_duplicate",
                "matched_document_id": exact["document_id"],
                "reason": "same fingerprint already exists",
                "confidence": 1.0,
                "normalized_name": exact.get("normalized_name") or normalize_filename(filename),
                "version_family_key": exact.get("version_family_key") or normalize_filename(filename),
                "version_label": exact.get("version_label"),
                "version_rank": float(exact.get("version_rank") or 0.0),
                "document_date": exact.get("document_date"),
                "effective_date": exact.get("effective_date"),
                "created_date": exact.get("created_date"),
            },
            "chunk_rows": [],
        }

    try:
        data = json.loads(file_bytes.decode("utf-8", errors="replace"))
    except Exception as exc:
        raise RuntimeError(f"Contact-directory JSON re-parse failed: {exc}")

    if not isinstance(data, list):
        raise RuntimeError("Contact-directory JSON must be a top-level array")

    enriched_rows: List[Dict[str, Any]] = []
    record_kind_counts: Dict[str, int] = {}

    for idx, record in enumerate(data):
        if not isinstance(record, dict):
            continue
        record_kind = record.get("kind")
        if record_kind not in _VALID_CONTACT_KINDS:
            # Skip malformed rows rather than abort — the validator script
            # is the enforcement path; ingestion is tolerant of the
            # occasional bad record slipping through.
            continue
        primary_id = record.get("contact_id")
        if not primary_id:
            # Without a stable ID we can't answer "who do I call for
            # Acme P1" queries deterministically; skip.
            continue
        primary_id = str(primary_id).strip()

        header = f"CONTACT: {primary_id} | KIND: {record_kind}"
        body = _render_contact_body(record)
        content = f"{header}\n\n{body}"
        person = record.get("person") or {}
        org = record.get("organization") or {}
        section_heading = f"{org.get('name') or ''}: {person.get('name') or primary_id}".strip(": ")

        record_kind_counts[record_kind] = record_kind_counts.get(record_kind, 0) + 1

        row_metadata_json = {
            "primary_id": primary_id,
            "id_type": "contact_id",
            "doc_kind": record_kind,
            "title": filename,
            "source_type": file_type,
            "document_type": "Contact Directory",
            "vendor": org.get("name") if record_kind == "contact_vendor" else None,
            "product": None,
            "domain": None,
            "version": None,
            "document_date": record.get("last_verified"),
            "effective_date": None,
            "created_date": None,
            "purpose_description": None,
            "operational_context": "contact_directory",
            # Contact-specific filters that future retrieval lanes
            # (Escalation mode, Vendor mode) can narrow on.
            "organization_name": org.get("name"),
            "organization_type": org.get("type"),
            "organization_segment": org.get("segment"),
            "organization_region": org.get("region"),
            "team_name": (record.get("team") or {}).get("name"),
            "escalation_level": (record.get("team") or {}).get("escalation_level"),
            "person_name": person.get("name"),
            "person_role": person.get("role"),
            "person_email": person.get("email"),
            "person_phone_primary": person.get("phone_primary"),
        }

        enriched_rows.append(
            {
                "chunk_index": idx,
                "content": content,
                "contextualized_content": content,
                "summary": None,
                "section_heading": section_heading or f"Contact {primary_id}",
                "chunk_type": "contact_record",
                "page_number": None,
                "token_estimate": max(1, len(content) // 4),
                "source_order": idx,
                "labels_json": {
                    "tags": [record_kind],
                    "entities": [
                        e for e in [org.get("name"), person.get("name")] if e
                    ],
                    "keywords": [],
                    "operational_context": "contact_directory",
                },
                "metadata_json": row_metadata_json,
            }
        )

    logger.info(
        "[ingest] structured fast-path: %d contact records, schema=ContactSchema, kinds=%s",
        len(enriched_rows), record_kind_counts,
    )

    # When the file mixes customer + vendor records, pick the majority
    # kind for the document-level doc_kind; the per-row metadata_json
    # still carries each record's true kind for retrieval filtering.
    if record_kind_counts:
        doc_level_kind = max(record_kind_counts.items(), key=lambda kv: kv[1])[0]
    else:
        doc_level_kind = "contact_customer"

    doc_metadata = {
        "title": filename,
        "document_type": "Contact Directory",
        "file_type": "contact_directory",
        "vendor": None,
        "product": None,
        "domain": None,
        "version_label": None,
        "document_date": None,
        "effective_date": None,
        "created_date": None,
        "section_count": len(enriched_rows),
        "chunk_count": len(enriched_rows),
        "metadata_version": "contact-v1",
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "metadata_json": {
            "title": filename,
            "document_type": "Contact Directory",
            "vendor": None,
            "product": None,
            "domain": None,
            "version": None,
            "document_date": None,
            "effective_date": None,
            "created_date": None,
            "glossary": {},
            "doc_kind": doc_level_kind,
            "record_kind_counts": record_kind_counts,
        },
    }

    candidates = version_candidate_lookup(
        owner_id=owner_id,
        normalized_name=normalize_filename(filename),
        title=filename,
    )
    version_decision = decide_version(
        filename=filename,
        owner_id=owner_id,
        preliminary_doc_metadata={
            "title": filename,
            "version": None,
            "document_date": None,
            "effective_date": None,
            "created_date": None,
        },
        candidates=candidates,
    )

    return {
        "status": "ready",
        "document_metadata": doc_metadata,
        "version_decision": version_decision,
        "chunk_rows": enriched_rows,
        # Let PREP-A's post-processor know the doc-level kind inferred
        # from the records themselves — this trumps the caller-supplied
        # doc_kind when a mismatch happens (e.g., operator passes
        # --doc-kind contact_customer but the file holds vendor rows).
        "doc_kind": doc_level_kind,
    }


# ---------------------------------------------------------------------------
# Schema registry — ordered; first match wins. Each class is a pure
# dispatcher: detect() decides, ingest() emits the same dict shape as the
# legacy path.
# ---------------------------------------------------------------------------

class ContactSchema:
    """Canonical customer/vendor contact-directory JSON (Sprint 3-PREP-C).

    Detect fires only when the top-level array's first record declares
    kind ∈ {contact_customer, contact_vendor}. Must be registered BEFORE
    GenericArraySchema in STRUCTURED_SCHEMAS so generic-array detection
    doesn't claim contact files first.
    """

    name = "ContactSchema"

    @staticmethod
    def detect(file_bytes: bytes) -> bool:
        return _is_contact_json(file_bytes)

    @staticmethod
    def ingest(**kwargs) -> Dict[str, Any]:
        return _ingest_contact_array(**kwargs)


class GoldTicketSchema:
    """Gold-ticket JSON: Metadata.Incident_Number + (Executive_Sharable_RCA | ITIL_5_Why)."""

    name = "GoldTicketSchema"

    @staticmethod
    def detect(file_bytes: bytes) -> bool:
        return _is_gold_ticket_json(file_bytes)

    @staticmethod
    def ingest(**kwargs) -> Dict[str, Any]:
        return _ingest_gold_ticket_json(**kwargs)


class GenericArraySchema:
    """Any list-of-dicts (or {records|items|data|...: [...]}) with ID-like fields."""

    name = "GenericArraySchema"

    @staticmethod
    def detect(file_bytes: bytes) -> bool:
        try:
            data = json.loads(file_bytes.decode("utf-8", errors="replace"))
        except Exception:
            return False
        records = _extract_generic_array(data)
        if not records:
            return False
        # Require a detectable ID on the first record and on at least 60% of
        # records overall. Prevents arbitrary JSON objects from being
        # misrouted into the structured fast-path.
        if _find_generic_id_field(records[0]) is None:
            return False
        hits = sum(1 for r in records if _find_generic_id_field(r) is not None)
        return hits / len(records) >= 0.6

    @staticmethod
    def ingest(**kwargs) -> Dict[str, Any]:
        return _ingest_generic_array(**kwargs)


STRUCTURED_SCHEMAS = [ContactSchema, GoldTicketSchema, GenericArraySchema]


def process_document(
    *,
    local_path: Path,
    filename: str,
    file_type: str,
    owner_id: str,
    fingerprint: str,
    exact_duplicate_lookup,
    version_candidate_lookup,
    doc_kind: Optional[str] = None,            # Sprint 3-PREP-A
) -> Dict[str, Any]:
    # Sprint 3-PREP-A — resolve effective doc_kind.
    # Precedence: explicit kwarg (validated) > content-detection on the
    # actual file > "kb" safe default.
    #
    # The previous fallback was a hardcoded "ticket", which caused every
    # PDF / DOCX KB upload to land as doc_kind=ticket and silently break
    # the KB-search / Discuss-with-LogIQ separation. The detector below
    # uses magic-byte sniffing + a tiny JSON parse probe to decide what
    # the file actually IS, then maps to "ticket" (JSON / CSV / TSV) or
    # "kb" (PDF / DOCX / TXT / etc.).
    _kind_candidate = (doc_kind or "").strip().lower()
    if _kind_candidate and _kind_candidate in settings.VALID_DOC_KINDS:
        resolved_kind = _kind_candidate
    else:
        from backend.ingestion.file_type_detector import detect_file_kind
        detection = detect_file_kind(local_path)
        resolved_kind = detection.doc_kind
        logger.info(
            "[doc_kind] auto-detected %s for %s (detected=%s source=%s agreement=%s)",
            resolved_kind, local_path.name,
            detection.detected, detection.source, detection.agreement,
        )
    # Structured-schema detection runs BEFORE generic parse so we never burn
    # Haiku calls re-guessing fields already present in the source JSON.
    # Every schema that fails detection falls through to the legacy pipeline.
    try:
        file_bytes = local_path.read_bytes()
    except Exception as exc:
        logger.warning("Could not read bytes for structured-schema detection: %s", exc)
        file_bytes = b""

    if file_bytes:
        # Sprint 2.9 — JSON structure validation. Runs BEFORE schema
        # detection so malformed JSON files are rejected with a specific
        # error instead of silently falling through to text chunking.
        diag = _diagnose_json_structure(file_bytes)
        if diag["kind"] == "malformed_json":
            logger.warning(
                "[json_validator] REJECT file=%r line=%d col=%d reason=%s",
                filename, diag["line"], diag["col"], diag["reason"],
            )
            return {
                "status": "rejected",
                "ingestion_status": "invalid_json",
                "error_kind": "malformed_json",
                "error_line": diag["line"],
                "error_col": diag["col"],
                "error_reason": diag["reason"],
                "filename": filename,
                "chunks_created": 0,
            }
        # kind == "valid_json" → proceed through schema loop (fast-path
        #   will re-parse and route to gold-ticket ingest).
        # kind == "not_json"   → skip validator, proceed to PDF/DOCX/text
        #   pipeline as before.

        for schema_cls in STRUCTURED_SCHEMAS:
            try:
                if not schema_cls.detect(file_bytes):
                    continue
            except Exception as exc:
                logger.warning(
                    "Schema %s.detect raised (%s) — trying next schema",
                    schema_cls.name, exc,
                )
                continue
            try:
                schema_result = schema_cls.ingest(
                    file_bytes=file_bytes,
                    filename=filename,
                    file_type=file_type,
                    owner_id=owner_id,
                    fingerprint=fingerprint,
                    exact_duplicate_lookup=exact_duplicate_lookup,
                    version_candidate_lookup=version_candidate_lookup,
                )
                # Sprint 3-PREP-A — stamp the resolved doc_kind onto the
                # result so api.py's insert_document_and_chunks call
                # writes the documents.doc_kind column correctly.
                if isinstance(schema_result, dict) and "doc_kind" not in schema_result:
                    schema_result["doc_kind"] = resolved_kind
                return schema_result
            except Exception as exc:
                logger.warning(
                    "Schema %s.ingest raised (%s) — falling through to generic pipeline",
                    schema_cls.name, exc,
                )
                break

    # CSV/TSV: use column-aware parser that emits one ParsedChunk per row.
    # Short-circuits the generic parse_file/build_chunks path (which would
    # treat the file as flat text). Gated by CSV_COLUMN_AWARE_PARSING_ENABLED
    # inside parse_csv; when disabled it falls back to a single flat-text chunk.
    ext = local_path.suffix.lower()
    if ext in (".csv", ".tsv"):
        from backend.ingestion.csv_parser import parse_csv
        org_schema_mapping: Optional[Dict[str, str]] = None
        if getattr(settings, "DYNAMIC_SCHEMA_INFERENCE_ENABLED", False):
            try:
                from backend.services.schema_inference import get_or_infer_schema_from_csv
                info = get_or_infer_schema_from_csv(
                    organization_id=owner_id,
                    local_path=local_path,
                )
                if info and info.get("schema_mapping"):
                    org_schema_mapping = info["schema_mapping"]
            except Exception as exc:  # non-fatal — fall back to heuristics
                logger.warning(
                    "[schema_inference] lookup failed for org=%s file=%s: %s",
                    owner_id, filename, exc,
                )
        chunks = parse_csv(local_path, org_schema=org_schema_mapping)
    else:
        blocks = parse_file(local_path)
        chunks = build_chunks(blocks)

    if not chunks:
        raise RuntimeError("No parsable content found in file")

    exact = None
    if settings.ENABLE_DUPLICATE_CHECK:
        exact = exact_duplicate_lookup(owner_id=owner_id, fingerprint=fingerprint)

    if exact:
        return {
            "status": "exact_duplicate",
            "document_metadata": {},
            "version_decision": {
                "decision": "exact_duplicate",
                "matched_document_id": exact["document_id"],
                "reason": "same fingerprint already exists",
                "confidence": 1.0,
                "normalized_name": exact.get("normalized_name") or normalize_filename(filename),
                "version_family_key": exact.get("version_family_key") or normalize_filename(filename),
                "version_label": exact.get("version_label"),
                "version_rank": float(exact.get("version_rank") or 0.0),
                "document_date": exact.get("document_date"),
                "effective_date": exact.get("effective_date"),
                "created_date": exact.get("created_date"),
            },
            "chunk_rows": [],
        }

    all_chunk_meta: Dict[int, Dict[str, Any]] = {}

    # Build batches — adaptive sizing when enabled, fixed stride otherwise
    batches: List[List[ParsedChunk]] = []
    if settings.ADAPTIVE_INGESTION_BATCHING_ENABLED and chunks:
        adaptive_size = _compute_adaptive_batch_size(chunks)
        avg_out = sum(_estimate_output_tokens_per_chunk(c) for c in chunks) // max(1, len(chunks))
        logger.info(
            "[adaptive_batch] doc=%s chunks=%d avg_out_tok/chunk=%d batch_size=%d (min=%d max=%d target=%d)",
            filename,
            len(chunks),
            avg_out,
            adaptive_size,
            settings.ADAPTIVE_BATCH_SIZE_MIN,
            settings.ADAPTIVE_BATCH_SIZE_MAX,
            settings.ADAPTIVE_TARGET_OUTPUT_TOKENS,
        )
        stride = adaptive_size
    else:
        stride = settings.CHUNK_BATCH_SIZE
    for start in range(0, len(chunks), stride):
        batches.append(chunks[start : start + stride])

    # Extract metadata concurrently across batches
    max_workers = min(settings.METADATA_CONCURRENCY, len(batches))
    if max_workers <= 1 or not settings.ENABLE_METADATA_EXTRACTION:
        # Sequential fallback
        for batch in batches:
            all_chunk_meta.update(
                batch_extract_chunk_metadata(
                    document_name=filename,
                    source_type=file_type,
                    chunks=batch,
                )
            )
    else:
        def _extract_batch(batch: List[ParsedChunk]) -> Dict[int, Dict[str, Any]]:
            return batch_extract_chunk_metadata(
                document_name=filename,
                source_type=file_type,
                chunks=batch,
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_extract_batch, b): b for b in batches}
            for future in as_completed(futures):
                try:
                    all_chunk_meta.update(future.result())
                except Exception as exc:
                    batch = futures[future]
                    logger.warning("Metadata extraction failed for batch starting at chunk %s: %s",
                                   batch[0].chunk_index if batch else "?", exc)
                    for chunk in batch:
                        all_chunk_meta[chunk.chunk_index] = _fallback_chunk_metadata(chunk, filename, file_type)

    doc_title = filename
    doc_type = infer_document_type(filename, None)
    vendor = None
    product = None
    domain = None
    version = None
    document_date = None
    effective_date = None
    created_date = None

    enriched_rows: List[Dict[str, Any]] = []
    for chunk in chunks:
        meta = all_chunk_meta.get(chunk.chunk_index) or _fallback_chunk_metadata(chunk, filename, file_type)

        doc_title = meta.get("title") or doc_title
        doc_type = meta.get("document_type") or doc_type
        vendor = vendor or meta.get("vendor")
        product = product or meta.get("product")
        domain = domain or meta.get("domain")
        version = version or meta.get("version")
        document_date = document_date or _safe_date(meta.get("document_date")) or _safe_date(meta.get("date"))
        effective_date = effective_date or _safe_date(meta.get("effective_date"))
        created_date = created_date or _safe_date(meta.get("created_date"))

        contextualized = chunk.text
        if settings.ENABLE_CHUNK_SUMMARY and meta.get("summary"):
            contextualized = (
                f"[summary] {meta['summary']}\n"
                f"[section] {meta.get('section') or chunk.section_heading or 'unknown'}\n"
                f"[chunk_type] {meta.get('chunk_type') or chunk.chunk_type}\n"
                f"{chunk.text}"
            )

        enriched_rows.append(
            {
                "chunk_index": chunk.chunk_index,
                "content": chunk.text,
                "contextualized_content": contextualized,
                "summary": meta.get("summary"),
                "section_heading": meta.get("section") or chunk.section_heading,
                "chunk_type": meta.get("chunk_type") or chunk.chunk_type,
                "page_number": chunk.page_number,
                "token_estimate": chunk.token_estimate,
                "source_order": chunk.source_order,
                "labels_json": {
                    "tags": meta.get("tags", []),
                    "entities": meta.get("entities", []),
                    "keywords": meta.get("keywords", []),
                    "operational_context": meta.get("operational_context"),
                },
                "metadata_json": {
                    "title": meta.get("title") or doc_title,
                    "source_type": file_type,
                    "document_type": meta.get("document_type") or doc_type,
                    "vendor": meta.get("vendor"),
                    "product": meta.get("product"),
                    "domain": meta.get("domain"),
                    "version": meta.get("version"),
                    "document_date": meta.get("document_date") or meta.get("date"),
                    "effective_date": meta.get("effective_date"),
                    "created_date": meta.get("created_date"),
                    "purpose_description": meta.get("purpose_description"),
                    "operational_context": meta.get("operational_context"),
                    "parser_metadata": chunk.metadata,
                },
            }
        )

    # ── Extract glossary/abbreviations from document content ──
    doc_glossary = {}
    full_text = ""
    try:
        full_text = "\n".join(chunk.text for chunk in chunks)
        doc_glossary = extract_glossary_from_text(full_text)
        if doc_glossary:
            logger.info(
                "Extracted %d glossary entries from %s (e.g. %s)",
                len(doc_glossary), filename, list(doc_glossary.keys())[:5],
            )
    except Exception as e:
        logger.warning("Glossary extraction during ingestion failed (non-fatal): %s", e)

    # Hotfix: learn customer-specific vocabulary (identifiers, field names,
    # enum values) from the ingested text so query_expansion preserves these
    # tokens verbatim. Fail-safe — must never block ingestion.
    if full_text:
        try:
            from backend.services.vocabulary_learner import (
                learn_from_content,
                persist as _vocab_persist,
            )
            learned = learn_from_content(full_text, filename)
            _vocab_persist(learned, filename)
        except Exception as e:
            logger.warning("[vocab_learner] ingestion hook failed (non-fatal): %s", e)

    doc_metadata = {
        "title": doc_title,
        "document_type": doc_type,
        "vendor": vendor,
        "product": product,
        "domain": domain,
        "version_label": version,
        "document_date": document_date,
        "effective_date": effective_date,
        "created_date": created_date,
        "section_count": len({row.get("section_heading") for row in enriched_rows if row.get("section_heading")}),
        "chunk_count": len(enriched_rows),
        "metadata_version": "phase2",
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "metadata_json": {
            "title": doc_title,
            "document_type": doc_type,
            "vendor": vendor,
            "product": product,
            "domain": domain,
            "version": version,
            "document_date": document_date,
            "effective_date": effective_date,
            "created_date": created_date,
            "glossary": doc_glossary,  # ← stored for persistence across restarts
        },
    }

    candidates = version_candidate_lookup(
        owner_id=owner_id,
        normalized_name=normalize_filename(filename),
        title=doc_title,
    )

    version_decision = decide_version(
        filename=filename,
        owner_id=owner_id,
        preliminary_doc_metadata={
            "title": doc_title,
            "version": version,
            "document_date": document_date,
            "effective_date": effective_date,
            "created_date": created_date,
        },
        candidates=candidates,
    )

    return {
        "status": "ready",
        "document_metadata": doc_metadata,
        "version_decision": version_decision,
        "chunk_rows": enriched_rows,
        "doc_kind": resolved_kind,   # Sprint 3-PREP-A
    }