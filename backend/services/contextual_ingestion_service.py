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
    result = haiku_client.invoke_json(system=CHUNK_METADATA_SYSTEM, prompt=prompt)

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

    if result is not None:
        for chunk in chunks:
            if chunk.chunk_index not in result:
                result[chunk.chunk_index] = _fallback_chunk_metadata(chunk, document_name, source_type)
        return result

    if len(chunks) > 1:
        merged: Dict[int, Dict[str, Any]] = {}
        for chunk in chunks:
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


def process_document(
    *,
    local_path: Path,
    filename: str,
    file_type: str,
    owner_id: str,
    fingerprint: str,
    exact_duplicate_lookup,
    version_candidate_lookup,
) -> Dict[str, Any]:
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

    # Build batches
    batches: List[List[ParsedChunk]] = []
    for start in range(0, len(chunks), settings.CHUNK_BATCH_SIZE):
        batches.append(chunks[start : start + settings.CHUNK_BATCH_SIZE])

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
    }