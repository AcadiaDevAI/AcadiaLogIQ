"""On-disk persistence for the Escalation KB.

A single JSON file under ``UPLOAD_DIR/escalation/kb.json`` holds the
filename, upload timestamp, per-section page ranges, and every chunk
with its precomputed Titan embedding. One file = one consolidated KB;
re-uploading replaces it atomically.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Dict, List, Optional

from backend.config import settings


logger = logging.getLogger("acadia-log-iq")


_KB_DIR = settings.UPLOAD_DIR / "escalation"
_LOCK = RLock()


def _safe_org_key(org_id: Optional[str]) -> str:
    """Filesystem-safe per-org folder name. org_id is our internal org
    UUID; we still sanitize defensively so a bad value can't escape the
    escalation directory. Raises if no org is provided — the escalation
    KB is strictly per-tenant and must never fall back to a shared path.
    """
    key = (org_id or "").strip()
    if not key:
        raise ValueError("escalation KB access requires an org_id")
    cleaned = "".join(ch for ch in key if ch.isalnum() or ch in "-_")
    if not cleaned:
        raise ValueError(f"invalid org_id for escalation KB: {org_id!r}")
    return cleaned


def _kb_dir(org_id: str) -> Path:
    return _KB_DIR / _safe_org_key(org_id)


def _kb_path(org_id: str) -> Path:
    return _kb_dir(org_id) / "kb.json"


# Bump this whenever the parser / chunking / embedding model changes
# in a way that should invalidate any kb.json saved by an older
# version. ``load_kb`` returns ``None`` when the stored ``parser_version``
# doesn't match, which triggers the bootstrap to rebuild from the PDF.
PARSER_VERSION = 2


def _ensure_dir(org_id: str) -> None:
    _kb_dir(org_id).mkdir(parents=True, exist_ok=True)


def save_kb(
    *,
    org_id: str,
    filename: str,
    sections: Dict[str, Dict[str, int]],
    chunks: List[Dict],
) -> Dict:
    """Atomically replace the KB on disk FOR ONE ORG.

    ``chunks`` items: ``{"section": str, "page": int, "text": str,
    "embedding": List[float]}``.
    """
    _ensure_dir(org_id)
    kb_path = _kb_path(org_id)
    payload = {
        "filename": filename,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parser_version": PARSER_VERSION,
        "sections": sections,
        "chunks": chunks,
    }
    with _LOCK:
        # Write to a sibling temp file then rename so a crash mid-write
        # leaves the previous KB intact.
        fd, tmp = tempfile.mkstemp(prefix="kb-", suffix=".json", dir=str(_kb_dir(org_id)))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            os.replace(tmp, kb_path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    return payload


def load_kb(org_id: str) -> Optional[Dict]:
    kb_path = _kb_path(org_id)
    with _LOCK:
        if not kb_path.exists():
            return None
        try:
            with kb_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            logger.warning("[escalation] failed to read kb.json: %s", exc)
            return None
        # Stale on-disk KB → treat as missing so the bootstrap rebuilds
        # from the source PDF with the current parser logic.
        if int(data.get("parser_version", 0)) < PARSER_VERSION:
            logger.info(
                "[escalation] kb.json parser_version=%s < %s; rebuilding",
                data.get("parser_version"), PARSER_VERSION,
            )
            return None
        return data


def kb_status(org_id: str) -> Dict:
    kb = load_kb(org_id)
    if not kb:
        return {"ready": False, "filename": None, "sections": {}, "updated_at": None}
    sections_summary = {
        sid: int(meta.get("chunks", 0)) for sid, meta in kb.get("sections", {}).items()
    }
    return {
        "ready": True,
        "filename": kb.get("filename"),
        "sections": sections_summary,
        "updated_at": kb.get("updated_at"),
    }


def chunks_for_section(org_id: str, section_id: str) -> List[Dict]:
    kb = load_kb(org_id)
    if not kb:
        return []
    return [c for c in kb.get("chunks", []) if c.get("section") == section_id]


def all_chunks(org_id: str) -> List[Dict]:
    """Every chunk for this org, across all sections. Used by the catch-all
    "general" query (US Pharma) which searches the whole KB rather than one
    vendor section."""
    kb = load_kb(org_id)
    if not kb:
        return []
    return list(kb.get("chunks", []))


def delete_kb(org_id: str) -> bool:
    """Remove the persisted KB JSON for ONE ORG so the next /status call
    shows the upload step again. Returns True if a file was deleted,
    False if nothing was on disk."""
    kb_path = _kb_path(org_id)
    with _LOCK:
        if not kb_path.exists():
            return False
        try:
            os.unlink(kb_path)
        except OSError as exc:
            logger.warning("[escalation] failed to delete kb.json: %s", exc)
            raise
        return True
