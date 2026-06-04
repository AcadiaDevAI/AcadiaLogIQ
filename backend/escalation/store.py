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
_KB_PATH = _KB_DIR / "kb.json"
_LOCK = RLock()


# Bump this whenever the parser / chunking / embedding model changes
# in a way that should invalidate any kb.json saved by an older
# version. ``load_kb`` returns ``None`` when the stored ``parser_version``
# doesn't match, which triggers the bootstrap to rebuild from the PDF.
PARSER_VERSION = 2


def _ensure_dir() -> None:
    _KB_DIR.mkdir(parents=True, exist_ok=True)


def save_kb(
    *,
    filename: str,
    sections: Dict[str, Dict[str, int]],
    chunks: List[Dict],
) -> Dict:
    """Atomically replace the KB on disk.

    ``chunks`` items: ``{"section": str, "page": int, "text": str,
    "embedding": List[float]}``.
    """
    _ensure_dir()
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
        fd, tmp = tempfile.mkstemp(prefix="kb-", suffix=".json", dir=str(_KB_DIR))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            os.replace(tmp, _KB_PATH)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    return payload


def load_kb() -> Optional[Dict]:
    with _LOCK:
        if not _KB_PATH.exists():
            return None
        try:
            with _KB_PATH.open("r", encoding="utf-8") as fh:
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


def kb_status() -> Dict:
    kb = load_kb()
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


def chunks_for_section(section_id: str) -> List[Dict]:
    kb = load_kb()
    if not kb:
        return []
    return [c for c in kb.get("chunks", []) if c.get("section") == section_id]


def delete_kb() -> bool:
    """Remove the persisted KB JSON so the next /status call shows
    the upload step again. Returns True if a file was deleted, False if
    nothing was on disk."""
    with _LOCK:
        if not _KB_PATH.exists():
            return False
        try:
            os.unlink(_KB_PATH)
        except OSError as exc:
            logger.warning("[escalation] failed to delete kb.json: %s", exc)
            raise
        return True
