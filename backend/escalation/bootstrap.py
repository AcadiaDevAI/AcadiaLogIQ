"""Auto-load the Escalation Procedures KB from the uploads directory.

When the modal opens, ``/escalation/status`` calls
:func:`ensure_kb_loaded`. We look for the consolidated PDF in this
order:

1. ``UPLOAD_DIR`` (recursive) — covers local-storage deployments and
   the legacy ``/upload`` multipart pipeline that always writes to disk.
2. S3 (when ``STORAGE_TYPE=s3``) — covers EC2 deployments that route
   uploads through the ``/upload/presign`` browser-direct pipeline.
   We first look under the caller's tenant prefix
   (``{S3_KEY_PREFIX}/{slug}/``) and then fall back to the whole
   ``{S3_KEY_PREFIX}/`` for admin uploads.

If found, we parse + embed + persist on the spot. If not, the route
returns ``ready=false`` with ``error="No document is uploaded."`` and
the modal shows that copy verbatim — plus a diagnostic block listing
what the backend can see on disk so the operator can correct the
deployment.

The work is idempotent: once ``kb.json`` exists with the current
parser version, the bootstrap is a no-op.
"""

from __future__ import annotations

import logging
from pathlib import Path
from threading import RLock
from typing import List, Optional

from backend.config import settings

from .bedrock_client import embed_text
from .parser import parse_pdf
from .s3_lookup import find_in_s3
from .sections import KB_FILENAME
from .store import kb_status, save_kb


logger = logging.getLogger("acadia-log-iq")


_BOOTSTRAP_LOCK = RLock()


def _upload_root() -> Path:
    return Path(settings.UPLOAD_DIR)


def _name_matches_kb(name: str) -> bool:
    """True if ``name`` (a filename) looks like the consolidated KB PDF.

    Tolerant of the chat /upload pipeline's ``{job_id}_{filename}``
    prefix, and tolerant of small naming variations like spaces vs.
    underscores or a leading "the-".
    """
    target = KB_FILENAME.lower()
    candidate = name.lower()
    if candidate == target or candidate.endswith("_" + target):
        return True
    if candidate.endswith(target):
        return True
    # Normalise separators so "Escalation Procedures KB.pdf" matches.
    normalised_target = target.replace("_", " ").replace("-", " ")
    normalised_candidate = candidate.replace("_", " ").replace("-", " ")
    if normalised_candidate.endswith(normalised_target):
        return True
    return False


def _iter_pdfs(root: Path) -> List[Path]:
    """Yield every regular .pdf file beneath ``root``, recursively.

    Symlinks + permission errors are swallowed so a single unreadable
    subdir can't poison the whole scan.
    """
    out: List[Path] = []
    try:
        for entry in root.rglob("*"):
            try:
                if entry.is_file() and entry.suffix.lower() == ".pdf":
                    out.append(entry)
            except OSError:
                continue
    except OSError as exc:
        logger.warning("[escalation] scan failed for %s: %s", root, exc)
    return out


def scan_report() -> dict:
    """Snapshot of the uploads tree for the /escalation/debug endpoint."""
    root = _upload_root()
    exists = False
    try:
        exists = root.is_dir()
    except OSError:
        pass
    pdfs = _iter_pdfs(root) if exists else []
    return {
        "upload_dir": str(root),
        "upload_dir_exists": exists,
        "pdfs_seen": [
            {"path": str(p), "name": p.name, "matches_kb": _name_matches_kb(p.name)}
            for p in pdfs
        ],
    }


def find_source_pdf() -> Optional[Path]:
    """Locate the Escalation Procedures KB PDF anywhere under UPLOAD_DIR.

    The scan is recursive so the file is found regardless of how the
    upload pipeline saved it: exact name at the root, ``{job_id}_``
    prefix from the chat /upload route, or nested under an
    ``escalation/`` (or any other) sub-directory.

    Tie-break: newest-mtime wins so re-uploading the PDF automatically
    supersedes the previous copy.
    """
    root = _upload_root()
    try:
        if not root.is_dir():
            logger.warning(
                "[escalation] UPLOAD_DIR does not exist or is not a directory: %s",
                root,
            )
            return None
    except OSError as exc:
        logger.warning("[escalation] UPLOAD_DIR check failed for %s: %s", root, exc)
        return None

    all_pdfs = _iter_pdfs(root)
    logger.info(
        "[escalation] scanning %s — %d pdf(s) found",
        root,
        len(all_pdfs),
    )

    matches = [p for p in all_pdfs if _name_matches_kb(p.name)]
    if not matches:
        if all_pdfs:
            logger.info(
                "[escalation] no PDF matched '%s' under %s; saw: %s",
                KB_FILENAME, root, [p.name for p in all_pdfs[:20]],
            )
        return None

    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    logger.info("[escalation] found KB PDF at %s", matches[0])
    return matches[0]


def ensure_kb_loaded(org_id: Optional[str], user_id: Optional[str] = None) -> dict:
    """Return the escalation KB status FOR ONE ORG.

    Per-tenant isolation: the KB is populated ONLY by that org's own
    ``POST /escalation/upload`` (saved under ``escalation/{org_id}/``).
    We deliberately NO LONGER auto-bootstrap by scanning UPLOAD_DIR / S3
    for any matching PDF — that recursive scan crossed tenant boundaries
    (it would load Acadia's PDF for US Pharma). If an org hasn't uploaded
    its own KB, the modal shows the upload step.

    Returns the status dict (same shape as :func:`store.kb_status`), with
    an ``error`` field when the KB isn't ready so the route can surface a
    friendly "upload required" message.
    """
    if not org_id:
        # No active org on the request → nothing to scope to. Treat as
        # not-ready rather than leaking a shared KB.
        return {
            "ready": False,
            "filename": None,
            "sections": {},
            "updated_at": None,
            "error": "No active organization. Pick an organization first.",
        }

    status = kb_status(org_id)
    if status.get("ready"):
        return status
    return {**status, "error": "No document is uploaded."}
