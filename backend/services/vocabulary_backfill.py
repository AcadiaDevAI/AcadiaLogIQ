"""
Vocabulary Backfill — one-shot scan of existing chunks to populate
learned_vocabulary without re-uploading files.

Run from a Python REPL or as a startup hook:
    from backend.services.vocabulary_backfill import backfill_all
    backfill_all()

Safe to re-run; ON CONFLICT updates occurrence count.
"""
from __future__ import annotations

import logging
from typing import Iterable, Tuple

from sqlalchemy import text

from backend.config import settings
from backend.db.connection import engine
from backend.services.vocabulary_learner import (
    learn_from_content,
    persist,
    reload_cache,
)

logger = logging.getLogger("acadia-log-iq")


def _iter_chunk_groups(batch_size: int = 50) -> Iterable[Tuple[str, str]]:
    """Yield (document_id, joined_content) per document so vocab learning
    runs on the full document context, not chunk-by-chunk noise."""
    with engine.connect() as conn:
        doc_rows = conn.execute(
            text("SELECT DISTINCT document_id FROM chunks ORDER BY document_id")
        ).fetchall()
    doc_ids = [r[0] for r in doc_rows]
    for i in range(0, len(doc_ids), batch_size):
        batch = doc_ids[i:i + batch_size]
        with engine.connect() as conn:
            for did in batch:
                rows = conn.execute(
                    text(
                        "SELECT content FROM chunks "
                        "WHERE document_id = :did ORDER BY id"
                    ),
                    {"did": did},
                ).fetchall()
                yield did, " ".join(r[0] or "" for r in rows)


def backfill_all() -> int:
    """Walk every document in the chunks table, learn vocabulary,
    refresh the cache. Returns number of documents processed."""
    count = 0
    for doc_id, content in _iter_chunk_groups():
        try:
            learned = learn_from_content(content, str(doc_id))
            persist(learned, str(doc_id))
            count += 1
        except Exception as exc:
            logger.warning(
                "[vocab_backfill] doc=%s failed: %s - continuing", doc_id, exc,
            )
    reload_cache()
    logger.info("[vocab_backfill] complete - processed %d documents", count)
    return count
