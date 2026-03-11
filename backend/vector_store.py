"""
Phase 1 vector store for PostgreSQL + pgvector.

What this version adds:
- document_versions table support
- separate embeddings table
- separate metadata table
- active/superseded retrieval policy
- ingestion job persistence helpers
- database-backed file listing helpers
"""

import json
import logging
import math
import re
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from sqlalchemy import bindparam, text

from backend.db.connection import SessionLocal

logger = logging.getLogger("acadia-log-iq")


# ============================================================================
# BM25 KEYWORD INDEX
# ============================================================================
class BM25Index:
    """
    In-memory BM25 index rebuilt from PostgreSQL active chunks at startup.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: Dict[str, str] = {}
        self.metadata: Dict[str, Dict] = {}
        self.doc_tokens: Dict[str, List[str]] = {}
        self.inverted_index: Dict[str, set] = defaultdict(set)
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_len = 0.0
        self.total_docs = 0
        self.df: Counter = Counter()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", (text or "").lower())

    def _recalculate_stats(self):
        self.total_docs = len(self.documents)
        self.avg_doc_len = (
            sum(self.doc_lengths.values()) / self.total_docs
            if self.total_docs
            else 0.0
        )

    def clear(self):
        self.documents.clear()
        self.metadata.clear()
        self.doc_tokens.clear()
        self.inverted_index.clear()
        self.doc_lengths.clear()
        self.df.clear()
        self.avg_doc_len = 0.0
        self.total_docs = 0

    def add_documents_batch(self, ids: List[str], docs: List[str], metas: List[Dict]):
        for doc_id, doc_text, meta in zip(ids, docs, metas):
            tokens = self._tokenize(doc_text)
            if not tokens:
                continue

            if doc_id in self.documents:
                self.remove_documents_by_file_id(meta.get("file_id"))

            self.documents[doc_id] = doc_text
            self.metadata[doc_id] = meta
            self.doc_tokens[doc_id] = tokens
            self.doc_lengths[doc_id] = len(tokens)

            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.inverted_index[token].add(doc_id)
                self.df[token] += 1

        self._recalculate_stats()

    def search(
        self,
        query: str,
        n_results: int = 10,
        file_type: Optional[str] = None,
    ) -> List[Tuple[str, str, Dict, float]]:
        query_tokens = self._tokenize(query)
        if not query_tokens or not self.total_docs:
            return []

        candidate_ids = set()
        for token in query_tokens:
            candidate_ids.update(self.inverted_index.get(token, set()))

        scored = []
        for doc_id in candidate_ids:
            meta = self.metadata.get(doc_id, {})
            if file_type and meta.get("file_type") != file_type:
                continue

            tokens = self.doc_tokens.get(doc_id, [])
            if not tokens:
                continue

            tf = Counter(tokens)
            doc_len = self.doc_lengths.get(doc_id, len(tokens))
            score = 0.0

            for token in query_tokens:
                if token not in tf:
                    continue

                df = self.df.get(token, 0)
                if df == 0:
                    continue

                idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)
                freq = tf[token]
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (
                    1 - self.b + self.b * (doc_len / max(self.avg_doc_len, 1))
                )
                score += idf * (numerator / denominator)

            if score > 0:
                scored.append((doc_id, self.documents[doc_id], meta, score))

        scored.sort(key=lambda x: x[3], reverse=True)
        return scored[:n_results]

    def _remove_documents_by_predicate(
        self, predicate: Callable[[str, Dict], bool]
    ) -> int:
        to_remove = [
            doc_id for doc_id, meta in self.metadata.items()
            if predicate(doc_id, meta)
        ]
        if not to_remove:
            return 0

        for doc_id in to_remove:
            tokens = set(self.doc_tokens.get(doc_id, []))
            for token in tokens:
                if doc_id in self.inverted_index.get(token, set()):
                    self.inverted_index[token].discard(doc_id)
                    if not self.inverted_index[token]:
                        self.inverted_index.pop(token, None)
                    if self.df.get(token, 0) > 0:
                        self.df[token] -= 1
                        if self.df[token] <= 0:
                            self.df.pop(token, None)

            self.documents.pop(doc_id, None)
            self.metadata.pop(doc_id, None)
            self.doc_tokens.pop(doc_id, None)
            self.doc_lengths.pop(doc_id, None)

        self._recalculate_stats()
        return len(to_remove)

    def remove_documents_by_source(self, source_name: str) -> int:
        return self._remove_documents_by_predicate(
            lambda _doc_id, meta: meta.get("source") == source_name
        )

    def remove_documents_by_file_id(self, file_id: str) -> int:
        return self._remove_documents_by_predicate(
            lambda _doc_id, meta: meta.get("file_id") == file_id
        )

    @property
    def size(self) -> int:
        return self.total_docs


bm25_index = BM25Index()


def get_bm25_index() -> BM25Index:
    return bm25_index


# ============================================================================
# GENERIC HELPERS
# ============================================================================
def _vector_literal(values: List[float]) -> str:
    return "[" + ",".join(f"{float(v):.8f}" for v in values) + "]"


def _now():
    return datetime.now(timezone.utc)


# ============================================================================
# FILE + VERSION HELPERS
# ============================================================================
def list_active_files(owner_id: str) -> List[Dict[str, Any]]:
    """
    List only active documents for a given owner.
    """
    with SessionLocal() as db:
        rows = db.execute(
            text(
                """
                SELECT
                    d.id::text AS id,
                    d.name,
                    d.file_type,
                    d.status,
                    d.owner_id,
                    d.created_at,
                    dv.id::text AS version_id,
                    COALESCE(dv.file_size_mb, '0') AS file_size_mb,
                    ij.job_id,
                    COALESCE(ij.status, 'indexed') AS job_status
                FROM documents d
                LEFT JOIN document_versions dv ON dv.id = d.current_version_id
                LEFT JOIN LATERAL (
                    SELECT job_id, status
                    FROM ingestion_jobs
                    WHERE file_id = d.id
                    ORDER BY created_at DESC
                    LIMIT 1
                ) ij ON TRUE
                WHERE d.owner_id = :owner_id
                  AND d.status = 'active'
                ORDER BY d.updated_at DESC, d.created_at DESC
                """
            ),
            {"owner_id": owner_id},
        ).mappings().all()

    results = []
    for row in rows:
        size_value = 0.0
        try:
            size_value = float(row["file_size_mb"] or 0)
        except Exception:
            size_value = 0.0

        results.append(
            {
                "id": row["id"],
                "name": row["name"],
                "file_type": row["file_type"],
                "size_mb": round(size_value, 2),
                "status": "indexed"
                if row["job_status"] in ("done", "indexed", None)
                else row["job_status"],
                "job_id": row["job_id"],
                "uploaded_at": row["created_at"].isoformat()
                if row["created_at"]
                else None,
                "owner_id": row["owner_id"],
            }
        )
    return results


def create_ingestion_job(
    *,
    job_id: str,
    file_id: str,
    owner_id: str,
    file_name: str,
    file_type: str,
    file_hash: str,
):
    with SessionLocal() as db:
        db.execute(
            text(
                """
                INSERT INTO ingestion_jobs(
                    job_id, file_id, owner_id, file_name, file_type, file_hash, status
                )
                VALUES (
                    :job_id, :file_id, :owner_id, :file_name, :file_type, :file_hash, 'queued'
                )
                ON CONFLICT (job_id) DO NOTHING
                """
            ),
            {
                "job_id": job_id,
                "file_id": file_id,
                "owner_id": owner_id,
                "file_name": file_name,
                "file_type": file_type,
                "file_hash": file_hash,
            },
        )
        db.commit()


def update_ingestion_job(job_id: str, **fields):
    if not fields:
        return

    set_parts = []
    params = {"job_id": job_id}

    allowed = {
        "status",
        "processed_chunks",
        "total_chunks",
        "successful_chunks",
        "error",
        "completed_at",
    }

    for key, value in fields.items():
        if key not in allowed:
            continue
        set_parts.append(f"{key} = :{key}")
        params[key] = value

    if not set_parts:
        return

    with SessionLocal() as db:
        db.execute(
            text(
                f"""
                UPDATE ingestion_jobs
                SET {", ".join(set_parts)}
                WHERE job_id = :job_id
                """
            ),
            params,
        )
        db.commit()


def get_ingestion_job(job_id: str) -> Optional[Dict[str, Any]]:
    with SessionLocal() as db:
        row = db.execute(
            text(
                """
                SELECT job_id, status, processed_chunks, total_chunks, successful_chunks,
                       file_name, file_type, file_hash, error, created_at, completed_at,
                       owner_id, file_id::text AS file_id
                FROM ingestion_jobs
                WHERE job_id = :job_id
                """
            ),
            {"job_id": job_id},
        ).mappings().first()

    if not row:
        return None

    return {
        "job_id": row["job_id"],
        "status": row["status"],
        "processed_chunks": int(row["processed_chunks"] or 0),
        "total_chunks": int(row["total_chunks"] or 0),
        "successful_chunks": int(row["successful_chunks"] or 0),
        "file": row["file_name"],
        "file_type": row["file_type"],
        "file_hash": row["file_hash"],
        "error": row["error"],
        "created_at": row["created_at"],
        "completed_at": row["completed_at"],
        "owner_id": row["owner_id"],
        "file_id": row["file_id"],
    }


def _supersede_existing_active_document(owner_id: str, filename: str):
    """
    Mark any same-owner active document with the same visible name as superseded.
    """
    with SessionLocal() as db:
        rows = db.execute(
            text(
                """
                SELECT d.id::text AS document_id,
                       d.current_version_id::text AS version_id
                FROM documents d
                WHERE d.owner_id = :owner_id
                  AND d.name = :filename
                  AND d.status = 'active'
                """
            ),
            {"owner_id": owner_id, "filename": filename},
        ).mappings().all()

        if not rows:
            return

        for row in rows:
            db.execute(
                text(
                    """
                    UPDATE documents
                    SET status = 'superseded',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = :document_id
                    """
                ),
                {"document_id": row["document_id"]},
            )

            if row["version_id"]:
                db.execute(
                    text(
                        """
                        UPDATE document_versions
                        SET is_active = FALSE,
                            superseded_at = CURRENT_TIMESTAMP
                        WHERE id = :version_id
                        """
                    ),
                    {"version_id": row["version_id"]},
                )

        db.commit()


def insert_document_and_chunks(
    *,
    document_id: str,
    filename: str,
    fingerprint: str,
    chunk_rows: List[Dict[str, Any]],
    owner_id: str,
    file_type: str,
    storage_uri: Optional[str] = None,
    file_size_mb: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Create a new active document + version + metadata + chunks + embeddings.

    Important Phase 1 behavior:
    - Any existing active same-name document for this owner becomes superseded.
    - Retrieval only uses active documents/current_version chunks.
    """
    _supersede_existing_active_document(owner_id, filename)

    version_id = str(uuid.uuid4())

    with SessionLocal() as db:
        try:
            # 1) Insert document first with NULL current_version_id
            db.execute(
                text(
                    """
                    INSERT INTO documents(
                        id, owner_id, name, file_type, status,
                        current_version_id, created_at, updated_at
                    )
                    VALUES (
                        :document_id, :owner_id, :filename, :file_type, 'active',
                        NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                    )
                    """
                ),
                {
                    "document_id": document_id,
                    "owner_id": owner_id,
                    "filename": filename,
                    "file_type": file_type,
                },
            )

            # 2) Insert version row referencing the document
            db.execute(
                text(
                    """
                    INSERT INTO document_versions(
                        id, document_id, version_number, fingerprint, storage_uri,
                        file_size_mb, is_active, uploaded_at, created_at
                    )
                    VALUES (
                        :version_id, :document_id, 1, :fingerprint, :storage_uri,
                        :file_size_mb, TRUE, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                    )
                    """
                ),
                {
                    "version_id": version_id,
                    "document_id": document_id,
                    "fingerprint": fingerprint,
                    "storage_uri": storage_uri,
                    "file_size_mb": str(file_size_mb or 0),
                },
            )

            # 3) Update document to point to the current version
            db.execute(
                text(
                    """
                    UPDATE documents
                    SET current_version_id = :version_id,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = :document_id
                    """
                ),
                {
                    "document_id": document_id,
                    "version_id": version_id,
                },
            )

            # 4) Metadata
            db.execute(
                text(
                    """
                    INSERT INTO document_metadata(document_id, metadata_json, created_at, updated_at)
                    VALUES (:document_id, CAST(:metadata_json AS jsonb), CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT (document_id)
                    DO UPDATE SET metadata_json = EXCLUDED.metadata_json,
                                  updated_at = CURRENT_TIMESTAMP
                    """
                ),
                {
                    "document_id": document_id,
                    "metadata_json": json.dumps(metadata or {}),
                },
            )

            inserted = 0

            # 5) Chunks + embeddings
            for row in chunk_rows:
                db.execute(
                    text(
                        """
                        INSERT INTO chunks(
                            id, document_id, document_version_id, chunk_index, content, created_at
                        )
                        VALUES (
                            :chunk_id, :document_id, :version_id, :chunk_index, :content, CURRENT_TIMESTAMP
                        )
                        """
                    ),
                    {
                        "chunk_id": row["id"],
                        "document_id": document_id,
                        "version_id": version_id,
                        "chunk_index": row["chunk_index"],
                        "content": row["content"],
                    },
                )

                db.execute(
                    text(
                        """
                        INSERT INTO embeddings(chunk_id, embedding, created_at)
                        VALUES (
                            :chunk_id,
                            CAST(:embedding AS vector),
                            CURRENT_TIMESTAMP
                        )
                        """
                    ),
                    {
                        "chunk_id": row["id"],
                        "embedding": _vector_literal(row["embedding"]),
                    },
                )

                inserted += 1

            db.commit()
            return inserted

        except Exception:
            db.rollback()
            raise


def delete_document_and_chunks(document_id: str) -> int:
    """
    Delete one logical document and everything connected to it.
    """
    with SessionLocal() as db:
        try:
            row = db.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM chunks
                    WHERE document_id = :document_id
                    """
                ),
                {"document_id": document_id},
            ).scalar_one()

            # Break documents -> current_version reference first
            db.execute(
                text(
                    """
                    UPDATE documents
                    SET current_version_id = NULL,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = :document_id
                    """
                ),
                {"document_id": document_id},
            )

            db.execute(
                text(
                    """
                    DELETE FROM embeddings
                    WHERE chunk_id IN (
                        SELECT id FROM chunks WHERE document_id = :document_id
                    )
                    """
                ),
                {"document_id": document_id},
            )

            db.execute(
                text(
                    """
                    DELETE FROM chunks
                    WHERE document_id = :document_id
                    """
                ),
                {"document_id": document_id},
            )

            db.execute(
                text(
                    """
                    DELETE FROM document_metadata
                    WHERE document_id = :document_id
                    """
                ),
                {"document_id": document_id},
            )

            db.execute(
                text(
                    """
                    DELETE FROM ingestion_jobs
                    WHERE file_id = :document_id
                    """
                ),
                {"document_id": document_id},
            )

            db.execute(
                text(
                    """
                    DELETE FROM document_versions
                    WHERE document_id = :document_id
                    """
                ),
                {"document_id": document_id},
            )

            db.execute(
                text(
                    """
                    DELETE FROM documents
                    WHERE id = :document_id
                    """
                ),
                {"document_id": document_id},
            )

            db.commit()
            return int(row or 0)

        except Exception:
            db.rollback()
            raise


def reset_pg_data() -> int:
    """
    Full reset used by /reset.
    """
    with SessionLocal() as db:
        count = db.execute(text("SELECT COUNT(*) FROM chunks")).scalar_one()
        db.execute(text("DELETE FROM chat_messages"))
        db.execute(text("DELETE FROM chat_sessions"))
        db.execute(text("DELETE FROM embeddings"))
        db.execute(text("DELETE FROM chunks"))
        db.execute(text("DELETE FROM document_metadata"))
        db.execute(text("DELETE FROM document_versions"))
        db.execute(text("DELETE FROM documents"))
        db.execute(text("DELETE FROM ingestion_jobs"))
        db.commit()
    return int(count or 0)


def count_pg_chunks() -> int:
    with SessionLocal() as db:
        try:
            return int(db.execute(text("SELECT COUNT(*) FROM chunks")).scalar_one())
        except Exception:
            return -1


def purge_orphan_chunks_db() -> int:
    """
    Defensive cleanup: delete chunks whose parent document or version is missing.
    """
    with SessionLocal() as db:
        before = db.execute(text("SELECT COUNT(*) FROM chunks")).scalar_one()

        db.execute(
            text(
                """
                DELETE FROM chunks c
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM documents d
                    WHERE d.id = c.document_id
                )
                OR NOT EXISTS (
                    SELECT 1
                    FROM document_versions dv
                    WHERE dv.id = c.document_version_id
                )
                """
            )
        )

        after = db.execute(text("SELECT COUNT(*) FROM chunks")).scalar_one()
        db.commit()

    return int((before or 0) - (after or 0))


# ============================================================================
# BM25 REBUILD FROM ACTIVE POSTGRES CHUNKS
# ============================================================================
def rebuild_bm25_from_postgres() -> int:
    bm25_index.clear()

    with SessionLocal() as db:
        rows = db.execute(
            text(
                """
                SELECT
                    c.id,
                    c.content,
                    d.id::text AS file_id,
                    d.owner_id,
                    d.name AS source,
                    d.file_type
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                JOIN document_versions dv ON dv.id = c.document_version_id
                WHERE d.status = 'active'
                  AND dv.is_active = TRUE
                  AND d.current_version_id = dv.id
                ORDER BY c.created_at ASC, c.chunk_index ASC
                """
            )
        ).mappings().all()

    ids, docs, metas = [], [], []
    for row in rows:
        ids.append(row["id"])
        docs.append(row["content"])
        metas.append(
            {
                "file_id": row["file_id"],
                "owner_id": row["owner_id"],
                "source": row["source"],
                "file_type": row["file_type"],
            }
        )

    if ids:
        bm25_index.add_documents_batch(ids, docs, metas)

    return bm25_index.size


# ============================================================================
# PGVECTOR SEARCH
# ============================================================================
def pgvector_search(
    *,
    query_embedding: List[float],
    n_results: int = 10,
    allowed_file_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    sql = """
        SELECT
            c.id,
            c.content,
            d.id::text AS file_id,
            d.owner_id,
            d.name AS source,
            d.file_type,
            (e.embedding <=> CAST(:query_embedding AS vector)) AS distance
        FROM embeddings e
        JOIN chunks c ON c.id = e.chunk_id
        JOIN documents d ON d.id = c.document_id
        JOIN document_versions dv ON dv.id = c.document_version_id
        WHERE d.status = 'active'
          AND dv.is_active = TRUE
          AND d.current_version_id = dv.id
    """

    params: Dict[str, Any] = {
        "query_embedding": _vector_literal(query_embedding),
        "limit": n_results,
    }

    if allowed_file_ids:
        sql += " AND d.id IN :allowed_ids"
        params["allowed_ids"] = tuple(allowed_file_ids)

    sql += """
        ORDER BY e.embedding <=> CAST(:query_embedding AS vector)
        LIMIT :limit
    """

    with SessionLocal() as db:
        stmt = text(sql)
        if allowed_file_ids:
            stmt = stmt.bindparams(bindparam("allowed_ids", expanding=True))

        rows = db.execute(stmt, params).mappings().all()

    hits = []
    for row in rows:
        hits.append(
            {
                "id": row["id"],
                "text": row["content"],
                "distance": float(row["distance"]),
                "metadata": {
                    "file_id": row["file_id"],
                    "owner_id": row["owner_id"],
                    "source": row["source"],
                    "file_type": row["file_type"],
                },
            }
        )
    return hits


# ============================================================================
# CHAT SESSION HELPERS (DB-BACKED)
# ============================================================================
def save_message_to_session(
    *,
    session_id: Optional[str],
    role: str,
    content: str,
    owner_id: str,
    sources: Optional[Dict[str, Any]] = None,
) -> str:
    session_id = session_id or uuid.uuid4().hex
    now = _now()

    with SessionLocal() as db:
        existing = db.execute(
            text("SELECT id FROM chat_sessions WHERE id = :session_id"),
            {"session_id": session_id},
        ).first()

        if not existing:
            title = content[:60] + "..." if len(content) > 60 else content
            db.execute(
                text(
                    """
                    INSERT INTO chat_sessions(id, owner_id, title, created_at, updated_at)
                    VALUES (:session_id, :owner_id, :title, :now, :now)
                    """
                ),
                {
                    "session_id": session_id,
                    "owner_id": owner_id,
                    "title": title,
                    "now": now,
                },
            )

        db.execute(
            text(
                """
                INSERT INTO chat_messages(session_id, role, content, sources_json, created_at)
                VALUES (:session_id, :role, :content, CAST(:sources_json AS jsonb), :now)
                """
            ),
            {
                "session_id": session_id,
                "role": role,
                "content": content,
                "sources_json": json.dumps(sources) if sources is not None else None,
                "now": now,
            },
        )

        db.execute(
            text(
                """
                UPDATE chat_sessions
                SET updated_at = :now
                WHERE id = :session_id
                """
            ),
            {"session_id": session_id, "now": now},
        )
        db.commit()

    return session_id


def list_chat_sessions(owner_id: str) -> List[Dict[str, Any]]:
    with SessionLocal() as db:
        rows = db.execute(
            text(
                """
                SELECT
                    s.id,
                    s.title,
                    s.updated_at,
                    COUNT(m.id) AS message_count
                FROM chat_sessions s
                LEFT JOIN chat_messages m ON m.session_id = s.id
                WHERE s.owner_id = :owner_id
                GROUP BY s.id, s.title, s.updated_at
                ORDER BY s.updated_at DESC
                """
            ),
            {"owner_id": owner_id},
        ).mappings().all()

    return [
        {
            "id": row["id"],
            "title": row["title"],
            "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
            "message_count": int(row["message_count"] or 0),
        }
        for row in rows
    ]


def get_chat_session(session_id: str, owner_id: str) -> Optional[Dict[str, Any]]:
    with SessionLocal() as db:
        session = db.execute(
            text(
                """
                SELECT id, title, created_at, updated_at
                FROM chat_sessions
                WHERE id = :session_id AND owner_id = :owner_id
                """
            ),
            {"session_id": session_id, "owner_id": owner_id},
        ).mappings().first()

        if not session:
            return None

        messages = db.execute(
            text(
                """
                SELECT role, content, sources_json, feedback, created_at
                FROM chat_messages
                WHERE session_id = :session_id
                ORDER BY id ASC
                """
            ),
            {"session_id": session_id},
        ).mappings().all()

    parsed_messages = []
    for row in messages:
        parsed_messages.append(
            {
                "role": row["role"],
                "content": row["content"],
                "sources": row["sources_json"],
                "feedback": row["feedback"],
                "timestamp": row["created_at"].isoformat()
                if row["created_at"]
                else None,
            }
        )

    return {
        "id": session["id"],
        "title": session["title"],
        "messages": parsed_messages,
        "created_at": session["created_at"].isoformat()
        if session["created_at"]
        else None,
        "updated_at": session["updated_at"].isoformat()
        if session["updated_at"]
        else None,
    }


def delete_chat_session(session_id: str, owner_id: str) -> bool:
    with SessionLocal() as db:
        row = db.execute(
            text(
                """
                DELETE FROM chat_sessions
                WHERE id = :session_id AND owner_id = :owner_id
                RETURNING id
                """
            ),
            {"session_id": session_id, "owner_id": owner_id},
        ).first()
        db.commit()
    return bool(row)


def delete_all_chat_sessions(owner_id: str) -> int:
    with SessionLocal() as db:
        count = db.execute(
            text("SELECT COUNT(*) FROM chat_sessions WHERE owner_id = :owner_id"),
            {"owner_id": owner_id},
        ).scalar_one()

        db.execute(
            text("DELETE FROM chat_sessions WHERE owner_id = :owner_id"),
            {"owner_id": owner_id},
        )
        db.commit()

    return int(count or 0)


def update_message_feedback(
    session_id: str,
    owner_id: str,
    message_index: int,
    feedback: Optional[str],
) -> bool:
    """
    Update feedback on the Nth message in a session ordered by id.
    """
    with SessionLocal() as db:
        session_exists = db.execute(
            text(
                """
                SELECT id
                FROM chat_sessions
                WHERE id = :session_id AND owner_id = :owner_id
                """
            ),
            {"session_id": session_id, "owner_id": owner_id},
        ).first()

        if not session_exists:
            return False

        message_row = db.execute(
            text(
                """
                SELECT id
                FROM chat_messages
                WHERE session_id = :session_id
                ORDER BY id ASC
                OFFSET :offset LIMIT 1
                """
            ),
            {"session_id": session_id, "offset": message_index},
        ).first()

        if not message_row:
            return False

        db.execute(
            text(
                """
                UPDATE chat_messages
                SET feedback = :feedback
                WHERE id = :message_id
                """
            ),
            {
                "feedback": feedback,
                "message_id": message_row[0],
            },
        )
        db.commit()
    return True