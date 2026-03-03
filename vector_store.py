"""
Vector Store + BM25 Keyword Index for Hybrid Search.

Two retrieval methods:
  1. ChromaDB (semantic / vector similarity)
  2. BM25 (keyword / term-frequency relevance)

The /ask endpoint in api.py merges results from both.
"""

import os
import logging
import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import chromadb

from config import settings

logger = logging.getLogger("acadia-log-iq")


# ============================================================================
# CHROMA VECTOR STORE — SINGLETON CLIENT
# ============================================================================
# IMPORTANT: We keep a single PersistentClient instance so that reset can
# safely call client.delete_collection() through the SAME SQLite connection.
# Creating multiple PersistentClient instances to the same path causes
# WAL conflicts and "readonly database" errors.
# ============================================================================
_chroma_client = None


def _get_chroma_client():
    """Return the singleton Chroma PersistentClient."""
    global _chroma_client
    if _chroma_client is None:
        chroma_path = settings.CHROMA_PERSIST_DIR
        os.makedirs(chroma_path, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=chroma_path,
            settings=chromadb.Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )
    return _chroma_client


def get_collection() -> Any:
    """Return the persistent Chroma collection (creates if needed)."""
    client = _get_chroma_client()
    return client.get_or_create_collection(
        name=settings.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def reset_chroma_collection() -> int:
    """
    Safely reset ChromaDB by deleting and recreating the collection.

    This uses ChromaDB's own API (not filesystem deletion), which:
    - Goes through proper SQLite transactions
    - Doesn't touch file handles or mount points
    - Won't corrupt the WAL journal
    - Won't cause "readonly database" errors

    Returns: number of chunks that were deleted.
    """
    client = _get_chroma_client()
    deleted_count = 0

    try:
        # Get current count before deletion
        existing = client.get_or_create_collection(name=settings.COLLECTION_NAME)
        deleted_count = existing.count()
    except Exception:
        pass

    try:
        # Delete the collection (drops all vectors, metadata, documents)
        client.delete_collection(name=settings.COLLECTION_NAME)
        logger.info("reset_chroma: Deleted collection '%s' (%d chunks)",
                     settings.COLLECTION_NAME, deleted_count)
    except Exception as e:
        logger.warning("reset_chroma: delete_collection failed: %s", e)

    # Recreate empty collection
    new_coll = client.get_or_create_collection(
        name=settings.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("reset_chroma: Recreated empty collection '%s'", settings.COLLECTION_NAME)

    return deleted_count


# ============================================================================
# BM25 KEYWORD INDEX
# ============================================================================
class BM25Index:
    """
    Okapi BM25 index for keyword-based retrieval.

    Catches exact term matches (error codes, IPs, device IDs, specific log
    patterns) that semantic/vector search can miss.

    Stored in-memory. Rebuilt from ChromaDB at startup, updated live during
    indexing.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: Dict[str, str] = {}
        self.metadata: Dict[str, Dict] = {}
        self.doc_tokens: Dict[str, List[str]] = {}
        self.inverted_index: Dict[str, set] = defaultdict(set)
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length: float = 0.0
        self.total_docs: int = 0

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenizer that preserves error codes, IPs, paths, technical terms."""
        if not text:
            return []
        text = text.lower()
        return re.findall(r'[a-z0-9][a-z0-9._:/-]*[a-z0-9]|[a-z0-9]+', text)

    def add_document(self, doc_id: str, text: str, metadata: Optional[Dict] = None):
        tokens = self.tokenize(text)
        if not tokens:
            return
        self.documents[doc_id] = text
        self.metadata[doc_id] = metadata or {}
        self.doc_tokens[doc_id] = tokens
        self.doc_lengths[doc_id] = len(tokens)
        for token in set(tokens):
            self.inverted_index[token].add(doc_id)
        self.total_docs = len(self.documents)
        if self.total_docs > 0:
            self.avg_doc_length = sum(self.doc_lengths.values()) / self.total_docs

    def add_documents_batch(
        self, doc_ids: List[str], texts: List[str],
        metadatas: Optional[List[Dict]] = None,
    ):
        if metadatas is None:
            metadatas = [{}] * len(doc_ids)
        for doc_id, text, meta in zip(doc_ids, texts, metadatas):
            tokens = self.tokenize(text)
            if not tokens:
                continue
            self.documents[doc_id] = text
            self.metadata[doc_id] = meta
            self.doc_tokens[doc_id] = tokens
            self.doc_lengths[doc_id] = len(tokens)
            for token in set(tokens):
                self.inverted_index[token].add(doc_id)
        self.total_docs = len(self.documents)
        if self.total_docs > 0:
            self.avg_doc_length = sum(self.doc_lengths.values()) / self.total_docs

    def _bm25_score(self, query_tokens: List[str], doc_id: str) -> float:
        if doc_id not in self.doc_tokens:
            return 0.0
        doc_counts = Counter(self.doc_tokens[doc_id])
        doc_len = self.doc_lengths[doc_id]
        score = 0.0
        for qt in query_tokens:
            if qt not in self.inverted_index:
                continue
            df = len(self.inverted_index[qt])
            idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)
            tf = doc_counts.get(qt, 0)
            tf_norm = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_doc_length, 1))
            )
            score += idf * tf_norm
        return score

    def search(
        self, query: str, n_results: int = 10, file_type: Optional[str] = None,
    ) -> List[Tuple[str, str, Dict, float]]:
        """Returns list of (doc_id, text, metadata, score) sorted by relevance."""
        if self.total_docs == 0:
            return []
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        candidates = set()
        for qt in query_tokens:
            if qt in self.inverted_index:
                candidates.update(self.inverted_index[qt])

        if file_type:
            candidates = {
                d for d in candidates
                if self.metadata.get(d, {}).get("file_type") == file_type
            }
        if not candidates:
            return []

        scored = []
        for doc_id in candidates:
            s = self._bm25_score(query_tokens, doc_id)
            if s > 0:
                scored.append((doc_id, self.documents[doc_id], self.metadata.get(doc_id, {}), s))
        scored.sort(key=lambda x: x[3], reverse=True)
        return scored[:n_results]

    def clear(self):
        self.documents.clear()
        self.metadata.clear()
        self.doc_tokens.clear()
        self.inverted_index.clear()
        self.doc_lengths.clear()
        self.avg_doc_length = 0.0
        self.total_docs = 0

    @property
    def size(self) -> int:
        return self.total_docs


# Global instance
bm25_index = BM25Index()


def get_bm25_index() -> BM25Index:
    return bm25_index


def rebuild_bm25_from_chroma(collection) -> int:
    """Rebuild BM25 from all ChromaDB documents. Called at startup."""
    global bm25_index
    bm25_index.clear()

    try:
        count = collection.count()
        if count == 0:
            logger.info("ChromaDB empty, BM25 index empty")
            return 0

        batch_size = 500
        total = 0
        for offset in range(0, count, batch_size):
            limit = min(batch_size, count - offset)
            results = collection.get(
                limit=limit, offset=offset,
                include=["documents", "metadatas"],
            )
            ids = results.get("ids", [])
            docs = results.get("documents", [])
            metas = results.get("metadatas", [])
            if ids and docs:
                bm25_index.add_documents_batch(ids, docs, metas)
                total += len(ids)

        logger.info("BM25 rebuilt: %d docs, %d terms", bm25_index.size, len(bm25_index.inverted_index))
        return total
    except Exception as e:
        logger.exception("BM25 rebuild failed: %s", e)
        return 0