"""
Phase 3 — Hybrid Retrieval Package.
Exposes the orchestrator as the single entry point for all retrieval.
Sub-modules: query_classifier, keyword_search, reranker, fusion.
"""

from backend.retrieval.orchestrator import retrieve  # noqa: F401
