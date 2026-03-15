"""
Phase 3 Retrieval Tests — validates query classification, fusion logic, and reranker.
Run with: pytest tests/test_retrieval.py -v
No database or AWS credentials needed for these unit tests.
"""

import pytest
import sys
import os

# ---------------------------------------------------------------------------
# Ensure backend is importable when running tests from project root
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ===========================================================================
# 1. Query Classifier Tests
# ===========================================================================
class TestQueryClassifier:
    """Tests for backend.retrieval.query_classifier.classify_query"""

    def _classify(self, query):
        from backend.retrieval.query_classifier import classify_query
        return classify_query(query)

    # --- Keyword-heavy queries ---
    def test_error_code_classified_as_keyword(self):
        """Error codes like ORA-00942 should trigger keyword strategy."""
        result = self._classify("What does ORA-00942 mean?")
        assert result.keyword_score > 0
        assert any("ORA" in t or "00942" in t for t in result.extracted_terms)

    def test_ip_address_classified_as_keyword(self):
        """IP addresses should trigger keyword extraction."""
        result = self._classify("Connection refused from 10.0.0.1 to 192.168.1.50")
        assert result.keyword_score > 0
        assert any("10.0.0.1" in t for t in result.extracted_terms)

    def test_cli_command_classified_as_keyword(self):
        """CLI commands like kubectl should trigger keyword strategy."""
        result = self._classify("kubectl get pods not working")
        assert result.keyword_score > 0

    def test_http_status_code_detected(self):
        """HTTP status codes like 500 or 404 should be extracted."""
        result = self._classify("Getting HTTP 500 errors on the API")
        assert len(result.extracted_terms) > 0

    def test_quoted_phrase_extracted(self):
        """Quoted phrases should be extracted as exact terms."""
        result = self._classify('Search for "connection timeout" in logs')
        assert any("connection timeout" in t for t in result.extracted_terms)

    # --- Semantic-heavy queries ---
    def test_how_question_is_semantic(self):
        """How/why questions should lean toward semantic strategy."""
        result = self._classify("How do I troubleshoot network latency issues?")
        assert result.semantic_score > 0

    def test_explain_question_is_semantic(self):
        """Explain questions should lean toward semantic strategy."""
        result = self._classify("Explain the difference between TCP and UDP")
        assert result.semantic_score > 0

    def test_best_practices_is_semantic(self):
        """Best practices queries should be semantic."""
        result = self._classify("What are best practices for database backup?")
        assert result.semantic_score > 0

    # --- Mixed queries ---
    def test_mixed_query(self):
        """Query with both error code and troubleshooting intent → mixed."""
        result = self._classify("How to fix ORA-00942 table not found error")
        # Should have signals from both channels
        assert result.keyword_score > 0
        assert result.semantic_score > 0

    # --- Metadata hint extraction ---
    def test_vendor_extraction(self):
        """Vendor names like AWS, Cisco should be extracted as metadata hints."""
        result = self._classify("Configure AWS S3 bucket policy for public access")
        assert "vendors" in result.metadata_hints
        assert "aws" in result.metadata_hints["vendors"]

    def test_multiple_vendors(self):
        """Multiple vendor names should all be extracted."""
        result = self._classify("Compare Kubernetes and Docker networking")
        vendors = result.metadata_hints.get("vendors", [])
        assert "kubernetes" in vendors or "k8s" in vendors
        assert "docker" in vendors

    # --- Edge cases ---
    def test_empty_query(self):
        """Empty query should return mixed with empty terms."""
        result = self._classify("")
        assert result.strategy == "mixed"

    def test_short_query(self):
        """Very short queries should still classify gracefully."""
        result = self._classify("DNS")
        assert result.strategy in ("keyword", "semantic", "mixed")
        assert len(result.extracted_terms) > 0


# ===========================================================================
# 2. Fusion Tests
# ===========================================================================
class TestFusion:
    """Tests for backend.retrieval.fusion.fuse_results"""

    def _make_intent(self, strategy="mixed"):
        from backend.retrieval.query_classifier import QueryIntent
        return QueryIntent(strategy=strategy)

    def _fuse(self, **kwargs):
        from backend.retrieval.fusion import fuse_results
        return fuse_results(**kwargs)

    def test_empty_inputs(self):
        """Fusion with no results from any channel returns empty."""
        result = self._fuse(
            vector_results=[],
            bm25_results=[],
            keyword_results=[],
            intent=self._make_intent(),
        )
        assert result == []

    def test_single_channel_vector(self):
        """Results from only vector channel should still produce output."""
        vector = [
            {"id": "chunk1", "text": "hello world", "metadata": {"source": "doc1"}},
            {"id": "chunk2", "text": "foo bar", "metadata": {"source": "doc1"}},
        ]
        result = self._fuse(
            vector_results=vector,
            bm25_results=[],
            keyword_results=[],
            intent=self._make_intent(),
        )
        assert len(result) == 2
        # First result should have higher score (rank 1 in vector)
        assert result[0][3] >= result[1][3]

    def test_multi_channel_boost(self):
        """Chunks found by multiple channels should score higher."""
        vector = [{"id": "shared", "text": "shared text", "metadata": {"source": "d1"}}]
        bm25 = [("shared", "shared text", {"source": "d1"}, 5.0)]
        keyword = [{"id": "shared", "text": "shared text", "metadata": {"source": "d1"}, "search_type": "fulltext", "rank": 1.0}]

        only_vector = [{"id": "solo", "text": "solo text", "metadata": {"source": "d2"}}]

        result = self._fuse(
            vector_results=vector + only_vector,
            bm25_results=bm25,
            keyword_results=keyword,
            intent=self._make_intent(),
        )
        # "shared" should rank higher than "solo" due to multi-channel boost
        ids = [r[0] for r in result]
        assert ids[0] == "shared"

    def test_keyword_strategy_boosts_keyword_channel(self):
        """Keyword strategy should give more weight to keyword channel."""
        keyword_only = [{"id": "kw1", "text": "error ORA-001", "metadata": {"source": "d1"}, "search_type": "ilike_fallback", "rank": 1.0}]
        vector_only = [{"id": "v1", "text": "some conceptual text", "metadata": {"source": "d2"}}]

        result_kw = self._fuse(
            vector_results=vector_only,
            bm25_results=[],
            keyword_results=keyword_only,
            intent=self._make_intent("keyword"),
        )

        result_sem = self._fuse(
            vector_results=vector_only,
            bm25_results=[],
            keyword_results=keyword_only,
            intent=self._make_intent("semantic"),
        )

        # In keyword strategy, kw1 should score relatively higher vs semantic strategy
        kw_scores = {r[0]: r[3] for r in result_kw}
        sem_scores = {r[0]: r[3] for r in result_sem}

        # kw1's score in keyword strategy should be >= its score in semantic strategy
        assert kw_scores.get("kw1", 0) >= sem_scores.get("kw1", 0)

    def test_deduplication(self):
        """Same chunk_id from multiple channels should appear only once."""
        vector = [{"id": "dup", "text": "text", "metadata": {}}]
        bm25 = [("dup", "text", {}, 3.0)]
        result = self._fuse(
            vector_results=vector,
            bm25_results=bm25,
            keyword_results=[],
            intent=self._make_intent(),
        )
        assert len(result) == 1


# ===========================================================================
# 3. Reranker Tests
# ===========================================================================
class TestReranker:
    """Tests for backend.retrieval.reranker module."""

    def test_noop_reranker(self):
        """NoopReranker should return candidates as-is, truncated to top_k."""
        from backend.retrieval.reranker import NoopReranker

        reranker = NoopReranker()
        candidates = [
            ("c1", "text1", {}, 0.9),
            ("c2", "text2", {}, 0.8),
            ("c3", "text3", {}, 0.7),
        ]
        result = reranker.rerank("test query", candidates, top_k=2)
        assert len(result) == 2
        assert result[0][0] == "c1"

    def test_noop_reranker_empty(self):
        """NoopReranker with empty input returns empty."""
        from backend.retrieval.reranker import NoopReranker
        assert NoopReranker().rerank("q", [], top_k=5) == []

    def test_factory_creates_noop(self):
        """Factory should return NoopReranker when backend='none'."""
        from backend.retrieval.reranker import create_reranker, NoopReranker
        import backend.config as cfg
        original = cfg.settings.RERANKER_BACKEND
        try:
            cfg.settings.RERANKER_BACKEND = "none"
            reranker = create_reranker()
            assert isinstance(reranker, NoopReranker)
        finally:
            cfg.settings.RERANKER_BACKEND = original

    def test_factory_creates_llm_with_fn(self):
        """Factory should return LLMReranker when backend='llm' and fn provided."""
        from backend.retrieval.reranker import create_reranker, LLMReranker
        import backend.config as cfg
        original = cfg.settings.RERANKER_BACKEND
        try:
            cfg.settings.RERANKER_BACKEND = "llm"
            reranker = create_reranker(generate_fn=lambda p, t: "[]")
            assert isinstance(reranker, LLMReranker)
        finally:
            cfg.settings.RERANKER_BACKEND = original

    def test_llm_reranker_fallback_on_bad_response(self):
        """LLM reranker should fall back to fusion order on bad LLM output."""
        from backend.retrieval.reranker import LLMReranker

        def bad_generate(prompt, max_tokens):
            return "this is not valid json at all"

        reranker = LLMReranker(bad_generate)
        candidates = [
            ("c1", "text1", {"source": "d1"}, 0.9),
            ("c2", "text2", {"source": "d2"}, 0.8),
        ]
        result = reranker.rerank("test", candidates, top_k=2)
        # Should still return results (fallback to fusion order)
        assert len(result) == 2


# ===========================================================================
# 4. Integration / Evaluation Examples
# ===========================================================================
class TestEvaluationExamples:
    """
    Example queries showing expected behavior for different query types.
    These document the expected classification and retrieval strategy
    but don't require a running database.
    """

    def _classify(self, query):
        from backend.retrieval.query_classifier import classify_query
        return classify_query(query)

    def test_eval_error_code_lookup(self):
        """
        Query: 'What does error ORA-00942 mean?'
        Expected: keyword strategy, ORA-00942 extracted as search term
        """
        r = self._classify("What does error ORA-00942 mean?")
        assert r.keyword_score > 0
        assert any("ORA" in t for t in r.extracted_terms)

    def test_eval_troubleshooting_guide(self):
        """
        Query: 'How to troubleshoot high CPU usage on production servers?'
        Expected: semantic strategy, conceptual retrieval
        """
        r = self._classify("How to troubleshoot high CPU usage on production servers?")
        assert r.semantic_score > 0

    def test_eval_command_lookup(self):
        """
        Query: 'Show me the kubectl command to restart a pod'
        Expected: keyword strategy with kubectl extracted
        """
        r = self._classify("Show me the kubectl command to restart a pod")
        assert r.keyword_score > 0
        assert any("kubectl" in t for t in r.extracted_terms)

    def test_eval_vendor_specific(self):
        """
        Query: 'AWS S3 bucket versioning configuration steps'
        Expected: semantic + keyword mixed, AWS extracted as vendor hint
        """
        r = self._classify("AWS S3 bucket versioning configuration steps")
        assert "vendors" in r.metadata_hints
        assert "aws" in r.metadata_hints["vendors"]

    def test_eval_config_key_lookup(self):
        """
        Query: 'What is the default value of max.poll.interval.ms in Kafka?'
        Expected: keyword strategy, dotted config key extracted
        """
        r = self._classify("What is the default value of max.poll.interval.ms in Kafka?")
        assert r.keyword_score > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
