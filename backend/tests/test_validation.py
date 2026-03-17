"""
Phase 6 Validation Tests — confidence scoring, grounding checks,
validation pipeline, and eval harness. No database or AWS needed.
Run with: pytest tests/test_validation.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ===========================================================================
# 1. Confidence Scorer Tests
# ===========================================================================
class TestConfidenceScorer:
    """Tests for backend.validation.confidence_scorer.score_confidence"""

    def _score(self, query="test", answer="test answer", doc_context="test context", **kwargs):
        from backend.validation.confidence_scorer import score_confidence
        defaults = {"ranked_chunks": [], "source_names": ["doc.pdf"]}
        defaults.update(kwargs)
        return score_confidence(query=query, answer=answer, doc_context=doc_context, **defaults)

    def test_well_grounded_answer_high_confidence(self):
        """Answer with good term overlap should score well."""
        r = self._score(
            query="What is the maximum file size?",
            answer="The maximum file size is 100MB as configured in the system settings.",
            doc_context="Configuration: MAX_FILE_SIZE_MB = 100. System settings allow up to 100MB uploads.",
        )
        assert r.grounding_score > 0.3
        assert r.coverage_score > 0.3

    def test_empty_answer_low_consistency(self):
        """Empty answer should have zero consistency."""
        r = self._score(answer="")
        assert r.consistency_score == 0.0

    def test_short_answer_penalized(self):
        """Very short answer should reduce consistency."""
        r = self._score(answer="Yes.")
        assert r.consistency_score < 1.0

    def test_hallucination_phrase_penalized(self):
        """Answer with hallucination phrases should be penalized."""
        r = self._score(
            answer="Based on my training data, the system uses PostgreSQL.",
            doc_context="Database: PostgreSQL 15 with pgvector extension.",
        )
        assert "hallucination_phrases" in r.penalties

    def test_refusal_gets_moderate_consistency(self):
        """Honest refusal should get moderate (not zero) consistency."""
        r = self._score(
            answer="I could not find supporting information for that question in the currently uploaded files.",
        )
        assert 0.3 <= r.consistency_score <= 0.7

    def test_high_retrieval_score(self):
        """Chunks with high fusion scores should boost retrieval signal."""
        chunks = [("id1", "text", {"source": "doc"}, 0.08)]  # 0.08 is high for RRF
        r = self._score(ranked_chunks=chunks)
        assert r.retrieval_score > 0.5

    def test_zero_retrieval_score(self):
        """No chunks should give zero retrieval score."""
        r = self._score(ranked_chunks=[])
        assert r.retrieval_score == 0.0


# ===========================================================================
# 2. Grounding Checker Tests
# ===========================================================================
class TestGroundingChecker:
    """Tests for backend.validation.grounding_checker.check_grounding"""

    def _check(self, **kwargs):
        from backend.validation.grounding_checker import check_grounding
        defaults = {
            "query": "test",
            "answer": "test answer",
            "doc_context": "test context document",
            "ranked_chunks": [],
            "source_names": ["doc.pdf"],
        }
        defaults.update(kwargs)
        return check_grounding(**defaults)

    def test_fabricated_url_detected(self):
        """URL in answer but not in context should be flagged."""
        r = self._check(
            answer="Visit https://fake-site.com/docs for more info.",
            doc_context="The documentation is available internally.",
        )
        assert len(r.fabrications) > 0
        assert not r.passed

    def test_legitimate_url_not_flagged(self):
        """URL present in both answer and context should be fine."""
        r = self._check(
            answer="See https://real-site.com/guide for details.",
            doc_context="Reference: https://real-site.com/guide has the full procedure.",
        )
        url_fabs = [f for f in r.fabrications if "URL" in f]
        assert len(url_fabs) == 0

    def test_fabricated_email_detected(self):
        """Email in answer but not in context should be flagged."""
        r = self._check(
            answer="Contact admin@fake-company.com for help.",
            doc_context="Escalate issues to the engineering team.",
        )
        assert len(r.fabrications) > 0

    def test_well_grounded_answer_passes(self):
        """Answer with terms from context should pass grounding."""
        r = self._check(
            answer="The system uses PostgreSQL database with pgvector extension for embeddings.",
            doc_context="Architecture: PostgreSQL 15 database with pgvector extension. Embeddings stored in vector column.",
        )
        assert r.grounding_score > 0.3

    def test_completely_ungrounded_fails(self):
        """Answer with no terms from context should fail."""
        r = self._check(
            answer="The quantum entanglement processor operates at 4 kelvin with superconducting qubits.",
            doc_context="Architecture: FastAPI backend, PostgreSQL database, React frontend.",
        )
        assert r.grounding_score < 0.5


# ===========================================================================
# 3. Validator Pipeline Tests
# ===========================================================================
class TestValidator:
    """Tests for backend.validation.validator.validate_answer"""

    def _validate(self, **kwargs):
        from backend.validation.validator import validate_answer
        defaults = {
            "query": "What is the file size limit?",
            "answer": "The file size limit is 100MB.",
            "doc_context": "Configuration: MAX_FILE_SIZE_MB = 100.",
            "ranked_chunks": [("id1", "text", {"source": "doc"}, 0.05)],
            "source_names": ["config.pdf"],
            "model_used": "test",
        }
        defaults.update(kwargs)
        return validate_answer(**defaults)

    def test_valid_answer_passes(self):
        """Well-grounded answer should pass validation."""
        r = self._validate()
        assert r.passed
        assert r.confidence > 0

    def test_fabricated_url_triggers_fallback(self):
        """Fabricated URL should trigger grounding failure and fallback."""
        r = self._validate(
            answer="See https://totally-fake.example.com/nonexistent for details.",
            doc_context="The documentation is maintained in the internal wiki.",
        )
        assert r.was_modified
        assert "could not be adequately verified" in r.answer or "fabricat" in " ".join(r.issues).lower()

    def test_empty_answer_fails(self):
        """Empty answer should fail validation."""
        r = self._validate(answer="")
        assert not r.passed or r.confidence < 0.3

    def test_validation_disabled_passthrough(self):
        """When validation disabled, should pass through with simple confidence."""
        import backend.config as cfg
        original = cfg.settings.ENABLE_ANSWER_VALIDATION
        try:
            cfg.settings.ENABLE_ANSWER_VALIDATION = False
            r = self._validate(answer="Any answer here.")
            assert r.passed
        finally:
            cfg.settings.ENABLE_ANSWER_VALIDATION = original

    def test_version_warning_appended(self):
        """Superseded source should add version warning to answer."""
        chunks = [("id1", "text", {
            "source": "old_runbook.pdf",
            "metadata_json": {"status": "superseded"},
        }, 0.05)]
        r = self._validate(
            ranked_chunks=chunks,
            answer="The procedure is to restart the service and check logs.",
            doc_context="Procedure: restart service, then check logs for errors.",
        )
        # If grounding passes, version warning should be appended
        if r.passed:
            assert r.version_warning or "superseded" in r.answer.lower() or r.version_warning != ""

    def test_confidence_breakdown_available(self):
        """Confidence detail should be populated."""
        r = self._validate()
        assert r.confidence_detail is not None
        assert r.confidence_detail.retrieval_score >= 0
        assert r.confidence_detail.coverage_score >= 0


# ===========================================================================
# 4. Eval Harness Tests
# ===========================================================================
class TestEvalHarness:
    """Tests for the evaluation harness utility."""

    def test_eval_suite_runs(self):
        """The built-in eval suite should run without errors."""
        from backend.validation.eval_harness import run_eval_suite, EVAL_SUITE
        results = run_eval_suite(EVAL_SUITE)
        assert len(results) == len(EVAL_SUITE)
        # At least some cases should be correct
        correct = sum(1 for r in results if r.correct)
        assert correct > 0

    def test_well_grounded_case_passes(self):
        """The well-grounded eval case should pass."""
        from backend.validation.eval_harness import run_eval_suite, EvalCase
        cases = [EvalCase(
            description="Simple grounded test",
            query="What is the limit?",
            answer="The limit is 100MB as specified in the configuration document.",
            doc_context="Configuration document: MAX_FILE_SIZE_MB = 100. This limits uploads to 100MB.",
            expected_pass=True,
        )]
        results = run_eval_suite(cases)
        assert results[0].correct, f"Expected pass but got: {results[0].issues}"

    def test_fabricated_url_case_fails(self):
        """The fabricated URL eval case should be caught."""
        from backend.validation.eval_harness import run_eval_suite, EvalCase
        cases = [EvalCase(
            description="Fabricated URL",
            query="Where are the docs?",
            answer="Visit https://fake-nonexistent-url.com/docs for documentation.",
            doc_context="Internal docs are on the shared drive.",
            expected_pass=False,
            expected_issues=["fabricat"],
        )]
        results = run_eval_suite(cases)
        assert results[0].correct, f"Expected fail but got: passed={results[0].passed}, issues={results[0].issues}"


# ===========================================================================
# 5. Integration / Evaluation Examples
# ===========================================================================
class TestValidationEvalExamples:
    """Documents expected validation behavior for representative cases."""

    def _validate(self, **kwargs):
        from backend.validation.validator import validate_answer
        defaults = {
            "ranked_chunks": [("id1", "text", {"source": "doc"}, 0.05)],
            "source_names": ["doc.pdf"],
            "model_used": "test",
        }
        defaults.update(kwargs)
        return validate_answer(**defaults)

    def test_eval_accurate_answer_passes(self):
        """Accurate, well-grounded answer → passes validation."""
        r = self._validate(
            query="What database does the system use?",
            answer="- The system uses PostgreSQL 15 with the pgvector extension for storing embeddings.",
            doc_context="Architecture: PostgreSQL 15 database with pgvector extension. Vector embeddings stored in dedicated table.",
        )
        assert r.passed
        assert r.confidence > 0.3

    def test_eval_hallucinated_url_rejected(self):
        """Answer with made-up URL → rejected."""
        r = self._validate(
            query="Where is the admin panel?",
            answer="- Access the admin panel at https://admin.fake-portal.io/dashboard",
            doc_context="Admin functions are available through the main API endpoints.",
        )
        assert not r.passed or r.was_modified

    def test_eval_honest_refusal_accepted(self):
        """Honest refusal when no info → accepted with moderate confidence."""
        r = self._validate(
            query="What is the CEO's salary?",
            answer="- I could not find supporting information for that question in the currently uploaded files.",
            doc_context="Company handbook: general HR policies.",
        )
        assert r.passed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
