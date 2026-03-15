"""
Phase 4 Routing Tests — validates complexity classification, context building,
and model selection policy. No database or AWS credentials needed.
Run with: pytest tests/test_routing.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ===========================================================================
# 1. Complexity Classifier Tests
# ===========================================================================
class TestComplexityClassifier:
    """Tests for backend.routing.complexity_classifier.classify_complexity"""

    def _classify(self, query, **kwargs):
        from backend.routing.complexity_classifier import classify_complexity
        defaults = {
            "ranked_chunks": [],
            "retrieval_confidence": 0.5,
            "source_count": 1,
            "context_chars": 3000,
        }
        defaults.update(kwargs)
        return classify_complexity(query=query, **defaults)

    # --- Simple queries ---
    def test_definition_query_is_simple(self):
        """'What is X' queries should score low complexity."""
        r = self._classify("What is a VPN?", retrieval_confidence=0.8)
        assert r.score < 0.5
        assert r.tier in ("simple", "moderate")

    def test_single_fact_lookup(self):
        """Direct fact lookup should be simple."""
        r = self._classify("Where is the log file located?", retrieval_confidence=0.9)
        assert r.score < 0.5

    def test_simple_list_request(self):
        """'List the X' should be low complexity."""
        r = self._classify("List the supported file formats", retrieval_confidence=0.8)
        assert r.score < 0.5

    # --- Complex queries ---
    def test_comparison_is_complex(self):
        """Comparison queries should score higher."""
        r = self._classify("Compare TCP and UDP protocols and their trade-offs")
        assert r.signals["multi_step"] > 0

    def test_root_cause_is_complex(self):
        """Root cause analysis should trigger reasoning signal."""
        r = self._classify("Why does the service crash under high load? What is the root cause?")
        assert r.signals["reasoning"] > 0

    def test_multi_step_workflow(self):
        """Step-by-step workflow should trigger multi-step signal."""
        r = self._classify("Walk me through the end-to-end deployment workflow step by step")
        assert r.signals["multi_step"] > 0

    def test_recommendation_is_complex(self):
        """Recommendation queries require synthesis."""
        r = self._classify("What do you recommend for improving database performance?")
        assert r.signals["reasoning"] > 0

    # --- Context signals ---
    def test_low_confidence_increases_complexity(self):
        """Low retrieval confidence should push complexity up."""
        low = self._classify("Some ambiguous question", retrieval_confidence=0.1)
        high = self._classify("Some ambiguous question", retrieval_confidence=0.9)
        assert low.score >= high.score

    def test_large_context_increases_complexity(self):
        """More context chars should increase complexity slightly."""
        small = self._classify("Explain this", context_chars=1000)
        large = self._classify("Explain this", context_chars=18000)
        assert large.signals["context_size"] > small.signals["context_size"]

    def test_multi_doc_increases_complexity(self):
        """Multiple source documents should increase complexity."""
        single = self._classify("Explain this", source_count=1)
        multi = self._classify("Explain this", source_count=4)
        assert multi.signals["multi_doc"] > single.signals["multi_doc"]

    # --- Tier assignment ---
    def test_tier_simple(self):
        """Very simple query with high confidence → simple tier."""
        r = self._classify(
            "What is DNS?",
            retrieval_confidence=0.95,
            source_count=1,
            context_chars=500,
        )
        assert r.tier == "simple"

    def test_tier_complex(self):
        """Complex multi-step comparison with low confidence → complex tier."""
        r = self._classify(
            "Compare and contrast the advantages and disadvantages of approach A vs B, "
            "then recommend the best approach and analyze the implications",
            retrieval_confidence=0.2,
            source_count=4,
            context_chars=15000,
        )
        assert r.tier == "complex"


# ===========================================================================
# 2. Context Builder Tests
# ===========================================================================
class TestContextBuilder:
    """Tests for backend.routing.context_builder.build_prompt"""

    def _build(self, **kwargs):
        from backend.routing.context_builder import build_prompt
        defaults = {
            "query": "Test question",
            "doc_context": "Some document text here.",
            "target_model": "mistral",
        }
        defaults.update(kwargs)
        return build_prompt(**defaults)

    def test_basic_prompt_structure(self):
        """Prompt should contain query, documents, and answer marker."""
        prompt = self._build()
        assert "USER QUESTION: Test question" in prompt
        assert "DOCUMENTS:" in prompt
        assert "ANSWER:" in prompt

    def test_grounding_rules_present(self):
        """Grounding rules must always be in the prompt."""
        prompt = self._build()
        assert "Answer ONLY from the DOCUMENTS" in prompt
        assert "Do NOT use outside knowledge" in prompt

    def test_session_history_included(self):
        """Recent messages should appear when provided."""
        messages = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]
        prompt = self._build(recent_messages=messages)
        assert "RECENT CONVERSATION:" in prompt
        assert "Previous question" in prompt

    def test_session_history_omitted_when_empty(self):
        """No session history block when no messages."""
        prompt = self._build(recent_messages=[])
        assert "RECENT CONVERSATION:" not in prompt

    def test_metadata_hints_included(self):
        """Metadata hints from ranked chunks should appear."""
        chunks = [
            ("id1", "text", {
                "metadata_json": {"vendor": "AWS", "product": "S3", "domain": "Cloud Storage"}
            }, 0.9),
        ]
        prompt = self._build(ranked_chunks=chunks)
        assert "DOCUMENT CONTEXT HINTS:" in prompt
        assert "AWS" in prompt

    def test_confidence_signal_high(self):
        """High confidence should produce confident signal."""
        prompt = self._build(retrieval_confidence=0.9, source_count=2)
        assert "HIGH" in prompt

    def test_confidence_signal_low(self):
        """Low confidence should produce cautious signal."""
        prompt = self._build(retrieval_confidence=0.2, source_count=1)
        assert "LOW" in prompt

    def test_sonnet_prompt_has_reasoning_instruction(self):
        """Sonnet prompt should include step-by-step reasoning instruction."""
        prompt = self._build(target_model="sonnet")
        assert "step by step" in prompt.lower()

    def test_mistral_prompt_is_simpler(self):
        """Mistral prompt should be more direct."""
        prompt = self._build(target_model="mistral")
        # Should not have the verbose sonnet instructions
        assert "break down your reasoning" not in prompt


# ===========================================================================
# 3. Model Router Policy Tests
# ===========================================================================
class TestModelRouter:
    """Tests for backend.routing.model_router routing policy."""

    def _select(self, tier, score=0.5):
        from backend.routing.model_router import _select_model
        from backend.routing.complexity_classifier import ComplexityResult
        cx = ComplexityResult(score=score, tier=tier)
        return _select_model(cx)

    def test_simple_routes_to_mistral(self):
        """Simple tier should route to Mistral."""
        model, reason = self._select("simple", 0.1)
        assert model == "mistral"
        assert "simple" in reason.lower()

    def test_moderate_routes_to_haiku(self):
        """Moderate tier should route to Haiku."""
        model, reason = self._select("moderate", 0.5)
        assert model == "haiku"
        assert "moderate" in reason.lower()

    def test_complex_routes_to_sonnet(self):
        """Complex tier should route to Sonnet."""
        model, reason = self._select("complex", 0.8)
        assert model == "sonnet"
        assert "complex" in reason.lower()

    def test_routing_disabled_uses_default(self):
        """When routing disabled, should use default model."""
        import backend.config as cfg
        original = cfg.settings.ENABLE_MODEL_ROUTING
        try:
            cfg.settings.ENABLE_MODEL_ROUTING = False
            model, reason = self._select("complex", 0.9)
            assert model == cfg.settings.ROUTING_DEFAULT_MODEL
            assert "disabled" in reason
        finally:
            cfg.settings.ENABLE_MODEL_ROUTING = original

    def test_mistral_fallback_on_failure(self):
        """Router should fall back to Mistral if Claude fails."""
        from backend.routing.model_router import route_and_generate

        def mock_generate(prompt, max_tokens):
            return "Mistral fallback answer"

        # Create a mock bedrock client that always fails
        class FailingClient:
            def invoke_model(self, **kwargs):
                raise RuntimeError("Simulated Bedrock failure")

        result = route_and_generate(
            query="Complex compare question",
            doc_context="Some document text",
            ranked_chunks=[],
            source_names=["doc1"],
            retrieval_confidence=0.5,
            generate_fn=mock_generate,
            bedrock_client=FailingClient(),
        )

        # Should have fallen back to Mistral
        assert "fallback" in result.model_used.lower() or result.model_used == "mistral"
        assert result.answer == "Mistral fallback answer"


# ===========================================================================
# 4. Integration / Evaluation Examples
# ===========================================================================
class TestRoutingEvalExamples:
    """
    Example queries documenting expected routing behavior.
    Validates that the full classify → route pipeline produces
    sensible model selections for representative query types.
    """

    def _route_model(self, query, confidence=0.5, sources=1, ctx_chars=3000):
        from backend.routing.complexity_classifier import classify_complexity
        from backend.routing.model_router import _select_model
        cx = classify_complexity(
            query=query,
            ranked_chunks=[],
            retrieval_confidence=confidence,
            source_count=sources,
            context_chars=ctx_chars,
        )
        model, reason = _select_model(cx)
        return model, cx

    def test_eval_definition_goes_to_mistral(self):
        """'What is X?' with high confidence → Mistral."""
        model, cx = self._route_model("What is a firewall?", confidence=0.9)
        assert model == "mistral", f"Expected mistral, got {model} ({cx.reason})"

    def test_eval_comparison_goes_to_sonnet(self):
        """Complex comparison with low confidence → Sonnet."""
        model, cx = self._route_model(
            "Compare the pros and cons of approach A versus B and recommend which to use",
            confidence=0.3, sources=3, ctx_chars=12000,
        )
        assert model == "sonnet", f"Expected sonnet, got {model} ({cx.reason})"

    def test_eval_troubleshooting_goes_to_haiku_or_higher(self):
        """Troubleshooting query → at least Haiku."""
        model, cx = self._route_model(
            "How to troubleshoot high memory usage on the production server?",
            confidence=0.6,
        )
        assert model in ("haiku", "sonnet"), f"Expected haiku+, got {model} ({cx.reason})"

    def test_eval_simple_lookup_stays_cheap(self):
        """Simple lookup with high confidence → Mistral."""
        model, cx = self._route_model("Where is the config file?", confidence=0.95)
        assert model == "mistral", f"Expected mistral, got {model} ({cx.reason})"

    def test_eval_synthesis_goes_to_sonnet(self):
        """Multi-doc synthesis → Sonnet."""
        model, cx = self._route_model(
            "Summarize and synthesize the key findings from all the uploaded documents "
            "and analyze their implications for our infrastructure",
            confidence=0.3, sources=5, ctx_chars=18000,
        )
        assert model == "sonnet", f"Expected sonnet, got {model} ({cx.reason})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
