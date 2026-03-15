"""
Model Router — selects and invokes the optimal model for each query.
Routes to Mistral (cheap/fast), Haiku (mid-tier), or Sonnet (premium)
based on complexity score, retrieval confidence, and config thresholds.
Logs routing decisions for cost tracking and debugging.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.config import settings
from backend.routing.complexity_classifier import ComplexityResult, classify_complexity
from backend.routing.context_builder import build_prompt

logger = logging.getLogger("acadia-log-iq")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class RoutingResult:
    """
    Complete result from the model router.

    Fields:
        answer       — the generated answer text
        model_used   — 'mistral' | 'haiku' | 'sonnet'
        complexity   — the ComplexityResult from classification
        reason       — why this model was selected
        generation_ms — time spent on LLM generation
        prompt_chars — total prompt size sent to the model
    """
    answer: str = ""
    model_used: str = "mistral"
    complexity: Optional[ComplexityResult] = None
    reason: str = ""
    generation_ms: int = 0
    prompt_chars: int = 0


# ---------------------------------------------------------------------------
# Routing policy — decides which model to use
# ---------------------------------------------------------------------------
def _select_model(complexity: ComplexityResult) -> Tuple[str, str]:
    """
    Apply routing policy based on complexity tier.

    Returns:
        (model_name, reason) tuple

    Policy:
        simple   → Mistral  (fast, cheap, good for definitions/lookups)
        moderate → Haiku    (mid-cost, better reasoning than Mistral)
        complex  → Sonnet   (premium, multi-step reasoning, synthesis)
    """
    if not settings.ENABLE_MODEL_ROUTING:
        return settings.ROUTING_DEFAULT_MODEL, "routing disabled, using default"

    tier = complexity.tier
    score = complexity.score

    if tier == "simple":
        return "mistral", f"simple query (score={score:.3f}): fast Mistral path"

    if tier == "complex":
        return "sonnet", f"complex query (score={score:.3f}): Sonnet for deep reasoning"

    # moderate tier → Haiku by default
    return "haiku", f"moderate query (score={score:.3f}): Haiku for balanced cost/quality"


# ---------------------------------------------------------------------------
# Model invocation functions
# ---------------------------------------------------------------------------
def _invoke_mistral(
    prompt: str,
    generate_fn: Callable[[str, int], str],
) -> str:
    """
    Generate answer using Mistral via the existing safe_generate function.
    Mistral is the cheapest option — used for straightforward lookups.
    """
    return generate_fn(prompt, settings.HAIKU_ANSWER_MAX_TOKENS)


def _invoke_claude(
    prompt: str,
    bedrock_client: Any,
    model_id: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """
    Generate answer using Claude (Haiku or Sonnet) via Bedrock.
    Uses the Anthropic Messages API format (same as bedrock_haiku.py).

    Args:
        prompt         — the full prompt text
        bedrock_client — boto3 bedrock-runtime client
        model_id       — Bedrock model ID for Haiku or Sonnet
        max_tokens     — max output tokens
        temperature    — generation temperature
    """
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "system": (
            "You are a strict document-grounded AI assistant. "
            "Answer only from the provided documents. "
            "Use bullet-point format."
        ),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    }

    response = bedrock_client.invoke_model(
        modelId=model_id,
        body=json.dumps(body).encode("utf-8"),
        accept="application/json",
        contentType="application/json",
    )

    payload = json.loads(response["body"].read().decode("utf-8"))
    content = payload.get("content", [])

    # Extract text from Claude's response blocks
    text_parts = [
        item.get("text", "")
        for item in content
        if item.get("type") == "text"
    ]
    result = "\n".join(text_parts).strip()

    if not result:
        stop_reason = payload.get("stop_reason", "unknown")
        logger.warning("Claude returned empty response | model=%s stop_reason=%s", model_id, stop_reason)
        return "No response generated."

    return result


# ---------------------------------------------------------------------------
# Main entry point: route_and_generate
# ---------------------------------------------------------------------------
def route_and_generate(
    *,
    query: str,
    doc_context: str,
    ranked_chunks: List[Tuple[str, str, Dict[str, Any], float]],
    source_names: List[str],
    retrieval_confidence: float,
    recent_messages: Optional[List[Dict[str, str]]] = None,
    generate_fn: Callable[[str, int], str],
    bedrock_client: Any,
) -> RoutingResult:
    """
    Main routing entry point. Called by the /ask endpoint.

    Pipeline:
    1. Classify query complexity using retrieval signals
    2. Select model tier (Mistral / Haiku / Sonnet)
    3. Build enriched prompt with context signals
    4. Invoke the selected model
    5. Return answer + routing metadata

    Args:
        query                — user's question
        doc_context          — assembled document text from retrieval
        ranked_chunks        — reranked retrieval results
        source_names         — list of source document names
        retrieval_confidence — confidence from retrieval pipeline (0.0-1.0)
        recent_messages      — recent session messages for context
        generate_fn          — Mistral generation function (safe_generate)
        bedrock_client       — boto3 bedrock-runtime client (for Claude calls)

    Returns:
        RoutingResult with answer, model used, complexity info, and timing.
    """
    result = RoutingResult()

    # ==================================================================
    # Step 1: Classify complexity
    # ==================================================================
    complexity = classify_complexity(
        query=query,
        ranked_chunks=ranked_chunks,
        retrieval_confidence=retrieval_confidence,
        source_count=len(set(source_names)),
        context_chars=len(doc_context),
    )
    result.complexity = complexity

    # ==================================================================
    # Step 2: Select model
    # ==================================================================
    model_name, reason = _select_model(complexity)
    result.model_used = model_name
    result.reason = reason

    logger.info(
        "Model routing: query='%.80s' → model=%s | %s | complexity=%s",
        query, model_name, reason, complexity.reason,
    )

    # ==================================================================
    # Step 3: Build enriched prompt
    # ==================================================================
    prompt = build_prompt(
        query=query,
        doc_context=doc_context,
        target_model=model_name,
        recent_messages=recent_messages,
        ranked_chunks=ranked_chunks,
        retrieval_confidence=retrieval_confidence,
        source_count=len(set(source_names)),
    )
    result.prompt_chars = len(prompt)

    # ==================================================================
    # Step 4: Invoke the selected model
    # ==================================================================
    t_start = time.perf_counter()

    try:
        if model_name == "sonnet":
            answer = _invoke_claude(
                prompt=prompt,
                bedrock_client=bedrock_client,
                model_id=settings.BEDROCK_SONNET_MODEL,
                max_tokens=settings.SONNET_MAX_TOKENS,
                temperature=settings.SONNET_TEMPERATURE,
            )
        elif model_name == "haiku":
            answer = _invoke_claude(
                prompt=prompt,
                bedrock_client=bedrock_client,
                model_id=settings.BEDROCK_HAIKU_MODEL,
                max_tokens=settings.HAIKU_ANSWER_MAX_TOKENS,
                temperature=settings.HAIKU_ANSWER_TEMPERATURE,
            )
        else:
            # Default: Mistral via existing safe_generate
            answer = _invoke_mistral(prompt, generate_fn)

    except Exception as exc:
        logger.warning(
            "Model %s failed (%s), falling back to Mistral",
            model_name, exc,
        )
        # Fallback: always try Mistral if Claude fails
        result.model_used = "mistral (fallback)"
        result.reason += f" | {model_name} failed: {exc}"
        try:
            answer = _invoke_mistral(prompt, generate_fn)
        except Exception as fallback_exc:
            logger.error("Mistral fallback also failed: %s", fallback_exc)
            answer = "Error generating response. Please try again."

    t_end = time.perf_counter()
    result.answer = answer
    result.generation_ms = int((t_end - t_start) * 1000)

    logger.info(
        "Generation complete: model=%s, %dms, %d prompt chars, complexity=%.3f (%s)",
        result.model_used, result.generation_ms,
        result.prompt_chars, complexity.score, complexity.tier,
    )

    return result
