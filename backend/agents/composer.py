"""
Response Composer Agent — synthesizes analysis findings into a final answer.
Takes the per-step findings from the Analysis Agent and composes a coherent,
grounded, bullet-point answer. Uses Haiku by default. Enforces grounding
rules so the final answer stays document-faithful.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List

from backend.config import settings
from backend.agents.base import AgentStepResult, TokenBudget, invoke_llm

logger = logging.getLogger("acadia-log-iq")


def run_composer(
    *,
    query: str,
    findings: List[str],
    source_names: List[str],
    budget: TokenBudget,
    generate_fn: Callable,
    bedrock_client: Any,
) -> AgentStepResult:
    """
    Composer Agent: synthesizes step-by-step findings into a final answer.

    How it works:
    1. Receives all findings from the Analysis Agent
    2. Asks the composer model to merge them into a coherent answer
    3. Enforces grounding rules: no outside knowledge, bullet-point format
    4. Returns the composed answer as an AgentStepResult

    If the composer fails or budget is exhausted, falls back to
    concatenating the raw findings directly.
    """
    # --- Prepare the findings block ---
    findings_text = "\n\n".join(findings) if findings else "[No analysis findings available]"
    sources_str = ", ".join(sorted(set(source_names))[:6]) if source_names else "uploaded documents"

    # Conversational composer voice — see CONVERSATIONAL_REFACTOR_BRIEF goal 1.2.
    # Length-proportional, bullets-only-when-asked, no meta-phrasing about findings.
    prompt = f"""You are a senior operations engineer synthesizing findings from a multi-step analysis into a single conversational answer for a trainee.

The analysis findings below were produced by sub-agents reading the source documents. Use them as your factual basis but answer the user's question in your own words, like a human expert talking to a colleague.

Rules:
- Match response length to the question. Short question → short answer. Broad question → fuller answer.
- Use natural prose. Bullets only when the question asks for a list or comparison.
- Do not add section headers unless the user asked for a structured breakdown.
- Do not say "based on the findings" or "according to the analysis". Just answer.
- If the findings don't fully cover the question, say what's missing in one plain sentence and give the best partial answer you can.

AVAILABLE SOURCES: {sources_str}

ANALYSIS FINDINGS:
{findings_text}

USER QUESTION: {query}

FINAL ANSWER:"""

    # Brief 5 / Part 2 — composer always produces analytical synthesis output,
    # regardless of the input query's class. Cap at the analytical tier so we
    # still trim the old 2048 blanket.
    _composer_max_tokens = (
        settings.RESPONSE_TOKENS_ANALYTICAL
        if settings.RESPONSE_TOKEN_CAPS_ENABLED
        else settings.AGENT_COMPOSER_MAX_TOKENS
    )
    logger.info(
        "[resp_class] composer class=analytical max_tokens=%d", _composer_max_tokens,
    )
    step_result = invoke_llm(
        prompt=prompt,
        model=settings.AGENT_COMPOSER_MODEL,
        max_tokens=_composer_max_tokens,
        budget=budget,
        agent_name="composer",
        generate_fn=generate_fn,
        bedrock_client=bedrock_client,
    )

    # --- Fallback: if composer fails, concatenate raw findings ---
    if not step_result.success or not step_result.output.strip():
        logger.warning("Composer failed, falling back to raw findings")
        fallback = "Based on the analysis of the uploaded documents:\n\n"
        for finding in findings:
            fallback += f"{finding}\n\n"
        step_result.output = fallback.strip()
        step_result.agent_name = "composer (fallback)"

    logger.info(
        "Composer complete: %d chars, model=%s, %dms",
        len(step_result.output), step_result.model_used, step_result.duration_ms,
    )

    return step_result
