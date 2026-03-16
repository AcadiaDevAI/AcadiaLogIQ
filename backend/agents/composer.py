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

    prompt = f"""You are a response composer agent. Synthesize the analysis findings below
into a clear, well-structured answer to the user's question.

STRICT RULES:
- Use ONLY the analysis findings below. Do NOT add outside knowledge.
- If findings say "insufficient evidence", reflect that honestly.
- Every answer must be in bullet-point format.
- Cite which document source supports each point when possible.
- Be concise but complete. Do not repeat the same point.

AVAILABLE SOURCES: {sources_str}

ANALYSIS FINDINGS:
{findings_text}

USER QUESTION: {query}

FINAL ANSWER:"""

    step_result = invoke_llm(
        prompt=prompt,
        model=settings.AGENT_COMPOSER_MODEL,
        max_tokens=settings.AGENT_COMPOSER_MAX_TOKENS,
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
