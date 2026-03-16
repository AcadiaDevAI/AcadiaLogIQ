"""
Analysis Agent — executes plan steps against document context.
Runs each step from the Planner against the retrieved documents,
producing a grounded finding per step. Uses Haiku by default
to keep costs low. Skips steps if token budget runs out.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Tuple

from backend.config import settings
from backend.agents.base import AgentStepResult, TokenBudget, invoke_llm

logger = logging.getLogger("acadia-log-iq")


def run_analysis(
    *,
    steps: List[str],
    doc_context: str,
    query: str,
    budget: TokenBudget,
    generate_fn: Callable,
    bedrock_client: Any,
) -> Tuple[List[str], List[AgentStepResult]]:
    """
    Analysis Agent: executes each planned step against the document context.

    How it works:
    1. For each step from the Planner, sends the step + full document context
    2. Asks the model to produce a grounded finding (evidence from the docs only)
    3. Collects findings into a list
    4. Stops early if token budget is exhausted

    Returns:
        (findings_list, step_results_list)
    """
    findings: List[str] = []
    step_results: List[AgentStepResult] = []

    for i, step in enumerate(steps):
        # Check budget before each step
        if budget.exhausted:
            logger.warning("Analysis agent stopping at step %d/%d: budget exhausted", i + 1, len(steps))
            break

        prompt = f"""You are a technical analysis agent. Answer the following analysis step
using ONLY the document context provided below. Do NOT use outside knowledge.

If the documents do not contain enough information for this step, say:
"Insufficient evidence in documents for this step."

Keep your response concise (3-6 bullet points max).

DOCUMENT CONTEXT:
{doc_context}

ORIGINAL USER QUESTION: {query}

ANALYSIS STEP {i + 1}/{len(steps)}: {step}

FINDINGS:"""

        step_result = invoke_llm(
            prompt=prompt,
            model=settings.AGENT_ANALYSIS_MODEL,
            max_tokens=settings.AGENT_ANALYSIS_MAX_TOKENS,
            budget=budget,
            agent_name=f"analyst_step_{i + 1}",
            generate_fn=generate_fn,
            bedrock_client=bedrock_client,
        )

        step_results.append(step_result)

        if step_result.success and step_result.output:
            findings.append(f"Step {i + 1} — {step}:\n{step_result.output}")
        else:
            findings.append(f"Step {i + 1} — {step}:\n[Analysis failed: {step_result.error or 'no output'}]")

        logger.debug(
            "Analysis step %d/%d complete: success=%s, %d tokens, %dms",
            i + 1, len(steps), step_result.success, step_result.tokens_used, step_result.duration_ms,
        )

    logger.info(
        "Analysis complete: %d/%d steps executed, %d findings",
        len(step_results), len(steps), len(findings),
    )

    return findings, step_results
