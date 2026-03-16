"""
Planner Agent — decomposes complex queries into concrete sub-steps.
Uses Sonnet (or configurable model) to create a retrieval-grounded plan.
Produces a compact list of analysis steps, each referencing document context.
Capped at AGENT_MAX_STEPS to control cost.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.config import settings
from backend.agents.base import AgentStepResult, TokenBudget, invoke_llm

logger = logging.getLogger("acadia-log-iq")


def run_planner(
    *,
    query: str,
    doc_context_preview: str,
    source_names: List[str],
    budget: TokenBudget,
    generate_fn: Callable,
    bedrock_client: Any,
) -> Tuple[List[str], AgentStepResult]:
    """
    Planner Agent: breaks a complex query into sub-steps.

    How it works:
    1. Sends the query + a preview of available document context to the planner model
    2. Asks for a JSON array of concrete analysis steps
    3. Parses the steps and caps at AGENT_MAX_STEPS
    4. Returns (steps_list, agent_step_result)

    The plan is grounded: steps reference the document content, not outside knowledge.
    If planning fails, returns a single fallback step: "answer the full question directly".
    """
    max_steps = settings.AGENT_MAX_STEPS

    # Build a compact source list for the planner
    sources_str = ", ".join(sorted(set(source_names))[:6]) if source_names else "unknown"

    prompt = f"""You are a technical planning agent. The user asked a complex question
that requires multi-step analysis from the provided documents.

Break the question into {max_steps} or fewer concrete analysis steps.
Each step must be answerable from the document context — do NOT plan steps
that require outside knowledge.

Return ONLY a JSON array of step strings, like:
["Step 1: ...", "Step 2: ...", "Step 3: ..."]

No commentary. No markdown fences. Just the JSON array.

Available document sources: {sources_str}

Document context preview (first 1500 chars):
{doc_context_preview[:1500]}

User question: {query}

Plan:"""

    step_result = invoke_llm(
        prompt=prompt,
        model=settings.AGENT_PLANNER_MODEL,
        max_tokens=settings.AGENT_PLANNER_MAX_TOKENS,
        budget=budget,
        agent_name="planner",
        generate_fn=generate_fn,
        bedrock_client=bedrock_client,
    )

    # --- Parse the plan ---
    steps: List[str] = []

    if step_result.success and step_result.output:
        raw = step_result.output.strip()

        # Try to extract a JSON array from the response
        try:
            # Handle markdown fences
            if "```" in raw:
                raw = raw.split("```json")[-1].split("```")[0] if "```json" in raw else raw.split("```")[1]

            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start >= 0 and end > start:
                parsed = json.loads(raw[start:end])
                if isinstance(parsed, list):
                    steps = [str(s).strip() for s in parsed if str(s).strip()]
        except (json.JSONDecodeError, IndexError):
            # Fall back to line-based parsing
            for line in raw.split("\n"):
                line = line.strip().lstrip("-•*0123456789.)").strip()
                if len(line) > 10:
                    steps.append(line)

    # --- Cap steps ---
    steps = steps[:max_steps]

    # --- Fallback if planning produced nothing ---
    if not steps:
        steps = [f"Analyze the documents to answer: {query}"]
        logger.warning("Planner produced no steps, using single fallback step")

    logger.info("Planner produced %d steps: %s", len(steps), [s[:60] for s in steps])

    return steps, step_result
