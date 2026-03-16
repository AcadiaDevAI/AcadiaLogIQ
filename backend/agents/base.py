"""
Agent Base — shared types, LLM invoker, and token budget tracker.
All agents import from here. Provides a unified invoke_llm() that
routes to the correct Bedrock model and tracks cumulative token usage.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from backend.config import settings

logger = logging.getLogger("acadia-log-iq")


# ---------------------------------------------------------------------------
# Token budget tracker — shared across all agents in a single pipeline run
# ---------------------------------------------------------------------------
@dataclass
class TokenBudget:
    """
    Tracks cumulative token usage across all agent steps in one pipeline run.
    Enforces the AGENT_MAX_TOTAL_TOKENS hard ceiling.
    """
    max_total: int = 0
    used: int = 0
    steps: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        self.max_total = self.max_total or settings.AGENT_MAX_TOTAL_TOKENS

    @property
    def remaining(self) -> int:
        return max(0, self.max_total - self.used)

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0

    def record(self, agent_name: str, tokens_used: int, model: str, ms: int):
        """Record a step's token usage."""
        self.used += tokens_used
        self.steps.append({
            "agent": agent_name,
            "model": model,
            "tokens": tokens_used,
            "cumulative": self.used,
            "ms": ms,
        })
        logger.debug(
            "TokenBudget: %s used %d tokens (%s), cumulative=%d/%d",
            agent_name, tokens_used, model, self.used, self.max_total,
        )


# ---------------------------------------------------------------------------
# Agent step result
# ---------------------------------------------------------------------------
@dataclass
class AgentStepResult:
    """Result from a single agent execution step."""
    agent_name: str = ""
    output: str = ""
    model_used: str = ""
    tokens_used: int = 0
    duration_ms: int = 0
    success: bool = True
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Full pipeline result
# ---------------------------------------------------------------------------
@dataclass
class AgentPipelineResult:
    """
    Complete result from the multi-agent pipeline.
    Returned to the /ask endpoint.
    """
    answer: str = ""
    agent_mode: bool = False
    steps: List[AgentStepResult] = field(default_factory=list)
    plan: Optional[List[str]] = None
    total_tokens: int = 0
    total_ms: int = 0
    reasoning_summary: str = ""  # internal log only, never exposed to user


# ---------------------------------------------------------------------------
# Unified LLM invoker
# ---------------------------------------------------------------------------
def invoke_llm(
    *,
    prompt: str,
    model: str,
    max_tokens: int,
    budget: TokenBudget,
    agent_name: str,
    generate_fn: Callable[[str, int], str],
    bedrock_client: Any,
) -> AgentStepResult:
    """
    Invoke the appropriate LLM based on model name.

    Checks token budget before calling. Routes to:
    - 'mistral'  → existing safe_generate function
    - 'haiku'    → Bedrock Claude Haiku
    - 'sonnet'   → Bedrock Claude Sonnet

    Returns an AgentStepResult with output, tokens used, and timing.
    """
    result = AgentStepResult(agent_name=agent_name, model_used=model)

    # --- Check budget ---
    effective_max = min(max_tokens, budget.remaining)
    if effective_max <= 50:
        result.success = False
        result.error = f"Token budget exhausted ({budget.used}/{budget.max_total})"
        logger.warning("Agent %s skipped: token budget exhausted", agent_name)
        return result

    t_start = time.perf_counter()

    try:
        if model == "mistral":
            # Use existing Mistral generate function
            output = generate_fn(prompt, effective_max)
        elif model in ("haiku", "sonnet"):
            # Use Bedrock Claude API
            model_id = (
                settings.BEDROCK_SONNET_MODEL if model == "sonnet"
                else settings.BEDROCK_HAIKU_MODEL
            )
            temp = (
                settings.SONNET_TEMPERATURE if model == "sonnet"
                else settings.HAIKU_ANSWER_TEMPERATURE
            )
            output = _invoke_claude(
                prompt=prompt,
                bedrock_client=bedrock_client,
                model_id=model_id,
                max_tokens=effective_max,
                temperature=temp,
            )
        else:
            raise ValueError(f"Unknown model: {model}")

        result.output = output
        result.success = True

        # Estimate tokens used (rough: 1 token ≈ 4 chars for output)
        est_tokens = len(output) // 4 + len(prompt) // 4
        result.tokens_used = est_tokens

    except Exception as exc:
        result.success = False
        result.error = str(exc)
        result.tokens_used = len(prompt) // 4  # count input even on failure
        logger.warning("Agent %s LLM call failed (%s): %s", agent_name, model, exc)

    result.duration_ms = int((time.perf_counter() - t_start) * 1000)

    # Record in budget tracker
    budget.record(agent_name, result.tokens_used, model, result.duration_ms)

    return result


def _invoke_claude(
    *,
    prompt: str,
    bedrock_client: Any,
    model_id: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Invoke Claude (Haiku or Sonnet) via Bedrock Messages API."""
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "system": "You are an expert technical analyst. Be precise and grounded in evidence.",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
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
    return "\n".join(
        item.get("text", "") for item in content if item.get("type") == "text"
    ).strip() or "No response generated."
