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


# ─────────────────────────────────────────────────────────────
# Composer markdown formatting addendum (Fix 5 — Rich Formatting Polish)
#
# The brief specifies this helper sit in orchestrator.py, but Composer's
# prompt construction is delegated here to run_composer() — which means
# orchestrator.py already imports this module. Placing the helper here
# avoids a circular import while still wrapping the Composer system
# prompt at its actual construction site.
# ─────────────────────────────────────────────────────────────

_COMPOSER_MARKDOWN_ADDENDUM = """

Markdown formatting rules (use only when they genuinely help readability):
- When the user asks to COMPARE specific items (e.g., "Compare INC-10005 and INC-10006"), lead the answer with a markdown table showing key attributes side-by-side (Customer, Priority, Duration, Resolution Approach, Outcome, etc.), then follow with a brief prose paragraph explaining the key insight or lesson. Always include a table for compare queries unless the items have no comparable attributes.
- Ticket IDs, customer names, product names, component names → use **bold** emphasis (e.g., **INC-10037**, **Enterprise-617**, **ADTRAN 908E**). Do NOT wrap identifiers in backticks.
- Inline code (`backticks`) is ONLY for literal commands (`show bgp summary`), file paths (`/etc/config`), or variable names. Never for ticket IDs.
- Step-by-step procedures → use numbered lists with clear action verbs.
- Key recommendations or important takeaways → use > blockquote on its own paragraph.
- Commands, configs, code → fenced code blocks with language hint (```bash, ```python, etc.).
- Default to flowing prose for non-compare questions. Do not force structure when prose is clearer.

CRITICAL — Confident synthesis: When the user asks for lessons, insights, takeaways, recommendations, biggest learnings, or "what should I learn from X", you MUST synthesize a confident answer from the RESOLUTION DETAIL, ROOT CAUSE, ITIL 5-WHY, SOP EXECUTION STEPS, and QA AUDITOR GAPS sections. The lesson is implicit in how the incident was resolved and what went wrong — extract it and present it confidently in your own words. Do NOT refuse by saying "not explicitly stated" or "I could not find this" just because the literal word "lesson" is absent from the document. The documents contain everything needed to answer insight questions through inference.

CRITICAL — Cross-cutting pattern synthesis: When the user asks analytical questions spanning multiple tickets (e.g., "common root causes", "recurring themes", "most frequently recommended improvements"), structure your answer as grouped insights with counts. Format each pattern as:

- **Pattern name** (N tickets) — brief description with representative examples

Example: "Hardware failures (14 tickets) — predominantly Cisco router SPE modules and switch stack failures; examples include INC-10005, INC-10006, INC-10032."

After the grouped patterns, add a brief 1-2 sentence synthesis paragraph explaining the dominant theme or recommendation. Never return raw ticket ID lists for analytical questions."""


def _build_composer_prompt_with_markdown(base_prompt: str) -> str:
    """
    Append markdown formatting guidance to the Composer base prompt.

    Controlled by AGENT_COMPOSER_MARKDOWN_ENABLED flag. When disabled,
    returns base_prompt unchanged — exactly pre-fix behavior, byte-for-byte.
    """
    if not getattr(settings, "AGENT_COMPOSER_MARKDOWN_ENABLED", True):
        return base_prompt
    return base_prompt + _COMPOSER_MARKDOWN_ADDENDUM


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
    #
    # Fix 5 — the rules section is wrapped with the markdown addendum helper
    # (flag-gated). The content section below the wrap is appended verbatim
    # so that when AGENT_COMPOSER_MARKDOWN_ENABLED=False the final prompt is
    # byte-for-byte identical to the pre-fix composer prompt.
    _composer_rules = """You are a senior operations engineer synthesizing findings from a multi-step analysis into a single conversational answer for a trainee.

The analysis findings below were produced by sub-agents reading the source documents. Use them as your factual basis but answer the user's question in your own words, like a human expert talking to a colleague.

Rules:
- Match response length to the question. Short question → short answer. Broad question → fuller answer.
- Use natural prose. Bullets only when the question asks for a list or comparison.
- Do not add section headers unless the user asked for a structured breakdown.
- Do not say "based on the findings" or "according to the analysis". Just answer.
- If the findings don't fully cover the question, say what's missing in one plain sentence and give the best partial answer you can."""

    _composer_content = f"""

AVAILABLE SOURCES: {sources_str}

ANALYSIS FINDINGS:
{findings_text}

USER QUESTION: {query}

FINAL ANSWER:"""

    prompt = _build_composer_prompt_with_markdown(_composer_rules) + _composer_content

    # Brief 5 / Part 2 — composer always produces analytical synthesis output,
    # regardless of the input query's class. Cap at the analytical tier so we
    # still trim the old 2048 blanket.
    _composer_max_tokens = (
        settings.RESPONSE_TOKENS_ANALYTICAL
        if settings.RESPONSE_TOKEN_CAPS_ENABLED
        else settings.AGENT_COMPOSER_MAX_TOKENS
    )

    # Fix 4 — analytical budget awareness. When the pipeline ran with the
    # elevated analytical ceiling (AGENT_ANALYTICAL_BUDGET) and there is
    # ample headroom remaining, prefer the full analytical response cap so
    # synthesis completes end-to-end instead of truncating. The effective
    # max is still bounded by budget.remaining inside invoke_llm(), so this
    # never over-spends; it just prevents premature truncation when room
    # exists. Flag-gated implicitly via AGENT_ANALYTICAL_BUDGET (0 = off).
    _analytical_floor = getattr(settings, "AGENT_ANALYTICAL_BUDGET", 0)
    if (
        _analytical_floor
        and budget.max_total >= _analytical_floor
        and budget.remaining >= settings.AGENT_BUDGET_COMPOSER_TOKENS
    ):
        _prev_max = _composer_max_tokens
        _composer_max_tokens = max(
            _composer_max_tokens, settings.RESPONSE_TOKENS_ANALYTICAL,
        )
        if _composer_max_tokens != _prev_max:
            logger.info(
                "[composer_budget] analytical floor applied: %d → %d",
                _prev_max, _composer_max_tokens,
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
