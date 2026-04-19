"""
Agent Orchestrator — decides whether agent mode is needed and runs the pipeline.
Entry points: should_escalate_to_agents() and run_agent_pipeline().
Only complex queries matching specific patterns trigger agents.
Simple and moderate queries are never touched by this module.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from backend.config import settings
from backend.agents.base import AgentPipelineResult, TokenBudget
from backend.agents.clarifier import run_clarifier
from backend.agents.planner import run_planner
from backend.agents.analyst import run_analysis
from backend.agents.composer import run_composer
from backend.routing.stage_enforcer import STAGE_DOCS, UNRESOLVED_ESCALATE_THRESHOLD

logger = logging.getLogger("acadia-log-iq")


# ---------------------------------------------------------------------------
# Patterns that indicate agent-eligible complex queries
# ---------------------------------------------------------------------------
_AGENT_ELIGIBLE_PATTERNS = [
    # Multi-step troubleshooting
    re.compile(r"\b(?:troubleshoot|diagnose|debug|step.by.step|walk me through)\b", re.I),
    # Comparison / synthesis
    re.compile(r"\b(?:compare|contrast|versus|vs\.?|differ(?:ence|ent)|pros?.and.cons?)\b", re.I),
    # Multi-document synthesis
    re.compile(r"\b(?:across|all documents?|every|each of|summarize all|consolidate)\b", re.I),
    # Guided remediation
    re.compile(r"\b(?:remediat|fix.+and.+verify|resolve.+then|after.+check)\b", re.I),
    # Root cause + recommendation
    re.compile(r"\b(?:root cause.+recommend|why.+and.+how|analyze.+then.+suggest)\b", re.I),
    # End-to-end workflows
    re.compile(r"\b(?:end.to.end|complete process|full workflow|entire procedure)\b", re.I),
]


# ---------------------------------------------------------------------------
# Escalation gate — decides whether to use agents
# ---------------------------------------------------------------------------
def should_escalate_to_agents(
    *,
    query: str,
    complexity_score: float,
    complexity_tier: str,
    source_count: int,
) -> Tuple[bool, str]:
    """
    Decide whether a query should escalate to multi-agent mode.

    Gate logic (ALL must be true):
    1. ENABLE_AGENT_MODE is True in config
    2. Complexity tier is 'complex' (from Phase 4 classifier)
    3. Complexity score exceeds AGENT_COMPLEXITY_THRESHOLD
    4. Query matches at least one agent-eligible pattern
    5. At least AGENT_MIN_SOURCES source documents available

    Returns:
        (should_escalate: bool, reason: str)
    """
    # Gate 1: feature flag
    if not settings.ENABLE_AGENT_MODE:
        return False, "agent mode disabled"

    # Gate 2: complexity tier must be 'complex'
    if complexity_tier != "complex":
        return False, f"tier={complexity_tier} (not complex)"

    # Gate 3: score threshold
    if complexity_score < settings.AGENT_COMPLEXITY_THRESHOLD:
        return False, f"score={complexity_score:.3f} below threshold {settings.AGENT_COMPLEXITY_THRESHOLD}"

    # Gate 4: pattern match
    pattern_match = any(p.search(query) for p in _AGENT_ELIGIBLE_PATTERNS)
    if not pattern_match:
        return False, "no agent-eligible patterns matched"

    # Gate 5: minimum sources
    if source_count < settings.AGENT_MIN_SOURCES:
        return False, f"source_count={source_count} below minimum {settings.AGENT_MIN_SOURCES}"

    return True, f"complex query (score={complexity_score:.3f}) with agent-eligible pattern"


# ---------------------------------------------------------------------------
# Main agent pipeline
# ---------------------------------------------------------------------------
def run_agent_pipeline(
    *,
    query: str,
    doc_context: str,
    ranked_chunks: List[Tuple[str, str, Dict[str, Any], float]],
    source_names: List[str],
    generate_fn: Callable,
    bedrock_client: Any,
    step_retriever_fn: Optional[Callable[[str], Any]] = None,
    prior_messages: Optional[List[Dict[str, str]]] = None,
    stage: Optional[str] = None,
    unresolved_count: int = 0,
) -> AgentPipelineResult:
    """
    Run the full multi-agent pipeline: Planner → Analyst → Composer.

    Pipeline:
    1. Planner (Sonnet): decomposes the query into concrete sub-steps
    2. Analyst (Haiku): executes each step against document context
    3. Composer (Haiku): synthesizes findings into a coherent answer

    Cost controls:
    - Token budget shared across all agents (AGENT_MAX_TOTAL_TOKENS)
    - Wall-clock timeout (AGENT_TIMEOUT_SECONDS)
    - Steps capped at AGENT_MAX_STEPS
    - Early termination if budget exhausted

    Fallback:
    - If any agent fails, the pipeline falls back gracefully
    - Planner failure → single "answer directly" step
    - Analyst failure → partial findings noted
    - Composer failure → raw findings concatenated

    Returns AgentPipelineResult with answer, step details, and reasoning summary.
    """
    result = AgentPipelineResult(agent_mode=True)
    budget = TokenBudget()
    t_start = time.perf_counter()

    # --- Reasoning log (internal only, never exposed to user) ---
    reasoning_log: List[str] = []

    try:
        # ==================================================================
        # Step 0: CLARIFIER — ask for clarification if the query is ambiguous
        # Fails safely: on any error, continue to Planner.
        # ==================================================================
        try:
            clar = run_clarifier(
                query=query,
                doc_context_preview=doc_context[:1200],
                source_names=source_names,
                budget=budget,
                generate_fn=generate_fn,
                bedrock_client=bedrock_client,
                prior_messages=prior_messages,
            )
            if clar.step is not None:
                result.steps.append(clar.step)
            reasoning_log.append(f"[Clarifier] {clar.reason}")

            if clar.needs_clarification and clar.questions:
                # Short-circuit: return clarifying questions as the answer.
                # Response structure unchanged — only the `answer` text differs.
                result.answer = clar.to_answer_text()
                result.plan = []
                result.total_ms = int((time.perf_counter() - t_start) * 1000)
                result.total_tokens = budget.used
                reasoning_log.append(
                    f"[Clarifier] Emitting {len(clar.questions)} question(s); "
                    "skipping Planner/Analyst/Composer this turn."
                )
                result.reasoning_summary = " | ".join(reasoning_log)
                logger.info(
                    "Agent pipeline short-circuited by Clarifier: %d question(s), %d tokens, %dms",
                    len(clar.questions), result.total_tokens, result.total_ms,
                )
                return result
        except Exception as clar_exc:
            # Never let a Clarifier bug break the pipeline.
            logger.warning("Clarifier raised, continuing to Planner: %s", clar_exc)
            reasoning_log.append(f"[Clarifier] error — continuing: {clar_exc}")

        # ==================================================================
        # Step 1: PLANNER — decompose the query
        # ==================================================================
        reasoning_log.append(f"[Planner] Decomposing query: '{query[:80]}...'")

        plan_steps, plan_result = run_planner(
            query=query,
            doc_context_preview=doc_context[:2000],
            source_names=source_names,
            budget=budget,
            generate_fn=generate_fn,
            bedrock_client=bedrock_client,
        )

        # ── Fix 7 (b): tiered escalation. When the user has signalled
        # "still not resolved" enough times during the docs stage, the
        # ticket pool has been exhausted — we route the Analyst's first
        # step across runbook/KB chunks by prepending a synthetic step.
        # step_retriever already honours file_type filters keyed on the
        # words "runbook" / "KB" in the step text, so no retriever change
        # is needed here.
        if (
            plan_steps
            and stage == STAGE_DOCS
            and unresolved_count >= UNRESOLVED_ESCALATE_THRESHOLD
        ):
            synthetic = (
                "Step 1: Search runbooks and KBs for the originally-reported "
                "issue and identify the next troubleshooting action not "
                "already attempted."
            )
            plan_steps = [synthetic, *plan_steps]
            reasoning_log.append(
                f"[Planner] Escalation prepend: unresolved={unresolved_count} "
                f">= {UNRESOLVED_ESCALATE_THRESHOLD} — routing to runbooks/KBs first."
            )

        result.steps.append(plan_result)
        result.plan = plan_steps
        reasoning_log.append(f"[Planner] Produced {len(plan_steps)} steps: {[s[:50] for s in plan_steps]}")

        # ── Goal 4: dynamic budget sizing now that we know the plan ──
        # The static AGENT_MAX_TOTAL_TOKENS cap was regularly blowing up on
        # 4-step plans. Compute a plan-aware budget (Clarifier + Planner +
        # N*PerStep + Composer), clamped by AGENT_BUDGET_HARD_CAP_TOKENS so
        # a runaway planner with 50 steps can't explode cost. max() with the
        # existing budget ensures we never shrink what's already in flight.
        if settings.ENABLE_DYNAMIC_AGENT_BUDGET:
            step_count = len(plan_steps or [])
            computed = (
                settings.AGENT_BUDGET_CLARIFIER_TOKENS
                + settings.AGENT_BUDGET_PLANNER_TOKENS
                + step_count * settings.AGENT_BUDGET_PER_STEP_TOKENS
                + settings.AGENT_BUDGET_COMPOSER_TOKENS
            )
            dynamic_max = min(computed, settings.AGENT_BUDGET_HARD_CAP_TOKENS)
            budget.max_total = max(budget.max_total, dynamic_max)
            logger.info(
                "[agents] dynamic budget: steps=%d computed=%d applied=%d",
                step_count, computed, dynamic_max,
            )

        # Check timeout
        elapsed = time.perf_counter() - t_start
        if elapsed > settings.AGENT_TIMEOUT_SECONDS:
            reasoning_log.append(f"[Timeout] Pipeline timed out after planner ({elapsed:.1f}s)")
            logger.warning("Agent pipeline timed out after planner (%.1fs)", elapsed)
            # Fall through to composer with just the plan
            plan_steps = []

        # ==================================================================
        # Step 2: ANALYST — execute each step
        # ==================================================================
        if plan_steps and not budget.exhausted:
            reasoning_log.append(f"[Analyst] Executing {len(plan_steps)} steps against document context")

            findings, analysis_results = run_analysis(
                steps=plan_steps,
                doc_context=doc_context,
                query=query,
                budget=budget,
                generate_fn=generate_fn,
                bedrock_client=bedrock_client,
                step_retriever_fn=step_retriever_fn,
            )

            result.steps.extend(analysis_results)
            reasoning_log.append(f"[Analyst] Completed {len(analysis_results)} steps, {len(findings)} findings")
        else:
            # Budget exhausted or no steps — use plan as pseudo-findings
            findings = [f"Direct analysis needed: {query}"]
            reasoning_log.append("[Analyst] Skipped (budget exhausted or no steps)")

        # Check timeout
        elapsed = time.perf_counter() - t_start
        if elapsed > settings.AGENT_TIMEOUT_SECONDS:
            reasoning_log.append(f"[Timeout] Pipeline timed out after analyst ({elapsed:.1f}s)")
            logger.warning("Agent pipeline timed out after analyst (%.1fs)", elapsed)

        # ==================================================================
        # Step 3: COMPOSER — synthesize findings into final answer
        # ==================================================================
        if not budget.exhausted:
            reasoning_log.append(f"[Composer] Synthesizing {len(findings)} findings into answer")

            compose_result = run_composer(
                query=query,
                findings=findings,
                source_names=source_names,
                budget=budget,
                generate_fn=generate_fn,
                bedrock_client=bedrock_client,
            )

            result.steps.append(compose_result)
            result.answer = compose_result.output
            reasoning_log.append(f"[Composer] Produced {len(compose_result.output)} char answer")
        else:
            # Budget exhausted — concatenate raw findings
            reasoning_log.append("[Composer] Skipped (budget exhausted), using raw findings")
            result.answer = "Based on the document analysis:\n\n" + "\n\n".join(findings)

    except Exception as exc:
        logger.exception("Agent pipeline failed: %s", exc)
        reasoning_log.append(f"[Error] Pipeline failed: {exc}")
        result.answer = "Error during multi-step analysis. Please try again."

    # --- Finalize ---
    result.total_ms = int((time.perf_counter() - t_start) * 1000)
    result.total_tokens = budget.used
    result.reasoning_summary = " | ".join(reasoning_log)

    logger.info(
        "Agent pipeline complete: %d steps, %d tokens, %dms | %s",
        len(result.steps), result.total_tokens, result.total_ms,
        result.reasoning_summary,
    )

    return result
