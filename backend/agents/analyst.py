"""
Analysis Agent — executes plan steps against document context.
Per-step hybrid retrieval is optional: when a step_retriever_fn is passed in,
each step runs its own BM25 + vector + RRF + rerank retrieval. Otherwise the
original global doc_context path is used (fully backward-compatible).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    step_retriever_fn: Optional[Callable[[str], Any]] = None,
) -> Tuple[List[str], List[AgentStepResult]]:
    """
    Execute each planned step and collect findings.

    If `step_retriever_fn` is provided, each step's prompt is built from
    per-step hybrid retrieval results. Otherwise the supplied global
    `doc_context` is used for every step (original behavior).

    Returns:
        (findings_list, step_results_list)
    """
    findings: List[str] = []
    step_results: List[AgentStepResult] = []
    step_retrieval_meta: List[Dict[str, Any]] = []

    # ── Bug 2 fix — identifier scope lock ─────────────────────────────
    # The Planner decomposes "Compare INC-10015 and INC-10033" into generic
    # step strings like "Step 3: Compare the two incidents on key resolution
    # quality dimensions…" — no ticket IDs. When the Analyst hands that raw
    # step text to per-step retrieval, the orchestrator's identifier
    # short-circuit can't fire and BM25/vector returns random tickets.
    # Extract identifiers from the original query once and, for any step
    # that names none of its own, inject the locked ones into the step text
    # passed to retrieval. That makes the orchestrator's short-circuit fire
    # with the correct scope.
    locked_identifiers: List[Tuple[str, str]] = []
    if getattr(settings, "ANALYST_IDENTIFIER_LOCK_ENABLED", True):
        try:
            from backend.retrieval.orchestrator import _extract_identifiers
            locked_identifiers = _extract_identifiers(query) or []
        except Exception as _id_exc:
            logger.warning("[analyst] identifier extraction failed (%s)", _id_exc)
            locked_identifiers = []
    locked_canonical = [cid for cid, _ in locked_identifiers]
    logger.info(
        "[analyst] identifier scope: %s (carried across all %d steps)",
        locked_canonical, len(steps),
    )

    for i, step in enumerate(steps):
        if budget.exhausted:
            logger.warning("Analysis agent stopping at step %d/%d: budget exhausted", i + 1, len(steps))
            break

        # ── Goal 4: reserve Composer tokens ──
        # Always leave at least AGENT_BUDGET_COMPOSER_TOKENS in the budget
        # so the Composer can run and synthesize findings. Without this
        # reserve, long Analyst loops burn the pool and the user sees raw
        # per-step findings instead of a clean answer.
        if (
            settings.ENABLE_DYNAMIC_AGENT_BUDGET
            and budget.remaining <= settings.AGENT_BUDGET_COMPOSER_TOKENS
        ):
            logger.info(
                "[agents] analyst stopped early at step %d/%d to reserve Composer tokens",
                i + 1, len(steps),
            )
            break

        # ── Per-step retrieval (optional) ──
        step_ctx = doc_context
        step_sources: List[str] = []
        step_applied = False
        step_reason = "fallback_global_context"

        # Bug 2 fix — inject locked identifiers into the step text when the
        # step itself names none. The retriever (and downstream orchestrator)
        # re-runs _extract_identifiers on whatever string we hand it, so the
        # cheapest thread-through is to augment the text. The Analyst prompt
        # below still receives the original `step` text so the LLM isn't
        # confused by the injected tag.
        step_for_retrieval = step
        if locked_identifiers:
            try:
                from backend.retrieval.orchestrator import _extract_identifiers
                step_has_ids = bool(_extract_identifiers(step))
            except Exception:
                step_has_ids = False
            if not step_has_ids:
                logger.info(
                    "[analyst] step has no identifiers — reusing locked scope: %s",
                    locked_canonical,
                )
                step_for_retrieval = f"{step} [scope: {', '.join(locked_canonical)}]"

        if step_retriever_fn is not None:
            try:
                sr = step_retriever_fn(step_for_retrieval)
                if sr and getattr(sr, "applied", False) and getattr(sr, "doc_context", ""):
                    step_ctx = sr.doc_context
                    step_sources = list(getattr(sr, "source_names", []) or [])
                    step_applied = True
                    step_reason = getattr(sr, "reason", "ok")
                else:
                    step_reason = getattr(sr, "reason", "not_applied") if sr else "no_result"
            except Exception as exc:
                logger.warning("Step retrieval raised for step %d (%s) — falling back", i + 1, exc)
                step_reason = f"error: {exc}"

        step_retrieval_meta.append({
            "index": i + 1,
            "applied": step_applied,
            "reason": step_reason,
            "source_count": len(step_sources),
        })

        sources_hint = ""
        if step_applied and step_sources:
            sources_hint = f"\nSOURCES FOR THIS STEP: {', '.join(sorted(set(step_sources))[:6])}"

        # Bug 2 safety belt — when the original query named specific tickets,
        # surface them in the prompt so generic step phrasings like
        # "Compare the two incidents" ground correctly.
        scope_hint = ""
        if locked_canonical:
            scope_hint = (
                f"\nLOCKED TICKET SCOPE: {', '.join(locked_canonical)}"
                f"\nWhen the step refers to \"the two incidents\", \"each ticket\","
                f" or similar, those refer to the locked scope above."
            )

        prompt = f"""You are a technical analysis agent. Answer the following analysis step
using ONLY the document context provided below. Do NOT use outside knowledge.

If the documents do not contain enough information for this step, say:
"Insufficient evidence in documents for this step."

Keep your response concise (3-6 bullet points max).

DOCUMENT CONTEXT:
{step_ctx}{sources_hint}

ORIGINAL USER QUESTION: {query}{scope_hint}

ANALYSIS STEP {i + 1}/{len(steps)}: {step}

FINDINGS:"""

        # Brief 5 / Part 2 — each analyst step is explanatory extraction
        # (bullet findings), not synthesis. Cap at the explanation tier to
        # drop the old 1500 blanket per step.
        _analyst_max_tokens = (
            settings.RESPONSE_TOKENS_EXPLANATION
            if settings.RESPONSE_TOKEN_CAPS_ENABLED
            else settings.AGENT_ANALYSIS_MAX_TOKENS
        )
        step_result = invoke_llm(
            prompt=prompt,
            model=settings.AGENT_ANALYSIS_MODEL,
            max_tokens=_analyst_max_tokens,
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
            "Analysis step %d/%d complete: success=%s, per_step_retrieval=%s, %d tokens, %dms",
            i + 1, len(steps), step_result.success, step_applied,
            step_result.tokens_used, step_result.duration_ms,
        )

    logger.info(
        "Analysis complete: %d/%d steps executed, %d findings, per_step_retrievals=%d",
        len(step_results), len(steps), len(findings),
        sum(1 for m in step_retrieval_meta if m["applied"]),
    )

    # Attach per-step retrieval metadata to the first step_result for downstream access
    if step_results:
        try:
            setattr(step_results[0], "step_retrieval_meta", step_retrieval_meta)
        except Exception:
            pass

    return findings, step_results
