"""
Agent Orchestrator — decides whether agent mode is needed and runs the pipeline.
Entry points: should_escalate_to_agents() and run_agent_pipeline().
Only complex queries matching specific patterns trigger agents.
Simple and moderate queries are never touched by this module.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
# Sprint 3C — mode → corpus (doc_kind) policy.
#
# Returns the doc_kinds filter list the /ask retrieval call should pass
# to orchestrator_retrieve, or None to leave retrieval unfiltered (i.e.
# the pre-3C ticket-history default). Today only Escalation mode has a
# corpus route; future sprints (3D ticket_handling, 3E vendor_oem) will
# add their own branches here. Kept as a standalone helper so the
# policy lives in agents/orchestrator.py (where the rest of mode logic
# lives) while the retrieval call itself stays in api.py.
#
# Flag-off: always returns None — pre-3C byte-identical behavior.
# ---------------------------------------------------------------------------
def resolve_mode_doc_kinds(session_mode: Any) -> Optional[List[str]]:
    if session_mode is None:
        return None

    mode_name: Optional[str] = None
    if isinstance(session_mode, str):
        mode_name = session_mode
    else:
        mode_name = getattr(session_mode, "selected_mode", None)

    if not mode_name:
        return None

    if mode_name == "escalation":
        return ["contact_customer"]

    # Sprint 3D — Ticket Handling mode routes to the SOP/runbook corpus.
    # Sub-mode (create/update/close/validate) does NOT change the corpus
    # filter — it only shapes the composer voice. All four sub-modes
    # retrieve from the same doc_kind=sop body.
    if mode_name == "ticket_handling":
        return ["sop"]

    # Sprint 3E — Vendor/OEM mode routes to the vendor corpora: contact
    # records (contact_vendor) AND prior vendor case records
    # (vendor_case). The composer voice then produces the three-part
    # response (case writeup + contact card + prior cases). Both
    # doc_kinds are registered in VALID_DOC_KINDS.
    if mode_name == "vendor_oem":
        return ["contact_vendor", "vendor_case"]

    return None


# ---------------------------------------------------------------------------
# Agent pipeline performance additions (additive, flag-gated).
# All behavior preserved when the matching feature flag is False.
# ---------------------------------------------------------------------------

# Dependency markers — steps mentioning these phrases are treated as
# dependent on earlier findings and run sequentially after the parallel wave.
_ANALYST_DEPENDENCY_PHRASES = (
    "based on above",
    "from previous",
    "from step",
    "using findings",
    "synthesize",
    "combine",
    "cross-reference",
)

# Analytical / compare regex — triggers the AGENT_ANALYTICAL_BUDGET bump
# when present in the user query and no pattern_context is active. Matches
# the analytical synthesis queries the Composer would otherwise truncate.
_ANALYTICAL_BUDGET_TRIGGERS = re.compile(
    r"\b(?:compare|contrast|versus|vs\.?|common|recurring|pattern|patterns|"
    r"themes?|trends?|typical|across|overall|root causes|improvements|"
    r"takeaways|insights|findings|gaps|weaknesses|recommendations|"
    r"deep\s+analysis|comprehensive\s+analysis|full\s+analysis)\b",
    re.I,
)

# Deep-mode analytical triggers. Mirrors the cross_cutting_detector's deep
# markers — duplicated here to avoid a new import dependency and keep this
# module self-contained.
_ANALYTICAL_DEEP_TRIGGERS = (
    "deep analysis",
    "deeper analysis",
    "detailed analysis",
    "comprehensive analysis",
    "full analysis",
    "all tickets in detail",
)


def _detect_analytical_mode(query: str) -> str:
    """Return 'deep' when the query explicitly requests depth, else 'fast'."""
    if not query:
        return "fast"
    q = query.lower()
    for trigger in _ANALYTICAL_DEEP_TRIGGERS:
        if trigger in q:
            return "deep"
    return "fast"


def _compute_dynamic_agent_budget(
    *,
    ticket_count: int,
    step_count: int,
    mode: str = "fast",
) -> int:
    """
    Compute a query-specific agent budget from complexity signals.

    Formula:
        base + (ticket_count × per_ticket) + (step_count × per_step)
        × deep_multiplier when mode == "deep"
        Clamped to [floor, ceiling]

    All six parameters (base / per_ticket / per_step / floor / ceiling /
    deep_multiplier) are independently tunable via config flags so the
    formula can be retuned without code changes.

    When AGENT_DYNAMIC_BUDGET_ENABLED is False, returns the static
    AGENT_ANALYTICAL_BUDGET — byte-identical pre-brief behavior.
    """
    if not getattr(settings, "AGENT_DYNAMIC_BUDGET_ENABLED", True):
        static_budget = getattr(settings, "AGENT_ANALYTICAL_BUDGET", 25000)
        logger.info(
            "[dynamic_budget] DISABLED -> using static budget=%d",
            static_budget,
        )
        return static_budget

    base = getattr(settings, "AGENT_DYNAMIC_BUDGET_BASE", 8000)
    per_ticket = getattr(settings, "AGENT_DYNAMIC_BUDGET_PER_TICKET", 1500)
    per_step = getattr(settings, "AGENT_DYNAMIC_BUDGET_PER_STEP", 2000)
    floor = getattr(settings, "AGENT_DYNAMIC_BUDGET_FLOOR", 10000)
    ceiling = getattr(settings, "AGENT_DYNAMIC_BUDGET_CEILING", 35000)
    deep_multiplier = getattr(
        settings, "AGENT_DYNAMIC_BUDGET_DEEP_MULTIPLIER", 1.5,
    )

    safe_ticket_count = max(0, int(ticket_count or 0))
    safe_step_count = max(1, int(step_count or 1))

    computed = base + (safe_ticket_count * per_ticket) + (safe_step_count * per_step)

    if mode == "deep":
        computed = int(computed * deep_multiplier)

    final = max(floor, min(computed, ceiling))

    logger.info(
        "[dynamic_budget] tickets=%d steps=%d mode=%s -> computed=%d final=%d "
        "(floor=%d ceiling=%d)",
        safe_ticket_count, safe_step_count, mode, computed, final, floor, ceiling,
    )

    return final


def _get_step_dependencies(step: Any) -> List[str]:
    """
    Extract dependency signals from a planner step's text.
    Accepts either a str or a dict (description/text fields).
    Steps referencing prior findings run sequentially after wave 1.
    """
    if isinstance(step, dict):
        step_text = (step.get("description") or step.get("text", "") or "")
    else:
        step_text = str(step or "")
    step_text = step_text.lower()
    return [phrase for phrase in _ANALYST_DEPENDENCY_PHRASES if phrase in step_text]


def _make_retrieval_cache_key(identifiers: List[str]) -> str:
    """Order-independent cache key from identifier list."""
    if not identifiers:
        return ""
    return "|".join(sorted(str(i) for i in identifiers))


class PipelineRetrievalCache:
    """
    In-memory retrieval cache scoped to a single pipeline run.

    Cleared between pipeline runs — no cross-query contamination. Reduces
    redundant retrieval work when analyst steps reference the same
    identifier set. Thread-safe for concurrent analyst steps.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._hit_count = 0
        self._miss_count = 0

    def get(self, identifiers: List[str]) -> Optional[Any]:
        if not getattr(settings, "AGENT_RETRIEVAL_CACHE_ENABLED", True):
            return None
        key = _make_retrieval_cache_key(identifiers)
        if not key:
            return None
        with self._lock:
            if key in self._cache:
                self._hit_count += 1
                logger.debug(
                    "[retrieval_cache] hit key=%s (total hits=%d)",
                    key[:40], self._hit_count,
                )
                return self._cache[key]
            self._miss_count += 1
            return None

    def put(self, identifiers: List[str], value: Any) -> None:
        if not getattr(settings, "AGENT_RETRIEVAL_CACHE_ENABLED", True):
            return
        key = _make_retrieval_cache_key(identifiers)
        if not key:
            return
        with self._lock:
            self._cache[key] = value

    def stats(self) -> Dict[str, float]:
        with self._lock:
            total = self._hit_count + self._miss_count
            return {
                "hits": self._hit_count,
                "misses": self._miss_count,
                "hit_rate": self._hit_count / total if total > 0 else 0.0,
            }


def _wrap_step_retriever_with_session_scope(
    original_fn: Optional[Callable[[str], Any]],
    session_scope: Optional[List[str]],
) -> Optional[Callable[[str], Any]]:
    """Hotfix: prepend session-scope tokens (customer, product, identifier)
    to each analyst step's retrieval text so cross-step context doesn't drift.

    Byte-identical behavior when session_scope is empty OR original_fn is None.
    """
    if original_fn is None:
        return None
    tokens = [t for t in (session_scope or []) if t]
    if not tokens:
        return original_fn
    prefix = " ".join(tokens)

    def _scoped(step_text: str) -> Any:
        try:
            # Only inject when the step doesn't already mention a scope token.
            lower_step = (step_text or "").lower()
            missing = [t for t in tokens if t.lower() not in lower_step]
            if missing:
                enriched = f"{' '.join(missing)} {step_text or ''}".strip()
            else:
                enriched = step_text
            # Sprint 2.7 Bug A — pass ORIGINAL step text as raw_query so
            # retrieve()'s Sprint 2.5 raw-query extraction fires on the
            # dash-preserving form (INC-TITAN-812), not the scope-enriched
            # text. Fall back if the underlying fn doesn't accept raw_query.
            try:
                return original_fn(enriched, raw_query=step_text)
            except TypeError:
                return original_fn(enriched)
        except Exception:
            return original_fn(step_text)

    return _scoped


def _wrap_step_retriever_with_cache(
    original_fn: Optional[Callable[[str], Any]],
    cache: PipelineRetrievalCache,
) -> Optional[Callable[[str], Any]]:
    """
    Return a step-retriever that checks/populates the pipeline-scoped cache
    using identifier lists extracted from the step text. When the original
    fn is None or the feature flag is disabled, returns the original fn
    unchanged (byte-identical behavior).
    """
    if original_fn is None:
        return None
    if not getattr(settings, "AGENT_RETRIEVAL_CACHE_ENABLED", True):
        return original_fn

    def _cached(step_text: str) -> Any:
        try:
            from backend.retrieval.orchestrator import _extract_identifiers
            ids = [cid for cid, _t in (_extract_identifiers(step_text) or [])]
        except Exception:
            ids = []

        if ids:
            cached = cache.get(ids)
            if cached is not None:
                return cached

        # Sprint 2.7 Bug A — pass step text as raw_query so the retriever's
        # Sprint 2.5 raw-query extraction fires on analyst sub-steps and
        # preserves dash-delimited identifiers (INC-TITAN-812). Fall back
        # to single-arg invocation if underlying fn doesn't accept kwarg.
        try:
            result = original_fn(step_text, raw_query=step_text)
        except TypeError:
            result = original_fn(step_text)

        if ids and result is not None:
            cache.put(ids, result)
        return result

    return _cached


def _execute_analyst_steps_parallel(
    *,
    plan_steps: List[str],
    doc_context: str,
    query: str,
    budget: TokenBudget,
    generate_fn: Callable,
    bedrock_client: Any,
    step_retriever_fn: Optional[Callable[[str], Any]],
) -> Tuple[List[str], List[Any]]:
    """
    Wave-based parallel analyst execution.

    Wave 1: independent steps run concurrently (semaphore-limited).
    Wave 2: dependent steps (synthesize/combine/cross-reference) run
            sequentially after Wave 1, enriched with prior findings.

    When ANALYST_PARALLEL_EXECUTION_ENABLED is False, this function
    delegates to the original sequential run_analysis() — byte-identical
    behavior to pre-fix pipeline.
    """
    if not getattr(settings, "ANALYST_PARALLEL_EXECUTION_ENABLED", True):
        return run_analysis(
            steps=plan_steps,
            doc_context=doc_context,
            query=query,
            budget=budget,
            generate_fn=generate_fn,
            bedrock_client=bedrock_client,
            step_retriever_fn=step_retriever_fn,
        )

    if not plan_steps:
        return [], []

    independent: List[Tuple[int, Any]] = []
    dependent: List[Tuple[int, Any]] = []
    for idx, step in enumerate(plan_steps):
        if _get_step_dependencies(step):
            dependent.append((idx, step))
        else:
            independent.append((idx, step))

    concurrency = max(1, int(getattr(settings, "ANALYST_PARALLEL_CONCURRENCY", 3)))
    logger.info(
        "[analyst_parallel] %d independent + %d dependent steps (concurrency=%d)",
        len(independent), len(dependent), concurrency,
    )

    findings_by_idx: Dict[int, str] = {}
    results_by_idx: Dict[int, Any] = {}

    def _run_single(step_text: Any, sub_ctx: str) -> Tuple[List[str], List[Any]]:
        return run_analysis(
            steps=[step_text],
            doc_context=sub_ctx,
            query=query,
            budget=budget,
            generate_fn=generate_fn,
            bedrock_client=bedrock_client,
            step_retriever_fn=step_retriever_fn,
        )

    # Wave 1 — independent steps in parallel
    if independent:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_map = {
                executor.submit(_run_single, step, doc_context): idx
                for idx, step in independent
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    sub_findings, sub_results = future.result()
                    if sub_findings:
                        findings_by_idx[idx] = sub_findings[0]
                    if sub_results:
                        results_by_idx[idx] = sub_results[0]
                except Exception as exc:
                    logger.warning(
                        "[analyst_parallel] step %d failed: %s", idx + 1, exc,
                    )

    # Wave 2 — dependent steps sequential, enriched with prior findings
    for idx, step in dependent:
        prior = [findings_by_idx[k] for k in sorted(findings_by_idx)]
        enriched_ctx = (
            f"{doc_context}\n\nPRIOR FINDINGS:\n" + "\n\n".join(prior)
            if prior else doc_context
        )
        try:
            sub_findings, sub_results = _run_single(step, enriched_ctx)
            if sub_findings:
                findings_by_idx[idx] = sub_findings[0]
            if sub_results:
                results_by_idx[idx] = sub_results[0]
        except Exception as exc:
            logger.warning(
                "[analyst_parallel] dependent step %d failed: %s", idx + 1, exc,
            )

    ordered_findings = [findings_by_idx[i] for i in sorted(findings_by_idx)]
    ordered_results = [results_by_idx[i] for i in sorted(results_by_idx)]

    logger.info(
        "[analyst_parallel] completed %d/%d steps",
        len(ordered_results), len(plan_steps),
    )
    return ordered_findings, ordered_results


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
    pattern_context: Optional[Dict[str, Any]] = None,
    session_scope: Optional[List[str]] = None,
    session_mode: Optional[Any] = None,
    doc_kinds: Optional[List[str]] = None,
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

    # Pipeline-scoped retrieval cache (Fix 2). Wraps the incoming
    # step_retriever_fn so identifier-keyed retrieval hits are deduplicated
    # within this single run. When AGENT_RETRIEVAL_CACHE_ENABLED=False the
    # wrapper returns the original fn unchanged — byte-identical behavior.
    pipeline_retrieval_cache = PipelineRetrievalCache()
    step_retriever_fn = _wrap_step_retriever_with_cache(
        step_retriever_fn, pipeline_retrieval_cache,
    )
    # Hotfix: session-scope anchoring. Wrap after the cache wrapper so cache
    # keys remain identifier-based; scope just enriches step text.
    step_retriever_fn = _wrap_step_retriever_with_session_scope(
        step_retriever_fn, session_scope,
    )

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

        # ── Pattern Analytics Polish — Fix 2: pattern-aware budget tuning ──
        # When pattern_context is active, Composer was regularly getting
        # starved ("Skipped (budget exhausted)") because 4 analyst steps ate
        # the whole 19K default. For pattern queries we cap planner steps,
        # bump the hard budget, and reserve composer tokens explicitly.
        # Flag-gated so pattern queries revert to the legacy dynamic budget
        # path below whenever it's flipped False.
        if (
            getattr(settings, "PATTERN_ANALYTICS_BUDGET_TUNING_ENABLED", True)
            and pattern_context
        ):
            max_steps_for_patterns = getattr(
                settings, "PATTERN_ANALYTICS_MAX_PLANNER_STEPS", 3,
            )
            if plan_steps and len(plan_steps) > max_steps_for_patterns:
                logger.info(
                    "[agents] pattern query: reducing planner steps from %d to %d",
                    len(plan_steps), max_steps_for_patterns,
                )
                plan_steps = plan_steps[:max_steps_for_patterns]
                result.plan = plan_steps

            pattern_budget = getattr(
                settings, "PATTERN_ANALYTICS_AGENT_BUDGET", 25000,
            )
            if budget.max_total < pattern_budget:
                logger.info(
                    "[agents] pattern query: bumping budget from %d to %d",
                    budget.max_total, pattern_budget,
                )
                budget.max_total = pattern_budget

            composer_reserve = getattr(
                settings, "PATTERN_ANALYTICS_COMPOSER_RESERVE", 4000,
            )
            step_cnt = max(1, len(plan_steps or []))
            per_step_budget = max(2000, (budget.max_total - composer_reserve) // step_cnt)
            logger.info(
                "[agents] pattern query: budget breakdown total=%d reserve=%d per_step=%d",
                budget.max_total, composer_reserve, per_step_budget,
            )

        # ── Fix 4 / Dynamic Budget: analytical / compare budget bump ──
        # When the query is analytical (cross-cutting synthesis, compare,
        # deep analysis) and the pattern_context path didn't already bump
        # the budget, compute a query-specific budget from complexity
        # signals (ticket count, planner step count, analysis mode) and
        # raise the ceiling so the Composer isn't starved.
        #
        # The dynamic formula replaces the earlier flat AGENT_ANALYTICAL_BUDGET
        # override. AGENT_ANALYTICAL_BUDGET is preserved as the fallback
        # that _compute_dynamic_agent_budget returns when the master
        # AGENT_DYNAMIC_BUDGET_ENABLED flag is False. The trigger gate,
        # pattern_context short-circuit, and log-tag format stay
        # byte-compatible with prior observability tooling.
        _analytical_dynamic_applied = False
        if not pattern_context:
            if _ANALYTICAL_BUDGET_TRIGGERS.search(query or ""):
                ticket_count = len(ranked_chunks) if ranked_chunks else 0
                step_count = len(plan_steps) if plan_steps else 1
                mode = _detect_analytical_mode(query)
                analytical_budget = _compute_dynamic_agent_budget(
                    ticket_count=ticket_count,
                    step_count=step_count,
                    mode=mode,
                )
                if analytical_budget and budget.max_total < analytical_budget:
                    logger.info(
                        "[agents] analytical query: bumping budget from %d to %d",
                        budget.max_total, analytical_budget,
                    )
                    budget.max_total = analytical_budget
                    _analytical_dynamic_applied = True

        # ── Goal 4: dynamic budget sizing now that we know the plan ──
        # The static AGENT_MAX_TOTAL_TOKENS cap was regularly blowing up on
        # 4-step plans. Compute a plan-aware budget (Clarifier + Planner +
        # N*PerStep + Composer), clamped by AGENT_BUDGET_HARD_CAP_TOKENS so
        # a runaway planner with 50 steps can't explode cost. max() with the
        # existing budget ensures we never shrink what's already in flight.
        #
        # DYNAMIC_BUDGET_OVERRIDE_FIX — Fix 1: skip this legacy sizing when
        # the analytical dynamic budget already fired. Its hard-cap clamp
        # (AGENT_BUDGET_HARD_CAP_TOKENS, ~20000) was silently reversing the
        # analytical bump back to ~19000, starving the Composer on deep
        # cross-cutting synthesis. For non-analytical queries, this path is
        # unchanged.
        if settings.ENABLE_DYNAMIC_AGENT_BUDGET and not _analytical_dynamic_applied:
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

        # Hotfix: composer starvation on multi-step non-pattern queries.
        # Reserve an explicit slice for the Composer regardless of whether
        # the query hit pattern_context or the analytical path, so the
        # synthesis step isn't "Skipped (budget exhausted)" on 3+ step plans.
        if (
            not pattern_context
            and plan_steps
            and len(plan_steps) >= 2
        ):
            composer_reserve = int(getattr(
                settings, "HOTFIX_COMPOSER_RESERVE_TOKENS", 5000,
            ))
            needed = budget.used + composer_reserve + 1000
            if budget.max_total < needed:
                logger.info(
                    "[agents] hotfix composer reserve: bumping budget from %d to %d "
                    "(used=%d reserve=%d steps=%d)",
                    budget.max_total, needed, budget.used, composer_reserve,
                    len(plan_steps),
                )
                budget.max_total = needed

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

            findings, analysis_results = _execute_analyst_steps_parallel(
                plan_steps=plan_steps,
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
        # v2 Bug #4: two-pool budget architecture. The Analyst phase may
        # legitimately burn through budget.max_total on deep multi-step
        # work; without a reservation the Composer would be skipped
        # ("budget exhausted"). When AGENT_BUDGET_SEPARATE_COMPOSER_POOL is
        # on, extend max_total by a fresh composer pool right before the
        # synthesis call so the Composer always gets a guaranteed slice
        # regardless of Analyst overrun.
        _two_pool_on = bool(
            getattr(settings, "AGENT_BUDGET_SEPARATE_COMPOSER_POOL", False)
        )
        if _two_pool_on:
            composer_pool = int(getattr(
                settings, "HOTFIX_COMPOSER_RESERVE_TOKENS",
                getattr(settings, "AGENT_BUDGET_COMPOSER_TOKENS", 5000),
            ))
            analyst_used = budget.used
            prior_max = budget.max_total
            new_max = max(prior_max, analyst_used + composer_pool)
            if new_max > prior_max:
                logger.info(
                    "[budget_two_pool] analyst_used=%d composer_pool=%d "
                    "prior_max=%d new_max=%d",
                    analyst_used, composer_pool, prior_max, new_max,
                )
                budget.max_total = new_max

        if not budget.exhausted:
            reasoning_log.append(f"[Composer] Synthesizing {len(findings)} findings into answer")

            compose_result = run_composer(
                query=query,
                findings=findings,
                source_names=source_names,
                budget=budget,
                generate_fn=generate_fn,
                bedrock_client=bedrock_client,
                session_mode=session_mode,
                # doc_kinds activates the KB-Search composer voice when
                # the request restricted retrieval to KB / SOP content.
                # None / ticket-only / mixed kinds leave the composer on
                # its default conversational voice (byte-identical to
                # the pre-change behavior).
                doc_kinds=doc_kinds,
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

    # Fix 2: emit pipeline-scoped retrieval cache stats.
    try:
        _cache_stats = pipeline_retrieval_cache.stats()
        if _cache_stats["hits"] + _cache_stats["misses"] > 0:
            logger.info(
                "[retrieval_cache] pipeline stats: hits=%d misses=%d hit_rate=%.0f%%",
                int(_cache_stats["hits"]), int(_cache_stats["misses"]),
                _cache_stats["hit_rate"] * 100,
            )
    except Exception:
        pass

    logger.info(
        "Agent pipeline complete: %d steps, %d tokens, %dms | %s",
        len(result.steps), result.total_tokens, result.total_ms,
        result.reasoning_summary,
    )

    return result


# ---------------------------------------------------------------------------
# Sprint 3B — 👎 KB/Runbook pivot pipeline.
#
# Fires only when the user thumbs-downs a Troubleshooting-mode answer.
# Re-runs retrieval filtered to doc_kinds=["sop","kb"] using the original
# user query and composes a 5-section runbook-style response via the
# kb_pivot voice override. Bypasses planner and analyst on purpose —
# the KB pivot is a single-shot "here is what the runbook says", not a
# multi-step synthesis. Keeps the pivot cheap and predictable.
#
# Flag-off: raises RuntimeError — callers must flag-gate before invoking.
# ---------------------------------------------------------------------------
def run_kb_pivot_pipeline(
    *,
    query: str,
    session_mode: Any,
    generate_fn: Callable,
    bedrock_client: Any,
    embed_fn: Callable,
    owner_id: str,
    allowed_file_ids: Set[str],
    bm25_search_fn: Optional[Callable] = None,
    vector_search_fn: Optional[Callable] = None,
    budget: Optional[TokenBudget] = None,
) -> AgentPipelineResult:
    """Sprint 3B — retrieve filtered to sop+kb, compose with KB voice.

    Reuses orchestrator_retrieve + run_composer with a different doc_kinds
    filter and voice_override="kb_pivot". Budget is a fresh half of
    AGENT_MAX_TOTAL_TOKENS (so a 👎 does not exhaust the session).
    """
    if budget is None:
        budget = TokenBudget(max_total=max(1, settings.AGENT_MAX_TOTAL_TOKENS // 2))

    result = AgentPipelineResult(agent_mode=True)
    t_start = time.perf_counter()

    # Local import to keep backend.agents.orchestrator free of a
    # compile-time dep on the retrieval package (avoids cycles when
    # tests import this module in isolation).
    from backend.retrieval.orchestrator import retrieve as _retrieve

    try:
        q_emb = embed_fn(query)
        if not q_emb:
            logger.warning("[kb_pivot] embedding failed for query=%r", query[:120])
            result.kb_pivot_empty = True
            result.answer = ""
            return result

        retrieval = _retrieve(
            query=query,
            raw_query=query,
            query_embedding=q_emb,
            owner_id=owner_id,
            allowed_file_ids=set(allowed_file_ids or set()),
            file_type="kb",
            generate_fn=generate_fn,
            bm25_search_fn=bm25_search_fn,
            vector_search_fn=vector_search_fn,
            doc_kinds=["sop", "kb"],
        )
        ranked = list(retrieval.ranked or [])
        logger.info("[kb_pivot] retrieval returned %d chunks", len(ranked))

        if not ranked:
            result.kb_pivot_empty = True
            result.answer = ""
            result.total_ms = int((time.perf_counter() - t_start) * 1000)
            return result

        # Concatenate the top chunks into a single findings block. Each chunk
        # gets a lightweight header so the composer can cite the right KB
        # article by name. We intentionally skip planner/analyst — KB pivot
        # is a single-shot runbook lookup.
        findings_parts: List[str] = []
        source_names: List[str] = []
        for idx, (_cid, ctext, cmeta, _cscore) in enumerate(ranked[:6], start=1):
            doc_name = (cmeta or {}).get("document_name") or (cmeta or {}).get(
                "file_name"
            ) or f"kb_source_{idx}"
            kind = (cmeta or {}).get("doc_kind") or "kb"
            findings_parts.append(
                f"[KB chunk {idx} — {doc_name} (doc_kind={kind})]\n{ctext}"
            )
            source_names.append(doc_name)

        compose_result = run_composer(
            query=query,
            findings=findings_parts,
            source_names=source_names,
            budget=budget,
            generate_fn=generate_fn,
            bedrock_client=bedrock_client,
            session_mode=session_mode,
            voice_override="kb_pivot",
        )
        result.steps.append(compose_result)
        _pivot_answer = (compose_result.output or "").strip()

        # ── Bring kb_pivot under the same validator as /ask ──
        # Before this gate, the 👎 path produced answers that bypassed
        # confidence / grounding / relevancy checks — which is why a
        # thumbs-down sometimes returned a "real" answer when /ask's
        # first attempt got replaced by the safe fallback. That
        # asymmetry trains users to thumbs-down to skip the safety
        # check, which is exactly the wrong behaviour. Run the same
        # validate_answer pass here so kb_pivot inherits identical
        # fabrication-detection and substitution semantics. If the
        # validator rejects (or replaces) the answer, that becomes the
        # kb_pivot output.
        try:
            from backend.validation.validator import validate_answer as _validate_answer
            # doc_context for the validator is the same chunk text the
            # composer saw — concat the findings to reconstruct it.
            _doc_ctx_for_val = "\n\n".join(findings_parts) if findings_parts else ""
            _validation = _validate_answer(
                query=query,
                answer=_pivot_answer,
                doc_context=_doc_ctx_for_val,
                ranked_chunks=ranked[:6],
                source_names=source_names,
                model_used="agents (kb_pivot)",
            )
            _pivot_answer = getattr(_validation, "answer", _pivot_answer)
            logger.info(
                "[kb_pivot] validation passed=%s was_modified=%s confidence=%.3f",
                bool(getattr(_validation, "passed", False)),
                bool(getattr(_validation, "was_modified", False)),
                float(getattr(_validation, "confidence", 0.0) or 0.0),
            )
        except Exception as _val_exc:
            logger.warning(
                "[kb_pivot] validator pass failed (%s) -- returning composer "
                "output unmodified", _val_exc,
            )

        result.answer = _pivot_answer
        result.kb_pivot_empty = False

    except Exception as exc:
        logger.exception("[kb_pivot] pipeline failed: %s", exc)
        result.answer = ""
        result.kb_pivot_empty = True

    result.total_ms = int((time.perf_counter() - t_start) * 1000)
    result.total_tokens = budget.used
    return result
