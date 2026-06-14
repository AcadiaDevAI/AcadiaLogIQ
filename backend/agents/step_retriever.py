"""
Step Retriever — wraps hybrid retrieval (expand_query + RRF + rerank) so the
Analyst can fetch fresh, step-specific chunks per plan step. All dependencies
(embed/vector/bm25/retrieve/expand) are injected to avoid import cycles.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("acadia-log-iq")


# expand_query / the glossary store is designed for HUMAN queries — it
# injects acronym definitions and uppercases canonical tokens. When run
# on Planner-generated sub-step text (e.g. "Step 2: Locate documentation
# on BGP attributes…") it produces nonsense like
# "Step 2: Locate LOCATE documentation on BGP attributes AS (Internal
# peers are BGP peers that are in the same Autonomous System) Override".
# Those parenthetical injections then bias the Analyst's vector
# embedding and BM25 channels toward chunks that aren't actually about
# the user's question. The fix: detect Planner-formatted sub-step text
# and skip glossary expansion. Match a leading "Step N:" or "Step N -"
# at the start of the trimmed text.
_PLANNER_STEP_PREFIX_RE = re.compile(r"^\s*Step\s+\d+\s*[:\-]\s", re.IGNORECASE)


def _looks_like_planner_step(text: str) -> bool:
    """True when `text` is a Planner-generated sub-step string."""
    return bool(text) and bool(_PLANNER_STEP_PREFIX_RE.match(text))


# Planner steps frequently name source files inline ("Step 3: Check the
# Routing_and_Switching_Knowledge_Base_Volume_2CCIE_short_notes.pdf for
# ..."). The orchestrator's identifier extractor treats those filename
# tokens as record IDs and fires its short-circuit (`identifier_exact_
# search`) against a `primary_id` that does not exist. The short-circuit
# inevitably fails over to the hybrid pipeline, but two bad things have
# already happened by then: (a) the per-step retrieval has been slowed
# by a doomed exact lookup, and (b) the model has been primed to think
# it is working from a named, structured source — which encourages it
# to invent ticket-shaped identifiers ("INC-LAN-88902") to look
# concrete. The sanitizer below strips obvious filename / source-doc
# tokens from the `raw_query` passed to the orchestrator JUST FOR
# Planner sub-step text. The query parameter (expanded_text) keeps the
# filename so semantic retrieval still benefits from it; only the
# identifier-extraction input is cleaned. Real identifiers the user
# scoped (e.g. INC-TITAN-812) are still preserved via the
# Analyst's `locked_identifiers` injection in analyst.py.
_FILENAME_TOKEN_RE = re.compile(
    r"\b[\w\-]+\.(?:pdf|docx?|txt|json|csv|ya?ml|md|html?|xml|tsv|xlsx?)\b",
    re.IGNORECASE,
)
# Long underscore-chained names like
# Routing_and_Switching_Knowledge_Base_Volume_2CCIE_short_notes —
# 4+ underscore-joined word parts. Real record IDs almost never have
# more than 2 underscores.
_LONG_UNDERSCORE_TOKEN_RE = re.compile(r"\b[A-Za-z]\w*(?:_\w+){3,}\b")


def _sanitize_for_identifier_extraction(text: str) -> str:
    """Strip filename / source-document tokens from `text` so they do not
    end up extracted as record identifiers downstream. Safe on None / empty."""
    if not text:
        return text or ""
    cleaned = _FILENAME_TOKEN_RE.sub(" ", text)
    cleaned = _LONG_UNDERSCORE_TOKEN_RE.sub(" ", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
RankedChunk = Tuple[str, str, Dict[str, Any], float]  # (text, source_name, metadata, score)


@dataclass
class StepRetrievalResult:
    """
    step_text        — the step string the retrieval was for
    ranked           — per-step ranked chunks (same shape as /ask retrieval)
    doc_context      — assembled context string ready to drop into the prompt
    source_names     — unique sources used in this step
    applied          — True if per-step retrieval ran successfully (else fallback)
    reason           — short human-readable explanation (for telemetry)
    """
    step_text: str = ""
    ranked: List[RankedChunk] = field(default_factory=list)
    doc_context: str = ""
    source_names: List[str] = field(default_factory=list)
    applied: bool = False
    reason: str = ""


# ---------------------------------------------------------------------------
# Factory — builds a closure Analyst can call with just a step string
# ---------------------------------------------------------------------------
def build_step_retriever(
    *,
    owner_id: Optional[str],
    allowed_file_ids: Optional[Set[str]],
    file_type: str,
    embed_fn: Callable[[str], Optional[List[float]]],
    bm25_search_fn: Optional[Callable],
    vector_search_fn: Callable,
    generate_fn: Callable,
    retrieve_fn: Callable,
    expand_query_fn: Callable,
    assemble_context_fn: Callable,
    max_chunks: int = 6,
    max_context_chars: int = 6000,
) -> Callable[[str], StepRetrievalResult]:
    """
    Build a callable: `retriever(step_text) -> StepRetrievalResult`.

    Uses the same expand_query + hybrid (BM25 + vector + RRF + rerank) stack
    as the /ask endpoint. Never raises — any failure yields applied=False so
    the Analyst can fall back to the pre-assembled global doc_context.
    """
    def _retrieve(
        step_text: str,
        raw_query: Optional[str] = None,
    ) -> StepRetrievalResult:
        # `raw_query` was added to fix the silent TypeError that was firing
        # on every agent step. The orchestrator's cache wrapper at
        # `backend/agents/orchestrator.py:327` calls this function with
        # `raw_query=step_text` so the underlying orchestrator can extract
        # dash-delimited identifiers (INC-TITAN-812) WITHOUT the query
        # rewriter's normalization eating the dashes. Before this fix the
        # call raised TypeError every time → orchestrator caught it and
        # retried without raw_query → identifier preservation was lost on
        # all agent steps AND each step paid for an extra exception cycle.
        # Defaulting `raw_query=None` keeps direct callers (non-cache path)
        # working unchanged.
        step_text = (step_text or "").strip()
        if not step_text:
            return StepRetrievalResult(step_text=step_text, reason="empty_step")

        try:
            # --- 1. expand query (acronyms, normalization, variants) ---
            # Skip expansion for Planner-generated sub-step text — the
            # glossary store is calibrated for human queries and pollutes
            # planner output with mid-sentence acronym definitions and
            # uppercase canonical tokens (see module-level comment on
            # `_looks_like_planner_step`).
            if _looks_like_planner_step(step_text):
                expanded_text = step_text
                logger.debug(
                    "Step retriever: skipping glossary expansion for "
                    "planner-step text (%s...)", step_text[:60],
                )
            else:
                try:
                    expanded = expand_query_fn(step_text)
                    expanded_text = getattr(expanded, "expanded_text", step_text) or step_text
                except Exception as exc:
                    logger.warning("Step retriever: expand_query failed (%s) -- using raw step", exc)
                    expanded_text = step_text

            # --- 2. embed the expanded step ---
            q_emb = embed_fn(expanded_text) if expanded_text else None
            if not q_emb and expanded_text != step_text:
                q_emb = embed_fn(step_text)
            if not q_emb:
                return StepRetrievalResult(
                    step_text=step_text,
                    reason="embed_failed",
                )

            # --- 3. hybrid retrieval ---
            # For Planner sub-step text, strip filename / source-doc
            # tokens out of the raw_query the orchestrator uses for
            # identifier extraction. Keeps semantic retrieval intact
            # (the `query=expanded_text` arg still carries the
            # filename), but stops the identifier extractor from
            # treating "Routing_and_Switching_Knowledge_Base_Volume_
            # 2CCIE_short_notes" as a record ID and firing a doomed
            # short-circuit lookup. Non-Planner callers pass through
            # unchanged so genuine identifiers (INC-FOO-123) are still
            # extracted.
            _raw_for_extract = raw_query or step_text
            if _looks_like_planner_step(step_text):
                _raw_for_extract = _sanitize_for_identifier_extraction(_raw_for_extract)
            retrieval = retrieve_fn(
                query=expanded_text,
                query_embedding=q_emb,
                owner_id=owner_id,
                allowed_file_ids=allowed_file_ids,
                file_type=file_type,
                generate_fn=generate_fn,
                bm25_search_fn=bm25_search_fn,
                vector_search_fn=vector_search_fn,
                # Pass through the raw (pre-normalization) text so the
                # orchestrator's identifier extractor sees INC-FOO-123
                # rather than 'INC FOO 123' (expand_query normalizes
                # dashes to spaces). For Planner sub-step text the
                # filename / source-doc tokens are stripped first
                # (see comment above).
                raw_query=_raw_for_extract,
            )
            ranked = list(getattr(retrieval, "ranked", []) or [])
            if not ranked:
                return StepRetrievalResult(
                    step_text=step_text,
                    reason="no_hits",
                )

            # --- 4. cap to top-N for cost control ---
            ranked = ranked[: max(1, int(max_chunks))]

            # --- 5. assemble a step-sized context string ---
            try:
                doc_ctx, doc_src = assemble_context_fn(ranked, max_context_chars)
            except Exception as exc:
                logger.warning("Step retriever: assemble_context failed (%s) — naive join", exc)
                doc_ctx = "\n\n".join(row[0] for row in ranked if row and row[0])
                doc_src = sorted({row[1] for row in ranked if row and len(row) > 1 and row[1]})

            return StepRetrievalResult(
                step_text=step_text,
                ranked=ranked,
                doc_context=doc_ctx or "",
                source_names=list(doc_src or []),
                applied=True,
                reason="ok",
            )

        except Exception as exc:
            logger.warning("Step retriever failed for step=%r (%s)", step_text[:80], exc)
            return StepRetrievalResult(
                step_text=step_text,
                reason=f"error: {exc}",
            )

    return _retrieve
