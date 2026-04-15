"""
Step Retriever — wraps hybrid retrieval (expand_query + RRF + rerank) so the
Analyst can fetch fresh, step-specific chunks per plan step. All dependencies
(embed/vector/bm25/retrieve/expand) are injected to avoid import cycles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("acadia-log-iq")


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
    def _retrieve(step_text: str) -> StepRetrievalResult:
        step_text = (step_text or "").strip()
        if not step_text:
            return StepRetrievalResult(step_text=step_text, reason="empty_step")

        try:
            # --- 1. expand query (acronyms, normalization, variants) ---
            try:
                expanded = expand_query_fn(step_text)
                expanded_text = getattr(expanded, "expanded_text", step_text) or step_text
            except Exception as exc:
                logger.warning("Step retriever: expand_query failed (%s) — using raw step", exc)
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
            retrieval = retrieve_fn(
                query=expanded_text,
                query_embedding=q_emb,
                owner_id=owner_id,
                allowed_file_ids=allowed_file_ids,
                file_type=file_type,
                generate_fn=generate_fn,
                bm25_search_fn=bm25_search_fn,
                vector_search_fn=vector_search_fn,
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
