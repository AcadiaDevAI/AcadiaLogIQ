"""
Empty-Result Fallback Chain — Closes the "zero-chunks → hallucination" hole.

Problem
-------
When hybrid retrieval (vector + BM25 + keyword + metadata) returns zero
chunks, today's chat endpoint passes an empty context to the LLM and the
model fabricates an answer that sounds confident. This module replaces
that path with an explicit three-stage fallback:

  Stage 1 — LLM query rewrite + retry
    Ask Haiku to generate N semantically-different phrasings of the
    original query (different keywords, alternate framings) and rerun
    the existing orchestrator on each. First non-empty result wins.

  Stage 2 — BM25-only last-ditch sweep
    If vector + hybrid all returned nothing, try pure lexical search
    one more time on the original query. Useful for very specific
    identifiers or vocabulary that vector embeddings under-weight.

  Stage 3 — Graceful decline
    Return a sentinel `RetrievalResult` whose stats carry
    `search_mode="empty_after_fallback"`. The chat endpoint MUST honor
    this sentinel and return the canned `EMPTY_FALLBACK_DECLINE_MESSAGE`
    rather than calling the LLM with empty context.

Design notes
------------
* This module deliberately does NOT replace the existing
  `expand_query` variant-retry path in `api.py`. That path runs first
  (acronym/glossary variants are cheap and often sufficient). This
  fallback runs AFTER variants have failed — so we only pay the Haiku
  rewrite cost when both vector and lexical search have already lost.

* The orchestrator function is injected as a callable (`retrieve_fn`)
  rather than imported at module-load time. This keeps the module
  unit-testable and avoids any circular import risk with
  `backend.retrieval.orchestrator`.

* All three stages are individually disable-able via config flags so
  operators can isolate behavior on a regression.

* Failure of stage 1 (Haiku call exception) or stage 2 (BM25 search
  exception) is non-fatal: we log and proceed to the next stage. The
  worst case is "all three stages logged failures" → we still return
  the decline sentinel and the chat endpoint declines gracefully.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from backend.config import settings

logger = logging.getLogger("acadia-log-iq")


# Constant lifted out so the api.py short-circuit can check against a
# stable string instead of typing "empty_after_fallback" inline.
SEARCH_MODE_EMPTY_AFTER_FALLBACK = "empty_after_fallback"


@dataclass
class FallbackOutcome:
    """
    Why a fallback attempt is being returned to the caller. Carries
    enough diagnostics to populate `[fallback_chain]` log lines and
    the RetrievalResult.stats dict.
    """

    stage_reached: str            # "rewrite", "bm25_only", "decline"
    rewrites_tried: int = 0
    bm25_tried: bool = False
    chunks_recovered: int = 0
    rewrite_used: Optional[str] = None
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 1 — LLM query rewrite
# ---------------------------------------------------------------------------
def generate_query_rewrites(
    original_query: str,
    *,
    count: int = 2,
) -> List[str]:
    """
    Ask Claude Haiku for `count` alternative phrasings of the failed
    query. The rewrites should use DIFFERENT keywords and reframe the
    question — not just normalize acronyms (that's `expand_query`'s job).

    Returns
    -------
    List[str]
        Validated rewrites. May be shorter than `count` on partial
        success; returns [] on Haiku failure. The function never raises
        — fallback proceeds to stage 2 on error.
    """
    if count <= 0:
        return []

    # Local import keeps this module importable in tests without boto3.
    try:
        from backend.services.bedrock_haiku import haiku_client
    except Exception as exc:  # boto3 missing, etc.
        logger.warning("[fallback_chain] haiku import failed: %s", exc)
        return []

    system = (
        "You generate alternative phrasings of search queries. "
        "Return strict JSON only. No markdown fences. No commentary."
    )
    prompt = (
        f"A semantic search of an internal knowledge base returned zero "
        f"results for this user question:\n\n"
        f"\"{original_query}\"\n\n"
        f"Generate {count} alternative phrasings that:\n"
        f"- Use different keywords or synonyms\n"
        f"- Reframe the question (e.g. statement form vs. question form)\n"
        f"- Keep the original intent and any named entities/IDs\n"
        f"- Are concise (one sentence each)\n\n"
        f"Return a JSON array of {count} strings. Example output:\n"
        f"[\"first rewrite\", \"second rewrite\"]\n\n"
        f"JSON array only:"
    )

    try:
        result = haiku_client.invoke_json(
            system=system,
            prompt=prompt,
            max_tokens=512,
            context="retrieval_fallback",
        )
    except Exception as exc:
        logger.warning("[fallback_chain] haiku rewrite failed: %s", exc)
        return []

    # Be lenient with the response shape — Haiku may return a list, or
    # a dict with one of several common keys. Mirrors the parsing done
    # by `_llm_discover_sections` in structured_parser.
    candidates: List[Any] = []
    if isinstance(result, list):
        candidates = result
    elif isinstance(result, dict):
        for key in ("rewrites", "queries", "phrasings", "results"):
            if isinstance(result.get(key), list):
                candidates = result[key]
                break

    rewrites: List[str] = []
    seen = {original_query.strip().lower()}
    for item in candidates:
        text = str(item).strip()
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        rewrites.append(text)
        if len(rewrites) >= count:
            break

    logger.info(
        "[fallback_chain] generated %d rewrite(s) for query=%r",
        len(rewrites), original_query[:80],
    )
    return rewrites


# ---------------------------------------------------------------------------
# Stage 2 — BM25-only sweep
# ---------------------------------------------------------------------------
def bm25_only_sweep(
    query: str,
    *,
    bm25_search_fn: Optional[Callable],
    allowed_file_ids: Optional[Set[str]],
    file_type: str = "kb",
    top_k: Optional[int] = None,
) -> List[Tuple[str, str, Dict[str, Any], float]]:
    """
    Run pure BM25 search bypassing the full orchestrator pipeline.
    The return shape matches `FusedResult` so the caller can drop it
    straight into a RetrievalResult.ranked list.

    Caller is responsible for honoring allowed_file_ids; we filter here
    too as a defence-in-depth (BM25 is corpus-wide by default).
    """
    if bm25_search_fn is None:
        logger.info("[fallback_chain] bm25 search unavailable — skipping stage 2")
        return []

    k = top_k if top_k is not None else settings.EMPTY_FALLBACK_BM25_TOP_K

    try:
        raw = bm25_search_fn(query, n_results=k, file_type=file_type)
    except Exception as exc:
        logger.warning("[fallback_chain] bm25 sweep raised: %s", exc)
        return []

    hits: List[Tuple[str, str, Dict[str, Any], float]] = []
    for chunk_id, text_value, meta, score in raw or []:
        if allowed_file_ids is not None:
            meta_file_id = (meta or {}).get("file_id")
            if meta_file_id not in allowed_file_ids:
                continue
        hits.append((chunk_id, text_value, meta or {}, float(score or 0.0)))

    logger.info(
        "[fallback_chain] bm25-only sweep produced %d hits for query=%r",
        len(hits), query[:80],
    )
    return hits


# ---------------------------------------------------------------------------
# High-level orchestrator — runs all three stages in sequence
# ---------------------------------------------------------------------------
def attempt_fallback(
    *,
    original_query: str,
    original_retrieval: Any,           # RetrievalResult from the failed call
    retrieve_fn: Callable,             # backend.retrieval.orchestrator.retrieve
    embed_fn: Callable,                # callable(text) -> List[float]
    bm25_search_fn: Optional[Callable],
    owner_id: str,
    allowed_file_ids: Set[str],
    file_type: str = "kb",
    generate_fn: Optional[Callable] = None,
    vector_search_fn: Optional[Callable] = None,
    doc_kinds: Optional[List[str]] = None,
) -> Any:
    """
    Run the three-stage fallback chain. Returns either:
      * A `RetrievalResult` with `.ranked` populated (stages 1 or 2 hit).
      * The `original_retrieval` mutated with
        `stats["search_mode"] = "empty_after_fallback"` and an empty
        `.ranked` list (stage 3 — caller declines gracefully).

    The function NEVER raises — every external call is wrapped, every
    failure logged and routed to the next stage.

    Parameters
    ----------
    original_query : str
        The user's question (post-history-enrichment) that produced
        zero chunks. Used by stage 1 and stage 2.
    original_retrieval : RetrievalResult
        The failed RetrievalResult. We mutate this in stage 3 rather
        than constructing a new one, so the caller's downstream code
        keeps seeing the same object with the same .stats / .intent.
    retrieve_fn : callable
        backend.retrieval.orchestrator.retrieve — called per rewrite.
    embed_fn : callable
        Text → embedding vector. Used to embed each rewrite.
    bm25_search_fn : callable | None
        bm25.search bound method. None disables stage 2.

    All remaining kwargs are passed through to `retrieve_fn` so the
    rewrite-retry path uses the same access control / doc_kinds filters
    as the original call.
    """
    outcome = FallbackOutcome(stage_reached="rewrite")

    if not settings.ENABLE_EMPTY_RESULT_FALLBACK:
        outcome.notes.append("fallback_disabled")
        return _mark_declined(original_retrieval, outcome)

    # ---- Stage 1: Haiku rewrite + retry ------------------------------
    rewrites = generate_query_rewrites(
        original_query,
        count=settings.EMPTY_FALLBACK_LLM_REWRITE_COUNT,
    )
    outcome.rewrites_tried = len(rewrites)

    for rewrite in rewrites:
        try:
            rewrite_emb = embed_fn(rewrite)
        except Exception as exc:
            logger.warning("[fallback_chain] embed failed for rewrite: %s", exc)
            continue
        if not rewrite_emb:
            continue

        try:
            retried = retrieve_fn(
                query=rewrite,
                raw_query=original_query,    # keep original for identifier detection
                query_embedding=rewrite_emb,
                owner_id=owner_id,
                allowed_file_ids=allowed_file_ids,
                file_type=file_type,
                generate_fn=generate_fn,
                bm25_search_fn=bm25_search_fn,
                vector_search_fn=vector_search_fn,
                doc_kinds=doc_kinds,
            )
        except Exception as exc:
            logger.warning("[fallback_chain] retrieve_fn raised on rewrite: %s", exc)
            continue

        if retried is not None and getattr(retried, "ranked", None):
            outcome.chunks_recovered = len(retried.ranked)
            outcome.rewrite_used = rewrite
            outcome.stage_reached = "rewrite"
            _annotate_stats(retried, outcome, recovered_via="llm_rewrite")
            logger.info(
                "[fallback_chain] stage=rewrite recovered=%d via=%r",
                outcome.chunks_recovered, rewrite[:80],
            )
            return retried

    # ---- Stage 2: BM25-only sweep ------------------------------------
    outcome.bm25_tried = True
    bm25_hits = bm25_only_sweep(
        original_query,
        bm25_search_fn=bm25_search_fn,
        allowed_file_ids=allowed_file_ids,
        file_type=file_type,
    )
    if bm25_hits:
        # Mutate original_retrieval rather than constructing a new
        # dataclass so the caller's downstream code keeps seeing the
        # same RetrievalResult instance.
        original_retrieval.ranked = bm25_hits
        outcome.chunks_recovered = len(bm25_hits)
        outcome.stage_reached = "bm25_only"
        _annotate_stats(original_retrieval, outcome, recovered_via="bm25_only")
        logger.info(
            "[fallback_chain] stage=bm25_only recovered=%d",
            outcome.chunks_recovered,
        )
        return original_retrieval

    # ---- Stage 3: Decline gracefully ---------------------------------
    outcome.stage_reached = "decline"
    logger.info(
        "[fallback_chain] stage=decline — all stages empty for query=%r",
        original_query[:80],
    )
    return _mark_declined(original_retrieval, outcome)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _annotate_stats(retrieval: Any, outcome: FallbackOutcome, *, recovered_via: str) -> None:
    """Stamp fallback diagnostics onto RetrievalResult.stats."""
    if not hasattr(retrieval, "stats") or retrieval.stats is None:
        retrieval.stats = {}
    retrieval.stats["fallback_recovered_via"] = recovered_via
    retrieval.stats["fallback_rewrites_tried"] = outcome.rewrites_tried
    retrieval.stats["fallback_bm25_tried"] = outcome.bm25_tried
    if outcome.rewrite_used:
        retrieval.stats["fallback_rewrite_used"] = outcome.rewrite_used


def _mark_declined(retrieval: Any, outcome: FallbackOutcome) -> Any:
    """
    Mutate the retrieval object to signal the chat endpoint that we
    have explicitly given up — DO NOT call the LLM.
    """
    if not hasattr(retrieval, "stats") or retrieval.stats is None:
        retrieval.stats = {}
    retrieval.stats["search_mode"] = SEARCH_MODE_EMPTY_AFTER_FALLBACK
    retrieval.stats["fallback_recovered_via"] = "decline"
    retrieval.stats["fallback_rewrites_tried"] = outcome.rewrites_tried
    retrieval.stats["fallback_bm25_tried"] = outcome.bm25_tried
    # Belt-and-braces: guarantee no chunks leak through.
    retrieval.ranked = []
    return retrieval
