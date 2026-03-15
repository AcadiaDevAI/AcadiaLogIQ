"""
Modular Reranker — scores fused candidates by relevance to the query.
Default backend: LLM (Mistral via Bedrock) for relevance scoring.
Swappable via RERANKER_BACKEND config to 'cross_encoder' or 'none'.
The reranker sits between fusion and final context assembly.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Tuple

from backend.config import settings

logger = logging.getLogger("acadia-log-iq")

# Result tuple: (chunk_id, text, metadata, score)
RankedResult = Tuple[str, str, Dict[str, Any], float]


# ---------------------------------------------------------------------------
# Abstract base class — all reranker backends implement this
# ---------------------------------------------------------------------------
class BaseReranker(ABC):
    """Interface for reranker backends. Subclass and implement rerank()."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[RankedResult],
        top_k: int = 6,
    ) -> List[RankedResult]:
        """
        Score and re-order candidates by relevance to query.
        Returns top_k results sorted by final blended score.
        """
        ...


# ---------------------------------------------------------------------------
# No-op reranker — just passes through fusion scores
# ---------------------------------------------------------------------------
class NoopReranker(BaseReranker):
    """Pass-through reranker: returns candidates as-is, truncated to top_k."""

    def rerank(self, query: str, candidates: List[RankedResult], top_k: int = 6) -> List[RankedResult]:
        return candidates[:top_k]


# ---------------------------------------------------------------------------
# LLM reranker — uses Mistral (or any Bedrock LLM) for relevance scoring
# ---------------------------------------------------------------------------
class LLMReranker(BaseReranker):
    """
    Uses the LLM (Mistral via Bedrock) to score chunk relevance 0-10.
    Blends the LLM relevance score with the original fusion score
    using configurable weights (RERANK_SCORE_WEIGHT / RERANK_FUSION_WEIGHT).
    """

    def __init__(self, generate_fn: Callable[[str, int], str]):
        """
        Args:
            generate_fn: function(prompt, max_tokens) -> str response text.
                         This is safe_generate from api.py.
        """
        self._generate = generate_fn

    def rerank(self, query: str, candidates: List[RankedResult], top_k: int = 6) -> List[RankedResult]:
        if not candidates or len(candidates) <= 1:
            return candidates[:top_k]

        # Limit to RERANK_CANDIDATES to control LLM cost
        pool = candidates[: min(len(candidates), settings.RERANK_CANDIDATES)]

        # --- Build the scoring prompt ---
        # Show the LLM a preview of each chunk and ask for relevance scores
        previews = []
        for i, (_, chunk_text, meta, _) in enumerate(pool):
            preview = chunk_text[:500].replace("\n", " ").strip()
            src = meta.get("source", "?")
            previews.append(f"[{i + 1}] (source: {src}) {preview}")

        prompt = (
            f"Rate each chunk's relevance to the question (0=irrelevant, 10=perfect match).\n"
            f"Question: {query}\n\n"
            f"Chunks:\n" + "\n".join(previews) + "\n\n"
            f"Respond ONLY with a JSON array: [{{\"chunk\":1,\"score\":8}}, ...]"
        )

        try:
            # Call the LLM for scoring
            resp = self._generate(prompt, 512)
            raw = resp.strip()

            # Parse the JSON response
            if "```" in raw:
                raw = raw.split("```json")[-1].split("```")[0] if "```json" in raw else raw.split("```")[1]
            start, end = raw.find("["), raw.rfind("]") + 1
            if start < 0 or end <= start:
                raise ValueError("No JSON array in reranker response")

            scores = json.loads(raw[start:end])

            # --- Blend LLM scores with fusion scores ---
            scored: List[RankedResult] = []
            scored_ids = set()

            for item in scores:
                idx = item.get("chunk", 0) - 1
                relevance = float(item.get("score", 0))
                if 0 <= idx < len(pool):
                    cid, text_val, meta, fusion_score = pool[idx]
                    # Weighted blend: LLM relevance (normalized 0-1) + original fusion score
                    blended = (
                        (relevance / 10.0) * settings.RERANK_SCORE_WEIGHT
                        + fusion_score * 30 * settings.RERANK_FUSION_WEIGHT
                    )
                    scored.append((cid, text_val, meta, blended))
                    scored_ids.add(cid)

            # Add any candidates the LLM missed (keep their original fusion score)
            for candidate in pool:
                if candidate[0] not in scored_ids:
                    scored.append(candidate)

            # Sort by blended score descending
            scored.sort(key=lambda x: x[3], reverse=True)

            logger.debug("LLM reranker: %d candidates → %d scored, returning top %d",
                         len(pool), len(scored), top_k)
            return scored[:top_k]

        except Exception as exc:
            logger.warning("LLM reranker failed (%s), falling back to fusion order", exc)
            return pool[:top_k]


# ---------------------------------------------------------------------------
# Cross-encoder reranker placeholder — for future sentence-transformers integration
# ---------------------------------------------------------------------------
class CrossEncoderReranker(BaseReranker):
    """
    Placeholder for cross-encoder reranking (e.g., sentence-transformers).
    When implemented, this would load a local model and score
    (query, passage) pairs directly without an LLM call.

    To enable: set RERANKER_BACKEND=cross_encoder and install
    sentence-transformers with a compatible model.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model_name
        self._model = None
        logger.info("CrossEncoderReranker initialized (model=%s) — not yet loaded", model_name)

    def _load_model(self):
        """Lazy-load the cross-encoder model on first use."""
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
            logger.info("Cross-encoder model loaded: %s", self._model_name)
        except ImportError:
            logger.error("sentence-transformers not installed. pip install sentence-transformers")
            raise

    def rerank(self, query: str, candidates: List[RankedResult], top_k: int = 6) -> List[RankedResult]:
        if not candidates:
            return []

        if self._model is None:
            self._load_model()

        pool = candidates[: min(len(candidates), settings.RERANK_CANDIDATES)]

        # Build (query, passage) pairs for the cross-encoder
        pairs = [(query, text[:512]) for _, text, _, _ in pool]

        try:
            ce_scores = self._model.predict(pairs)

            scored = []
            for i, (cid, text_val, meta, fusion_score) in enumerate(pool):
                ce_score = float(ce_scores[i])
                blended = (
                    ce_score * settings.RERANK_SCORE_WEIGHT
                    + fusion_score * 30 * settings.RERANK_FUSION_WEIGHT
                )
                scored.append((cid, text_val, meta, blended))

            scored.sort(key=lambda x: x[3], reverse=True)
            return scored[:top_k]

        except Exception as exc:
            logger.warning("Cross-encoder rerank failed (%s), fallback to fusion order", exc)
            return pool[:top_k]


# ---------------------------------------------------------------------------
# Factory: create the right reranker based on config
# ---------------------------------------------------------------------------
def create_reranker(generate_fn: Callable[[str, int], str] = None) -> BaseReranker:
    """
    Factory function that returns the appropriate reranker backend
    based on settings.RERANKER_BACKEND.

    Args:
        generate_fn: The LLM generate function (required for 'llm' backend)

    Returns:
        A BaseReranker instance
    """
    backend = settings.RERANKER_BACKEND.lower()

    if backend == "none":
        logger.info("Reranker: none (pass-through)")
        return NoopReranker()

    if backend == "cross_encoder":
        logger.info("Reranker: cross-encoder")
        return CrossEncoderReranker()

    if backend == "llm":
        if generate_fn is None:
            logger.warning("LLM reranker requested but no generate_fn provided, using no-op")
            return NoopReranker()
        logger.info("Reranker: LLM (Mistral via Bedrock)")
        return LLMReranker(generate_fn)

    logger.warning("Unknown reranker backend '%s', using no-op", backend)
    return NoopReranker()
