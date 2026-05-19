"""
Retrieval-quality metrics — Recall@K, NDCG@K, MRR.

All functions are pure (no DB, no LLM, no globals) so they're trivially
unit-testable and can be applied to any (predicted, expected) chunk-ID
pair regardless of where the retrieval ran.

Conventions
-----------
* ``predicted`` is the ordered list of chunk IDs the system under
  test returned (most-relevant first).
* ``expected`` is the unordered SET of chunk IDs we consider relevant
  (typically the snapshot from a known-good baseline run).
* ``k`` cuts ``predicted`` at the top K positions before scoring.
* All metrics return a float in ``[0.0, 1.0]``; higher is better.

Why these three metrics
-----------------------
* **Recall@K** — "Did the right chunks make the cut?" Insensitive to
  rank ORDER within the top K. Best at detecting catastrophic misses.
* **NDCG@K** — "Are the right chunks at the top?" Penalises a correct
  chunk that drops from rank 1 to rank 5. Best at detecting ranking
  degradation (the kind FTS-vs-BM25 differences usually produce).
* **MRR** — "How deep do we have to look to find any correct chunk?"
  Sanity check that we don't blow up rank 1 entirely.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Set


def recall_at_k(predicted: Sequence[str], expected: Iterable[str], k: int) -> float:
    """Fraction of expected items that appear in ``predicted[:k]``.

    Returns 1.0 when ``expected`` is empty (vacuous truth — no
    expectations were set so nothing can be wrong).
    """
    expected_set: Set[str] = set(expected)
    if not expected_set:
        return 1.0
    top_k = set(predicted[:k]) if predicted else set()
    hits = len(top_k & expected_set)
    return hits / len(expected_set)


def ndcg_at_k(predicted: Sequence[str], expected: Iterable[str], k: int) -> float:
    """Normalised Discounted Cumulative Gain @ K.

    Uses binary relevance (1 if chunk is in expected, 0 otherwise) so
    the implementation matches what we can produce from a baseline
    snapshot. Denominator is the ideal DCG assuming all relevant items
    would have been at the top.
    """
    expected_set: Set[str] = set(expected)
    if not expected_set:
        return 1.0
    if not predicted:
        return 0.0

    # DCG over the model's actual ranking.
    dcg = 0.0
    for i, chunk_id in enumerate(predicted[:k]):
        if chunk_id in expected_set:
            # log2(i+2) because positions are 1-indexed and DCG uses
            # log(position+1) — i=0 → log2(2)=1.0 → full credit.
            dcg += 1.0 / math.log2(i + 2)

    # Ideal DCG: relevant items in rank order at positions 1..min(|expected|,k).
    ideal_hits = min(len(expected_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def mrr(predicted: Sequence[str], expected: Iterable[str]) -> float:
    """Mean Reciprocal Rank of the first correct hit (or 0 if none).

    Single-query MRR — averaging across queries happens at the harness
    level.
    """
    expected_set: Set[str] = set(expected)
    if not expected_set or not predicted:
        return 0.0
    for i, chunk_id in enumerate(predicted):
        if chunk_id in expected_set:
            return 1.0 / (i + 1)
    return 0.0


def summarise(per_query: list[dict]) -> dict:
    """Aggregate per-query metric dicts into a corpus-level summary.

    Each entry in ``per_query`` is expected to carry ``recall_at_5``,
    ``ndcg_at_5``, and ``mrr`` keys (produced by the test harness).
    Returns macro averages — every query weighed equally regardless
    of how many relevant chunks it has.
    """
    if not per_query:
        return {"n": 0, "recall_at_5": 0.0, "ndcg_at_5": 0.0, "mrr": 0.0}
    n = len(per_query)
    return {
        "n": n,
        "recall_at_5": sum(q.get("recall_at_5", 0.0) for q in per_query) / n,
        "ndcg_at_5": sum(q.get("ndcg_at_5", 0.0) for q in per_query) / n,
        "mrr": sum(q.get("mrr", 0.0) for q in per_query) / n,
    }
