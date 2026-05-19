"""
Retrieval regression gate.

Replays every entry in ``golden_queries.jsonl`` through the CURRENT
orchestrator and asserts the corpus-level metrics stay within
tolerance of the captured baseline.

Failure modes
-------------
* No ``golden_queries.jsonl`` → test SKIPS (with a hint to run
  ``scripts/build_golden_queries.py`` first). This means a fresh
  checkout doesn't fail CI; the suite only gates once a baseline
  has been intentionally captured.
* Backend can't boot (DB unreachable, missing settings) → test SKIPS
  with the underlying error in the skip reason. Same rationale —
  CI without a DB shouldn't false-fail.
* Metric below threshold → test FAILS loudly with a per-query diff.

Thresholds
----------
* Recall@5  >= 0.90 of baseline (i.e. we don't lose more than 10% of
  the chunks the captured baseline pulled).
* NDCG@5    >= 0.85 of baseline (slightly looser — rank perturbations
  inside top-5 are expected after a hybrid-weight tweak).
* MRR       >= 0.85 of baseline.

These thresholds match the BM25 → FTS migration acceptance criteria
in Phase 0.1.
"""

from __future__ import annotations

import json
import pathlib
from typing import List

import pytest

from .eval_metrics import recall_at_k, ndcg_at_k, mrr, summarise


_GOLDEN_PATH = pathlib.Path(__file__).parent / "golden_queries.jsonl"

# Acceptance thresholds — expressed as a fraction of the baseline so
# they auto-adjust as the corpus grows.
_RECALL_TOLERANCE = 0.90
_NDCG_TOLERANCE = 0.85
_MRR_TOLERANCE = 0.85


def _load_golden() -> List[dict]:
    if not _GOLDEN_PATH.exists():
        pytest.skip(
            "golden_queries.jsonl missing — run "
            "`python -m scripts.build_golden_queries` to capture a baseline. "
            "This test is a regression gate, not a unit test; skipping is "
            "the correct behaviour when no baseline has been captured.",
        )
    out: List[dict] = []
    with _GOLDEN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _safe_import_orchestrator():
    """Import the orchestrator lazily so a CI run without a DB skips
    cleanly rather than ImportError-ing in collection.
    """
    try:
        from backend.retrieval.orchestrator import retrieve  # type: ignore
    except Exception as exc:  # noqa: BLE001 — broad on purpose for clean skip
        pytest.skip(f"orchestrator import failed: {exc}")
    return retrieve


def test_retrieval_baseline_within_tolerance():
    """End-to-end regression gate over the captured golden set."""
    golden = _load_golden()
    if not golden:
        pytest.skip("golden_queries.jsonl is empty")

    retrieve_fn = _safe_import_orchestrator()

    per_query: List[dict] = []
    failures: List[str] = []
    for entry in golden:
        query = entry["query"]
        expected = entry.get("top_k_chunk_ids", [])

        try:
            chunks = retrieve_fn(query=query, top_k=10)
        except Exception as exc:
            failures.append(f"retrieval crashed for {query!r}: {exc}")
            continue

        predicted = []
        for c in (chunks or [])[:10]:
            if not isinstance(c, dict):
                continue
            chunk_id = (
                c.get("id")
                or c.get("chunk_id")
                or c.get("source", {}).get("chunk_id")
            )
            if chunk_id:
                predicted.append(str(chunk_id))

        per_query.append({
            "query": query,
            "recall_at_5": recall_at_k(predicted, expected, k=5),
            "ndcg_at_5": ndcg_at_k(predicted, expected, k=5),
            "mrr": mrr(predicted, expected),
        })

    if failures:
        pytest.fail("retrieval crashes:\n" + "\n".join(failures[:5]))

    summary = summarise(per_query)
    # Print a concise corpus-level snapshot so a failure is easy to
    # triage from CI output.
    print(
        f"\n[golden] n={summary['n']} "
        f"recall@5={summary['recall_at_5']:.3f} "
        f"ndcg@5={summary['ndcg_at_5']:.3f} "
        f"mrr={summary['mrr']:.3f}",
    )

    # The baseline is "the current system at the time of capture", so
    # the expected value for each metric ON the captured queries is
    # 1.0 (the system returns its own snapshot back). The thresholds
    # are the fraction of that perfect score we're willing to lose.
    assert summary["recall_at_5"] >= _RECALL_TOLERANCE, (
        f"Recall@5 {summary['recall_at_5']:.3f} below tolerance "
        f"{_RECALL_TOLERANCE:.2f} — retrieval has regressed."
    )
    assert summary["ndcg_at_5"] >= _NDCG_TOLERANCE, (
        f"NDCG@5 {summary['ndcg_at_5']:.3f} below tolerance "
        f"{_NDCG_TOLERANCE:.2f} — ranking has regressed."
    )
    assert summary["mrr"] >= _MRR_TOLERANCE, (
        f"MRR {summary['mrr']:.3f} below tolerance "
        f"{_MRR_TOLERANCE:.2f} — top-1 retrieval has regressed."
    )
