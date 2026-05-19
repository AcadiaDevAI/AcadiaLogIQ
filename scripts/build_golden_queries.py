"""
Build / refresh the golden-query baseline for the retrieval regression suite.

Usage::

    python -m scripts.build_golden_queries [--limit 100] [--top-k 10]

What it does
------------
1. Pulls the most recent N user queries from ``chat_messages`` (de-PII'd
   by stripping anything that looks like an email, phone, or 16-digit
   number).
2. Filters out queries that are too short / too long to be useful as
   regression cases (≤ 3 tokens or > 400 chars).
3. For each surviving query, calls the current retrieval orchestrator
   and snapshots the top-K chunk IDs.
4. Writes ``tests/retrieval/golden_queries.jsonl`` — one JSON object
   per line — overwriting the previous file.

When to run
-----------
* Once at the start of any retrieval upgrade (Phase 0.1 BM25 → FTS)
  to capture the "current production truth" baseline.
* After an intentional retrieval improvement, when a human has
  reviewed the new behaviour and is ready to lock it in as the new
  baseline.
* NOT routinely — running this between baseline captures defeats the
  purpose of having a regression gate.

PII policy
----------
The user has approved storing de-PII'd query text + chunk IDs in
``golden_queries.jsonl``. We do NOT store the LLM-generated answers,
the retrieved chunk content, or any user IDs / session IDs. The jsonl
file is safe to commit to git.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent
sys.path.insert(0, str(_REPO_ROOT))


# Output path — versioned, committed, machine-readable.
_GOLDEN_PATH = _REPO_ROOT / "tests" / "retrieval" / "golden_queries.jsonl"


# Conservative PII scrubbers. Run in the order listed so longer patterns
# are caught before shorter overlapping ones (e.g. email before raw
# domain).
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)?\d{3}[\s-]?\d{4}\b")
_LONG_DIGITS_RE = re.compile(r"\b\d{12,}\b")  # credit-card-ish runs


def _scrub_pii(text: str) -> str:
    if not text:
        return ""
    out = _EMAIL_RE.sub("<email>", text)
    out = _PHONE_RE.sub("<phone>", out)
    out = _LONG_DIGITS_RE.sub("<id>", out)
    return out


def _is_useful_query(q: str) -> bool:
    """Filter out queries that won't be informative regression cases."""
    if not q:
        return False
    stripped = q.strip()
    if len(stripped) > 400:
        return False
    token_count = len(stripped.split())
    if token_count < 3:
        return False
    return True


def _pull_recent_queries(limit: int) -> List[str]:
    """Read recent user-role messages from ``chat_messages``."""
    from sqlalchemy import create_engine, text  # type: ignore
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        # Allow the .env fallback so this script can be run on a
        # laptop without the AWS Secrets bootstrap path.
        try:
            from dotenv import load_dotenv  # type: ignore
            load_dotenv()
            db_url = os.environ.get("DATABASE_URL")
        except Exception:
            pass
    if not db_url:
        raise RuntimeError(
            "DATABASE_URL not set — cannot pull queries. "
            "Either source backend/.env or export the var in your shell."
        )

    engine = create_engine(db_url)
    sql = text("""
        SELECT content
          FROM chat_messages
         WHERE role = 'user'
           AND length(coalesce(content, '')) > 5
         ORDER BY created_at DESC
         LIMIT :limit
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"limit": limit}).fetchall()
    return [r[0] for r in rows if r and r[0]]


def _capture_topk(
    query: str,
    top_k: int,
    orchestrator_fn,
) -> List[str]:
    """Run the current retrieval pipeline once and pull chunk IDs.

    Falls back to an empty list on any error so a single bad query
    can't kill the whole build run.
    """
    try:
        chunks = orchestrator_fn(query=query, top_k=top_k)
    except Exception as exc:
        logging.warning("[golden] retrieval failed for %r — %s", query[:60], exc)
        return []
    out: List[str] = []
    for c in (chunks or [])[:top_k]:
        # The orchestrator returns chunk dicts; chunk_id is the stable
        # key. Defensive about shape so a future schema tweak doesn't
        # silently produce empty baselines.
        chunk_id = (
            c.get("id")
            or c.get("chunk_id")
            or c.get("source", {}).get("chunk_id")
        ) if isinstance(c, dict) else None
        if chunk_id:
            out.append(str(chunk_id))
    return out


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=100,
        help="Max number of queries to pull (default 100).",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Top-K chunk IDs to snapshot per query (default 10).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be written without touching the file.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Pull + scrub + filter.
    raw = _pull_recent_queries(args.limit)
    scrubbed = [_scrub_pii(q) for q in raw]
    useful = [q for q in scrubbed if _is_useful_query(q)]
    logging.info(
        "[golden] pulled=%d  scrubbed=%d  useful=%d",
        len(raw), len(scrubbed), len(useful),
    )

    if not useful:
        logging.warning(
            "[golden] no usable queries found. The chat_messages table "
            "may be empty or all entries are too short/long. Aborting.",
        )
        return 1

    # Lazy import — only need the orchestrator when we're actually
    # capturing baselines.
    try:
        from backend.retrieval.orchestrator import retrieve as orchestrator_retrieve  # type: ignore
    except Exception as exc:
        logging.error(
            "[golden] could not import orchestrator (%s). Ensure the "
            "backend can boot before running this script.", exc,
        )
        return 2

    records = []
    for q in useful:
        chunk_ids = _capture_topk(q, args.top_k, orchestrator_retrieve)
        if not chunk_ids:
            # Skip queries that produce zero retrievals — they're not
            # informative regression cases.
            continue
        records.append({
            "query": q,
            "top_k_chunk_ids": chunk_ids,
        })

    logging.info("[golden] captured %d queries with non-empty retrievals", len(records))

    if args.dry_run:
        for r in records[:5]:
            print(json.dumps(r, ensure_ascii=False))
        print(f"... ({len(records)} total — dry run, file untouched)")
        return 0

    _GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _GOLDEN_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logging.info("[golden] wrote %s (%d records)", _GOLDEN_PATH, len(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
