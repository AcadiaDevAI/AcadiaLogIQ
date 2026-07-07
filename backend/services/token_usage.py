"""Per-org Bedrock token-usage accounting.

Every Bedrock ``invoke_model`` response carries the exact tokens the model
billed — in the JSON body (``usage`` for Claude, ``inputTextTokenCount`` for
Titan) and, universally, in the response headers
(``x-amzn-bedrock-input-token-count`` / ``-output-token-count``). Most call
sites historically discarded it. This module is the single choke point that:

  1. extracts usage uniformly across model families (extract_bedrock_usage),
  2. prices it (cost_usd) using a per-model rate table,
  3. buffers increments in-process keyed by (org, day, feature, model), and
  4. flushes them into ``token_usage_daily`` at request/job boundaries.

Buffering (rather than one DB write per call) keeps very high-volume paths —
Titan embeddings especially — from hammering a single daily-rollup row, and
keeps the write off the request/inference hot path.

Tenancy: the org is captured from ``current_org_id_var`` at record time and
carried on each buffer bucket, so a flush that happens at a boundary (where
the ContextVar may be resetting) still writes to the correct org. flush()
sets ``app.current_org`` to each bucket's org so the RLS WITH CHECK on
``token_usage_daily`` is satisfied.

Nothing here may raise into a caller — token accounting must never break an
LLM call or a request.
"""
from __future__ import annotations

import datetime as _dt
import logging
import threading
import uuid as _uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from backend.config import settings

logger = logging.getLogger("acadia-log-iq")


# ---------------------------------------------------------------------------
# Pricing — USD per 1,000,000 tokens (input, output), resolved by a substring
# match on the Bedrock model id. Anthropic rates (haiku/sonnet) are from the
# claude-api reference; Mistral 7B and Titan Embed v2 are AWS Bedrock list
# prices — verify against your negotiated rate and override here if different.
# Titan embeddings have no output tokens, so output rate is 0.
# ---------------------------------------------------------------------------
_DEFAULT_PRICING: Dict[str, Tuple[float, float]] = {
    "haiku": (1.00, 5.00),
    "sonnet": (3.00, 15.00),
    "mistral": (0.15, 0.20),
    "titan-embed": (0.02, 0.0),
    "titan": (0.02, 0.0),
}

# Cap on distinct buffer keys before a safety flush fires. Keys are
# (org, day, feature, model) so this is only reached under heavy multi-org /
# multi-feature load; normal flushing happens at request/job boundaries.
_BUFFER_MAX_KEYS = 500


def _pricing() -> Dict[str, Tuple[float, float]]:
    # Allow an ops override via settings.MODEL_PRICING without requiring it.
    override = getattr(settings, "MODEL_PRICING", None)
    if isinstance(override, dict) and override:
        return override
    return _DEFAULT_PRICING


def price_for(model_id: str) -> Tuple[float, float]:
    """(input_per_mtok, output_per_mtok) for a Bedrock model id. Longest
    matching key wins so 'titan-embed' beats 'titan'."""
    mid = (model_id or "").lower()
    best: Optional[Tuple[float, float]] = None
    best_len = -1
    for key, rate in _pricing().items():
        if key in mid and len(key) > best_len:
            best, best_len = rate, len(key)
    if best is None:
        logger.debug("[token_usage] no price for model=%s — cost recorded as 0", model_id)
        return (0.0, 0.0)
    return best


def cost_usd(model_id: str, input_tokens: int, output_tokens: int) -> float:
    in_rate, out_rate = price_for(model_id)
    return (input_tokens / 1_000_000.0) * in_rate + (output_tokens / 1_000_000.0) * out_rate


def extract_bedrock_usage(payload: Any, response: Any = None) -> Tuple[int, int]:
    """Return (input_tokens, output_tokens) from a Bedrock response.

    Order: body ``usage`` (Claude) → body ``inputTextTokenCount`` (Titan) →
    response HTTP headers (universal; the only source for Mistral)."""
    in_t = out_t = 0
    if isinstance(payload, dict):
        usage = payload.get("usage") or {}
        in_t = int(usage.get("input_tokens") or 0)
        out_t = int(usage.get("output_tokens") or 0)
        if not in_t and payload.get("inputTextTokenCount") is not None:
            in_t = int(payload.get("inputTextTokenCount") or 0)
    if not in_t and not out_t and isinstance(response, dict):
        headers = (response.get("ResponseMetadata") or {}).get("HTTPHeaders") or {}
        try:
            in_t = int(headers.get("x-amzn-bedrock-input-token-count") or 0)
            out_t = int(headers.get("x-amzn-bedrock-output-token-count") or 0)
        except (TypeError, ValueError):
            pass
    return in_t, out_t


# ---------------------------------------------------------------------------
# In-process buffer + cumulative process totals.
# ---------------------------------------------------------------------------
_lock = threading.Lock()
# key: (org_str|None, date_iso, feature, model_id) -> mutable counters
_buffer: Dict[Tuple[Optional[str], str, str, str], Dict[str, float]] = {}
_totals = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}

_UPSERT_SQL = """
INSERT INTO token_usage_daily
    (organization_id, usage_date, feature, model_id,
     call_count, input_tokens, output_tokens, cost_usd)
VALUES
    (:org, :usage_date, :feature, :model, :calls, :in_toks, :out_toks, :cost)
ON CONFLICT (organization_id, usage_date, feature, model_id) DO UPDATE SET
    call_count    = token_usage_daily.call_count    + EXCLUDED.call_count,
    input_tokens  = token_usage_daily.input_tokens  + EXCLUDED.input_tokens,
    output_tokens = token_usage_daily.output_tokens + EXCLUDED.output_tokens,
    cost_usd      = token_usage_daily.cost_usd       + EXCLUDED.cost_usd
"""


def _current_org() -> Optional[str]:
    """UUID string of the org in scope, or None. Lazy import so the pure
    pricing/extract helpers are importable without the DB engine."""
    try:
        from backend.db.connection import current_org_id_var

        org = current_org_id_var.get()
        return str(org) if org is not None else None
    except Exception:  # pragma: no cover - defensive
        return None


def _today_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).date().isoformat()


def record_token_usage(
    feature: str,
    model_id: str,
    input_tokens: int,
    output_tokens: int,
) -> None:
    """Log + buffer one Bedrock call's usage. Never raises."""
    if not getattr(settings, "TOKEN_USAGE_TRACKING_ENABLED", True):
        return
    try:
        feature = (feature or "unknown").split("#", 1)[0]  # strip retry suffix
        in_t = int(input_tokens or 0)
        out_t = int(output_tokens or 0)
        c = cost_usd(model_id, in_t, out_t)
        org = _current_org()
        key = (org, _today_iso(), feature, model_id or "unknown")

        with _lock:
            slot = _buffer.get(key)
            if slot is None:
                slot = {"calls": 0, "in": 0, "out": 0, "cost": 0.0}
                _buffer[key] = slot
            slot["calls"] += 1
            slot["in"] += in_t
            slot["out"] += out_t
            slot["cost"] += c

            _totals["calls"] += 1
            _totals["input_tokens"] += in_t
            _totals["output_tokens"] += out_t
            _totals["cost_usd"] += c
            overflow = len(_buffer) >= _BUFFER_MAX_KEYS

        logger.info(
            "[token_usage] feature=%s model=%s in=%d out=%d cost=$%.5f org=%s",
            feature, model_id, in_t, out_t, c, org,
        )
        if overflow:
            flush()
    except Exception as exc:  # pragma: no cover - accounting must never break calls
        logger.debug("[token_usage] record failed: %s", exc)


def _drain() -> List[Dict[str, Any]]:
    """Atomically empty the buffer into a flat list of row dicts. Pure — no
    DB — so the buffering/grouping logic is unit-testable on its own."""
    with _lock:
        rows = [
            {
                "org": org,
                "usage_date": date_iso,
                "feature": feature,
                "model": model,
                "calls": int(slot["calls"]),
                "in_toks": int(slot["in"]),
                "out_toks": int(slot["out"]),
                "cost": round(float(slot["cost"]), 6),
            }
            for (org, date_iso, feature, model), slot in _buffer.items()
        ]
        _buffer.clear()
    return rows


def flush() -> None:
    """Persist all buffered usage into token_usage_daily. Groups by org and
    stamps app.current_org per group so RLS WITH CHECK is satisfied. Never
    raises; rows for a null-org context are dropped (can't satisfy RLS)."""
    rows = _drain()
    if not rows:
        return
    by_org: Dict[Optional[str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_org[r["org"]].append(r)

    try:
        from sqlalchemy import text
        from backend.db.connection import SessionLocal, current_org_id_var
    except Exception as exc:  # pragma: no cover
        logger.warning("[token_usage] flush skipped (no DB): %s", exc)
        return

    for org, org_rows in by_org.items():
        if not org:
            logger.warning(
                "[token_usage] dropping %d buffered rows with no org context",
                len(org_rows),
            )
            continue
        token = current_org_id_var.set(_uuid.UUID(str(org)))
        try:
            db = SessionLocal()
            try:
                for r in org_rows:
                    db.execute(
                        text(_UPSERT_SQL),
                        {
                            "org": org,
                            "usage_date": r["usage_date"],
                            "feature": r["feature"],
                            "model": r["model"],
                            "calls": r["calls"],
                            "in_toks": r["in_toks"],
                            "out_toks": r["out_toks"],
                            "cost": r["cost"],
                        },
                    )
                db.commit()
            except Exception as exc:
                db.rollback()
                logger.warning(
                    "[token_usage] flush failed for org=%s (%d rows dropped): %s",
                    org, len(org_rows), exc,
                )
            finally:
                db.close()
        finally:
            current_org_id_var.reset(token)


def get_usage_totals() -> Dict[str, Any]:
    """Snapshot of cumulative usage since the last reset — used by the worker
    to print a per-job cost summary."""
    with _lock:
        snap = dict(_totals)
    snap["cost_usd"] = round(snap["cost_usd"], 6)
    return snap


def reset_usage_totals() -> None:
    with _lock:
        _totals.update({"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
