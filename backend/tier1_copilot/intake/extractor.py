"""Sprint 9 — Extractor: one Haiku call → JSON parse → candidates.

Reuses Sprint 6's `invoke_llm` helper from `backend.agents.base` so the
Bedrock model id, budget mechanism, and timing instrumentation are
unchanged. No parallel client.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, List, Optional

from backend.config import settings
from backend.tier1_copilot.intake.catalogs import IntakeCatalogs
from backend.tier1_copilot.intake.prompt_builder import build_extraction_prompt
from backend.tier1_copilot.intake.schemas import (
    FieldEvidence,
    ValidatedCandidate,
    ValidationStatus,
)

logger = logging.getLogger("acadia-log-iq")


# ─────────────────────────────────────────────────────────────
# JSON parser — tolerant of code-fence wrapping and stray text
# ─────────────────────────────────────────────────────────────
def parse_extraction_json(raw: str) -> Optional[List[dict]]:
    """Return the parsed JSON array, or None if it can't be recovered.

    Tolerates: ```json fences, leading/trailing prose, single trailing
    comma. Does NOT tolerate: structural malformation (missing brackets,
    bare object instead of array)."""
    if not raw:
        return None
    txt = raw.strip()
    # Strip ```json ... ``` fences.
    if txt.startswith("```"):
        lines = txt.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        txt = "\n".join(lines).strip()
    try:
        parsed = json.loads(txt)
    except Exception:
        # Recover the outermost JSON array if the LLM wrapped it in prose.
        start = txt.find("[")
        end = txt.rfind("]")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(txt[start:end + 1])
            except Exception:
                # Last-ditch: drop a trailing comma before the closing bracket.
                cleaned = re.sub(r",(\s*[\]\}])", r"\1", txt[start:end + 1])
                try:
                    parsed = json.loads(cleaned)
                except Exception:
                    return None
        else:
            return None
    if not isinstance(parsed, list):
        return None
    return parsed


# ─────────────────────────────────────────────────────────────
# Candidate construction (unvalidated — validator handles match step)
# ─────────────────────────────────────────────────────────────
def _build_candidate(item: dict) -> ValidatedCandidate:
    if not isinstance(item, dict):
        return ValidatedCandidate()
    ev = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
    severity = item.get("severity")
    if severity not in ("P1", "P2", "P3", "P4", None):
        # Non-enum severity — let the validator demote it.
        severity = None if severity is None else str(severity)
    return ValidatedCandidate(
        severity=severity if severity in ("P1", "P2", "P3", "P4") else None,
        asset_name=_str_or_none(item.get("asset_name")),
        alert_type=_str_or_none(item.get("alert_type")),
        customer=_str_or_none(item.get("customer")),
        location=_str_or_none(item.get("location")),
        users_impacted_count=_int_or_none(item.get("users_impacted_count")),
        evidence=FieldEvidence(
            severity=_str_or_none(ev.get("severity")),
            asset_name=_str_or_none(ev.get("asset_name")),
            alert_type=_str_or_none(ev.get("alert_type")),
            customer=_str_or_none(ev.get("customer")),
        ),
        validation=ValidationStatus(),
    )


def _str_or_none(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _int_or_none(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────
# Public entry
# ─────────────────────────────────────────────────────────────
def _evidence_empty(item: Any) -> bool:
    """Sprint 9.2 — true iff the candidate has at least one non-null
    field with no corresponding evidence substring. Used by the
    retry-on-missing-evidence loop."""
    if not isinstance(item, dict):
        return True
    ev = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
    for field in ("severity", "asset_name", "alert_type", "customer"):
        value = item.get(field)
        if value is None:
            continue
        evid = ev.get(field) if isinstance(ev, dict) else None
        if not evid or not str(evid).strip():
            return True
    return False


def extract_candidates(
    *,
    raw_text: str,
    source_type: str,
    catalogs: IntakeCatalogs = None,
    n_candidates: int = 4,
    llm_invoke: Optional[Callable[[str], str]] = None,
) -> List[ValidatedCandidate]:
    """Sprint 9.2 — content-only extraction.

    Catalogs are NOT injected into the prompt — that hint mechanism
    contaminated extractions in Sprint 9 (proven by reproduction
    testing). The validator handles catalog matching after the fact.
    The `catalogs` kwarg is accepted for backward-compatible call sites
    but ignored by the prompt builder.

    Two retry conditions:
      1. JSON parse failure → re-prompt with stricter "JSON only".
      2. Sprint 9.2: any candidate has a non-null field without a
         corresponding `evidence` substring → re-prompt instructing
         "every non-null field MUST have a verbatim evidence substring".
    Both retries are best-effort; if they also fail the original parsed
    output (or empty list) is returned and the validator's grounding
    pass nulls hallucinated fields downstream.
    """
    prompt = build_extraction_prompt(
        raw_text=raw_text,
        source_type=source_type,
        n_candidates=n_candidates,
        catalogs=catalogs,  # accepted-but-ignored — see prompt_builder
    )

    invoker = llm_invoke or _default_invoker
    raw = invoker(prompt) or ""
    parsed = parse_extraction_json(raw)
    if parsed is None:
        stricter = (
            prompt
            + "\n\nSTRICT: return ONLY a JSON array. No code fences, no "
            "explanation. The first character of your response MUST be `[`."
        )
        raw2 = invoker(stricter) or ""
        parsed = parse_extraction_json(raw2)
    if parsed is None:
        logger.warning(
            "[intake] extractor — JSON parse failed twice (source=%s, len=%d)",
            source_type, len(raw_text),
        )
        return []

    # Sprint 9.2 — retry once if any candidate is missing evidence.
    if any(_evidence_empty(item) for item in parsed[:n_candidates]):
        evidence_retry = (
            prompt
            + "\n\nIMPORTANT: Every non-null field MUST have a verbatim "
            "evidence substring from the content. Re-extract with "
            "evidence populated for every non-null field."
        )
        raw3 = invoker(evidence_retry) or ""
        retry_parsed = parse_extraction_json(raw3)
        if retry_parsed:
            parsed = retry_parsed

    cands = [_build_candidate(item) for item in parsed[:n_candidates]]
    logger.info(
        "[intake] extractor produced %d candidates (source=%s)",
        len(cands), source_type,
    )
    return cands


def _default_invoker(prompt: str) -> str:
    """Default LLM caller — wires Sprint 6's invoke_llm to Haiku via the
    parent app's bedrock client. Returns "" on any failure so the
    handler can return a graceful empty-candidates response."""
    try:
        from backend.agents.base import TokenBudget, invoke_llm
        from backend import api as _api
    except Exception as exc:
        logger.warning("[intake] default invoker import failed: %s", exc)
        return ""

    budget = TokenBudget(max_total=4000)
    try:
        from backend.config import settings as _settings
        _max_tokens = int(getattr(_settings, "INTAKE_EXTRACTION_MAX_TOKENS", 2000))
        step = invoke_llm(
            prompt=prompt,
            model="haiku",
            max_tokens=_max_tokens,
            budget=budget,
            agent_name="universal_intake_extractor",
            generate_fn=getattr(_api, "safe_generate", None),
            bedrock_client=getattr(_api, "bedrock", None),
        )
    except Exception as exc:
        logger.warning("[intake] invoke_llm raised: %s", exc)
        return ""

    return (step.output or "").strip() if getattr(step, "success", False) else ""
