"""Raw Tier-1 intake → stable alert signature + search payload.

A deterministic signature lets two engineers who phrase the same alert
slightly differently share the same cache row. The signature is a
pipe-joined, lowercase, space-collapsed string built from the three
mandatory fields; optional fields are folded into search_terms /
search_text for retrieval, not the signature (because a paste-in IP or
device ID varies across incidents for the same root issue).
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Set

from backend.tier1_copilot.schemas import Tier1AnalyzeRequest

_WS_RE = re.compile(r"\s+")


def _clean(value: str) -> str:
    return _WS_RE.sub(" ", (value or "").strip().lower())


def normalize_alert(req: Tier1AnalyzeRequest, alias_dict: Any) -> Dict[str, Any]:
    """Produce the normalization bundle used by cache + retrieval.

    Returns a dict with:
      alert_signature  : deterministic pipe-joined cache key
      signature_hash   : SHA-1 of signature + sorted optional-field tuple
      search_terms     : alias-expanded keyword list (sorted, unique)
      search_text      : space-joined free text for embedding + BM25
    """
    # Severity became optional at the API boundary — the landing form
    # no longer gates submit on it. Fall back to an empty string in the
    # signature so the cache key + hash stay deterministic without
    # dereferencing None.
    #
    # Previous (severity-mandatory) line preserved for reference:
    # sig_parts = [req.severity.lower(), _clean(req.asset_name), _clean(req.alert_type)]
    sig_parts = [
        (req.severity or "").lower(),
        _clean(req.asset_name),
        _clean(req.alert_type),
    ]
    alert_signature = " | ".join(sig_parts)

    # Signature hash — includes optional fields so genuinely different
    # alerts with the same required triple don't collide in the cache.
    hash_parts = (
        alert_signature,
        _clean(req.customer or ""),
        _clean(req.location or ""),
        _clean(req.technology or ""),
        _clean(req.ip_or_device_id or ""),
        _clean(req.error_code or ""),
    )
    # Store scope (US Pharma) — same symptom at different stores must not
    # collide in the cache. Appended ONLY when store_id is present so
    # Acadia's cache key stays byte-identical.
    _store = _clean(getattr(req, "store_id", None) or "")
    if _store:
        hash_parts = hash_parts + (_store,)
    signature_hash = hashlib.sha1(
        "||".join(hash_parts).encode("utf-8")
    ).hexdigest()

    # Base terms for alias expansion.
    base_terms: Set[str] = set()
    for raw in (req.asset_name, req.alert_type, req.technology, req.error_code):
        if raw:
            cleaned = _clean(raw)
            if cleaned:
                base_terms.add(cleaned)

    expanded: Set[str] = set(base_terms)
    if alias_dict is not None:
        for term in list(base_terms):
            for alias in alias_dict.get_aliases(term):
                if alias:
                    expanded.add(alias)

    # `req.severity` may be None now (optional at the API boundary);
    # coerce to "" so the `if p` filter below drops it cleanly instead
    # of trying to join NoneType into the search text.
    search_text_parts: List[str] = [
        req.severity or "",
        req.asset_name,
        req.alert_type,
        req.technology or "",
        req.error_code or "",
        req.notes or "",
    ]
    search_text = " ".join(p for p in search_text_parts if p).strip()

    return {
        "alert_signature": alert_signature,
        "signature_hash": signature_hash,
        "search_terms": sorted(expanded),
        "search_text": search_text,
    }
