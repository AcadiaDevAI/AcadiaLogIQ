"""Sprint 9 — Diversity reranker.

Drops candidates whose `diversity_signature` is already represented by
a higher-ranked candidate. Pads with the suppressed candidates if
filtering would leave fewer than `max_cards`.
"""
from __future__ import annotations

from typing import List, Optional, Set

from backend.tier1_copilot.intake.schemas import ValidatedCandidate


def _normalize_signature_part(value: Optional[str]) -> str:
    """Sprint 9.2 — lowercase + whitespace-collapse + strip.

    Sprint 9.0 logs showed `INC-ALPHA-207` appearing at rank 1 AND
    rank 2 because the validator-built signatures differed only in
    case/whitespace ("BGP Flap" vs "bgp flap") and the dedup compared
    them as raw strings. Normalising both sides before the seen-set
    lookup catches these variants.
    """
    if not value:
        return ""
    return " ".join(str(value).lower().split())


def _normalised_signature(sig: Optional[str]) -> str:
    """Apply normalisation to each of the three pipe-separated parts of
    the signature so dedup is case- and whitespace-insensitive."""
    if not sig:
        return ""
    parts = sig.split("|")
    return "|".join(_normalize_signature_part(p) for p in parts)


def diversify(
    candidates: List[ValidatedCandidate],
    *,
    max_cards: int = 4,
) -> List[ValidatedCandidate]:
    """Return up to `max_cards` candidates with distinct signatures.

    Order is preserved relative to input — the LLM produces them in
    descending likelihood, so the first occurrence of each signature is
    the most likely interpretation for that triple. Sprint 9.2: the
    signature comparison is now case- and whitespace-insensitive (see
    `_normalize_signature_part`).
    """
    selected: List[ValidatedCandidate] = []
    seen: Set[str] = set()

    for cand in candidates:
        raw_sig = cand.diversity_signature
        if not raw_sig:
            # Defensive: validator always sets it, but guard anyway.
            raw_sig = f"?|?|?|{id(cand)}"
        sig = _normalised_signature(raw_sig)
        if sig in seen:
            continue
        selected.append(cand)
        seen.add(sig)
        if len(selected) >= max_cards:
            return selected

    if len(selected) >= max_cards:
        return selected[:max_cards]

    # Pad with the rejected duplicates so the engineer always has
    # max_cards (when available). Order preserved.
    if len(selected) < max_cards:
        leftover_ids = {id(c) for c in selected}
        for cand in candidates:
            if len(selected) >= max_cards:
                break
            if id(cand) in leftover_ids:
                continue
            selected.append(cand)

    return selected[:max_cards]
