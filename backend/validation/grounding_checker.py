"""
Grounding Checker — verifies answer faithfulness to source documents.
Detects claims not supported by retrieved context, flags answers
sourced from superseded documents, and checks for fabricated specifics
(URLs, phone numbers, email addresses not in the context).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from backend.config import settings

logger = logging.getLogger("acadia-log-iq")


@dataclass
class GroundingResult:
    """
    Result of grounding verification.

    Fields:
        passed          — overall grounding check passed
        grounding_score — 0.0-1.0, fraction of answer grounded in context
        issues          — list of specific grounding issues found
        version_warning — non-empty if superseded sources detected
        fabrications    — specifics found in answer but not in context
    """
    passed: bool = True
    grounding_score: float = 1.0
    issues: List[str] = field(default_factory=list)
    version_warning: str = ""
    fabrications: List[str] = field(default_factory=list)


# Patterns for hard fabrications (URLs, emails, phone numbers, ticket
# IDs) — these trigger full Case B fallback (answer replaced) because
# they cannot be inferred from context and a wrong value is direct
# misinformation, not a soft elaboration.
_URL_PATTERN = re.compile(r"https?://[\w./-]+", re.I)
_EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w{2,}\b", re.I)
_PHONE_PATTERN = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")

# Ticket / record identifier shapes — anything matching this in the
# answer is a record-reference claim. If the same identifier isn't in
# the retrieved context, the LLM made it up (the broadcast-storm eval
# case invented "INC-LAN-88902"). This MUST be a hard fabrication, not
# a soft penalty, because users treat ticket IDs as authoritative
# references.
#
# Patterns covered:
#   INC-12345, CHG-001                  (letter-prefix + digits)
#   INC-LAN-88902, INC-TITAN-812        (letter-prefix + word + digits)
#   CHG-2024-001                         (letter-prefix + year + seq)
#   INC0012345                          (no separator)
# Tightened to require digits SOMEWHERE in the captured token so
# generic phrases like "BGP-NEIGHBOR" aren't flagged.
_TICKET_ID_PATTERN = re.compile(
    r"\b("
    r"[A-Z]{2,5}-[A-Z]{2,}-\d{2,}"          # INC-LAN-88902, INC-TITAN-812
    r"|[A-Z]{2,5}-\d{4}-\d{2,}"             # CHG-2024-001
    r"|[A-Z]{2,5}-\d{3,}"                   # INC-12345, CHG-001
    r"|[A-Z]{2,5}\d{4,}"                    # INC0012345
    r")\b"
)

# Genuine ticket / incident / change reference prefixes. The HARD
# fabrication fail (answer replacement) fires ONLY for these. The shape
# above also matches NETWORK DEVICE HOSTNAMES and CIRCUIT IDs
# (e.g. RTR-TN-3001, FW-TN-3001, LTE-TN-3001, VER-TN-3001, FRO-TN-3001-FIB01)
# which are legitimate CONFIG identifiers, not authoritative ticket
# pointers — flagging them nuked correct store-config / network answers.
# Restricting the hard fail to real ticket prefixes keeps the original
# protection (invented INC-/CHG-/TKT- refs) while ending the false
# positives on device names.
_TICKET_ID_PREFIXES = frozenset({
    "INC", "CHG", "TKT", "REQ", "PRB", "PROB", "TASK", "SR",
    "CASE", "RITM", "CR", "WO", "TICKET",
})


def _is_real_ticket_ref(token: str) -> bool:
    """True only when the identifier's leading alpha segment is a known
    ticket/incident/change prefix. Excludes device hostnames / circuit ids
    (RTR-, FW-, LTE-, VER-, FRO-, SW-, AP-, …)."""
    m = re.match(r"^([A-Z]+)", token or "")
    return bool(m and m.group(1) in _TICKET_ID_PREFIXES)


# ---------------------------------------------------------------------------
# Specifics-fabrication detector (soft signal — reduces grounding score
# rather than triggering full Case B fallback)
# ---------------------------------------------------------------------------
# Catches the failure pattern shown in the manual eval (Q3 timer
# hallucination, Q10 CLI expansion, Q12 invented warning): the LLM
# produces plausible technical specifics that are NOT present verbatim
# in the retrieved documents. The prompt-side "DO NOT FABRICATE
# SPECIFICS" directive is the primary defence; this detector catches
# what slips through.
#
# Why soft signal: a single fabricated timer value shouldn't nuke an
# otherwise-good answer (that's Case B's job, reserved for URLs/emails).
# Instead we count fabricated specifics and proportionally reduce the
# grounding score. If enough specifics are fabricated, the lowered
# score itself trips the gate via Case D.

# Numeric specifics with a unit suffix (timers, percentages, durations).
# Matches: "180s", "180 seconds", "5%", "10ms", "30 min", "port 179",
# "v1.4.2". Caller compares answer-side matches against context-side
# matches; deltas are flagged.
_NUMERIC_UNIT_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*"
    r"(?:s(?:ec(?:ond)?s?)?|ms|m(?:in(?:ute)?s?)?|h(?:ours?)?|"
    r"d(?:ays?)?|%|percent|"
    r"port|ports|"
    r"v\d|version)\b",
    re.IGNORECASE,
)

# Command/CLI tokens — common command verbs followed by an argument.
# Catches "show ip route", "router bgp 65000", "interface GigabitEthernet0/1",
# "configure terminal", "no shutdown", "frame-relay map bridge ...". A
# fabricated CLI snippet is the Q10 failure mode.
_CLI_RE = re.compile(
    r"\b(?:show|router|interface|no|configure|conf\s+t|"
    r"frame-relay|access-list|ip\s+route|"
    r"set|clear|debug|undebug|enable|disable|"
    r"copy|reload|write|tunnel|crypto)"
    r"\s+[\w./:-]+"
    # Optional second arg — MUST contain a digit, slash, colon, or
    # dot (i.e. look like a real CLI token, not an English word like
    # "the" / "a" / "interface"). This kills the false-positive where
    # "no shutdown the interface" was captured as a fake CLI command.
    r"(?:\s+[\w./:-]*[\d/:.][\w./:-]*)?",
    re.IGNORECASE,
)

# Version / RFC / model numbers — "v1.4.2", "RFC 4271", "ISR-4451".
_VERSION_RE = re.compile(
    r"\b(?:RFC\s*\d{3,5}|v\d+\.\d+(?:\.\d+)?|"
    r"\b[A-Z]{2,}-?\d{2,5}[A-Z]?)\b",
)


def _detect_fabricated_specifics(
    answer: str, doc_context: str
) -> List[str]:
    """
    Return a list of *suspicious* specifics — exact-string tokens that
    appear in the answer but NOT in the document context.

    Soft signal: caller treats each entry as a grounding-score penalty
    rather than a fail-the-answer event. This is deliberately tuned to
    have a low false-negative rate; some legitimate phrases (e.g.
    common synonyms) will get flagged. The grounding score still
    decides pass/fail.

    Comparison strategy: case-insensitive, whitespace-normalized exact
    substring presence in `doc_context`. We don't try fuzzy match —
    the goal is "did the LLM supply a verbatim specific that the
    documents don't contain?".
    """
    if not answer or not doc_context:
        return []

    ctx_normalized = " ".join(doc_context.lower().split())
    suspicious: List[str] = []
    seen: set = set()

    # Trailing punctuation to strip so "show bgp neighbors." doesn't
    # falsely fail to match "show bgp neighbors" in context.
    _TRIM_CHARS = ".,;:!?)]}'\""

    for pattern in (_NUMERIC_UNIT_RE, _CLI_RE, _VERSION_RE):
        for raw in pattern.findall(answer):
            token = raw.strip().lower().rstrip(_TRIM_CHARS)
            token_norm = " ".join(token.split())
            if not token_norm or token_norm in seen:
                continue
            seen.add(token_norm)
            if token_norm not in ctx_normalized:
                suspicious.append(raw.strip().rstrip(_TRIM_CHARS))

    return suspicious


def check_grounding(
    *,
    query: str,
    answer: str,
    doc_context: str,
    ranked_chunks: List[Tuple[str, str, Dict[str, Any], float]],
    source_names: List[str],
) -> GroundingResult:
    """
    Verify that the answer is faithful to the source documents.

    Checks performed:
    1. Fabricated specifics — URLs, emails, phone numbers in answer but not in context
    2. Version awareness — detect if top sources are from superseded documents
    3. Grounding coverage — what fraction of answer's substantive sentences
       have supporting evidence in the retrieved context
    4. Hallucination pattern — known LLM hallucination phrases

    Returns GroundingResult with pass/fail, score, and detailed issues.
    """
    result = GroundingResult()
    ctx_lower = (doc_context or "").lower()
    answer_lower = (answer or "").lower()

    # ==================================================================
    # Check 1: Fabricated specifics
    # URLs, emails, phone numbers in answer that aren't in the context
    # ==================================================================
    for pattern, label in [
        (_URL_PATTERN, "URL"),
        (_EMAIL_PATTERN, "email"),
        (_PHONE_PATTERN, "phone number"),
        # Ticket / record identifiers — invented record references are
        # the worst class of hallucination because users treat them as
        # authoritative pointers. Catching them here (hard fabrication
        # → Case B answer replacement) is the right severity.
        (_TICKET_ID_PATTERN, "ticket / record identifier"),
    ]:
        answer_matches = set(pattern.findall(answer or ""))
        context_matches = set(pattern.findall(doc_context or ""))

        fabricated = answer_matches - context_matches
        if pattern is _TICKET_ID_PATTERN:
            # Only genuine ticket/incident references are hard fabrications;
            # network device hostnames / circuit ids that share the shape
            # (RTR-TN-3001, FW-TN-3001, VER-TN-3001, FRO-TN-3001-FIB01) are
            # config identifiers and must not replace the answer.
            fabricated = {f for f in fabricated if _is_real_ticket_ref(f)}
        for fab in fabricated:
            result.fabrications.append(f"Fabricated {label}: {fab}")
            result.issues.append(f"Answer contains {label} '{fab}' not found in source documents")

    # ==================================================================
    # Check 2: Version awareness
    # Flag if top-ranked sources are from superseded document versions
    # ==================================================================
    if settings.VALIDATION_WARN_SUPERSEDED and ranked_chunks:
        superseded_sources = set()
        active_sources = set()

        for _, _, meta, _ in ranked_chunks[:5]:
            source_name = meta.get("source", "unknown")
            meta_json = meta.get("metadata_json", {})

            if isinstance(meta_json, dict):
                status = meta_json.get("status", "active")
                if status == "superseded":
                    superseded_sources.add(source_name)
                else:
                    active_sources.add(source_name)

        if superseded_sources and not active_sources:
            # ALL top sources are superseded — strong warning
            result.version_warning = (
                "Note: This answer is based on document versions that may have been superseded. "
                "A newer version may exist with updated information. "
                f"Superseded sources: {', '.join(sorted(superseded_sources))}"
            )
            result.issues.append("All top sources are from superseded document versions")
        elif superseded_sources:
            # Mix of superseded and active — mild warning
            result.version_warning = (
                "Note: Some source documents may have newer versions available. "
                f"Potentially outdated: {', '.join(sorted(superseded_sources))}"
            )

    # ==================================================================
    # Check 3: Sentence-level grounding
    # Split answer into sentences and check each has context support
    # ==================================================================
    sentences = re.split(r"[.!?\n]", answer or "")
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

    if sentences:
        grounded_count = 0
        for sentence in sentences:
            # Extract key terms from the sentence
            terms = {
                t for t in re.findall(r"\w+", sentence.lower())
                if len(t) > 3 and t not in {"this", "that", "with", "from", "have", "been",
                                              "would", "could", "should", "which", "their",
                                              "there", "these", "those", "about", "after",
                                              "before", "between", "through", "during"}
            }
            if not terms:
                grounded_count += 1  # skip trivial sentences
                continue

            # Check if at least 40% of terms appear in context
            hits = sum(1 for t in terms if t in ctx_lower)
            if len(terms) > 0 and hits / len(terms) >= 0.40:
                grounded_count += 1

        result.grounding_score = grounded_count / len(sentences) if sentences else 1.0
    else:
        result.grounding_score = 1.0  # no sentences to check

    # ==================================================================
    # Check 3b: Specifics-fabrication detector (SOFT signal)
    # ==================================================================
    # Catches the manual-eval failure pattern: the LLM produces a
    # plausible technical specific (timer value, CLI fragment, version
    # number) that is NOT present verbatim in retrieved chunks. This
    # is the Q3/Q10/Q12 pattern. We don't fail outright (that would
    # nuke borderline answers); instead each suspicious specific
    # subtracts 0.10 from the grounding score. If enough accumulate,
    # the lowered score trips the threshold via Case D in the validator.
    suspicious = _detect_fabricated_specifics(answer or "", doc_context or "")
    if suspicious:
        # Penalty tuning notes (see broadcast-storm eval failure):
        # Previously 0.10 per token, cap 0.50. A 3-fabrication answer
        # with high sentence-level grounding (0.90) only dropped to
        # 0.60 — still above the 0.40 threshold — and the answer with
        # invented "INC-LAN-88902" went out to the user.
        # Now 0.20 per token, cap 0.60: two fabricated specifics
        # drop 0.90 → 0.50; three drop to 0.30 (below threshold → fail).
        # The cap stops a torrent from bypassing the gate logic
        # entirely; it doesn't prevent a legitimate fail.
        penalty = min(0.60, 0.20 * len(suspicious))
        original_score = result.grounding_score
        result.grounding_score = max(0.0, result.grounding_score - penalty)
        # Surface the most likely culprits in issues so operators can
        # see what triggered the penalty without re-running the detector.
        preview = ", ".join(repr(s)[:40] for s in suspicious[:5])
        result.issues.append(
            f"Specifics-fabrication penalty: {len(suspicious)} suspicious "
            f"token(s) not in context (e.g. {preview}). Score "
            f"{original_score:.2f} → {result.grounding_score:.2f}"
        )
        logger.info(
            "[grounding] specifics-fabrication: %d tokens penalty=%.2f "
            "score %.3f -> %.3f",
            len(suspicious), penalty, original_score, result.grounding_score,
        )

    # ==================================================================
    # Check 4: Overall pass/fail
    # ==================================================================
    if result.fabrications:
        result.passed = False
        result.issues.append(f"Found {len(result.fabrications)} fabricated specifics")

    if result.grounding_score < settings.VALIDATION_MIN_GROUNDING:
        result.passed = False
        result.issues.append(
            f"Grounding score {result.grounding_score:.2f} below threshold "
            f"{settings.VALIDATION_MIN_GROUNDING}"
        )

    logger.info(
        "Grounding: score=%.3f, %d fabrications, version_warn=%s, %d issues -> %s",
        result.grounding_score, len(result.fabrications),
        bool(result.version_warning), len(result.issues),
        "PASS" if result.passed else "FAIL",
    )

    return result
