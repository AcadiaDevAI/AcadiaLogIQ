"""Sprint 10 Stage 1B — Do Not Chase aggregator.

Per spec §3.3, source field families (per cohort ticket):
    - Knowledge_Base[*].semantic_unit_educational.diagnostic_pathway
        .elimination_checklist[]                       (list of strings)
    - Knowledge_Base[*].semantic_unit_educational.diagnostic_logic
        .differential_diagnosis[]                      (list of strings)
    - Knowledge_Base[*].semantic_unit_educational.diagnostic_pathway
        .false_path_red_herrings[]                     (list of
        ``{misleading_signal, rule_out_logic}`` dicts — only source that
        carries real per-item rule-out copy)
    - Troubleshooting_Ledger.Diagnostic_Tests_Executed[] — only entries
        whose ``outcome`` matches normal | healthy | not_root_cause |
        ruled_out (case-insensitive contains). Currently absent from
        the production corpus (0 tickets carry it); the branch is kept
        for forward-compatibility with future enrichment.

Aggregation:
    - Concat into flat (signal, rule_out) tuples.
    - Normalize signal (lower, strip punctuation, collapse whitespace).
    - Group; keep groups with count >= MIN_OCCURRENCE_COUNT.
    - Sort desc by count. Cap at MAX_ENTRIES.

Compatibility:
    The current production gold-ticket schema nests these fields one
    level deeper, under ``Knowledge_Base[*].semantic_unit_educational``.
    A small wrapper (`_se_kb`) resolves both the nested layout and the
    legacy flat layout in one place so callers (and tests) stay shape-
    agnostic.

Logging:
    Emits one INFO line per call summarising cohort coverage:
    ``[stage1b] cohort=N with_pairs=M coverage=P% groups=K
    reason=... entries=E``. Lets prod monitoring spot ingestion
    regressions (a batch of tickets without populated false-path /
    elimination data shows up as a coverage drop) without requiring
    every caller to inspect the result object.

Pure function — no DB, no LLM.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .schemas import DoNotChaseEntry, Stage1bDoNotChase


logger = logging.getLogger("acadia-log-iq")


_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s\-]")
_RULED_OUT_KEYWORDS = ("normal", "healthy", "not_root_cause", "ruled_out", "ruled out")


# Sprint 10.4 §4.2 — surface single-occurrence signals. Real-corpus
# review showed valuable single-ticket red herrings being filtered out
# at the prior min_count=2 threshold. The Do-Not-Chase panel is more
# useful when it errs on the side of including a hint than hiding one.
MIN_OCCURRENCE_COUNT = 1
MAX_ENTRIES = 8


def _normalize(s: str) -> str:
    s = _PUNCT.sub("", str(s).lower())
    return _WS.sub(" ", s).strip()


def _is_ruled_out_outcome(outcome: Any) -> bool:
    if not outcome:
        return False
    s = str(outcome).lower()
    return any(k in s for k in _RULED_OUT_KEYWORDS)


def _incident_number(ticket: Dict[str, Any]) -> Optional[str]:
    meta = ticket.get("Metadata") if isinstance(ticket, dict) else None
    if not isinstance(meta, dict):
        return None
    inc = meta.get("Incident_Number")
    return str(inc).strip() if inc else None


def _kb_entries(ticket: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(ticket, dict):
        return []
    kb = ticket.get("Knowledge_Base")
    if isinstance(kb, dict):
        return [kb]
    if isinstance(kb, list):
        return [k for k in kb if isinstance(k, dict)]
    return []


def _se_kb(kb_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Return the KB sub-tree that holds diagnostic data, tolerating
    two known layouts:

    * Nested (current ingest):
        ``Knowledge_Base[*].semantic_unit_educational`` — diagnostic
        sections (``diagnostic_pathway``, ``diagnostic_logic``) live
        one level under this wrapper.
    * Flat (legacy / hand-written tickets): the diagnostic sections
        sit directly on the KB entry.

    Always returns a dict so callers can chain ``.get(...)`` without
    None-guards. Mirrors the same shape resolver used by
    ``stage1_smoking_gun._pivot_payload`` so the two stages stay in
    lock-step on schema interpretation.
    """
    if not isinstance(kb_entry, dict):
        return {}
    sue = kb_entry.get("semantic_unit_educational")
    if isinstance(sue, dict):
        return sue
    return kb_entry


def _harvest_pairs(ticket: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Pull (signal, rule_out) pairs from all four field families."""
    pairs: List[Tuple[str, str]] = []

    for kb in _kb_entries(ticket):
        sue = _se_kb(kb)

        diagnostic_pathway = sue.get("diagnostic_pathway") if isinstance(sue.get("diagnostic_pathway"), dict) else {}
        diagnostic_logic   = sue.get("diagnostic_logic")   if isinstance(sue.get("diagnostic_logic"),   dict) else {}

        # diagnostic_pathway.elimination_checklist[]
        # In the corpus this is a list of bare strings ("Power equipment
        # verified on-site …"). Per-item rule_out_logic isn't supplied
        # by the source schema for this field; the dict-shaped branch
        # remains for legacy / hand-written tickets that stamp it that
        # way.
        ec = diagnostic_pathway.get("elimination_checklist")
        if isinstance(ec, list):
            for item in ec:
                if isinstance(item, dict):
                    sig = item.get("signal") or item.get("misleading_signal") or item.get("item") or ""
                    rule = item.get("rule_out_logic") or item.get("reason") or ""
                    if sig:
                        pairs.append((str(sig), str(rule)))
                elif isinstance(item, str) and item.strip():
                    pairs.append((item, ""))

        # diagnostic_logic.differential_diagnosis[] (list of bare
        # diagnosis strings in the production corpus).
        dd = diagnostic_logic.get("differential_diagnosis")
        if isinstance(dd, list):
            for item in dd:
                if isinstance(item, dict):
                    sig = item.get("signal") or item.get("misleading_signal") or item.get("item") or ""
                    rule = item.get("rule_out_logic") or item.get("reason") or ""
                    if sig:
                        pairs.append((str(sig), str(rule)))
                elif isinstance(item, str) and item.strip():
                    pairs.append((item, ""))

        # diagnostic_pathway.false_path_red_herrings[] — the only
        # source that carries real per-item rule-out logic in the gold
        # schema. Dict shape: ``{misleading_signal, rule_out_logic}``.
        fp = diagnostic_pathway.get("false_path_red_herrings")
        if isinstance(fp, list):
            for item in fp:
                if isinstance(item, dict):
                    sig = item.get("misleading_signal") or ""
                    rule = item.get("rule_out_logic") or ""
                    if sig:
                        pairs.append((str(sig), str(rule)))
                elif isinstance(item, str) and item.strip():
                    pairs.append((item, ""))

    # Troubleshooting_Ledger.Diagnostic_Tests_Executed[]
    tl = ticket.get("Troubleshooting_Ledger") if isinstance(ticket.get("Troubleshooting_Ledger"), dict) else {}
    tests = tl.get("Diagnostic_Tests_Executed") if isinstance(tl, dict) else None
    if isinstance(tests, list):
        for entry in tests:
            if not isinstance(entry, dict):
                continue
            if not _is_ruled_out_outcome(entry.get("outcome")):
                continue
            sig = entry.get("test") or entry.get("test_name") or entry.get("name") or ""
            rule = (
                entry.get("conclusion")
                or entry.get("rule_out_logic")
                or entry.get("notes")
                or ""
            )
            if sig:
                pairs.append((str(sig), str(rule)))

    return pairs


def build_stage1b(
    cohort: List[Dict[str, Any]],
    min_count: int = MIN_OCCURRENCE_COUNT,
    max_entries: int = MAX_ENTRIES,
) -> Stage1bDoNotChase:
    """See module docstring.

    Sprint 10.4 — `min_count` defaults to MIN_OCCURRENCE_COUNT (1).
    Lowered from 2 so single-ticket red herrings surface in entries.
    Reason taxonomy collapsed to three values; `below_threshold` is
    structurally unreachable at min_count=1 and was dropped from the
    schema in 10.4.

      - "populated"    → entries returned, panel renders normally
      - "no_data"      → cohort had zero false_path / elimination
                         pairs to harvest
      - "no_recurring" → harvested pairs exist but every group is
                         count<min_count (kept as a safety branch —
                         only fires when callers pass min_count > 1)
    """
    n = len(cohort or [])
    if n == 0:
        return Stage1bDoNotChase(empty=True, reason="no_data")

    # group key = normalised signal → metadata
    groups: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "raw_signals": [],
        "rule_outs": [],
        "incidents": [],
    })

    raw_pairs_count = 0
    tickets_with_pairs = 0
    for ticket in cohort:
        inc = _incident_number(ticket) or ""
        ticket_contributed = False
        for sig, rule in _harvest_pairs(ticket):
            if not sig or not str(sig).strip():
                continue
            key = _normalize(sig)
            if not key:
                continue
            raw_pairs_count += 1
            ticket_contributed = True
            g = groups[key]
            g["raw_signals"].append(sig.strip())
            if rule and rule.strip():
                g["rule_outs"].append(rule.strip())
            if inc and inc not in g["incidents"]:
                g["incidents"].append(inc)
        if ticket_contributed:
            tickets_with_pairs += 1

    coverage_pct = round(100.0 * tickets_with_pairs / n) if n else 0

    def _emit(result: Stage1bDoNotChase) -> Stage1bDoNotChase:
        # One log line per call so prod monitoring can spot ingestion
        # regressions (a batch of tickets without populated false-path
        # / elimination data shows up as a coverage drop).
        logger.info(
            "[stage1b] cohort=%d with_pairs=%d coverage=%d%% "
            "groups=%d reason=%s entries=%d",
            n, tickets_with_pairs, coverage_pct,
            len(groups), result.reason, len(result.entries),
        )
        return result

    # Sprint 10.1 — classify the empty case so the frontend can pick
    # the right copy.
    if raw_pairs_count == 0 or not groups:
        return _emit(Stage1bDoNotChase(empty=True, reason="no_data"))

    # Filter by min_count (count = unique incident count) and sort
    qualifying = [
        (key, g)
        for key, g in groups.items()
        if len(g["incidents"]) >= min_count
    ]
    if not qualifying:
        # Sprint 10.4 — at the default min_count=1 every harvested
        # pair clears the threshold, so this branch is unreachable
        # in normal operation. It only fires when a caller explicitly
        # passes min_count > 1 and no group meets it; we still report
        # "no_recurring" so the frontend renders the appropriate
        # empty-state copy.
        return _emit(Stage1bDoNotChase(empty=True, reason="no_recurring"))

    # Primary sort: groups whose source tickets supplied a real
    # per-item rule_out_logic (i.e. came in via false_path_red_herrings
    # or a dict-shaped elimination_checklist entry) rank ahead of
    # groups that would otherwise fall back to the generic "checked
    # and found healthy" line. The §3.3 procedure asks for the *why*
    # alongside the *what* — an entry that carries it is strictly
    # more useful to the engineer regardless of occurrence count.
    #
    # Secondary sort: occurrence count (more incidents = stronger
    # evidence). Within each tier the most-supported entry leads.
    qualifying.sort(
        key=lambda kv: (bool(kv[1]["rule_outs"]), len(kv[1]["incidents"])),
        reverse=True,
    )

    out: List[DoNotChaseEntry] = []
    for _key, g in qualifying[:max_entries]:
        # Canonical signal — most-frequent verbatim.
        canonical = max(set(g["raw_signals"]), key=g["raw_signals"].count)
        # Canonical rule-out — longest non-empty string when the group
        # has any (most descriptive wins, mirroring Stage 1A's "longest
        # at tie" pattern). Otherwise the generic fallback fires.
        rule = (
            max(g["rule_outs"], key=len) if g["rule_outs"]
            else "Checked and found healthy in past incidents — verify but do not deep-dive."
        )
        out.append(DoNotChaseEntry(
            misleading_signal=canonical,
            rule_out_logic=rule,
            occurrence_count=len(g["incidents"]),
            seen_in_incidents=g["incidents"][:5],
        ))

    return _emit(Stage1bDoNotChase(entries=out, empty=False, reason="populated"))
