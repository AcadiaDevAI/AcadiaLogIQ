"""Sprint 10 Stage 1B — Do Not Chase aggregator.

Per spec §3.3:
  Source field families (per cohort ticket):
    - Knowledge_Base[*].diagnostic_pathway.elimination_checklist[]
    - Knowledge_Base[*].diagnostic_logic.differential_diagnosis[]
    - Knowledge_Base[*].false_path_red_herrings[].misleading_signal +
      .rule_out_logic
    - Troubleshooting_Ledger.Diagnostic_Tests_Executed[] — only entries
      whose `outcome` matches normal | healthy | not_root_cause |
      ruled_out (case-insensitive contains)

  Aggregation:
    - Concat into flat (signal, rule_out) tuples.
    - Normalize signal (lower, strip punctuation, collapse whitespace).
    - Group; keep groups with count >= 2.
    - Sort desc by count. Cap at 8.

Pure function — no DB, no LLM.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .schemas import DoNotChaseEntry, Stage1bDoNotChase


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


def _harvest_pairs(ticket: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Pull (signal, rule_out) pairs from all four field families."""
    pairs: List[Tuple[str, str]] = []

    for kb in _kb_entries(ticket):
        # diagnostic_pathway.elimination_checklist[]
        path = kb.get("diagnostic_pathway") if isinstance(kb.get("diagnostic_pathway"), dict) else {}
        ec = path.get("elimination_checklist") if isinstance(path, dict) else None
        if isinstance(ec, list):
            for item in ec:
                if isinstance(item, dict):
                    sig = item.get("signal") or item.get("misleading_signal") or item.get("item") or ""
                    rule = item.get("rule_out_logic") or item.get("reason") or ""
                    if sig:
                        pairs.append((str(sig), str(rule)))
                elif isinstance(item, str) and item.strip():
                    pairs.append((item, ""))

        # diagnostic_logic.differential_diagnosis[]
        dl = kb.get("diagnostic_logic") if isinstance(kb.get("diagnostic_logic"), dict) else {}
        dd = dl.get("differential_diagnosis") if isinstance(dl, dict) else None
        if isinstance(dd, list):
            for item in dd:
                if isinstance(item, dict):
                    sig = item.get("signal") or item.get("misleading_signal") or item.get("item") or ""
                    rule = item.get("rule_out_logic") or item.get("reason") or ""
                    if sig:
                        pairs.append((str(sig), str(rule)))
                elif isinstance(item, str) and item.strip():
                    pairs.append((item, ""))

        # false_path_red_herrings[]
        fp = kb.get("false_path_red_herrings")
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
    if not cohort:
        return Stage1bDoNotChase(empty=True, reason="no_data")

    # group key = normalised signal → metadata
    groups: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "raw_signals": [],
        "rule_outs": [],
        "incidents": [],
    })

    raw_pairs_count = 0
    for ticket in cohort:
        inc = _incident_number(ticket) or ""
        for sig, rule in _harvest_pairs(ticket):
            if not sig or not str(sig).strip():
                continue
            key = _normalize(sig)
            if not key:
                continue
            raw_pairs_count += 1
            g = groups[key]
            g["raw_signals"].append(sig.strip())
            if rule and rule.strip():
                g["rule_outs"].append(rule.strip())
            if inc and inc not in g["incidents"]:
                g["incidents"].append(inc)

    # Sprint 10.1 — classify the empty case so the frontend can pick
    # the right copy.
    if raw_pairs_count == 0 or not groups:
        return Stage1bDoNotChase(empty=True, reason="no_data")

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
        return Stage1bDoNotChase(empty=True, reason="no_recurring")

    qualifying.sort(key=lambda x: len(x[1]["incidents"]), reverse=True)

    out: List[DoNotChaseEntry] = []
    for _key, g in qualifying[:max_entries]:
        # Canonical signal — most-frequent verbatim
        canonical = max(set(g["raw_signals"]), key=g["raw_signals"].count)
        rule = (
            g["rule_outs"][0] if g["rule_outs"]
            else "Checked and found healthy in past incidents — verify but do not deep-dive."
        )
        out.append(DoNotChaseEntry(
            misleading_signal=canonical,
            rule_out_logic=rule,
            occurrence_count=len(g["incidents"]),
            seen_in_incidents=g["incidents"][:5],
        ))

    return Stage1bDoNotChase(entries=out, empty=False, reason="populated")
