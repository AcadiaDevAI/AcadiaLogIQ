"""
Brief 3 acceptance harness — simulates a single session by maintaining an
in-memory message list and invoking rewrite_query() for each turn. For
aggregation scenarios we additionally run detect_aggregation_intent_v2 on
the rewritten query to confirm downstream routing.

No real backend server is needed — this exercises the rewriter contract
directly, mirroring what api.py will hand it for each turn.
"""
from __future__ import annotations

import io
import os
import sys
from typing import Dict, List

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.query_rewriter import rewrite_query
from backend.retrieval.metadata_sql import detect_aggregation_intent_v2


# ---- Simulated session state ---------------------------------------------
session: List[Dict[str, str]] = []


def _append(role: str, content: str) -> None:
    session.append({"role": role, "content": content})


def _assistant_stub(turn_tag: str, rewritten: str) -> str:
    """
    Fake assistant answer used only to give the rewriter realistic context
    for subsequent turns. Uses the rewritten query text plus a generic
    acknowledgement so downstream turns have something concrete to point at.
    """
    return f"[assistant reply for turn {turn_tag} on: {rewritten}]"


def run_turn(label: str, user_text: str, *, fake_assistant: str | None = None) -> None:
    print(f"\n--- {label} ---")
    print(f"USER: {user_text!r}")
    history_before = list(session)
    result = rewrite_query(user_text, recent_messages=history_before)
    print(
        f"REWRITER -> was_rewritten={result.was_rewritten} "
        f"reason={result.reason} confidence={result.confidence:.2f}"
    )
    if result.was_rewritten:
        print(f"REWRITE: {result.rewritten_query!r}")
    else:
        print("REWRITE: (unchanged)")

    # For aggregation-style turns, also show detector routing.
    agg = detect_aggregation_intent_v2(result.rewritten_query)
    if agg is not None:
        filt_parts = []
        if agg.customer_name: filt_parts.append(f"customer={agg.customer_name}")
        if agg.priority: filt_parts.append(f"priority={agg.priority}")
        if agg.sla_target_met is not None: filt_parts.append(f"sla_met={agg.sla_target_met}")
        if agg.component: filt_parts.append(f"component={agg.component}")
        if agg.ranking_field: filt_parts.append(f"ranking_field={agg.ranking_field}")
        if agg.ranking_direction: filt_parts.append(f"ranking_direction={agg.ranking_direction}")
        filt = " ".join(filt_parts)
        print(f"AGG DETECT: op={agg.operation} {filt}".rstrip())
    else:
        print("AGG DETECT: None (falls through to retrieval / agents)")

    _append("user", user_text)
    assistant_text = fake_assistant or _assistant_stub(label, result.rewritten_query)
    _append("assistant", assistant_text)


# ========================================================================
# SCENARIOS 1-7 — single continuous session, no resets
# ========================================================================
print("\n" + "=" * 72)
print("Scenarios 1-7 (single continuous session)")
print("=" * 72)

# Scenario 1 — Pronoun resolution
run_turn(
    "Scenario 1a",
    "Tell me about INC-10015",
    fake_assistant=(
        "INC-10015 was a P1 incident at Enterprise-792. 50 barcode scanners "
        "failed after a firmware push; root cause was a bad driver bundle, "
        "resolved by rolling back to the prior firmware."
    ),
)
run_turn("Scenario 1b", "what caused it?")

# Scenario 2 — Ordinal resolution
run_turn(
    "Scenario 2a",
    "How many Nebula-Corp tickets?",
    fake_assistant=(
        "Nebula-Corp has 9 tickets: INC-10000, INC-10007, INC-10018, "
        "INC-10023, INC-10027, INC-10034, INC-10035, INC-10043, INC-10044."
    ),
)
run_turn("Scenario 2b", "what about the second one?")

# Scenario 3 — Ellipsis
run_turn(
    "Scenario 3a",
    "Tell me about INC-10005",
    fake_assistant=(
        "INC-10005 was a P2 incident for Enterprise-412 involving VDI session "
        "drops. The team restored service by restarting the broker."
    ),
)
run_turn("Scenario 3b", "and the resolution?")
run_turn("Scenario 3c", "and the customer?")

# Scenario 4 — Filter swap (aggregation routing follows through)
run_turn(
    "Scenario 4a",
    "How many Nebula-Corp tickets missed SLA?",
    fake_assistant="5 Nebula-Corp tickets missed SLA.",
)
run_turn("Scenario 4b", "same for Enterprise-859?")

# Scenario 5 — Ambiguity kept
# Brief says "hello" would trivial-short-circuit before rewriter in production;
# we keep the same pattern — simulate a trivial reply.
run_turn(
    "Scenario 5a",
    "hello",
    fake_assistant="Hello! Ask me anything about your uploaded documents.",
)
run_turn("Scenario 5b", "that thing we saw")

# Scenario 6 — Self-contained pass-through
run_turn(
    "Scenario 6a",
    "Tell me about INC-10000",
    fake_assistant="INC-10000 was a SIP/SBC outage at Nebula-Corp, resolved by licensing rotation.",
)
run_turn("Scenario 6b", "What was the root cause of INC-10015?")

# ========================================================================
# SCENARIO 7 — fresh session, first turn only
# ========================================================================
print("\n" + "=" * 72)
print("Scenario 7 (fresh session, no prior history)")
print("=" * 72)
session.clear()
run_turn("Scenario 7", "what about it?")


# ========================================================================
# SCENARIO 8 — 14-query regression matrix (isolated per query, no cross-turn
# carryover expected — each must be treated as self-contained)
# ========================================================================
print("\n" + "=" * 72)
print("Scenario 8 — 14-query regression matrix")
print("=" * 72)
session.clear()

regression_queries = [
    ("R1", "Tell me about INC-10001"),
    ("R2", "Tell me about INC-10005"),
    ("R3", "Tell me about INC-10008"),
    ("R4", "Tell me about INC-10041"),
    ("R5", "Tell me about INC-10002"),
    ("R6", "How many Nebula-Corp tickets?"),
    ("R7", "Which customer had the most incidents?"),
    ("R8", "How many tickets missed SLA?"),
    ("R9", "Which ticket had the highest resolution quality score?"),
    ("R10", "Compare INC-10005 and INC-10006"),
    ("R11", "What are common root causes for WiFi incidents?"),
    ("R12", "Tell me about Enterprise-338"),
    ("R13", "What resolutions were applied for Enterprise-338?"),
    ("R14", "What QA gaps were identified for Enterprise-338?"),
]

for label, q in regression_queries:
    # Each regression query runs in a FRESH session (no prior turns) — the
    # brief asserts every one is self-contained and must not be mutated.
    session.clear()
    run_turn(label, q)
