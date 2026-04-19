"""
Acceptance tests for Clarifier Refined-Query Grounding Fix brief.

Simulates the exact scenario: "How do I troubleshoot a router failure?"
Exercises the 3 patches: prompt-level rule, post-parse validator, enhanced log.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, List, Tuple
from unittest import mock

os.environ.setdefault("BEDROCK_SKIP_INIT", "1")

from backend.services import interactive_clarifier as ic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)


def _chunk(cid, text, meta, score):
    return (cid, text, meta, score)


def _router_failure_candidates():
    """Five candidates, all routery, distinct INC ids."""
    return [
        _chunk("c1", "Router crash rebooted after power failure at site.",
               {"primary_id": "INC-10036", "customer_name": "Enterprise-338",
                "component": "Router",
                "summary": "Equipment down after failed change request"}, 0.74),
        _chunk("c2", "Router OSPF adjacency flapping — misconfigured timers.",
               {"primary_id": "INC-10101", "customer_name": "Globex",
                "component": "Router",
                "summary": "OSPF flap on primary WAN link"}, 0.71),
        _chunk("c3", "Router BGP session reset every 90s, tracked to keepalive.",
               {"primary_id": "INC-10203", "customer_name": "Initech",
                "component": "Router",
                "summary": "BGP keepalive misconfig causing reset"}, 0.70),
        _chunk("c4", "Router packet drop during evening peak traffic.",
               {"primary_id": "INC-10312", "customer_name": "Umbrella",
                "component": "Router",
                "summary": "Peak-hour packet loss on edge router"}, 0.68),
        _chunk("c5", "Router interface down after maintenance window.",
               {"primary_id": "INC-10420", "customer_name": "Hooli",
                "component": "Router",
                "summary": "Interface flap post-maintenance"}, 0.66),
    ]


# -----------------------------------------------------------------------------
# Test 1: Happy path — Haiku emits correct refined_queries (with ids)
# -----------------------------------------------------------------------------
HAIKU_GOOD_JSON = {
    "context_summary": "Multiple router-failure tickets match — which one?",
    "options": [
        {"id": "opt_1",
         "label": "Globex: OSPF flap on primary WAN link",
         "refined_query": "How do I troubleshoot INC-10101 - the router OSPF flap for Globex on the primary WAN link?"},
        {"id": "opt_2",
         "label": "Initech: BGP keepalive reset loop",
         "refined_query": "How do I troubleshoot INC-10203 - the BGP keepalive reset loop for Initech?"},
        {"id": "opt_3",
         "label": "Umbrella: peak-hour packet loss",
         "refined_query": "How do I troubleshoot INC-10312 - peak-hour packet loss on Umbrella's edge router?"},
        {"id": "opt_4",
         "label": "Enterprise-338: Equipment down after failed change request",
         "refined_query": "How do I troubleshoot INC-10036 - the router failure for Enterprise-338 caused by a failed change request?"},
        {"id": "opt_other",
         "label": "Something else - let me clarify",
         "refined_query": ""},
    ],
}


def test_happy_path_ids_preserved():
    ic._SESSION_COUNTS.clear()
    ic._SESSION_COUNT_TIMESTAMPS.clear()
    with mock.patch.object(ic, "_invoke_haiku", return_value=json.dumps(HAIKU_GOOD_JSON)):
        res = ic.try_clarify(
            query="How do I troubleshoot a router failure?",
            ranked_chunks=_router_failure_candidates(),
            triage_confidence=0.45,
            carried_identifiers=None,
            session_id="sess_happy",
            recent_messages=[],
        )
    assert res.needs_clarification, f"expected trigger, got skip={res.skip_reason}"
    # All 4 content options should be present and carry INC-xxxx ids
    content = [o for o in res.options if o.id != "opt_other"]
    assert len(content) == 4, f"expected 4 content options, got {len(content)}"
    for o in content:
        assert "INC-" in o.refined_query, \
            f"option {o.id} missing INC- id in refined_query: {o.refined_query!r}"
    # The Enterprise-338 option must specifically carry INC-10036
    ent = next(o for o in content if "Enterprise-338" in o.label)
    assert "INC-10036" in ent.refined_query, \
        f"Enterprise-338 option lost INC-10036: {ent.refined_query!r}"
    print(f"  -> Enterprise-338 refined_query: {ent.refined_query}")


# -----------------------------------------------------------------------------
# Test 2: Repair path — Haiku drops one id, validator repairs via customer match
# -----------------------------------------------------------------------------
HAIKU_MISSING_ID_JSON = {
    "context_summary": "Multiple tickets match.",
    "options": [
        {"id": "opt_1",
         "label": "Globex: OSPF flap on primary WAN link",
         "refined_query": "How do I troubleshoot INC-10101 OSPF flap?"},
        {"id": "opt_2",
         "label": "Enterprise-338: Equipment down after failed change request",
         "refined_query": "How do I troubleshoot the Enterprise-338 router failure?"},
        {"id": "opt_other",
         "label": "Something else",
         "refined_query": ""},
    ],
}


def test_repair_path():
    ic._SESSION_COUNTS.clear()
    ic._SESSION_COUNT_TIMESTAMPS.clear()
    with mock.patch.object(ic, "_invoke_haiku", return_value=json.dumps(HAIKU_MISSING_ID_JSON)):
        res = ic.try_clarify(
            query="How do I troubleshoot a router failure?",
            ranked_chunks=_router_failure_candidates(),
            triage_confidence=0.45,
            carried_identifiers=None,
            session_id="sess_repair",
            recent_messages=[],
        )
    assert res.needs_clarification, f"skip={res.skip_reason}"
    opt2 = next((o for o in res.options if o.id == "opt_2"), None)
    assert opt2 is not None, "opt_2 unexpectedly dropped"
    assert "INC-10036" in opt2.refined_query, \
        f"repair failed — INC-10036 not present: {opt2.refined_query!r}"
    print(f"  -> repaired opt_2 refined_query: {opt2.refined_query}")


# -----------------------------------------------------------------------------
# Test 3: Drop path — Haiku emits bogus label with no id; validator drops it
# -----------------------------------------------------------------------------
HAIKU_BOGUS_JSON = {
    "context_summary": "Multiple tickets match.",
    "options": [
        {"id": "opt_1",
         "label": "Globex: OSPF flap on primary WAN link",
         "refined_query": "How do I troubleshoot INC-10101 OSPF flap?"},
        {"id": "opt_2",
         "label": "Something vague about networking",
         "refined_query": "How do I troubleshoot a network problem?"},
        {"id": "opt_other",
         "label": "Something else",
         "refined_query": ""},
    ],
}


def test_drop_path():
    ic._SESSION_COUNTS.clear()
    ic._SESSION_COUNT_TIMESTAMPS.clear()
    with mock.patch.object(ic, "_invoke_haiku", return_value=json.dumps(HAIKU_BOGUS_JSON)):
        res = ic.try_clarify(
            query="How do I troubleshoot a router failure?",
            ranked_chunks=_router_failure_candidates(),
            triage_confidence=0.45,
            carried_identifiers=None,
            session_id="sess_drop",
            recent_messages=[],
        )
    # After dropping opt_2, fewer than 2 real options remain → skip_reason should
    # be insufficient_distinct_options per existing logic.
    if res.needs_clarification:
        ids = [o.id for o in res.options]
        assert "opt_2" not in ids, f"opt_2 should have been dropped, got {ids}"
    else:
        assert res.skip_reason == "insufficient_distinct_options", \
            f"expected insufficient_distinct_options, got {res.skip_reason}"
    print(f"  -> drop path: needs_clar={res.needs_clarification} skip={res.skip_reason}")


# -----------------------------------------------------------------------------
# Test 4: Selection expansion carries identifier through to /ask
# -----------------------------------------------------------------------------
def test_selection_expansion_propagates_identifier():
    """Simulates the frontend click path: refined_query with INC-10036 lands in
    the next /ask request. Uses the api helper _expand_clarification_selection."""
    # Import lazily to avoid FastAPI init cost when not needed
    from backend import api as _api
    # Seed the store
    cid = _api._store_clarification(
        session_id="sess_sel",
        options=[
            {"id": "opt_1", "label": "Globex: OSPF flap",
             "refined_query": "How do I troubleshoot INC-10101 OSPF flap?"},
            {"id": "opt_4", "label": "Enterprise-338: Equipment down",
             "refined_query": "How do I troubleshoot INC-10036 - the router failure for Enterprise-338 caused by a failed change request?"},
            {"id": "opt_other", "label": "Something else", "refined_query": ""},
        ],
        original_query="How do I troubleshoot a router failure?",
    )

    sel = _api.ClarificationSelection(
        clarification_id=cid,
        selected_option_id="opt_4",
        free_text=None,
    )
    expanded = _api._expand_clarification_selection(
        session_id="sess_sel",
        selection=sel,
        fallback_query="How do I troubleshoot a router failure?",
    )
    assert "INC-10036" in expanded, \
        f"expanded query missing INC-10036: {expanded!r}"
    print(f"  -> expanded query: {expanded}")

    # Also verify _extract_identifiers picks it up downstream
    from backend.retrieval.orchestrator import _extract_identifiers
    ids = _extract_identifiers(expanded)
    assert any("INC-10036" in str(i) for i in ids), \
        f"_extract_identifiers returned: {ids}"
    print(f"  -> _extract_identifiers: {ids}")


def run(name, fn):
    print(f"\n===== {name} =====")
    try:
        fn()
        print(f"PASS: {name}")
        return True
    except Exception as e:
        print(f"FAIL: {name}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    results = []
    results.append(("happy_path_ids_preserved",
                    run("Happy path: ids preserved", test_happy_path_ids_preserved)))
    results.append(("repair_path_fills_missing_id",
                    run("Repair path: fills missing id", test_repair_path)))
    results.append(("drop_path_unrepairable",
                    run("Drop path: unrepairable option", test_drop_path)))
    results.append(("selection_expansion",
                    run("Selection expansion carries id", test_selection_expansion_propagates_identifier)))

    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"\n=========  TOTAL: {passed}/{total} PASSED =========")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
