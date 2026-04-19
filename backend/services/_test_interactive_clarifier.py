"""
Unit-level acceptance tests 1-9 for the Interactive Clarifier.

Tests 10-16 require the live FastAPI stack + frontend and are out of scope here.
"""
from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict, List, Tuple
from unittest import mock

import os
os.environ.setdefault("BEDROCK_SKIP_INIT", "1")

from backend.services import interactive_clarifier as ic
from backend.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)


def _chunk(cid: str, text: str, meta: Dict[str, Any], score: float) -> Tuple[str, str, Dict[str, Any], float]:
    return (cid, text, meta, score)


def _three_distinct_chunks():
    return [
        _chunk("c1", "Stripe API returned 500 when processing refund.",
               {"primary_id": "INC-1001", "customer_name": "Stripe", "component": "API"}, 0.72),
        _chunk("c2", "Shopify webhook retry storm caused queue backup.",
               {"primary_id": "INC-1002", "customer_name": "Shopify", "component": "Webhooks"}, 0.69),
        _chunk("c3", "Plaid balance endpoint returned stale cached data.",
               {"primary_id": "INC-1003", "customer_name": "Plaid", "component": "Balance API"}, 0.68),
        _chunk("c4", "Different preview.",
               {"primary_id": "INC-1004", "customer_name": "Adyen", "component": "Settlements"}, 0.65),
    ]


def _same_record_chunks(primary_id="INC-9001"):
    base_meta = {"primary_id": primary_id, "customer_name": "Acme", "component": "Billing"}
    return [
        _chunk(f"c{i}", f"Chunk {i} of {primary_id}", base_meta, 0.70 - 0.01 * i)
        for i in range(5)
    ]


HAIKU_JSON = {
    "context_summary": "Multiple tickets match the topic — which one?",
    "options": [
        {"id": "opt_1", "label": "Stripe API 500 refund error",
         "refined_query": "Stripe API refund 500 error root cause"},
        {"id": "opt_2", "label": "Shopify webhook retry storm",
         "refined_query": "Shopify webhook retry storm queue backup"},
        {"id": "opt_3", "label": "Plaid stale balance data",
         "refined_query": "Plaid balance endpoint stale cache root cause"},
        {"id": "opt_other", "label": "Something else - let me clarify",
         "refined_query": ""},
    ],
}


def run_test(name: str, fn):
    print(f"\n===== {name} =====")
    try:
        fn()
        print(f"PASS: {name}")
        return True
    except AssertionError as e:
        print(f"FAIL: {name}: {e}")
        return False
    except Exception as e:
        print(f"ERROR: {name}: {type(e).__name__}: {e}")
        return False


def reset_state():
    ic._SESSION_COUNTS.clear()
    ic._SESSION_COUNT_TIMESTAMPS.clear()


def t1_ambiguous_short_query():
    reset_state()
    with mock.patch.object(ic, "_invoke_haiku", return_value=json.dumps(HAIKU_JSON)):
        res = ic.try_clarify(
            query="refund error",
            ranked_chunks=_three_distinct_chunks(),
            triage_confidence=0.5,
            carried_identifiers=None,
            session_id="t1",
            recent_messages=[],
        )
    assert res.needs_clarification, f"expected trigger, got skip={res.skip_reason}"
    assert len(res.options) >= 3, f"expected >=3 options, got {len(res.options)}"
    assert any(o.id == "opt_other" for o in res.options), "opt_other missing"


def t2_vague_noun_no_antecedent():
    reset_state()
    with mock.patch.object(ic, "_invoke_haiku", return_value=json.dumps(HAIKU_JSON)):
        res = ic.try_clarify(
            query="the outage",
            ranked_chunks=_three_distinct_chunks(),
            triage_confidence=0.4,
            carried_identifiers=None,
            session_id="t2",
            recent_messages=[],
        )
    assert res.needs_clarification, f"expected trigger, got skip={res.skip_reason}"


def t3_generic_topic_multiple_records():
    reset_state()
    with mock.patch.object(ic, "_invoke_haiku", return_value=json.dumps(HAIKU_JSON)):
        res = ic.try_clarify(
            query="payment failure issue",
            ranked_chunks=_three_distinct_chunks(),
            triage_confidence=0.6,
            carried_identifiers=None,
            session_id="t3",
            recent_messages=[],
        )
    assert res.needs_clarification, f"expected trigger, got skip={res.skip_reason}"


def t4_explicit_identifier_skips():
    reset_state()
    res = ic.try_clarify(
        query="tell me about INC-1001",
        ranked_chunks=_three_distinct_chunks(),
        triage_confidence=0.9,
        carried_identifiers=None,
        session_id="t4",
        recent_messages=[],
    )
    assert not res.needs_clarification
    assert res.skip_reason == "specific_identifier", f"got {res.skip_reason}"


def t5_broad_intent_skips():
    reset_state()
    res = ic.try_clarify(
        query="show all tickets across customers",
        ranked_chunks=_three_distinct_chunks(),
        triage_confidence=0.5,
        carried_identifiers=None,
        session_id="t5",
        recent_messages=[],
    )
    assert not res.needs_clarification
    assert res.skip_reason == "broad_intent", f"got {res.skip_reason}"


def t6_cache_path_never_runs():
    # Cache hits short-circuit at the API layer before try_clarify is called.
    # Unit level: assert that the function respects the disabled flag.
    reset_state()
    with mock.patch.object(settings, "INTERACTIVE_CLARIFIER_ENABLED", False):
        res = ic.try_clarify(
            query="refund error",
            ranked_chunks=_three_distinct_chunks(),
            triage_confidence=0.5,
            carried_identifiers=None,
            session_id="t6",
            recent_messages=[],
        )
    assert not res.needs_clarification
    assert res.skip_reason == "disabled", f"got {res.skip_reason}"


def t7_answering_prior_clarification_skips():
    reset_state()
    prior = [
        {"role": "user", "content": "refund error"},
        {"role": "assistant", "content": "Can you clarify?",
         "context_stats": {"clarification_presented": True}},
    ]
    res = ic.try_clarify(
        query="the first one",
        ranked_chunks=_three_distinct_chunks(),
        triage_confidence=0.5,
        carried_identifiers=None,
        session_id="t7",
        recent_messages=prior,
    )
    assert not res.needs_clarification
    assert res.skip_reason == "answering_prior_clarification", f"got {res.skip_reason}"


def t8_rate_limited_after_max():
    reset_state()
    sid = "t8"
    # Bump session count to max
    for _ in range(int(settings.INTERACTIVE_CLARIFIER_MAX_PER_SESSION)):
        ic._bump_session_clarification_count(sid)
    res = ic.try_clarify(
        query="refund error",
        ranked_chunks=_three_distinct_chunks(),
        triage_confidence=0.5,
        carried_identifiers=None,
        session_id=sid,
        recent_messages=[],
    )
    assert not res.needs_clarification
    assert res.skip_reason == "rate_limited", f"got {res.skip_reason}"


def t9_single_record_below_threshold():
    reset_state()
    res = ic.try_clarify(
        query="refund error",
        ranked_chunks=_same_record_chunks(),
        triage_confidence=0.9,
        carried_identifiers=None,
        session_id="t9",
        recent_messages=[],
    )
    assert not res.needs_clarification
    # Could be score_below_threshold (if high triage_confidence dampens score)
    assert res.skip_reason in {"score_below_threshold"}, f"got {res.skip_reason}"


def main():
    results = []
    results.append(("T1_ambiguous_short_query", run_test("T1 ambiguous short query", t1_ambiguous_short_query)))
    results.append(("T2_vague_noun", run_test("T2 vague noun no antecedent", t2_vague_noun_no_antecedent)))
    results.append(("T3_generic_topic", run_test("T3 generic topic multiple records", t3_generic_topic_multiple_records)))
    results.append(("T4_identifier_skip", run_test("T4 explicit identifier skips", t4_explicit_identifier_skips)))
    results.append(("T5_broad_intent_skip", run_test("T5 broad intent skips", t5_broad_intent_skips)))
    results.append(("T6_disabled_flag", run_test("T6 disabled flag path", t6_cache_path_never_runs)))
    results.append(("T7_answering_prior", run_test("T7 answering prior clarification", t7_answering_prior_clarification_skips)))
    results.append(("T8_rate_limited", run_test("T8 rate-limited after max", t8_rate_limited_after_max)))
    results.append(("T9_single_record", run_test("T9 single-record below threshold", t9_single_record_below_threshold)))

    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"\n=========  TOTAL: {passed}/{total} PASSED =========")
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
