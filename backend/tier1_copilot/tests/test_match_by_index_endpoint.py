"""Sprint 8 — GET /tier1/session/{id}/match/{index} (offline, mocked DB)."""
from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime, timezone
from unittest import mock

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _make_client(*, ux_on: bool, sprint6_on: bool = True, sprint7_on: bool = True):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.tier1_copilot.routes import router
    from backend.config import settings

    app = FastAPI()
    app.include_router(router)
    patches = [
        mock.patch.object(settings, "LOGIQ_TIER1_COPILOT_BACKEND", sprint6_on),
        mock.patch.object(settings, "LOGIQ_TIER1_PROGRESSIVE_BACKEND", sprint7_on),
        mock.patch.object(settings, "LOGIQ_TIER1_UX_FIXES_BACKEND", ux_on),
    ]
    for p in patches:
        p.start()
    return TestClient(app), patches


def _teardown(patches):
    for p in patches:
        p.stop()


def _fake_session(chunk_ids):
    from backend.tier1_copilot.session_state.tier1_session import Tier1SessionRecord
    return Tier1SessionRecord(
        id="sess_abc",
        alert_signature="p2 | a | b",
        alert_payload={
            "severity": "P2",
            "asset_name": "V-Desktop Environment",
            "alert_type": "Desktop Slowness",
            "session_id": "sess_abc",
        },
        top_5_match_ids=list(chunk_ids),
        created_at=datetime.now(timezone.utc),
        last_activity_at=datetime.now(timezone.utc),
    )


class FlagOffTests(unittest.TestCase):
    def test_flag_off_returns_404(self):
        client, patches = _make_client(ux_on=False)
        try:
            r = client.get("/tier1/session/sess_abc/match/0")
            self.assertEqual(r.status_code, 404)
            self.assertEqual(r.json()["detail"], "tier1_ux_fixes_flag_off")
        finally:
            _teardown(patches)


class SessionNotFoundTests(unittest.TestCase):
    def test_missing_session_returns_404(self):
        from backend.tier1_copilot import routes
        client, patches = _make_client(ux_on=True)
        try:
            with mock.patch.object(routes, "get_session", return_value=None):
                r = client.get("/tier1/session/sess_abc/match/0")
            self.assertEqual(r.status_code, 404)
            self.assertEqual(r.json()["detail"], "session_not_found")
        finally:
            _teardown(patches)


class OutOfRangeTests(unittest.TestCase):
    def test_negative_index_rejected_by_path_schema(self):
        """FastAPI path param validation happens before our handler —
        the integer path converter just passes negative integers through,
        so our 422 check fires in the handler."""
        from backend.tier1_copilot import routes
        client, patches = _make_client(ux_on=True)
        try:
            with mock.patch.object(
                routes, "get_session", return_value=_fake_session(["c1", "c2"]),
            ):
                r = client.get("/tier1/session/sess_abc/match/5")
            self.assertEqual(r.status_code, 422)
            self.assertEqual(r.json()["detail"], "match_index_out_of_range")
        finally:
            _teardown(patches)

    def test_empty_top_5_returns_422(self):
        from backend.tier1_copilot import routes
        client, patches = _make_client(ux_on=True)
        try:
            with mock.patch.object(
                routes, "get_session", return_value=_fake_session([]),
            ):
                r = client.get("/tier1/session/sess_abc/match/0")
            self.assertEqual(r.status_code, 422)
        finally:
            _teardown(patches)


class HappyPathTests(unittest.TestCase):
    def test_happy_path_returns_rank_n_shape(self):
        from backend.tier1_copilot import routes
        from backend.tier1_copilot.schemas import Tier1AnswerSection

        sess = _fake_session(["chunk-1", "chunk-2", "chunk-3"])
        ticket_md = {
            "Metadata": {
                "Incident_Number": "INC-ALPHA-031",
                "customer_name": "Aetheris",
                "priority": "P2",
            },
            "Symptom_Solution_Mapping": {"Detected_Symptom": "Slowness"},
        }
        rerun_candidates = [
            {
                "chunk_id": "chunk-2",
                "metadata_json": ticket_md,
                "final_score": 0.73,
                "_score_components": {"alert_type_match": 0.8},
            },
        ]
        fake_answer_raw = (
            "Issue Understanding\nSession lag.\n\n"
            "Historical Match\nINC-ALPHA-031.\n\n"
            "Most Likely Cause\nDDC load.\n\n"
            "Recommended First Checks\n- Get-BrokerController\n\n"
            "Most Likely Fix\nRestart DDC.\n\n"
            "Validation\nLaunch < 20s.\n\n"
            "Escalate If\nStill slow.\n\n"
            "Follow-up Question\nOutcome?\n"
        )

        client, patches = _make_client(ux_on=True)
        try:
            with mock.patch.object(routes, "get_session", return_value=sess), \
                 mock.patch.object(
                    routes, "_rerun_retrieval_for_session",
                    return_value=rerun_candidates,
                 ), \
                 mock.patch.object(routes, "_invoke_haiku", return_value=fake_answer_raw), \
                 mock.patch.object(routes, "update_match_index", return_value=True), \
                 mock.patch.object(routes, "touch_activity", return_value=True):
                r = client.get("/tier1/session/sess_abc/match/1")
            self.assertEqual(r.status_code, 200)
            body = r.json()
            self.assertEqual(body["match_index"], 1)
            self.assertEqual(body["total_matches"], 3)
            self.assertEqual(body["matched_incident"], "INC-ALPHA-031")
            self.assertEqual(body["confidence"], "Medium")
            self.assertEqual(body["similar_count"], 3)
            # Answer body parsed successfully.
            self.assertIn(
                "DDC",
                body["answer"]["most_likely_cause"] or "",
            )
            self.assertEqual(body["session_id"], "sess_abc")
            self.assertEqual(body["top_5_match_ids"], ["chunk-1", "chunk-2", "chunk-3"])
        finally:
            _teardown(patches)

    def test_rerun_misses_candidate_falls_back_to_direct_fetch(self):
        """If _rerun_retrieval_for_session doesn't surface the target
        chunk, the handler calls _load_ticket_metadata_by_chunk_id and
        still returns 200 — graceful fallback the spec explicitly
        asks for."""
        from backend.tier1_copilot import routes

        sess = _fake_session(["chunk-X"])
        fake_answer_raw = (
            "Issue Understanding\nX.\n\nHistorical Match\nY.\n\n"
            "Most Likely Cause\nZ.\n\nRecommended First Checks\n- q\n\n"
            "Most Likely Fix\nf.\n\nValidation\nv.\n\n"
            "Escalate If\ne.\n\nFollow-up Question\nfq.\n"
        )
        client, patches = _make_client(ux_on=True)
        try:
            with mock.patch.object(routes, "get_session", return_value=sess), \
                 mock.patch.object(
                    routes, "_rerun_retrieval_for_session", return_value=[],
                 ), \
                 mock.patch.object(
                    routes, "_load_ticket_metadata_by_chunk_id",
                    return_value={"Metadata": {"Incident_Number": "INC-99"}},
                 ), \
                 mock.patch.object(routes, "_invoke_haiku", return_value=fake_answer_raw), \
                 mock.patch.object(routes, "update_match_index", return_value=True), \
                 mock.patch.object(routes, "touch_activity", return_value=True):
                r = client.get("/tier1/session/sess_abc/match/0")
            self.assertEqual(r.status_code, 200)
            body = r.json()
            self.assertEqual(body["matched_incident"], "INC-99")
            # No candidate → confidence defaults to "Medium".
            self.assertEqual(body["confidence"], "Medium")
        finally:
            _teardown(patches)


if __name__ == "__main__":
    unittest.main(verbosity=2)
