"""Sprint 7 — stuck detection threshold logic (offline, pure)."""
from __future__ import annotations

import os
import sys
import unittest
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from unittest import mock

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


@dataclass
class FakeSession:
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    thumbs_down_count: int = 0
    stuck_nudge_shown: bool = False
    resolved: bool = False
    escalated: bool = False


class StuckDetectorTests(unittest.TestCase):
    def test_fresh_session_under_threshold_no_nudge(self):
        from backend.tier1_copilot.diagnostics.stuck_detector import should_nudge
        sess = FakeSession()
        now = sess.created_at + timedelta(seconds=60)
        self.assertFalse(should_nudge(sess, now))

    def test_elapsed_over_threshold_nudges(self):
        from backend.config import settings
        from backend.tier1_copilot.diagnostics.stuck_detector import should_nudge
        sess = FakeSession()
        with mock.patch.object(settings, "TIER1_STUCK_THRESHOLD_SECONDS", 120):
            now = sess.created_at + timedelta(seconds=121)
            self.assertTrue(should_nudge(sess, now))

    def test_two_thumbs_down_trigger_before_threshold(self):
        from backend.tier1_copilot.diagnostics.stuck_detector import should_nudge
        sess = FakeSession(thumbs_down_count=2)
        now = sess.created_at + timedelta(seconds=10)
        self.assertTrue(should_nudge(sess, now))

    def test_already_shown_suppresses(self):
        from backend.tier1_copilot.diagnostics.stuck_detector import should_nudge
        sess = FakeSession(thumbs_down_count=5, stuck_nudge_shown=True)
        now = sess.created_at + timedelta(seconds=9999)
        self.assertFalse(should_nudge(sess, now))

    def test_resolved_suppresses(self):
        from backend.tier1_copilot.diagnostics.stuck_detector import should_nudge
        sess = FakeSession(thumbs_down_count=5, resolved=True)
        now = sess.created_at + timedelta(seconds=9999)
        self.assertFalse(should_nudge(sess, now))

    def test_escalated_suppresses(self):
        from backend.tier1_copilot.diagnostics.stuck_detector import should_nudge
        sess = FakeSession(thumbs_down_count=5, escalated=True)
        now = sess.created_at + timedelta(seconds=9999)
        self.assertFalse(should_nudge(sess, now))

    def test_elapsed_seconds_clamps_at_zero(self):
        from backend.tier1_copilot.diagnostics.stuck_detector import elapsed_seconds
        sess = FakeSession()
        past = sess.created_at - timedelta(seconds=30)
        self.assertEqual(elapsed_seconds(sess, past), 0)

    def test_elapsed_seconds_reasonable(self):
        from backend.tier1_copilot.diagnostics.stuck_detector import elapsed_seconds
        sess = FakeSession()
        future = sess.created_at + timedelta(seconds=125)
        self.assertEqual(elapsed_seconds(sess, future), 125)


if __name__ == "__main__":
    unittest.main(verbosity=2)
