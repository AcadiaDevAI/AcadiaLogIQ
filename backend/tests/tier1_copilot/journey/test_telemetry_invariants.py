"""Sprint 11 — Cross-module invariant tests for journey telemetry.

The journey telemetry has TWO gatekeepers for valid event_type values:
  1. JourneyEventRequest.event_type (Pydantic Literal in schemas.py)
     — rejects unknown values at request-validation time with HTTP 422.
  2. _VALID_EVENT_TYPES (set in telemetry.py) — checked inside
     record_event(); unknown values silently drop with 200 + a WARN
     log line (the route returns ok=True/False but the HTTP code stays
     200 for legacy reasons).

If those two drift, the symptom is silent data loss: events that pass
schema validation get dropped at the recorder. That's exactly the
Sprint 11 bug where `kb_chat_engaged` was added to the schema but the
recorder rejected it; the engagement counter was never populated and
the escalation traversal log couldn't distinguish "viewed Stage 4"
from "actually engaged the KB chat".

This invariant test locks the two surfaces together. If you add an
event_type to one, you must add it to the other or this test breaks
loudly.
"""
from __future__ import annotations

import typing
import unittest

from backend.tier1_copilot.journey.schemas import JourneyEventRequest
from backend.tier1_copilot.journey.telemetry import _VALID_EVENT_TYPES


def _literal_members(model_cls, field_name) -> set:
    """Pull the Literal[...] member set off a Pydantic model field.

    Robust across Pydantic v1/v2 — uses model_fields when present,
    falls back to typing.get_type_hints / get_args.
    """
    fields = getattr(model_cls, "model_fields", None) or getattr(model_cls, "__fields__", {})
    field = fields.get(field_name)
    annotation = getattr(field, "annotation", None) if field is not None else None
    if annotation is None:
        annotation = typing.get_type_hints(model_cls).get(field_name)
    args = typing.get_args(annotation)
    return set(args)


class TelemetryEventTypeInvariants(unittest.TestCase):
    """Schema Literal members must equal the recorder's allowlist set."""

    def test_schema_literal_matches_telemetry_allowlist(self):
        schema_members = _literal_members(JourneyEventRequest, "event_type")
        # Symmetric diff makes the failure message immediately readable:
        # left side = in schema but not allowlist (would silently drop)
        # right side = in allowlist but not schema (would 422 reject)
        only_in_schema = schema_members - _VALID_EVENT_TYPES
        only_in_recorder = _VALID_EVENT_TYPES - schema_members
        self.assertEqual(
            only_in_schema, set(),
            f"Event types {only_in_schema} pass schema validation but the "
            f"telemetry recorder rejects them (silent data loss). Add to "
            f"_VALID_EVENT_TYPES in backend/tier1_copilot/journey/telemetry.py.",
        )
        self.assertEqual(
            only_in_recorder, set(),
            f"Event types {only_in_recorder} are in the recorder allowlist "
            f"but not in JourneyEventRequest's Literal — clients can't post "
            f"them (request returns 422). Add to schemas.py.",
        )

    def test_kb_chat_engaged_is_in_both(self):
        """Sprint 11-specific regression — the event the user reported
        as missing must be plumbed through both gatekeepers."""
        schema_members = _literal_members(JourneyEventRequest, "event_type")
        self.assertIn("kb_chat_engaged", schema_members)
        self.assertIn("kb_chat_engaged", _VALID_EVENT_TYPES)


if __name__ == "__main__":
    unittest.main()
