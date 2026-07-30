"""Shared US Pharma store-scoped KB chat opener helpers (Idea A).

Both entry points into a store-scoped KB chat open with the SAME 2-line
store-orientation summary + the SAME clickable sample questions:

  1. Journey Stage 4 handoff (search-kb-handoff route).
  2. The standalone "Discuss Store Specific with LogIQ" store-scoped chat
     (POST /chat/sessions/store-scoped), launched from the landing tile,
     the stages quick-actions pill, and the intake form.

Keeping the summary builder + the sample questions here means one source
of truth, so the two flows never drift.
"""
from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger("acadia-log-iq")


def store_sample_questions(store_id: str) -> List[str]:
    """Clickable sample questions for the store opener.

    Intentionally empty — the store-scoped chat no longer surfaces
    predefined question chips. The opener is a plain welcome greeting
    (see :func:`build_store_summary`) and the engineer types their own
    question. Kept as a function (returning ``[]``) so both call sites
    (the standalone store-scoped chat and the Stage 4 handoff) keep their
    single source of truth without any call-site changes.
    """
    return []


def build_store_summary(store_id: str) -> Optional[str]:
    """The store-scoped KB chat opener — a simple welcome greeting.

    Previously this read the store-config chunk and asked Haiku for a
    2-line orientation. That (plus the sample-question chips) has been
    replaced by a plain, friendly greeting so the chat opens as an open
    "ask me anything" prompt scoped to the store. Never raises.
    """
    sid = (store_id or "").strip()
    if not sid:
        return None
    return f"Hi, happy to assist! Ask me anything about Store {sid}."
