"""Startup hook: build the alias dictionary once per process.

Called from the parent app's FastAPI lifespan when
LOGIQ_TIER1_COPILOT_BACKEND is True. Safe to call repeatedly — the
AliasDictionary.rebuild_from_db is idempotent and swaps the internal
map atomically.
"""
from __future__ import annotations

import logging

from backend.tier1_copilot.alias_dictionary import get_alias_dictionary

logger = logging.getLogger("acadia-log-iq")


def build_alias_dictionary_on_startup() -> None:
    """Pull ticket metadata_json rows and rebuild the alias dict.

    Never raises. Logs a warning on failure; /tier1/analyze will
    degrade to base-term-only matching until the next rebuild."""
    try:
        from backend.db.connection import engine
        ad = get_alias_dictionary()
        ad.rebuild_from_db(engine)
        logger.info(
            "[tier1_copilot] startup alias dict ready: %d terms",
            ad.term_count(),
        )
    except Exception as exc:
        logger.warning(
            "[tier1_copilot] startup alias dict build failed: %s", exc,
        )
