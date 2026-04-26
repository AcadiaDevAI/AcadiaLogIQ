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


# Sprint 9 — single startup hook that rebuilds BOTH the Sprint 6 alias
# dictionary AND the Sprint 9 IntakeCatalogs from the same chunks scan.
def build_tier1_runtime_state_on_startup() -> None:
    """Rebuild every in-memory tier1_copilot artefact in one DB pass.

    Used by `backend/api.py` lifespan when either Tier-1 flag is on.
    Never raises — each rebuild has its own try/except so a failure in
    one doesn't poison the other."""
    build_alias_dictionary_on_startup()
    try:
        from backend.config import settings
        if not getattr(settings, "LOGIQ_UNIVERSAL_INTAKE_BACKEND", False):
            return
        from backend.db.connection import engine
        from backend.tier1_copilot.intake.catalogs import get_intake_catalogs
        catalogs = get_intake_catalogs(lazy_build=False)
        catalogs.rebuild_from_db(engine)
        s, a, t, c = catalogs.health_snapshot()
        logger.info(
            "[intake] startup catalogs ready: severities=%d assets=%d "
            "alert_types=%d customers=%d",
            s, a, t, c,
        )
    except Exception as exc:
        logger.warning(
            "[intake] startup catalogs build failed: %s", exc,
        )
