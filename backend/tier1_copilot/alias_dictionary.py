"""In-memory alias dictionary for Tier-1 alert expansion.

Built once at app startup (or on demand) by scanning chunks.metadata_json
for the canonical alert-signal fields. Kept in memory because alias
lookup is on the request hot path; we don't want a SQL roundtrip per
`/tier1/analyze` call.

This module is intentionally thread-safe via a module-level singleton
plus a reentrant lock on mutation. Readers see an immutable snapshot
of the internal dict (`_aliases` is replaced atomically on rebuild).
"""
from __future__ import annotations

import logging
import threading
from typing import Dict, Iterable, Optional, Set

logger = logging.getLogger("acadia-log-iq")


class AliasDictionary:
    """Canonical term → set of aliases. All keys lowercased + trimmed."""

    def __init__(self) -> None:
        self._aliases: Dict[str, Set[str]] = {}
        self._lock = threading.RLock()
        self._built = False

    # ---------- read path ----------------------------------------------------
    def get_aliases(self, term: str) -> Set[str]:
        if not term:
            return set()
        key = term.lower().strip()
        return self._aliases.get(key, set())

    def term_count(self) -> int:
        return len(self._aliases)

    def is_built(self) -> bool:
        return self._built

    # ---------- construction -------------------------------------------------
    def rebuild_from_rows(self, rows: Iterable[dict]) -> None:
        """Build the alias map from an iterable of chunk rows.

        Each row must expose a `metadata_json` dict with the gold-schema
        shape (see Sprint 2.9 / 4 ingestion). Missing keys are tolerated;
        a row with none of the signal keys contributes nothing.

        The expansion rules are intentionally conservative — we want
        high-precision aliases (not a free-form synonym net). For each
        ticket we group:
          - Fingerprints (e.g. BGP-5-ADJCHANGE) as aliases of
            component_category
          - Affected_Assets as aliases of Target_Service
          - Detected_Symptom tokens as aliases of the parent Target_Service
          - Knowledge_Base.semantic_unit_educational.related_signals as
            aliases of the ticket's fingerprints
        """
        fresh: Dict[str, Set[str]] = {}

        for row in rows:
            md = row.get("metadata_json") if isinstance(row, dict) else None
            if not isinstance(md, dict):
                continue
            meta = md.get("Metadata") or {}
            if not isinstance(meta, dict):
                continue

            component = (meta.get("component_category") or "").strip().lower()
            target = (meta.get("Target_Service") or "").strip().lower()
            assets = meta.get("Affected_Assets") or []
            fingerprints = [
                str(f).strip().lower() for f in (meta.get("Fingerprints") or [])
                if f
            ]

            ssm = md.get("Symptom_Solution_Mapping") or {}
            symptom = (ssm.get("Detected_Symptom") or "").strip().lower() \
                if isinstance(ssm, dict) else ""

            # Fingerprint → component_category (and reverse)
            if component:
                for fp in fingerprints:
                    _pair(fresh, component, fp)

            # Affected asset → target_service (and reverse)
            if target:
                for a in assets:
                    if not isinstance(a, str):
                        continue
                    a_key = a.strip().lower()
                    if a_key:
                        _pair(fresh, target, a_key)

            # Detected symptom tokens → target service
            if target and symptom:
                _pair(fresh, target, symptom)

            # Related signals from Knowledge_Base → ticket fingerprints
            kb = md.get("Knowledge_Base") or []
            if isinstance(kb, list):
                for entry in kb:
                    if not isinstance(entry, dict):
                        continue
                    unit = entry.get("semantic_unit_educational") or {}
                    related = unit.get("related_signals") or [] \
                        if isinstance(unit, dict) else []
                    if not isinstance(related, list):
                        continue
                    for rel in related:
                        if not isinstance(rel, str):
                            continue
                        r_key = rel.strip().lower()
                        if not r_key:
                            continue
                        for fp in fingerprints:
                            _pair(fresh, fp, r_key)

        with self._lock:
            self._aliases = fresh
            self._built = True

        logger.info(
            "[tier1_copilot] alias dictionary built: %d canonical terms",
            len(fresh),
        )

    def rebuild_from_db(self, engine) -> None:
        """Fetch ticket metadata_json rows from the live DB and rebuild.

        Defensive: any SQL error leaves the existing dict intact and logs
        a warning. The /tier1/analyze endpoint degrades gracefully to
        base-term-only matching if no aliases are available.
        """
        rows: list[dict] = []
        try:
            from sqlalchemy import text
            with engine.connect() as conn:
                result = conn.execute(
                    text(
                        """
                        SELECT metadata_json
                        FROM chunks
                        WHERE metadata_json->>'doc_kind' = 'ticket'
                           OR metadata_json->'Metadata' IS NOT NULL
                        """
                    )
                ).mappings().all()
                rows = [dict(r) for r in result]
        except Exception as exc:
            logger.warning("[tier1_copilot] alias rebuild DB read failed: %s", exc)
            return

        self.rebuild_from_rows(rows)


def _pair(store: Dict[str, Set[str]], a: str, b: str) -> None:
    """Record a ↔ b as mutual aliases."""
    if not a or not b or a == b:
        return
    store.setdefault(a, set()).add(b)
    store.setdefault(b, set()).add(a)


# ─────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────
_SINGLETON: Optional[AliasDictionary] = None
_SINGLETON_LOCK = threading.Lock()


def get_alias_dictionary() -> AliasDictionary:
    global _SINGLETON
    if _SINGLETON is None:
        with _SINGLETON_LOCK:
            if _SINGLETON is None:
                _SINGLETON = AliasDictionary()
    return _SINGLETON
