#!/usr/bin/env python3
"""
Sprint 3-PREP-C — validate a contact-directory JSON file against the
canonical schema (docs/CONTACT_SCHEMA.md).

Exit codes:
  0 — all records valid
  1 — one or more records have validation errors
  2 — file is unparseable or not a top-level array

Usage:
  python -m backend.scripts.validate_contacts acme_contacts.json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

REQUIRED_TOP_KEYS = {"contact_id", "kind", "organization", "team", "person"}
VALID_KINDS = {"contact_customer", "contact_vendor"}
ORG_REQUIRED = {"name", "type"}
# phone_primary is RECOMMENDED but not required — some vendor contacts
# are email-only.
PERSON_REQUIRED = {"name", "role", "email"}


def validate_record(rec: Dict[str, Any], idx: int) -> List[str]:
    errors: List[str] = []

    missing = REQUIRED_TOP_KEYS - set(rec.keys())
    if missing:
        errors.append(f"[row {idx}] missing keys: {sorted(missing)}")

    kind = rec.get("kind")
    if kind not in VALID_KINDS:
        errors.append(
            f"[row {idx}] invalid kind={kind!r}. "
            f"Must be one of {sorted(VALID_KINDS)}"
        )

    org = rec.get("organization") or {}
    if not isinstance(org, dict):
        errors.append(f"[row {idx}] organization must be an object, got {type(org).__name__}")
    else:
        missing_org = ORG_REQUIRED - set(org.keys())
        if missing_org:
            errors.append(
                f"[row {idx}] organization missing keys: {sorted(missing_org)}"
            )
        if org.get("type") not in {"customer", "vendor"}:
            errors.append(
                f"[row {idx}] organization.type must be 'customer' or 'vendor', "
                f"got {org.get('type')!r}"
            )

    person = rec.get("person") or {}
    if not isinstance(person, dict):
        errors.append(f"[row {idx}] person must be an object, got {type(person).__name__}")
    else:
        missing_person = PERSON_REQUIRED - set(person.keys())
        if missing_person:
            errors.append(
                f"[row {idx}] person missing keys: {sorted(missing_person)}"
            )
        email = (person.get("email") or "").strip()
        if email and "@" not in email:
            errors.append(
                f"[row {idx}] person.email doesn't look valid: {email!r}"
            )

    team = rec.get("team") or {}
    if not isinstance(team, dict):
        errors.append(f"[row {idx}] team must be an object, got {type(team).__name__}")
    else:
        level = team.get("escalation_level")
        if level is not None and not isinstance(level, int):
            errors.append(
                f"[row {idx}] team.escalation_level must be int, got "
                f"{type(level).__name__}"
            )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a contact-directory JSON file.",
    )
    parser.add_argument("file", help="Path to the contacts JSON file")
    args = parser.parse_args()

    try:
        with open(args.file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        print(f"[FATAL] File not found: {args.file}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"[FATAL] Cannot parse JSON: {exc}", file=sys.stderr)
        return 2

    if not isinstance(data, list):
        print("[FATAL] Top-level must be a JSON array", file=sys.stderr)
        return 2

    all_errors: List[str] = []
    for idx, rec in enumerate(data):
        if not isinstance(rec, dict):
            all_errors.append(f"[row {idx}] not an object")
            continue
        all_errors.extend(validate_record(rec, idx))

    if all_errors:
        for e in all_errors:
            print(e, file=sys.stderr)
        print(
            f"\n{len(all_errors)} error(s) in {len(data)} record(s). "
            f"Fix and re-run.",
            file=sys.stderr,
        )
        return 1

    print(f"OK - {len(data)} contact record(s) valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
