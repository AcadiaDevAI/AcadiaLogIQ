#!/usr/bin/env python3
"""
Sprint 3-PREP-C — reference ETL: Excel contact sheet -> canonical JSON.

This is a TEMPLATE. Every customer's Excel layout is slightly different,
so expect to copy this script per-source and adjust column mappings /
`--org-name` / `--kind`. The output JSON shape is fixed by the
canonical contact schema (see docs/CONTACT_SCHEMA.md).

Usage:
  python -m backend.scripts.etl_excel_contacts \
    --in acme_escalation_matrix.xlsx \
    --sheet "Escalation Matrix" \
    --kind contact_customer \
    --org-name "Acme Corp" \
    --org-type customer \
    --out acme_contacts.json

Standard (template) column layout expected in row 1:
  A: Name
  B: Role / Title
  C: Team
  D: Escalation Level (1/2/3, integer)
  E: Phone Primary
  F: Email
  G: Hours (e.g., "24x7", "BH 9-5")
  H: Notes
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _slugify(value: str) -> str:
    """Turn 'Acme Corp' / 'Ring Communications, Inc.' into ACME-CORP."""
    if not value:
        return "UNKNOWN"
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").upper()
    return cleaned or "UNKNOWN"


def _coerce_level(value):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        # Sometimes Excel stores "Tier 2" or "L2" — pull first digit.
        match = re.search(r"\d+", str(value))
        return int(match.group()) if match else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Excel contact sheet -> canonical JSON (reference ETL).",
    )
    parser.add_argument("--in", dest="input", required=True, type=Path)
    parser.add_argument("--sheet", default=None, help="Sheet name (default: active sheet)")
    parser.add_argument(
        "--kind", required=True,
        choices=["contact_customer", "contact_vendor"],
    )
    parser.add_argument("--org-name", required=True)
    parser.add_argument(
        "--org-type", required=True,
        choices=["customer", "vendor"],
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    try:
        from openpyxl import load_workbook
    except ImportError:
        print(
            "[FATAL] openpyxl is not installed. Run: "
            "pip install openpyxl>=3.1.0",
            file=sys.stderr,
        )
        return 2

    if not args.input.exists():
        print(f"[FATAL] Input file not found: {args.input}", file=sys.stderr)
        return 2

    wb = load_workbook(args.input, data_only=True)
    ws = wb[args.sheet] if args.sheet else wb.active

    org_slug = _slugify(args.org_name)
    records = []

    for row_idx, row in enumerate(
        ws.iter_rows(min_row=2, values_only=True), start=2,
    ):
        # Pad the tuple so short rows (fewer than 8 cells) don't unpack-fail.
        padded = (tuple(row) + (None,) * 8)[:8]
        name, role, team, level, phone, email, hours, notes = padded

        # Skip blank / header-ish rows. We require AT LEAST one of name+email
        # so the record is addressable and reachable.
        name = (str(name).strip() if name is not None else "")
        email = (str(email).strip() if email is not None else "")
        if not name or not email:
            continue

        contact_id = f"CONT-{org_slug}-{row_idx:03d}"
        record = {
            "contact_id": contact_id,
            "kind": args.kind,
            "organization": {
                "name": args.org_name,
                "type": args.org_type,
            },
            "team": {
                "name": (str(team).strip() if team else "Unassigned"),
                "escalation_level": _coerce_level(level),
                "hours": (str(hours).strip() if hours else None),
            },
            "person": {
                "name": name,
                "role": (str(role).strip() if role else ""),
                "phone_primary": (str(phone).strip() if phone else None),
                "email": email,
            },
            "notes": (str(notes).strip() if notes else ""),
            "source_file": args.input.name,
        }
        records.append(record)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} records -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
