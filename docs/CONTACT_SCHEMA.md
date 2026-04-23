# Contact Directory — Canonical JSON Schema

> Sprint 3-PREP-C deliverable. This document defines the shape ingestion
> expects for `doc_kind=contact_customer` and `doc_kind=contact_vendor`
> files. ETL scripts (starting with `etl_excel_contacts.py`) emit this
> shape; the validator (`validate_contacts.py`) enforces it before
> ingestion.

---

## Top-level shape

One JSON array per file. Each element is **one contact record**
(one person or one team mailbox). The ingestion pipeline produces
**one chunk per record** — no merging, no narrative cramming. This is
deliberate: retrieval must return a precise contact card, not a
paragraph about where a phone number lives in a PDF.

```json
[
  { "contact_id": "CONT-ACME-T2-001", ... },
  { "contact_id": "CONT-ACME-T2-002", ... }
]
```

The top-level `[]` is REQUIRED. `{}` at the top level is rejected as
`[FATAL] Top-level must be a JSON array` by the validator.

---

## Record fields (customer contact)

```json
{
  "contact_id": "CONT-ACME-T2-001",
  "kind": "contact_customer",
  "organization": {
    "name": "Acme Corp",
    "type": "customer",
    "segment": "Enterprise",
    "region": "US-East"
  },
  "team": {
    "name": "NOC Tier 2",
    "escalation_level": 2,
    "hours": "24x7"
  },
  "person": {
    "name": "Jane Smith",
    "role": "Shift Lead",
    "phone_primary": "+1-555-0100",
    "phone_secondary": null,
    "email": "jane.smith@acme.com",
    "pager": null
  },
  "escalation": {
    "triggers": ["P1", "SLA breach"],
    "after_hours_path": "Call +1-555-0199 (on-call rotation)"
  },
  "notes": "Primary contact for P1 escalations. Prefers email for non-urgent.",
  "last_verified": "2025-09-15",
  "source_file": "acme_escalation_matrix_v3.xlsx"
}
```

### Required

| Field | Type | Notes |
|---|---|---|
| `contact_id` | string | Stable unique ID. Becomes `primary_id` in the chunk so exact-identifier queries route straight to the record. |
| `kind` | enum | `"contact_customer"` or `"contact_vendor"`. MUST match `doc_kind`. |
| `organization.name` | string | |
| `organization.type` | enum | `"customer"` or `"vendor"`. |
| `team` | object | Fields inside may be null, but the object itself must exist. |
| `person.name` | string | |
| `person.role` | string | May be empty `""`, but key must exist. |
| `person.email` | string | Must contain `@` if present. |

### Recommended

| Field | Notes |
|---|---|
| `person.phone_primary` | Vendor contacts are sometimes email-only; still fill where known. |
| `team.escalation_level` | Integer, typically 1/2/3. Non-integer strings like "Tier 2" get coerced by the ETL's `_coerce_level` to `2`. |
| `team.hours` | Free-text. Common values: `"24x7"`, `"BH 9-5"`, `"APAC BH"`. |
| `escalation.triggers` | Array of short tokens (e.g. `["P1", "SLA breach"]`). |
| `last_verified` | ISO date string. Aids stale-data pruning. |
| `source_file` | Origin Excel/PDF/Word filename. Round-trip debugging. |

---

## Record fields (vendor contact)

Identical to the customer schema plus an optional `vendor_details`
block:

```json
{
  "contact_id": "CONT-CISCO-TAC-001",
  "kind": "contact_vendor",
  "organization": {
    "name": "Cisco Systems",
    "type": "vendor"
  },
  "team": {
    "name": "TAC Sev-1",
    "escalation_level": 1,
    "hours": "24x7"
  },
  "person": {
    "name": "TAC Duty Engineer",
    "role": "TAC L2",
    "phone_primary": "+1-800-553-2447",
    "email": "tac@cisco.com"
  },
  "vendor_details": {
    "product_lines": ["Firewall", "SD-WAN"],
    "support_portal_url": "https://support.cisco.com",
    "tac_phone": "+1-800-553-2447",
    "sla_tier": "Enterprise 24x7",
    "account_manager": "John Doe"
  },
  "source_file": "cisco_support_matrix.xlsx"
}
```

All fields inside `vendor_details` are optional. When present, they are
flattened into the chunk's embedded text so phrases like "Cisco TAC
number" / "firewall vendor support portal" retrieve correctly.

---

## How records become chunks

Per record, ingestion emits:

- `content` — labeled prose card (`Contact: ...`, `Organization: ...`,
  `Phone: ...`, `Email: ...`, etc.). See `_render_contact_body()` in
  `backend/services/contextual_ingestion_service.py`.
- `metadata_json.primary_id` — the `contact_id`.
- `metadata_json.id_type` — `"contact_id"`.
- `metadata_json.doc_kind` — `"contact_customer"` or `"contact_vendor"`
  (taken from the record itself, not the filename).
- Additional metadata filters: `organization_name`, `organization_type`,
  `organization_segment`, `organization_region`, `team_name`,
  `escalation_level`, `person_name`, `person_role`, `person_email`,
  `person_phone_primary`. These feed future retrieval lanes (Escalation
  mode, Vendor/OEM mode).

Document-level `doc_kind` is set to the majority kind across records
when a file mixes customer and vendor rows; per-row `doc_kind` still
reflects each record's own type so retrieval filtering stays precise.

---

## Workflow (end-to-end)

```bash
# 1. Convert source Excel -> JSON
python -m backend.scripts.etl_excel_contacts \
  --in /data/contacts/acme.xlsx \
  --sheet "Escalation Matrix" \
  --kind contact_customer \
  --org-name "Acme Corp" \
  --org-type customer \
  --out /tmp/acme_contacts.json

# 2. Validate the produced JSON
python -m backend.scripts.validate_contacts /tmp/acme_contacts.json
# Expect: "OK - N contact record(s) valid."

# 3. Ingest via PREP-B CLI
python -m backend.scripts.bulk_ingest \
  --path /tmp/acme_contacts.json \
  --doc-kind contact_customer \
  --owner-id <uuid> \
  --resume-log /tmp/bulk_ingest.jsonl
```

For every new source format (PDF, Word, Google Sheet export, Notion
dump, ...), copy `etl_excel_contacts.py`, adjust the cell-to-field
mapping, and rerun the same two validate + ingest steps. The canonical
JSON shape is the integration contract; everything else is ETL detail.
