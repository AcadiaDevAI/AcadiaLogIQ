"""Acadia Escalation Contact Directory — structured lookup.

Mirrors `Acadia_Escalation_Contact_Directory.md` so Stage-5 escalation
can surface a one-line contact recommendation deterministically, even
before / alongside the PDF being retrieved by the RAG layer.

Editing guide
-------------
The three tables below are the only things you need to touch when the
directory doc changes:

    CUSTOMER_CONTACTS     — §3 of the doc (per-customer primary contact)
    VENDOR_CONTACTS       — §2 of the doc (vendor / OEM TAC lines)
    ASSET_FAMILY_TO_INTERNAL — §1 + §4 of the doc (Acadia internal Tier-2)

Each row has:
    - "aliases"  (CUSTOMER_CONTACTS only) or "keywords" (the others)
      that drive case-insensitive substring matching against the
      ticket's customer / alert_type / asset_name / technology fields.
    - "name"   — short label that appears as the bold contact line.
    - "detail" — phone / email / portal / entitlement, one line.

`lookup_directory_contacts(...)` returns at most one row per bucket
(Customer, Vendor, Internal) plus a fallback row when nothing matches.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from backend.tier1_copilot.schemas import Tier1DirectoryContact


# ─────────────────────────────────────────────────────────────
# §3 Customer technical contacts
# ─────────────────────────────────────────────────────────────
CUSTOMER_CONTACTS: List[Dict[str, Any]] = [
    {
        "aliases": ["aetheris corp", "aetheris"],
        "name": "Drew Holloway — Aetheris IT Operations Lead",
        "detail": "+1-555-1001 · drew.holloway@aetheris.example · 24x7 bridge +1-555-1002",
    },
    {
        "aliases": ["autocorp llc", "autocorp"],
        "name": "Sam Patel — AutoCorp Network Manager",
        "detail": "+1-555-1011 · spatel@autocorp.example · 24x7 escalation +1-555-1012",
    },
    {
        "aliases": ["bevcorp intl", "bevcorp"],
        "name": "Marisol Vega — BevCorp Distribution IT Lead",
        "detail": "+1-555-1021 · mvega@bevcorp.example · ops bridge +1-555-1022",
    },
    {
        "aliases": ["cybercorp"],
        "name": "Ren Tanaka — CyberCorp InfoSec Operations",
        "detail": "+1-555-1031 · rtanaka@cybercorp.example · security bridge +1-555-1032 (ISO clearance required)",
    },
    {
        "aliases": ["energynet inc", "energynet"],
        "name": "Jordan Brooks — EnergyNet Field Network Lead",
        "detail": "+1-555-1041 · jbrooks@energynet.example · NERC-CIP bridge +1-555-1042",
    },
    {
        "aliases": ["global-health-link", "global health link", "ghl"],
        "name": "Dr. Priya Anand — GHL Clinical Systems Operations",
        "detail": "+1-555-1051 · panand@ghl.example · HIPAA bridge +1-555-1052",
    },
    {
        "aliases": ["mediagiant corp", "mediagiant"],
        "name": "Riya Khan — MediaGiant Broadcast NetOps",
        "detail": "+1-555-1061 · rkhan@mediagiant.example · broadcast NOC +1-555-1062",
    },
    {
        "aliases": ["aetheris retail", "city-alpha", "city alpha"],
        "name": "Lin Park — Aetheris Retail Site IT Coordinator",
        "detail": "+1-555-1071 · lpark@aetheris-retail.example · retail dispatch +1-555-1072",
    },
]


# ─────────────────────────────────────────────────────────────
# §2 Vendor / OEM support
# ─────────────────────────────────────────────────────────────
VENDOR_CONTACTS: List[Dict[str, Any]] = [
    {
        "keywords": [
            "bgp", "ospf", "core switch", "core routing", "ios", "nx-os", "cisco",
        ],
        "name": "Cisco TAC — Core routing / IOS",
        "detail": "+1-800-553-2447 · tac@cisco.com · https://mycase.cloudapps.cisco.com/case · Smart Net 24x7 Premium",
    },
    {
        "keywords": [
            "mpls", "wan circuit", "wan", "carrier", "dark fiber", "last-mile", "providerx",
        ],
        "name": "Provider Carrier Ops",
        "detail": "+1-866-555-7777 (NOC) · +1-866-555-7778 (P1) · noc-escalations@providerx.com",
    },
    {
        "keywords": ["borealis", "managed router", "regional isp"],
        "name": "Borealis-ISP Engineering",
        "detail": "+1-555-0901 (NOC) · +1-555-0902 (P1) · engineering@borealis-isp.com",
    },
    {
        "keywords": [
            "aruba", "wireless controller", "clearpass", "cppm", "airwave",
            "wireless ap", "access point",
        ],
        "name": "Aruba TAC — HPE Aruba Networking",
        "detail": "+1-800-943-4526 · https://asp.arubanetworks.com · Foundation Care 24x7",
    },
    {
        "keywords": ["zebra", "handheld", "scanner", "mc scanner", "tc scanner"],
        "name": "Zebra Technologies Support",
        "detail": "+1-800-653-5350 · techsupport@zebra.com · OneCare Premier 24x7",
    },
    {
        "keywords": [
            "citrix", "v-desktop", "vdesktop", "vdi", "delivery controller",
            "workspace app", "daas",
        ],
        "name": "Citrix Cloud Software Group Support",
        "detail": "+1-800-424-8749 · https://support.citrix.com · Citrix Premier 24x7",
    },
    {
        "keywords": [
            "aetheris app", "aetheris directory", "aft fax", "aetheris server",
        ],
        "name": "Aetheris Support",
        "detail": "+1-555-0801 · +1-555-0802 (P1) · vendor-support@aetheris.example",
    },
    {
        "keywords": ["ghl-app", "ghl app", "global-health-link app", "healthcare app"],
        "name": "GHL-App Support",
        "detail": "+1-555-0811 · +1-555-0812 (P1) · support@ghl-app.example · Healthcare 24x7 (HIPAA)",
    },
    {
        "keywords": [
            "storage", "san", "nas", "data-link", "cyber-node", "replication",
            "backup array",
        ],
        "name": "Cyber-Node Storage Vendor",
        "detail": "+1-555-0850 · +1-555-0851 (P1) · tac@cybernode.example · Premium 4hr onsite",
    },
    {
        "keywords": ["z-tech", "ztech", "edge gateway"],
        "name": "Z-Tech Gateway Support",
        "detail": "+1-555-0860 · +1-555-0861 (P1) · support@ztech.example",
    },
    {
        "keywords": ["access-gate", "access gate", "cpe", "customer premises"],
        "name": "OEM Access-Gate Vendor",
        "detail": "+1-555-0870 · +1-555-0871 (P1) · support@accessgate.example",
    },
    {
        "keywords": ["voice", "sip", "sbc", "fax", "trunk"],
        "name": "Telecom Voice Carrier (T-Com Voice)",
        "detail": "+1-555-0700 · +1-555-0701 (24x7 voice NOC) · noc@t-com-voice.example",
    },
    {
        "keywords": ["ng911", "911", "ecms", "emergency voice", "public safety"],
        "name": "ECMS-VoIP / NG911 NOC",
        "detail": "+1-555-0710 · +1-555-0711 (24x7) · ng911-noc@ng911.example · P1 SLA 15min",
    },
    {
        "keywords": [
            "hardware", "rma", "depot", "smart-hands", "field replacement",
            "chassis swap",
        ],
        "name": "HW-REPAIR — Field Hardware Repair",
        "detail": "+1-555-0880 (dispatch) · +1-555-0881 (after-hours) · dispatch@hwrepair.example",
    },
]


# ─────────────────────────────────────────────────────────────
# §1 + §4 Internal Tier-2 routing
# ─────────────────────────────────────────────────────────────
ASSET_FAMILY_TO_INTERNAL: List[Dict[str, Any]] = [
    {
        "keywords": [
            "bgp", "mpls", "wan", "wireless", "controller", "ap", "z-tech",
            "cpe", "voice", "sip", "sbc", "borealis", "carrier", "fiber",
            "router", "firewall", "load balancer",
        ],
        "name": "NET-OPS — Network Tier-2",
        "detail": "+1-555-0201 · netops@acadia.internal · #netops-tier2",
    },
    {
        "keywords": [
            "citrix", "v-desktop", "vdesktop", "vdi", "aetheris app", "ghl",
            "handheld", "scanner", "zebra", "application",
        ],
        "name": "APP-SVC-POOL-99 — Application Services Tier-2",
        "detail": "+1-555-0202 · appsvc@acadia.internal · #appsvc-tier2",
    },
    {
        "keywords": ["storage", "san", "nas", "backup", "replication", "cyber-node"],
        "name": "Cyber-Node Storage Team — Storage Tier-2",
        "detail": "+1-555-0203 · storage@acadia.internal · #storage-tier2",
    },
    {
        "keywords": ["hardware", "swap", "rma", "field", "dispatch"],
        "name": "DISPATCH-CENTER — Field Tech Dispatch",
        "detail": "+1-555-0204 · dispatch@acadia.internal · #field-dispatch",
    },
    {
        "keywords": ["ng911", "911", "ecms", "outage bridge", "p1 multi-team"],
        "name": "Partner Major Incident Mgmt (P-MIM) — Tier-3",
        "detail": "+1-555-0302 · mim@acadia.internal · #mim-bridge",
    },
]


FALLBACK_CONTACT = Tier1DirectoryContact(
    label="Directory",
    name="Acadia Escalation Contact Directory",
    detail=(
        "No specific match for this customer or asset family — "
        "consult the full KB doc 'Acadia Escalation Contact Directory' for routing."
    ),
)


# ─────────────────────────────────────────────────────────────
# Lookup
# ─────────────────────────────────────────────────────────────
def _normalize(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _match_customer(customer: Optional[str]) -> Optional[Tier1DirectoryContact]:
    cust_n = _normalize(customer)
    if not cust_n:
        return None
    for entry in CUSTOMER_CONTACTS:
        for alias in entry["aliases"]:
            alias_n = alias.lower()
            if alias_n in cust_n or cust_n in alias_n:
                return Tier1DirectoryContact(
                    label="Customer",
                    name=entry["name"],
                    detail=entry["detail"],
                )
    return None


def _kw_match(kw: str, haystack: str) -> bool:
    """Word-boundary-aware substring match. Prevents false positives like
    `ap` matching inside `app` while still letting hyphenated keywords
    (`z-tech`, `v-desktop`) match cleanly."""
    pattern = r"(?<!\w)" + re.escape(kw.lower()) + r"(?!\w)"
    return re.search(pattern, haystack) is not None


def _match_keyword_table(
    table: List[Dict[str, Any]],
    haystack: str,
    label: str,
) -> Optional[Tier1DirectoryContact]:
    if not haystack:
        return None
    for entry in table:
        for kw in entry["keywords"]:
            if _kw_match(kw, haystack):
                return Tier1DirectoryContact(
                    label=label,
                    name=entry["name"],
                    detail=entry["detail"],
                )
    return None


def lookup_directory_contacts(
    *,
    customer: Optional[str] = None,
    alert_type: Optional[str] = None,
    asset_name: Optional[str] = None,
    technology: Optional[str] = None,
    notes: Optional[str] = None,
) -> List[Tier1DirectoryContact]:
    """Return up to 3 directory lines (Customer, Vendor, Internal Tier-2).

    Falls back to a single FALLBACK_CONTACT row when nothing matches so
    the escalation card always renders something useful."""
    haystack = " ".join(
        _normalize(x) for x in (alert_type, asset_name, technology, notes)
    ).strip()

    out: List[Tier1DirectoryContact] = []

    cust = _match_customer(customer)
    if cust:
        out.append(cust)

    vendor = _match_keyword_table(VENDOR_CONTACTS, haystack, "Vendor / OEM")
    if vendor:
        out.append(vendor)

    internal = _match_keyword_table(
        ASSET_FAMILY_TO_INTERNAL, haystack, "Internal Tier-2",
    )
    if internal:
        out.append(internal)

    if not out:
        out.append(FALLBACK_CONTACT)

    return out
