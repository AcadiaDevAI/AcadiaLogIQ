"""Section definitions for the Escalation Procedures KB.

One consolidated PDF (Escalation_Procedures_KB.pdf) carries four
labelled sections. Each leaf option in the frontend maps to exactly
one section id below. The anchor patterns are matched (case-
insensitive, punctuation-tolerant) as heading-style lines near the top
of a page to detect the section start.
"""

from __future__ import annotations

from typing import Dict, List


SECTION_IDS: List[str] = ["cisco", "microsoft", "verizon", "att", "vendor_dispatch"]

# Catch-all section for uploads that have no vendor-section headings — e.g.
# US Pharma's JSON escalation matrices, which are ingested whole and queried
# across the entire KB (no per-vendor picker). Kept OUT of SECTION_IDS so the
# PDF section detector is unaffected.
GENERAL_SECTION = "general"


SECTION_LABELS: Dict[str, str] = {
    "cisco": "Cisco",
    "microsoft": "Microsoft",
    "verizon": "Verizon",
    "att": "AT&T",
    "vendor_dispatch": "Vendor Dispatch",
    "general": "Escalation",
}


SECTION_CATEGORY: Dict[str, str] = {
    "cisco": "OEM Vendor",
    "microsoft": "OEM Vendor",
    "verizon": "Telco",
    "att": "Telco",
    "vendor_dispatch": "Third-party Coordinations",
}


SECTION_ANCHORS: Dict[str, List[str]] = {
    "cisco": ["cisco"],
    "microsoft": ["microsoft", "msft", "ms"],
    "verizon": ["verizon"],
    "att": ["at&t", "at & t", "at and t", "att"],
    "vendor_dispatch": ["vendor dispatch", "vendor-dispatch"],
}


KB_FILENAME = "Escalation_Procedures_KB.pdf"
