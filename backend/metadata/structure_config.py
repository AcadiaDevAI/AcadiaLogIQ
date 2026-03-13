"""
Phase 2 operational structure configuration.

What this file adds:
- central section taxonomy
- operational chunk type definitions
- future-friendly metadata rules
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class OperationalSectionRule:
    canonical_name: str
    aliases: List[str]
    chunk_type: str
    priority: int = 100


REQUIRED_OPERATIONAL_SECTIONS: List[OperationalSectionRule] = [
    OperationalSectionRule("title", ["title", "document title"], "heading_chunk", 1),
    OperationalSectionRule("document_type", ["document type", "type"], "classification_chunk", 2),
    OperationalSectionRule("purpose_summary", ["purpose", "summary", "overview"], "summary_chunk", 3),
    OperationalSectionRule("domain", ["domain", "service domain"], "classification_chunk", 4),
    OperationalSectionRule("vendor_product", ["vendor", "product", "vendor/product"], "classification_chunk", 5),
    OperationalSectionRule("symptoms", ["symptoms", "symptom", "trigger conditions", "issue", "problem"], "symptom_chunk", 6),
    OperationalSectionRule("diagnostic_steps", ["diagnostic steps", "troubleshooting", "diagnosis", "investigation"], "diagnostic_chunk", 7),
    OperationalSectionRule("resolution_steps", ["resolution", "fix", "solution", "remediation"], "resolution_chunk", 8),
    OperationalSectionRule("escalation_criteria", ["escalation", "escalation criteria", "when to escalate"], "escalation_chunk", 9),
    OperationalSectionRule("references", ["references", "links", "related docs"], "reference_chunk", 10),
]

OPTIONAL_OPERATIONAL_SECTIONS: List[OperationalSectionRule] = [
    OperationalSectionRule("preconditions", ["preconditions", "before you begin"], "precondition_chunk", 20),
    OperationalSectionRule("commands", ["commands", "cli", "command"], "command_chunk", 21),
    OperationalSectionRule("expected_results", ["expected results", "expected output"], "validation_chunk", 22),
    OperationalSectionRule("known_errors", ["known errors", "errors", "error codes"], "error_chunk", 23),
    OperationalSectionRule("root_cause_notes", ["root cause", "root cause notes"], "root_cause_chunk", 24),
    OperationalSectionRule("workarounds", ["workarounds", "temporary fix"], "workaround_chunk", 25),
    OperationalSectionRule("validation_post_checks", ["validation", "post checks", "verification"], "validation_chunk", 26),
]

SECTION_ALIAS_TO_RULE: Dict[str, OperationalSectionRule] = {}
for rule in REQUIRED_OPERATIONAL_SECTIONS + OPTIONAL_OPERATIONAL_SECTIONS:
    SECTION_ALIAS_TO_RULE[rule.canonical_name.lower()] = rule
    for alias in rule.aliases:
        SECTION_ALIAS_TO_RULE[alias.lower()] = rule


def match_operational_section(heading: str):
    heading_normalized = (heading or "").strip().lower()
    for alias, rule in SECTION_ALIAS_TO_RULE.items():
        if alias in heading_normalized:
            return rule
    return None