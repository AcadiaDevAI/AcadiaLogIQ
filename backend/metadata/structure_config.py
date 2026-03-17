"""
Operational structure configuration — Phase 6 accuracy fix.
Adds runbook scenario pattern matching, alert signature recognition,
and telecom/network-specific section aliases so the parser and
retrieval pipeline understand operational document structure.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class OperationalSectionRule:
    canonical_name: str
    aliases: List[str]
    chunk_type: str
    priority: int = 100


REQUIRED_OPERATIONAL_SECTIONS: List[OperationalSectionRule] = [
    OperationalSectionRule("title", ["title", "document title", "troubleshooting runbook"], "heading_chunk", 1),
    OperationalSectionRule("document_type", ["document type", "type"], "classification_chunk", 2),
    OperationalSectionRule("purpose_summary", ["purpose", "summary", "overview", "incident summary"], "summary_chunk", 3),
    OperationalSectionRule("domain", ["domain", "service domain"], "classification_chunk", 4),
    OperationalSectionRule("vendor_product", ["vendor", "product", "vendor/product"], "classification_chunk", 5),
    OperationalSectionRule("symptoms", [
        "symptoms", "symptom", "trigger conditions", "issue", "problem",
        "alert signatures", "alarm signatures",
    ], "symptom_chunk", 6),
    OperationalSectionRule("diagnostic_steps", [
        "diagnostic steps", "troubleshooting", "diagnosis", "investigation",
        "corrective actions", "diagnostic procedure",
    ], "diagnostic_chunk", 7),
    OperationalSectionRule("probable_causes", [
        "probable causes", "probable cause", "root cause", "possible causes",
        "failure causes", "cause analysis",
    ], "root_cause_chunk", 8),
    OperationalSectionRule("resolution_steps", [
        "resolution", "fix", "solution", "remediation",
        "corrective actions", "repair steps",
    ], "resolution_chunk", 9),
    OperationalSectionRule("escalation_criteria", [
        "escalation", "escalation criteria", "when to escalate",
        "escalation matrix",
    ], "escalation_chunk", 10),
    OperationalSectionRule("references", ["references", "links", "related docs"], "reference_chunk", 11),
]

OPTIONAL_OPERATIONAL_SECTIONS: List[OperationalSectionRule] = [
    OperationalSectionRule("preconditions", ["preconditions", "before you begin"], "precondition_chunk", 20),
    OperationalSectionRule("commands", ["commands", "cli", "command", "commands / tools"], "command_chunk", 21),
    OperationalSectionRule("expected_results", ["expected results", "expected output"], "validation_chunk", 22),
    OperationalSectionRule("known_errors", ["known errors", "errors", "error codes"], "error_chunk", 23),
    OperationalSectionRule("root_cause_notes", ["root cause", "root cause notes"], "root_cause_chunk", 24),
    OperationalSectionRule("workarounds", ["workarounds", "temporary fix"], "workaround_chunk", 25),
    OperationalSectionRule("validation_post_checks", [
        "validation", "post checks", "verification",
        "validation / post-check", "post-check",
    ], "validation_chunk", 26),
    OperationalSectionRule("severity", ["severity", "priority", "impact level"], "classification_chunk", 27),
    # --- Runbook scenario patterns ---
    OperationalSectionRule("scenario", [
        "scenario a", "scenario b", "scenario c", "scenario d",
        "scenario e", "scenario f",
    ], "diagnostic_chunk", 30),
]

SECTION_ALIAS_TO_RULE: Dict[str, OperationalSectionRule] = {}
for rule in REQUIRED_OPERATIONAL_SECTIONS + OPTIONAL_OPERATIONAL_SECTIONS:
    SECTION_ALIAS_TO_RULE[rule.canonical_name.lower()] = rule
    for alias in rule.aliases:
        SECTION_ALIAS_TO_RULE[alias.lower()] = rule


def match_operational_section(heading: str) -> Optional[OperationalSectionRule]:
    """
    Match a heading to an operational section rule.
    Checks both exact and substring matching against aliases.
    """
    heading_normalized = (heading or "").strip().lower()
    if not heading_normalized:
        return None

    # Exact match first
    if heading_normalized in SECTION_ALIAS_TO_RULE:
        return SECTION_ALIAS_TO_RULE[heading_normalized]

    # Substring match
    for alias, rule in SECTION_ALIAS_TO_RULE.items():
        if alias in heading_normalized:
            return rule

    return None