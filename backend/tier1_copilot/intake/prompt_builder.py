"""Sprint 9.2 — Evidence-grounded extraction prompt.

The Sprint 9 prompt sampled the catalog (top-30 most-frequent assets,
alert types, customers) as "hints" to the LLM. Reproduction testing
showed the LLM treated those hints as a closed vocabulary and picked
from them when its own extraction was uncertain — producing
hallucinated "matched" values like `v-bay-core-rtr` for an email
mentioning `NY4-CORE-RTR-01`.

Sprint 9.2 deletes the hint mechanism. The LLM is now a pure extractor
of what is *literally written in the content*. Catalog matching is a
separate, deterministic step performed by the validator AFTER
extraction. Every non-null field must come with `evidence` — an exact
substring of the input — which the validator independently verifies.
A field whose evidence isn't found in raw_text is rejected as a
hallucination.

This module exposes only `build_extraction_prompt(raw_text,
source_type, n_candidates)`. The `catalogs` parameter from Sprint 9
has been removed from the call surface entirely.
"""
from __future__ import annotations

from backend.config import settings


_EXTRACTION_PROMPT = """You are a NOC ticket extractor. Read the {source_type} content below. Extract structured fields ONLY from what is literally written in the content. Return JSON only — no prose, no explanation, no markdown.

CRITICAL RULES:
- Extract values that are LITERALLY PRESENT in the content. Do not infer, guess, or substitute "similar" terms.
- For each non-null field, you MUST also return an `evidence` substring — the EXACT TEXT from the content that justifies that field's value. Evidence must be a verbatim substring (case may differ but the words must match) of the input content.
- If a field cannot be cleanly extracted from the content, return null for that field. Returning null is correct and expected — do NOT guess.
- Do NOT consult any catalog, list, or known-values reference. Extract from the content, full stop.

Return up to {n_candidates} alternative interpretations, ordered by likelihood. Each must be MEANINGFULLY DIFFERENT from the others (different severity, different asset, different alert type, or different evidence span).

Output schema (JSON array):
[
  {{
    "severity": "P1" | "P2" | "P3" | "P4" | null,
    "asset_name": "<verbatim asset/system mentioned>" | null,
    "alert_type": "<verbatim log signature / error code / fingerprint, OR human description as fallback>" | null,
    "customer": "<verbatim customer/organization name>" | null,
    "location": "<verbatim location>" | null,
    "users_impacted_count": integer | null,
    "evidence": {{
      "severity": "<exact substring from content>" | null,
      "asset_name": "<exact substring from content>" | null,
      "alert_type": "<exact substring from content>" | null,
      "customer": "<exact substring from content>" | null
    }}
  }},
  ...
]

ALERT TYPE EXTRACTION — STRICT PRIORITY HIERARCHY:
The `alert_type` field must reflect the most MACHINE-GREPPABLE form of the issue mentioned in the content. Tier-2 engineers grep these strings against current-incident logs, so a log signature beats a human paraphrase every time.

When the content contains MORE THAN ONE form of the issue, pick the highest-priority form below:

  Priority 1 — LOG SIGNATURES (cisco-style or similar log lines).
    Format: %FACILITY-SEVERITY-MNEMONIC: <message>
    Examples that MUST be picked verbatim when present:
      "%BFD-6-ADJ_CHANGE: Adj Down - Control Timer Expired"
      "%BGP-5-ADJCHANGE: neighbor 10.255.0.1 Down - BFD down"
      "%LINEPROTO-5-UPDOWN: Line protocol on Interface ..."
      "%LINK-3-UPDOWN: Interface ... changed state to down"

  Priority 2 — ERROR CODES / TICKET FINGERPRINTS.
    Examples: "ERR-1234", "BGP-3-NOTIFICATION", "INSIDE_WIRING_FAIL",
    "Control Plane CPU > 95%", "TTL 252 (Actual)", "Authentication failure".

  Priority 3 — HUMAN DESCRIPTION (fallback ONLY when no log line / error code is in the content).
    Examples: "BGP neighbor down", "Slow checkout transaction",
    "Fax line not working".

If a log signature AND a human paraphrase both appear in the content, you MUST pick the log signature. Do NOT pick "BGP neighbor down" when the content also says "%BFD-6-ADJ_CHANGE: Adj Down - Control Timer Expired" — pick the latter.

The `evidence.alert_type` substring MUST be the verbatim source text for whichever priority you picked.

Severity decoding (apply only what's actually written):
- "P1" / "P2" / "P3" / "P4" / "Sev 1-4" / "Severity 1-4" → use directly
- "critical" / "outage" / "down" / "complete failure" → P1
- "urgent" / "multiple users affected" / "degraded service" → P2
- "single user" / "minor" / "low impact" → P3
- "informational" / "request" / "FYI" → P4
- nothing severity-related in content → null

Content ({source_type}):
<<<
{raw_text}
>>>
"""


def build_extraction_prompt(
    *,
    raw_text: str,
    source_type: str,
    n_candidates: int = 4,
    # Sprint 9 catalog kwargs accepted-but-ignored for backward
    # compatibility with the existing extractor call surface. They no
    # longer influence the prompt — that's the entire point of 9.2.
    catalogs=None,
) -> str:
    """Sprint 9.2 — content-only extraction prompt.

    The validator independently catalog-matches each extraction after
    the LLM returns. This prevents hint contamination: the LLM cannot
    pick from a list it never sees. `catalogs` is accepted for call-site
    compatibility but explicitly NOT injected into the prompt.
    """
    cap = int(getattr(settings, "INTAKE_MAX_RAW_CHARS", 10000))
    return _EXTRACTION_PROMPT.format(
        source_type=source_type,
        raw_text=(raw_text or "")[:cap],
        n_candidates=int(n_candidates),
    )
