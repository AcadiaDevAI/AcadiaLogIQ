"""
Shared infrastructure for the tier1_copilot sub-features.

Modules in here are imported by BOTH the RCA and Gap Analysis
packages. They MUST stay strictly utility-shaped — no business
logic that belongs to one of the two features. The compounding
rule from the original "fork-everything" decision: shared code
lives here only when it can't be cleanly forked without duplicate
DB rows or duplicate side effects.

Current residents:
    report_cache.py   — read / write / invalidate cached report
                        Markdown by (report_kind, incident_number),
                        and record 👍/👎 feedback events.
"""
