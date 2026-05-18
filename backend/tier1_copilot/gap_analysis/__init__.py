"""
Gap Analysis — generates two long-form reports from an existing
ingested ticket JSON, addressed to a single incident number.

Reports produced (in parallel, never sequenced):
    1. Gap Analysis Report      (LogIQ Master Prompt v1.0)
    2. Blameless Post-Mortem    (SRE-style 13-section report)

Design intent
-------------
The module is a deliberate FORK of ``backend/tier1_copilot/rca/``. It
reuses the same proven pattern (Bedrock Claude → two parallel prompts
→ Markdown output, failure-open per panel) without sharing any code
with RCA, so the two features can evolve independently and a change
to RCA's prompts / model / lookup never lands silently in Gap
Analysis (or vice versa).

Public surface
--------------
``routes.py`` mounts ``POST /gap-analysis/{incident_number}``. The
sidebar's "Gap Analysis" button hits it. Strict-auth gated via
``lazy_auth_dependency`` (Clerk JWT required, like every other
protected route in the app).
"""
