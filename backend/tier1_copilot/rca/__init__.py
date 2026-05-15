"""Sprint 13.32 — RCA-from-incident-number module.

Isolated from the Tier-1 Resolution Journey. The sidebar's new
"RCA" button opens a fresh right-pane flow that asks for an incident
number, looks up the ticket's structured JSON, and runs two LLM
calls in parallel to produce:

  * Customer-Facing External RCA  — strict 7-section markdown, plain
                                    English, safety-stripped.
  * Internal Incident RCA         — full 12-section technical markdown
                                    with hostnames, CLI commands,
                                    scores, and appendix.

Both prompts run against the same ticket dict (looked up by
``Metadata.Incident_Number`` in the existing ``chunks.metadata_json``
JSONB store the Stage 3 cohort harvester already uses).
"""
