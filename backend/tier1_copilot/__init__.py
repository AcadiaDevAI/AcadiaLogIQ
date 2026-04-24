"""Tier-1 Alert Copilot module.

Purpose: Convert structured alert intake from a NOC Tier-1 engineer
into a deterministic troubleshooting answer drawn from historical
incident JSON.

Responsibility: Self-contained module — own routes, own schemas,
own retrieval, own prompt. Shares only the chunks table, Bedrock
client, and config from the parent app.

Flow position: Sits parallel to the existing chat / fingerprint
flows; reachable from the landing page as a separate entry point.
Removable in one folder + one router-include line if rejected.
"""
