"""
Phase 5 — Multi-Agent Troubleshooting Package.
Exposes run_agent_pipeline() and should_escalate_to_agents() as entry points.
Only activates for complex queries; simple/moderate queries bypass entirely.
"""

from backend.agents.orchestrator import run_agent_pipeline, should_escalate_to_agents  # noqa: F401
