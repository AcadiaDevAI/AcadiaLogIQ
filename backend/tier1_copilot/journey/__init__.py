"""Sprint 10 — Tier-1 Resolution Journey.

After /tier1/analyze succeeds, the journey assembles a 6-stage view
from the same `tier1_sessions.top_5_match_ids` cohort the analyze
flow already ranked. No new ranking, no new embeddings, no LLM hot
path. Stage 0 + 1A + 1B always-visible on first paint; Stages 2-5
lazy-fetched per labeled next-stage button.

Public exports — keep this surface narrow; routes.py is the only
thing FastAPI consumes from this package.
"""
from .schemas import (
    JourneyInitial,
    PivotInsights,
    SearchKBHandoffResponse,
    Stage0BestTicketDistillation,
    Stage1aSmokingGun,
    Stage1bDoNotChase,
    Stage2HistoricalMatches,
    Stage3TroubleshootingApproach,
    Stage4SearchKB,
)

__all__ = [
    "JourneyInitial",
    "PivotInsights",
    "SearchKBHandoffResponse",
    "Stage0BestTicketDistillation",
    "Stage1aSmokingGun",
    "Stage1bDoNotChase",
    "Stage2HistoricalMatches",
    "Stage3TroubleshootingApproach",
    "Stage4SearchKB",
]
