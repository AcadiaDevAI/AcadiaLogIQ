"""Org-module router — exposes the active org's public config.

GET /orgs/me/config is the single source of truth the frontend reads for
per-org display name, theme (accent), enabled flows, and feature flags — so
that config lives in the org profile (backend/orgs/<org>.py), not hardcoded
twice on the frontend.
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends

from backend.orgs.base import OrgProfile
from backend.orgs.context import get_org_profile

router = APIRouter(prefix="/orgs", tags=["orgs"])


@router.get("/me/config")
async def my_org_config(
    profile: OrgProfile = Depends(get_org_profile),
) -> Dict[str, Any]:
    return profile.public_config()
