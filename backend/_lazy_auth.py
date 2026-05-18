"""
Lazy-import wrapper around ``backend.api.auth_dependency``.

Why this exists
---------------
Every FastAPI sub-router (``tier1_copilot/routes.py``,
``tier1_copilot/journey/routes.py``, ``tier1_copilot/intake/routes.py``)
needs the same Clerk-JWT auth check that the main ``backend/api.py``
routes use. They CAN'T do a top-level ``from backend.api import
auth_dependency`` because the sub-routers are imported BY ``api.py``
during app construction — a top-level import here would create a
circular import.

This tiny module defers the import to request time, which works at
runtime (by then ``api.py`` has finished executing its module body).

Usage
-----
::

    from backend._lazy_auth import lazy_auth_dependency

    @router.get("/something")
    async def handler(
        ...,
        user_id: Optional[str] = Depends(lazy_auth_dependency),
    ):
        ...

The handler receives the authenticated Clerk user_id (a string like
``user_2abc...``) or raises 401 if the JWT is missing/invalid — same
behavior as the main-app routes.
"""

from __future__ import annotations

from typing import Optional

from fastapi import Header, Request


async def lazy_auth_dependency(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> Optional[str]:
    """Defer the import of ``backend.api.auth_dependency`` to request time."""
    from backend.api import auth_dependency

    return await auth_dependency(request, x_api_key)
