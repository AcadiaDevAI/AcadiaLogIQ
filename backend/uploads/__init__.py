"""
Upload pipeline package.

Owns the logic for the presigned-PUT browser → S3 upload flow:

* ``keys``     — pure functions that build the S3 object key.
                 ``tenants/{tenant_id}/raw/{yyyy}/{mm}/{dd}/{job}_{name}``
* ``schemas``  — Pydantic request/response models for the two HTTP routes.
* ``service``  — orchestration that the route layer calls into.

The HTTP routes themselves live in ``backend/api.py`` (consistent with
the rest of the project). They are thin shells that delegate here.

Design intent
-------------
Phase 1 today treats each authenticated user as their own tenant
(``tenant_id == user_id``). When Phase 2 introduces a real tenants
table, only ``keys.derive_tenant_id`` changes; the S3 object layout,
the route contracts, and the ingestion code path all stay the same.
"""
