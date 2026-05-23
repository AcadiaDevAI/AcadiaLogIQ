"""
Ticket Filter — independent feature.

Lets engineers pull historical tickets matching a
``(SLA_Target_Met, Resolution_Quality_Score)`` filter. No
dependency on the RCA / Gap Analysis / Chat code paths — this
module owns its router, its SQL helpers, and its own response
schema so changes here can't disturb any other flow.

Surface
-------
* ``POST /ticket-filter``  — auth-required, rate-limited, returns
  paginated ticket summaries.
* SQL lives in ``query.py`` so the route is a thin HTTP wrapper
  around a pure function (unit-testable without FastAPI).
"""
