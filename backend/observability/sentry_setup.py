"""
Sentry initialisation + PII scrubbing.

Single entry point — ``init_sentry(dsn, env, release, traces_sample_rate)``
— that calls ``sentry_sdk.init(...)`` when a DSN is configured and is a
silent no-op otherwise. The "no-op when unset" semantics mean this
module is safe to import + call even before Sentry is provisioned for
the environment.

What's captured
---------------
* Every unhandled exception (FastAPI returns 500 → Sentry event).
* ``logger.warning("...")`` / ``logger.error("...")`` / ``logger.exception(...)``
  records (via ``LoggingIntegration(event_level=WARNING)``). Lower
  levels stay out of Sentry to avoid spending quota on INFO chatter.
* 10% of request traces by default (``traces_sample_rate=0.1``) — gives
  flame graphs of the request lifecycle for slow endpoints without
  paying for every request.

What's scrubbed (PII / secrets)
-------------------------------
Customer ticket text and Clerk JWTs are sensitive. The
``before_send`` hook redacts:

  * Headers: ``Authorization``, ``Cookie``, ``X-API-Key``, any
    ``X-Clerk-*`` header.
  * Cookies: the entire ``cookies`` map on the request context.
  * Request bodies on routes that ingest customer text — ``/ask``,
    ``/tier1/*``, ``/rca/*``, ``/gap-analysis/*``, ``/chat/*``.
    These routes receive Markdown / free-form English from the
    customer's tickets and emails; we never want it in a third-
    party SaaS dashboard. The full body is replaced with the
    string ``"<redacted: customer-content route>"``.

The scrubber is intentionally generous (allow-list of "safe" routes
to forward bodies → we have none today, so the default is "always
scrub bodies"). When a future route needs request bodies in Sentry,
add it to ``_BODY_FORWARD_ALLOWLIST`` explicitly.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional


logger = logging.getLogger("acadia-log-iq")


# Header names we always redact. Compared case-insensitively because
# different HTTP stacks normalise differently (uvicorn lowercases,
# tests sometimes don't).
_SENSITIVE_HEADERS = frozenset({
    "authorization",
    "cookie",
    "x-api-key",
})

# Header prefixes we redact wholesale. Any header whose name starts
# with one of these (case-insensitive) is treated as sensitive.
_SENSITIVE_HEADER_PREFIXES = (
    "x-clerk-",
)

# Routes whose request bodies are *allowed* to flow into Sentry. We
# default to "scrub everything" — adding an entry here is an explicit
# decision that the route's body contains no customer data. Keep
# empty until a route demonstrably needs it.
_BODY_FORWARD_ALLOWLIST: frozenset[str] = frozenset()

# Routes that always contain customer text in their bodies. Listed
# explicitly so a quick prefix match in ``_should_scrub_body`` is
# enough to redact them. Every other body is also scrubbed by
# default — this list is only here for future ``allowlist`` toggling.
_CUSTOMER_BODY_ROUTE_PREFIXES = (
    "/ask",
    "/tier1/",
    "/rca/",
    "/gap-analysis/",
    "/chat/",
)


def _redact_headers(headers: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return a copy of ``headers`` with sensitive entries replaced.

    Sentry's ``before_send`` is called on every event and trace, so
    this runs hot. The implementation is O(n) over the headers map
    with no regex / allocations beyond the result dict.
    """
    if not headers:
        return headers
    redacted: Dict[str, Any] = {}
    for name, value in headers.items():
        lower = name.lower() if isinstance(name, str) else ""
        if lower in _SENSITIVE_HEADERS or any(
            lower.startswith(p) for p in _SENSITIVE_HEADER_PREFIXES
        ):
            redacted[name] = "<redacted>"
        else:
            redacted[name] = value
    return redacted


def _should_scrub_body(route_path: Optional[str]) -> bool:
    """Return True iff this route's body might contain customer text."""
    if not route_path:
        return True  # safer default — unknown route is treated as sensitive
    if route_path in _BODY_FORWARD_ALLOWLIST:
        return False
    return any(route_path.startswith(p) for p in _CUSTOMER_BODY_ROUTE_PREFIXES) \
        or True  # see module docstring: default is always-scrub


def _scrub_event(event: Dict[str, Any], hint: Dict[str, Any]) -> Dict[str, Any]:
    """``before_send`` hook — mutate ``event`` to remove PII.

    Sentry calls this for *every* event (errors AND transactions) so
    it must be cheap and robust. We never raise from here — a
    scrubber bug must never block error reporting.
    """
    try:
        request_ctx = event.get("request") or {}
        # 1. Redact headers and cookies.
        if request_ctx:
            headers = request_ctx.get("headers")
            if headers:
                request_ctx["headers"] = _redact_headers(headers)
            if request_ctx.get("cookies"):
                request_ctx["cookies"] = "<redacted>"
            # 2. Redact request body on customer-content routes.
            route_path = request_ctx.get("url", "") or request_ctx.get("path", "")
            # Sentry sets ``url`` as the full URL; extract the path
            # portion for the route check.
            try:
                from urllib.parse import urlparse
                if "://" in route_path:
                    route_path = urlparse(route_path).path
            except Exception:
                pass
            if _should_scrub_body(route_path):
                if "data" in request_ctx:
                    request_ctx["data"] = "<redacted: customer-content route>"

        # 3. Strip ``Authorization``-like values out of breadcrumbs.
        for breadcrumb in event.get("breadcrumbs", {}).get("values", []) or []:
            data = breadcrumb.get("data") or {}
            if "Authorization" in data:
                data["Authorization"] = "<redacted>"
            if "headers" in data:
                data["headers"] = _redact_headers(data.get("headers"))
    except Exception as exc:  # noqa: BLE001 — defensive: scrubber bug ≠ data loss
        logger.warning("[sentry] scrubber error suppressed: %s", exc)

    return event


def init_sentry(
    *,
    dsn: Optional[str],
    environment: Optional[str] = None,
    release: Optional[str] = None,
    traces_sample_rate: float = 0.1,
) -> bool:
    """Initialise the Sentry SDK if a DSN is configured.

    Returns
    -------
    True   — SDK initialised, events will be sent.
    False  — DSN empty / Sentry SDK not installed; the rest of the
             application is unaffected.

    The function never raises. A misconfiguration logs a warning and
    falls through to the "Sentry disabled" path so a bad DSN can't
    prevent the API from starting.
    """
    dsn = (dsn or "").strip()
    if not dsn:
        logger.info(
            "[sentry] disabled (SENTRY_DSN unset) — error monitoring "
            "is not active; logs still ship to CloudWatch.",
        )
        return False

    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.logging import LoggingIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration
    except ImportError as exc:
        logger.warning(
            "[sentry] sentry-sdk not installed (%s) — error monitoring "
            "disabled. Add ``sentry-sdk[fastapi]`` to requirements.txt.",
            exc,
        )
        return False

    try:
        sentry_sdk.init(
            dsn=dsn,
            environment=environment or "dev",
            release=release or None,
            # 100% of errors are reported. Tracing is sampled to keep
            # spend predictable; bump if you want full request flame
            # graphs (and accept the higher event volume).
            sample_rate=1.0,
            traces_sample_rate=max(0.0, min(1.0, float(traces_sample_rate))),
            # PII scrubber. Mutates the event dict in place; see
            # ``_scrub_event`` for the full redaction rules.
            before_send=_scrub_event,
            # Same scrubber for transactions (perf events) so a slow
            # ``/ask`` trace doesn't carry the request body into the
            # transaction's "spans" payload.
            before_send_transaction=_scrub_event,
            # Capture customer text only after scrubbing, never raw.
            # ``send_default_pii=False`` is Sentry's belt to our
            # braces — even if our scrubber misses something, the
            # SDK itself won't attach IP / cookies / headers by
            # default.
            send_default_pii=False,
            integrations=[
                StarletteIntegration(),
                FastApiIntegration(),
                # event_level=WARNING means logger.warning + .error +
                # .exception flow into Sentry as events. level=INFO
                # means INFO breadcrumbs (lightweight context attached
                # to actual events). We never spend per-event budget
                # on INFO chatter.
                LoggingIntegration(level=logging.INFO, event_level=logging.WARNING),
            ],
        )
    except Exception as exc:  # noqa: BLE001 — init failures must not crash startup
        logger.warning(
            "[sentry] init failed (%s) — error monitoring disabled. "
            "Application boot continues normally.", exc,
        )
        return False

    logger.info(
        "[sentry] initialised env=%s release=%s traces_sample_rate=%.2f",
        environment, release, traces_sample_rate,
    )
    return True
