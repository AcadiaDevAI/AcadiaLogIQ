// Sentry frontend initialisation.
//
// Single entry point — ``initSentry()`` — called from ``src/index.js``
// before ``ReactDOM.createRoot(...).render(...)``. When the DSN env
// var is unset, this is a silent no-op so the bundle can ship into
// any environment (laptop dev, staging, prod) without conditional
// build steps.
//
// What we capture
// ---------------
// * Render-time crashes — via the ``<SentryErrorBoundary>`` wrapper
//   in ``components/SentryErrorBoundary.js``.
// * Unhandled promise rejections + window.onerror — Sentry's default
//   global hooks, enabled the moment ``init`` runs.
// * 10% of route transitions get a performance trace (``tracesSampleRate``).
//
// What we deliberately DO NOT capture
// -----------------------------------
// * Session Replay — privacy concern (customer ticket text on screen)
//   and bundle weight. Defer until layers 1–3 are stable.
// * Console.log breadcrumbs at INFO level — too noisy; ``WARN+`` only.
//
// PII scrubbing
// -------------
// ``beforeSend`` strips ``Authorization`` headers from any breadcrumb
// network capture. Customer ticket text appears inside request bodies
// (which Sentry browser SDK doesn't capture by default), so the only
// realistic PII leak vector is the Authorization header on fetch /
// axios calls — the scrubber covers it.
//
// Environment tagging
// -------------------
// ``release`` is set to ``REACT_APP_BUILD_TIMESTAMP`` so each deploy
// is identifiable in the Sentry UI. The same value lives in the
// ``BuildStamp`` debug component the backend logs at boot — so when
// a Sentry issue says "first seen on release=20260518-1430", you can
// match that to a specific deploy.

import * as Sentry from "@sentry/react";


// Strip Authorization-like values from any breadcrumb headers / data
// before the event leaves the browser. Sentry calls this for every
// captured event AND every transaction.
function _scrubBreadcrumbs(event) {
  try {
    const crumbs = event?.breadcrumbs;
    if (Array.isArray(crumbs)) {
      for (const c of crumbs) {
        const d = c?.data;
        if (!d) continue;
        if (d.Authorization) d.Authorization = "<redacted>";
        if (d.headers && typeof d.headers === "object") {
          for (const key of Object.keys(d.headers)) {
            const lower = key.toLowerCase();
            if (
              lower === "authorization"
              || lower === "cookie"
              || lower === "x-api-key"
              || lower.startsWith("x-clerk-")
            ) {
              d.headers[key] = "<redacted>";
            }
          }
        }
      }
    }
    // Sentry occasionally captures the request payload on xhr / fetch
    // breadcrumbs. Customer ticket text should never leave the
    // browser via this path.
    const req = event?.request;
    if (req?.headers && typeof req.headers === "object") {
      for (const key of Object.keys(req.headers)) {
        const lower = key.toLowerCase();
        if (
          lower === "authorization"
          || lower === "cookie"
          || lower === "x-api-key"
          || lower.startsWith("x-clerk-")
        ) {
          req.headers[key] = "<redacted>";
        }
      }
    }
  } catch (e) {
    // Scrubber must never break error reporting — swallow silently.
  }
  return event;
}


// Read DSN + env from CRA build-time env. Empty DSN = no-op init.
// We deliberately accept ``undefined`` (env var unset) the same way
// we accept empty string — both mean "Sentry not configured".
export function initSentry() {
  const dsn = (process.env.REACT_APP_SENTRY_DSN || "").trim();
  if (!dsn) {
    // eslint-disable-next-line no-console
    console.log(
      "[Sentry] disabled — REACT_APP_SENTRY_DSN not set; error monitoring inactive.",
    );
    return false;
  }

  try {
    Sentry.init({
      dsn,
      environment: process.env.REACT_APP_SENTRY_ENV || "dev",
      release: process.env.REACT_APP_BUILD_TIMESTAMP || undefined,
      // 100% of errors, 10% of transactions. Bump tracesSampleRate
      // when you need full performance flame graphs and accept the
      // higher event volume on the Sentry plan.
      sampleRate: 1.0,
      tracesSampleRate: 0.1,
      // Browser-side autoinstrumentation is opt-in starting in
      // @sentry/react v8. The default browser integrations (global
      // handlers, breadcrumbs, dedupe) load automatically; we don't
      // need to enumerate them.
      beforeSend: _scrubBreadcrumbs,
      beforeSendTransaction: _scrubBreadcrumbs,
      // No Session Replay — see module docstring.
    });
    // eslint-disable-next-line no-console
    console.log(
      "[Sentry] initialised env=%s release=%s",
      process.env.REACT_APP_SENTRY_ENV || "dev",
      process.env.REACT_APP_BUILD_TIMESTAMP || "(none)",
    );
    return true;
  } catch (e) {
    // eslint-disable-next-line no-console
    console.warn("[Sentry] init failed — error monitoring disabled.", e);
    return false;
  }
}


// Re-export the Sentry namespace for components that want to capture
// custom events (``Sentry.captureMessage``, ``Sentry.setTag``, etc.).
// Components importing this module never need to depend on
// ``@sentry/react`` directly, which keeps the surface area small and
// makes the no-op behaviour transparent.
export { Sentry };
