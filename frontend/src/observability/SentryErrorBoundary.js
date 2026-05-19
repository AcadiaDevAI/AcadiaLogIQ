// SentryErrorBoundary — catches render-time exceptions and reports
// them to Sentry while showing a friendly fallback UI.
//
// React's built-in error boundary mechanism only catches errors
// thrown during rendering / lifecycle methods. Async errors,
// event-handler errors, and unhandled promise rejections are
// captured by Sentry's global handlers (set up in ``initSentry``)
// — those need no extra wrapping.
//
// We use the official ``Sentry.ErrorBoundary`` from @sentry/react
// which automatically calls ``Sentry.captureException(error,
// { contexts: { react: { componentStack } } })`` for us. Our wrapper
// adds:
//   * A sensible default fallback (no React-default white-screen).
//   * ``showDialog`` is OFF so we never block the user — they see
//     the fallback, our ops gets the event, the user clicks Reload.

import React from "react";
import { Sentry } from "./sentryClient";


function DefaultFallback({ error, resetError }) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        height: "100vh",
        fontFamily: "'Poppins', 'Inter', system-ui, sans-serif",
        padding: 24,
        textAlign: "center",
        background: "var(--bg-primary, #ffffff)",
        color: "var(--text-primary, #1f2937)",
      }}
    >
      <h1 style={{ fontSize: 22, fontWeight: 700, marginBottom: 8 }}>
        Something went wrong.
      </h1>
      <p style={{ fontSize: 14, marginBottom: 16, maxWidth: 480 }}>
        We&apos;ve logged the problem. Try reloading the page; if it keeps
        happening, please report the issue to support with the time it
        occurred.
      </p>
      <button
        onClick={resetError}
        style={{
          padding: "8px 16px",
          borderRadius: 6,
          border: "1px solid var(--acadia-primary, #4f46e5)",
          background: "var(--acadia-primary, #4f46e5)",
          color: "#ffffff",
          cursor: "pointer",
        }}
      >
        Try again
      </button>
    </div>
  );
}


export default function SentryErrorBoundary({ children }) {
  // When Sentry isn't configured (DSN unset) ``Sentry.ErrorBoundary``
  // still works — it just doesn't ship the event anywhere. So we can
  // mount this unconditionally without a feature flag, and the
  // fallback UI still kicks in to protect the user from a white
  // screen.
  return (
    <Sentry.ErrorBoundary fallback={DefaultFallback}>
      {children}
    </Sentry.ErrorBoundary>
  );
}
