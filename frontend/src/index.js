import React from "react";
import ReactDOM from "react-dom/client";
import "./index.css";
import App from "./App";
import { ClerkProvider } from "@clerk/clerk-react";
import { initSentry } from "./observability/sentryClient";
import SentryErrorBoundary from "./observability/SentryErrorBoundary";

// Sentry MUST run before React renders so global handlers
// (window.onerror, unhandledrejection) are installed before any user
// code can throw. DSN-empty → no-op, no network, no overhead.
initSentry();

// Auto-recover from stale-chunk failures. Lazy routes (React.lazy) live in
// separate webpack chunks. After a redeploy — or a CRA dev-server recompile —
// a tab holding an OLD build requests a chunk hash that no longer exists,
// producing a "Loading chunk … failed" ChunkLoadError. We reload ONCE to pull
// the fresh build. A short-lived sessionStorage guard stops a reload loop when
// the chunk is genuinely broken (the retry fails within seconds of the reload).
function isChunkLoadError(err, fallbackMessage) {
  const name = err && err.name ? err.name : "";
  const msg = (err && err.message) || fallbackMessage || "";
  return (
    name === "ChunkLoadError" ||
    /ChunkLoadError/i.test(msg) ||
    /Loading (?:CSS )?chunk [\w-]+ failed/i.test(msg)
  );
}

function recoverFromChunkError() {
  const KEY = "chunkReloadAt";
  const last = Number(sessionStorage.getItem(KEY) || 0);
  // If we reloaded within the last 10s and STILL hit a chunk error, the fresh
  // build failed too — stop reloading and let the error surface (to Sentry /
  // the error boundary) instead of looping.
  if (Date.now() - last < 10000) return;
  try {
    sessionStorage.setItem(KEY, String(Date.now()));
  } catch (_) {
    // sessionStorage can throw in private-mode / sandboxed frames — reload
    // anyway; worst case is no loop guard.
  }
  window.location.reload();
}

window.addEventListener("error", (event) => {
  if (isChunkLoadError(event && event.error, event && event.message)) {
    recoverFromChunkError();
  }
});
window.addEventListener("unhandledrejection", (event) => {
  const reason = event && event.reason;
  if (isChunkLoadError(reason, reason && reason.message)) {
    recoverFromChunkError();
  }
});

const CLERK_KEY = process.env.REACT_APP_CLERK_PUBLISHABLE_KEY;

// Pin clerk-js to an immutable version so the main script AND its lazy-loaded
// chunks are always served from the same URL. Without this, clerk-react loads
// clerk.browser.js from the floating ".../@clerk/clerk-js@5/..." tag, which the
// browser/CDN caches aggressively. When Clerk repaves the "@5" tag to a new
// patch, the cached main script references chunk hashes the CDN no longer
// serves → intermittent "Loading chunk 344 failed / ChunkLoadError" on mount.
// Keep this in sync with the clerk-js version @clerk/clerk-react ships against;
// override via env if a specific pin is needed per environment.
const CLERK_JS_VERSION =
  process.env.REACT_APP_CLERK_JS_VERSION || "5.127.1";

const root = ReactDOM.createRoot(document.getElementById("root"));

if (CLERK_KEY) {
  console.log("[Clerk] Auth enabled — publishable key found");
} else {
  console.log("[Clerk] Auth disabled — no REACT_APP_CLERK_PUBLISHABLE_KEY set");
}

// SentryErrorBoundary is the outermost wrapper so a render crash
// anywhere in the tree shows our friendly fallback instead of a
// React-default white screen. ClerkProvider is nested inside so a
// Clerk init error is still caught by the boundary.
root.render(
  <React.StrictMode>
    <SentryErrorBoundary>
      {CLERK_KEY ? (
        <ClerkProvider
          publishableKey={CLERK_KEY}
          clerkJSVersion={CLERK_JS_VERSION}
          afterSignOutUrl="/"
        >
          <App />
        </ClerkProvider>
      ) : (
        <App />
      )}
    </SentryErrorBoundary>
  </React.StrictMode>
);
