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

const CLERK_KEY = process.env.REACT_APP_CLERK_PUBLISHABLE_KEY;

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
        <ClerkProvider publishableKey={CLERK_KEY} afterSignOutUrl="/">
          <App />
        </ClerkProvider>
      ) : (
        <App />
      )}
    </SentryErrorBoundary>
  </React.StrictMode>
);
