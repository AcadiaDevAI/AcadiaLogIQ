import React from "react";
import { SignedIn, SignedOut, SignIn } from "@clerk/clerk-react";
import useAuthInterceptor from "../hooks/useAuthInterceptor";
import useAutoRegister from "../hooks/useAutoRegister";
import { OrgContextProvider, useOrg } from "../hooks/OrgContext";
import OrgPickerPage from "./OrgPicker/OrgPickerPage";

const CLERK_KEY = process.env.REACT_APP_CLERK_PUBLISHABLE_KEY;

/**
 * AuthGate — orchestrates auth in strict order:
 *
 *   1. useAuthInterceptor() — sets token + verifies it works → isReady
 *   2. useAutoRegister(isReady) — waits for isReady, then registers user
 *   3. {isReady ? children : spinner} — blocks app until auth confirmed
 *
 * Phase 0 multi-tenant addition (failure-soft):
 *
 *   Once auth is ready, we wrap children with OrgContextProvider so the
 *   active organization is resolved via GET /organizations/me/active. If
 *   the backend POSITIVELY reports no active org, we render
 *   OrgPickerPage instead of children. On any failure (network error,
 *   malformed response), we fall through to children so a tenancy bug
 *   never locks users out of the existing app.
 */
export default function AuthGate({ children }) {
  const { isReady } = useAuthInterceptor();
  useAutoRegister(isReady);

  if (!CLERK_KEY) {
    return <>{children}</>;
  }

  return (
    <>
      <SignedOut>
        <div
          style={{
            minHeight: "100vh",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "#ffffff",
            fontFamily: "'Poppins', 'Inter', system-ui, sans-serif",
          }}
        >
          <SignIn routing="hash" />
        </div>
      </SignedOut>

      <SignedIn>
        {isReady ? (
          <OrgContextProvider>
            <OrgGate>{children}</OrgGate>
          </OrgContextProvider>
        ) : (
          <LoadingSpinner label="Loading..." />
        )}
      </SignedIn>
    </>
  );
}

/**
 * OrgGate — inner component that lives INSIDE OrgContextProvider so it
 * can consume useOrg(). Three states:
 *
 *   1. Not yet resolved   → spinner (matches the auth spinner so the
 *                            transition is invisible if /me/active is fast).
 *   2. Resolved, no org   → render OrgPickerPage. This only happens on a
 *                            CLEAN { has_active_org: false } response.
 *   3. Resolved, has org  → render children (the app).
 *
 * Failure-soft: a fetch error inside OrgContext still flips
 * hasResolvedActiveOrg=true with activeOrg=null. To avoid presenting the
 * picker on a transient error, we render children when the context
 * reports an error — the existing app is the safe fallback.
 */
function OrgGate({ children }) {
  const { activeOrg, hasResolvedActiveOrg, error } = useOrg();

  if (!hasResolvedActiveOrg) {
    return <LoadingSpinner label="Loading..." />;
  }
  if (error) {
    // Tenancy fetch broke — never block the existing app.
    return <>{children}</>;
  }
  if (!activeOrg) {
    return <OrgPickerPage />;
  }
  return <>{children}</>;
}

function LoadingSpinner({ label }) {
  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontFamily: "'Poppins', 'Inter', system-ui, sans-serif",
      }}
    >
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "12px" }}>
        <div
          style={{
            width: "32px",
            height: "32px",
            border: "3px solid #e5e7eb",
            borderTopColor: "var(--acadia-primary)",
            borderRadius: "50%",
            animation: "spin 0.8s linear infinite",
          }}
        />
        <span style={{ fontSize: "14px", color: "#6b7280" }}>{label}</span>
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      </div>
    </div>
  );
}
