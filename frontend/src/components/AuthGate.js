import React from "react";
import { SignedIn, SignedOut, SignIn } from "@clerk/clerk-react";
import useAuthInterceptor from "../hooks/useAuthInterceptor";
import useAutoRegister from "../hooks/useAutoRegister";

const CLERK_KEY = process.env.REACT_APP_CLERK_PUBLISHABLE_KEY;

/**
 * AuthGate — orchestrates auth in strict order:
 *
 *   1. useAuthInterceptor() — sets token + verifies it works → isReady
 *   2. useAutoRegister(isReady) — waits for isReady, then registers user
 *   3. {isReady ? children : spinner} — blocks app until auth confirmed
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
          children
        ) : (
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
              <span style={{ fontSize: "14px", color: "#6b7280" }}>Loading...</span>
              <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
            </div>
          </div>
        )}
      </SignedIn>
    </>
  );
}
