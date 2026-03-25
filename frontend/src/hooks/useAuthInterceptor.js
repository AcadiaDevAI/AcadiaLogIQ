import { useEffect, useState, useRef, useCallback } from "react";
import { useAuth } from "@clerk/clerk-react";
import { setTokenGetter } from "../services/api";

const CLERK_KEY = process.env.REACT_APP_CLERK_PUBLISHABLE_KEY;

/**
 * useAuthInterceptor
 *
 * Registers Clerk's getToken on axios so every request includes
 * the Authorization header.
 *
 * Design:
 *   1. getToken is stored in a REF (never stale — always the latest)
 *   2. stableTokenGetter has a fixed identity ([] deps) but reads the ref
 *   3. setTokenGetter() opens the gate in api.js — queued requests proceed
 *   4. isReady flips only after a token is successfully fetched
 */
export default function useAuthInterceptor() {
  if (!CLERK_KEY) {
    return { isLoaded: true, isSignedIn: false, userId: null, isReady: true };
  }

  return useAuthInterceptorInner();
}

function useAuthInterceptorInner() {
  const { getToken, isLoaded, isSignedIn, userId } = useAuth();
  const [isReady, setIsReady] = useState(false);
  const setupDone = useRef(false);
  const prevUserId = useRef(null);

  // ── Always keep the latest getToken in a ref ────────────
  const getTokenRef = useRef(getToken);
  getTokenRef.current = getToken;

  // ── Stable token fetcher (identity never changes) ───────
  // api.js holds a reference to THIS function forever.
  // Because it reads from the ref, it always calls the
  // current Clerk getToken — even after session refreshes.
  const stableTokenGetter = useCallback(() => {
    const fn = getTokenRef.current;
    if (!fn) return Promise.resolve(null);
    return fn({ skipCache: true });
  }, []);

  useEffect(() => {
    if (!isLoaded) return;

    // Handle user switch (sign out → sign in as different user)
    if (prevUserId.current !== null && prevUserId.current !== userId) {
      setupDone.current = false;
      setIsReady(false);
    }
    prevUserId.current = userId;

    if (isSignedIn && getToken && !setupDone.current) {
      setupDone.current = true;

      // Register the token getter on axios.
      // This OPENS the auth gate in api.js — all queued requests proceed.
      setTokenGetter(stableTokenGetter);

      // Verify we can actually get a token before declaring ready
      stableTokenGetter()
        .then((token) => {
          if (token) {
            console.log("[AuthInterceptor] Token verified, auth ready. User:", userId);
          } else {
            console.warn("[AuthInterceptor] getToken returned null");
          }
          setIsReady(true);
        })
        .catch((err) => {
          console.error("[AuthInterceptor] Initial token fetch failed:", err);
          // Still set ready — the 401 retry interceptor handles failures
          setIsReady(true);
        });

    } else if (isLoaded && !isSignedIn) {
      // Signed out — close the gate
      setTokenGetter(null);
      setupDone.current = false;
      setIsReady(true);
    }
  }, [isLoaded, isSignedIn, getToken, userId, stableTokenGetter]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      setTokenGetter(null);
      setupDone.current = false;
    };
  }, []);

  return { isLoaded, isSignedIn, userId, isReady };
}