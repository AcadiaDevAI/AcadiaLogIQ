/**
 * OrgContext — Frontend Phase 0 multi-tenant active-org context.
 *
 * Owns the application-wide knowledge of "which organization is the
 * current request operating against." Built on top of:
 *
 *   * The backend tenancy layer (see backend/tenancy/README.md). The
 *     JWT carries Clerk's org claims; backend resolves them to our
 *     internal organization UUID and surfaces the result via
 *     GET /organizations/me/active.
 *
 *   * Clerk's `useOrganization` hook. When the user picks a different
 *     org via Clerk's switcher widget (Phase 0 doesn't ship one, but
 *     Clerk-hosted account UI can also switch orgs), Clerk re-issues
 *     a fresh JWT and updates its session state. We refetch on those
 *     events so the active-org view in our UI stays in sync.
 *
 * Public surface:
 *
 *   <OrgContextProvider>
 *     {children}
 *   </OrgContextProvider>
 *
 *   const { activeOrg, isLoading, error, refetch } = useOrg();
 *
 * Failure-soft contract:
 *
 *   If the /organizations/me/active call fails or returns malformed
 *   data, we deliberately do NOT block the user from reaching the app.
 *   The convention is "treat unresolved as has-active-org=true," so
 *   the existing single-tenant flow continues to render. This keeps a
 *   tenancy bug from locking everyone out of the product on day 1.
 *
 *   The picker only renders when we POSITIVELY KNOW there's no
 *   active org (clear `has_active_org: false` response from a
 *   successful fetch).
 */

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useOrganization, useUser } from "@clerk/clerk-react";

import { getActiveOrganization } from "../services/api";

// ─────────────────────────────────────────────────────────────────
// Internal context shape
// ─────────────────────────────────────────────────────────────────
const OrgContext = createContext({
  activeOrg: null,           // null until the fetch completes
  hasResolvedActiveOrg: false, // true once /me/active returned (success or
                              //   failure); used by AuthGate to decide
                              //   whether to render the picker
  isLoading: true,
  error: null,
  refetch: async () => {},
  platformRole: "user",
});

// ─────────────────────────────────────────────────────────────────
// Provider
// ─────────────────────────────────────────────────────────────────
export function OrgContextProvider({ children }) {
  // We watch Clerk's hook so an external org-switch (e.g. via Clerk's
  // hosted account menu) triggers a refetch — otherwise our UI would
  // hold a stale active-org until next reload.
  const { organization: clerkOrg } = useOrganization();
  const { isSignedIn } = useUser();

  const [activeOrg, setActiveOrg] = useState(null);
  const [platformRole, setPlatformRole] = useState("user");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [hasResolvedActiveOrg, setHasResolvedActiveOrg] = useState(false);

  // Guard against state updates after unmount. Note: we set the ref
  // to true on every mount (not just at useRef init) because React 18
  // StrictMode dev-mode synthetically unmounts+remounts components,
  // and the cleanup from that first synthetic unmount would otherwise
  // leave the ref stuck at false — which would silently swallow every
  // setState in our async refetch and pin the UI on the loading spinner.
  const isMountedRef = useRef(true);
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  // ─────────────────────────────────────────────────────────────
  // Fetch active org from our backend.
  // We treat ALL non-success outcomes as "leave existing app
  // rendering" to avoid locking users out on a tenancy outage.
  // ─────────────────────────────────────────────────────────────
  const refetch = useCallback(async () => {
    if (!isSignedIn) {
      // Not signed in yet — nothing to fetch. Don't touch state.
      return;
    }
    setIsLoading(true);
    setError(null);
    try {
      const res = await getActiveOrganization();
      const payload = res?.data || {};
      if (!isMountedRef.current) return;
      if (payload.has_active_org && payload.organization) {
        setActiveOrg(payload.organization);
      } else {
        setActiveOrg(null);
      }
      setPlatformRole(payload.platform_role || "user");
      setHasResolvedActiveOrg(true);
    } catch (err) {
      if (!isMountedRef.current) return;
      // eslint-disable-next-line no-console
      console.warn("[OrgContext] /me/active fetch failed", err);
      setError(err);
      // Failure-soft: keep activeOrg null, but flip hasResolved so the
      // AuthGate falls through to the existing children rather than
      // hanging on a spinner.
      setHasResolvedActiveOrg(true);
    } finally {
      if (isMountedRef.current) setIsLoading(false);
    }
  }, [isSignedIn]);

  // Initial fetch when user signs in. Also re-fires when Clerk's
  // active org id changes (e.g. an org switch outside our UI).
  // `refetch` is memoized on [isSignedIn], so including it in deps
  // doesn't cause extra fires — it's identity-stable until isSignedIn
  // flips. Including it keeps the linter happy without a disable.
  useEffect(() => {
    refetch();
  }, [refetch, clerkOrg?.id]);

  const value = useMemo(
    () => ({
      activeOrg,
      hasResolvedActiveOrg,
      isLoading,
      error,
      refetch,
      platformRole,
    }),
    [activeOrg, hasResolvedActiveOrg, isLoading, error, refetch, platformRole]
  );

  return <OrgContext.Provider value={value}>{children}</OrgContext.Provider>;
}

// ─────────────────────────────────────────────────────────────────
// Consumer hook
// ─────────────────────────────────────────────────────────────────
export function useOrg() {
  return useContext(OrgContext);
}

export default OrgContext;
