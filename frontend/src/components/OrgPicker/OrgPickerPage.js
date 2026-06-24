/**
 * OrgPickerPage — landing page shown when the authenticated user has
 * no active organization.
 *
 * Two sections:
 *   * "Your organizations"  — clickable member tiles. Click → enter
 *     the org. We persist the choice server-side via
 *     PATCH /users/me/active-org AND (if Clerk has an org for it)
 *     call Clerk's setActive() so the JWT re-issues with the new
 *     org claim. Then we refetch the OrgContext.
 *
 *   * "Other organizations" — publicly-listable orgs the user is NOT
 *     a member of. Click → RequestAccessModal.
 *
 * Phase 0 scope: this page renders only when GET /organizations/me/active
 * returned has_active_org=false. AuthGate is the routing decision-maker.
 */

import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Empty, message, Spin, Typography } from "antd";
import { useOrganizationList } from "@clerk/clerk-react";

import {
  listOrganizations,
  setActiveOrganization,
} from "../../services/api";
import { useOrg } from "../../hooks/OrgContext";
import OrgTile from "./OrgTile";
import RequestAccessModal from "./RequestAccessModal";

const { Title, Text } = Typography;

export default function OrgPickerPage() {
  const { refetch: refetchOrgContext } = useOrg();

  // Clerk gives us a programmatic way to set the active org on the
  // Clerk session itself. We use this so the next JWT carries the
  // newly-chosen org claim — otherwise our backend would still see
  // the old (or absent) org.
  const { setActive } = useOrganizationList();

  const [yourOrgs, setYourOrgs] = useState([]);
  const [otherOrgs, setOtherOrgs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [enterBusyId, setEnterBusyId] = useState(null);

  // Request-access modal state.
  const [requestModal, setRequestModal] = useState({ open: false, org: null });

  const loadOrgs = useCallback(async () => {
    setLoading(true);
    try {
      const res = await listOrganizations();
      const payload = res?.data || {};
      setYourOrgs(payload.your_organizations || []);
      setOtherOrgs(payload.other_organizations || []);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error("[OrgPickerPage] listOrganizations failed", err);
      message.error("Could not load organizations. Refresh to retry.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadOrgs();
  }, [loadOrgs]);

  // ─────────────────────────────────────────────────────────────
  // Enter an org the user is a member of.
  // Sequence:
  //   1. Persist last_active_org_id server-side.
  //   2. If Clerk has a matching org (most common), call setActive()
  //      so Clerk re-issues a JWT with the new org claim.
  //   3. Trigger OrgContext refetch — picks up the new active org
  //      and AuthGate renders children (the app) on next paint.
  // Fail-soft: if step 2 has no matching Clerk org (e.g. our DB has
  // the org but the user accepted-then-left in Clerk), we still
  // continue. The OrgContext refetch will surface whatever Clerk
  // currently says is active.
  // ─────────────────────────────────────────────────────────────
  const handleEnter = useCallback(
    async (org) => {
      if (!org?.id) return;
      setEnterBusyId(org.id);
      try {
        await setActiveOrganization(org.id);
      } catch (err) {
        // eslint-disable-next-line no-console
        console.warn("[OrgPickerPage] setActiveOrganization failed", err);
        // Non-fatal — Clerk-side switch still has a chance.
      }
      try {
        if (setActive && org.id) {
          // Match Clerk's expected param shape.
          await setActive({ organization: org.id });
        }
      } catch (err) {
        // eslint-disable-next-line no-console
        console.warn("[OrgPickerPage] Clerk setActive failed", err);
      }
      try {
        await refetchOrgContext();
      } catch (_err) {
        // refetch is fail-soft inside OrgContext; ignore here.
      }
      setEnterBusyId(null);
    },
    [setActive, refetchOrgContext]
  );

  const handleRequestAccess = useCallback((org) => {
    setRequestModal({ open: true, org });
  }, []);

  const handleRequestModalClose = useCallback(() => {
    setRequestModal({ open: false, org: null });
  }, []);

  const hasYours = yourOrgs.length > 0;
  const hasOthers = otherOrgs.length > 0;

  const headerCopy = useMemo(() => {
    if (hasYours && hasOthers) {
      return "Pick an organization to continue, or request access to a new one.";
    }
    if (hasYours) return "Pick an organization to continue.";
    if (hasOthers) {
      return "You're not a member of any organization yet. Request access to one below.";
    }
    return "No organizations available. Contact your administrator.";
  }, [hasYours, hasOthers]);

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "flex-start",
        padding: "48px 24px",
        background: "var(--bg-primary, #ffffff)",
        fontFamily: "'Poppins', 'Inter', system-ui, sans-serif",
      }}
    >
      <img
        src="/logo.png"
        alt="Acadia LogIQ"
        style={{ height: 48, marginBottom: 24 }}
      />

      <Title level={3} style={{ marginBottom: 8, textAlign: "center" }}>
        Choose your workspace
      </Title>
      <Text type="secondary" style={{ marginBottom: 36, textAlign: "center" }}>
        {headerCopy}
      </Text>

      {loading ? (
        <Spin size="large" />
      ) : (
        <div
          style={{
            width: "100%",
            maxWidth: 1080,
            display: "flex",
            flexDirection: "column",
            gap: 40,
          }}
        >
          {hasYours && (
            <Section title="Your organizations">
              <Grid>
                {yourOrgs.map((org) => (
                  <OrgTile
                    key={org.id}
                    org={org}
                    onEnter={handleEnter}
                    busy={enterBusyId === org.id}
                  />
                ))}
              </Grid>
            </Section>
          )}

          {hasOthers && (
            <Section title="Other organizations">
              <Grid>
                {otherOrgs.map((org) => (
                  <OrgTile
                    key={org.id}
                    org={org}
                    onRequestAccess={handleRequestAccess}
                  />
                ))}
              </Grid>
            </Section>
          )}

          {!hasYours && !hasOthers && (
            <Empty description="No organizations to show." />
          )}
        </div>
      )}

      <RequestAccessModal
        open={requestModal.open}
        org={requestModal.org}
        onClose={handleRequestModalClose}
      />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Small layout helpers (kept inline — Phase 0 doesn't introduce
// new shared CSS / theme primitives).
// ─────────────────────────────────────────────────────────────
function Section({ title, children }) {
  return (
    <div>
      <Title level={5} style={{ marginBottom: 16 }}>
        {title}
      </Title>
      {children}
    </div>
  );
}

function Grid({ children }) {
  return (
    <div
      style={{
        display: "flex",
        flexWrap: "wrap",
        gap: 16,
        justifyContent: "flex-start",
      }}
    >
      {children}
    </div>
  );
}
