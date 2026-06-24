/**
 * OrgTile — Single tile on the OrgPickerPage.
 *
 * Renders ONE organization with two visual variants:
 *
 *   Member  — clickable card with bold name + role badge.
 *             Click invokes `onEnter(org)` (parent handles the switch).
 *
 *   Locked  — card with a lock icon + greyed name + "Request access"
 *             button. Click invokes `onRequestAccess(org)`.
 *
 * Visual style intentionally uses antd primitives to match the rest
 * of the frontend; no custom CSS files added in Phase 0.
 */

import React from "react";
import { Card, Tag, Button } from "antd";
import { LockOutlined, ArrowRightOutlined } from "@ant-design/icons";

const DEFAULT_LOGO_PLACEHOLDER = "/logo.png";

function RoleBadge({ role }) {
  if (!role) return null;
  const color = role === "admin" ? "blue" : "default";
  return (
    <Tag color={color} style={{ marginInlineStart: 8 }}>
      {role}
    </Tag>
  );
}

export default function OrgTile({ org, onEnter, onRequestAccess, busy }) {
  if (!org) return null;

  const isMember = Boolean(org.is_member);
  const logo = org.logo_url || DEFAULT_LOGO_PLACEHOLDER;

  // Card body shared by both variants.
  const body = (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 12,
        textAlign: "center",
      }}
    >
      <img
        src={logo}
        alt={org.name}
        style={{
          height: 56,
          width: "auto",
          objectFit: "contain",
          opacity: isMember ? 1 : 0.55,
        }}
        onError={(e) => {
          // Failsafe — if the org's branded logo URL 404s, fall
          // back to the platform default so the tile keeps its shape.
          if (e.currentTarget.src !== window.location.origin + DEFAULT_LOGO_PLACEHOLDER) {
            e.currentTarget.src = DEFAULT_LOGO_PLACEHOLDER;
          }
        }}
      />
      <div
        style={{
          fontWeight: 600,
          fontSize: 16,
          opacity: isMember ? 1 : 0.7,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        {isMember ? null : (
          <LockOutlined style={{ marginInlineEnd: 6, fontSize: 14 }} />
        )}
        {org.name}
        {isMember ? <RoleBadge role={org.role} /> : null}
      </div>

      {isMember ? (
        <Button
          type="primary"
          icon={<ArrowRightOutlined />}
          loading={busy}
          onClick={() => onEnter && onEnter(org)}
          style={{
            backgroundColor: "var(--acadia-primary)",
            borderColor: "var(--acadia-primary)",
          }}
        >
          Enter
        </Button>
      ) : (
        <Button
          onClick={() => onRequestAccess && onRequestAccess(org)}
          loading={busy}
        >
          Request access
        </Button>
      )}
    </div>
  );

  return (
    <Card
      hoverable={isMember}
      bodyStyle={{ padding: 24 }}
      style={{
        width: 240,
        backgroundColor: "var(--bg-secondary)",
        borderColor: "var(--border-color)",
        cursor: isMember ? "pointer" : "default",
      }}
      onClick={isMember ? () => onEnter && onEnter(org) : undefined}
    >
      {body}
    </Card>
  );
}
