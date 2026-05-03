import React from "react";
import { Button, Card, Divider, Space, Tag, message } from "antd";
import { CopyOutlined } from "@ant-design/icons";
import useAutoScrollIntoView from "../../hooks/useAutoScrollIntoView";
import { useTier1Theme } from "../../theme/ThemeProvider";

/**
 * Sprint 7 — EscalationPackageCard
 *
 * Receives the `package` object from POST /tier1/escalation-package
 * and renders it as a ready-to-paste bundle with a single Copy button.
 */
export default function EscalationPackageCard({ pkg, onClose }) {
  const { tokens, isModern } = useTier1Theme();
  const scrollRef = useAutoScrollIntoView(!!pkg);
  if (!pkg) return null;

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(pkg.formatted_text || "");
      message.success("Escalation package copied to clipboard.");
    } catch {
      message.warning(
        "Copy failed — select the text block below and copy manually.",
      );
    }
  };

  const cardStyle = isModern
    ? {
        backgroundColor: tokens.surfaceBase,
        borderColor: tokens.surfaceElevated,
        borderRadius: tokens.radiusLg,
        boxShadow: tokens.shadowMd,
      }
    : {
        backgroundColor: "var(--bg-secondary)",
        borderColor: "var(--border-color)",
      };

  return (
    <Card
      ref={scrollRef}
      title={
        <div className="flex justify-between items-center"> 
          <span>Escalation Package</span>
          <Space>
            <Tag color="blue">{pkg.priority || "P?"}</Tag>
            {pkg.affected_customer && <Tag>{pkg.affected_customer}</Tag>}
          </Space>
        </div>
      }
      extra={<Button onClick={onClose}>Close</Button>}
      bodyStyle={{ padding: 24 }}
      style={cardStyle}
    >
      {pkg.summary && (
        <div className="mb-3 t-text text-sm">
          <span className="t-text-muted">Summary: </span>
          {pkg.summary}
        </div>
      )}

      {Array.isArray(pkg.affected_assets) && pkg.affected_assets.length > 0 && (
        <div className="mb-3 t-text text-sm">
          <span className="t-text-muted">Affected assets: </span>
          {pkg.affected_assets.join(", ")}
        </div>
      )}

      {pkg.suggested_owner_team && (
        <div className="mb-3 t-text text-sm">
          <span className="t-text-muted">Suggested owner: </span>
          <strong>{pkg.suggested_owner_team}</strong>
        </div>
      )}

      {Array.isArray(pkg.escalation_path) && pkg.escalation_path.length > 0 && (
        <div className="mb-3 t-text text-sm">
          <span className="t-text-muted">Escalation path: </span>
          {pkg.escalation_path.join(" → ")}
        </div>
      )}

      {Array.isArray(pkg.customer_contacts) &&
        pkg.customer_contacts.length > 0 && (
          <div className="mb-3">
            <div className="t-text font-semibold text-sm mb-1">
              Customer contacts
            </div>
            <ul className="list-disc list-inside t-text text-sm">
              {pkg.customer_contacts.map((c, i) => (
                <li key={i}>
                  {[c.name, c.role, c.phone, c.email]
                    .filter(Boolean)
                    .join(" | ")}
                </li>
              ))}
            </ul>
          </div>
        )}

      {Array.isArray(pkg.what_was_tried) && pkg.what_was_tried.length > 0 && (
        <div className="mb-3">
          <div className="t-text font-semibold text-sm mb-1">
            What was tried
          </div>
          <ul className="list-disc list-inside t-text text-sm">
            {pkg.what_was_tried.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </div>
      )}

      {pkg.recommended_next_action && (
        <div className="mb-3 t-text text-sm">
          <span className="t-text-muted">Recommended next action: </span>
          {pkg.recommended_next_action}
        </div>
      )}

      {Array.isArray(pkg.directory_contacts) &&
        pkg.directory_contacts.length > 0 && (
          <div
            className="mb-3"
            style={{
              backgroundColor: isModern
                ? tokens.surfaceElevated
                : "var(--bg-primary)",
              border: `1px solid ${
                isModern ? tokens.borderSubtle : "var(--border-color)"
              }`,
              borderLeft: "3px solid var(--acadia-primary)",
              borderRadius: 6,
              padding: "10px 12px",
            }}
          >
            <div className="t-text font-semibold text-sm mb-1">
              Recommended contact
              <span
                className="t-text-muted"
                style={{ fontWeight: 400, marginLeft: 6, fontSize: 11 }}
              >
                (Acadia Escalation Directory)
              </span>
            </div>
            <ul
              className="t-text text-sm"
              style={{ listStyle: "none", paddingLeft: 0, margin: 0 }}
            >
              {pkg.directory_contacts.map((d, i) => (
                <li key={i} style={{ marginBottom: 4 }}>
                  {d.label && (
                    <Tag
                      color="geekblue"
                      style={{ marginRight: 6, fontSize: 11 }}
                    >
                      {d.label}
                    </Tag>
                  )}
                  {d.name && <strong>{d.name}</strong>}
                  {d.detail && (
                    <span className="t-text-muted">
                      {d.name ? " — " : ""}
                      {d.detail}
                    </span>
                  )}
                </li>
              ))}
            </ul>
          </div>
        )}

      <Divider className="my-3" />

      <div className="flex justify-between items-center mb-2">
        <span className="t-text-muted text-xs">
          Paste-ready text (ServiceNow / Jira / email)
        </span>
        <Button
          size="small"
          icon={<CopyOutlined />}
          type="primary"
          style={{ backgroundColor: "var(--acadia-primary)", borderColor: "var(--acadia-primary)" }}
          onClick={handleCopy}
        >
          Copy to clipboard
        </Button>
      </div>
      <pre
        style={{
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
          backgroundColor: "var(--bg-primary, #0d0d12)",
          color: "var(--text-primary, #e6e6e6)",
          padding: 12,
          borderRadius: 6,
          maxHeight: 320,
          overflow: "auto",
          fontSize: 12,
        }}
      >
        {pkg.formatted_text || "(empty)"}
      </pre>
    </Card>
  );
}
