import React from "react";
import { Button, Card, Space, Tag, Tooltip } from "antd";
import { CheckCircleOutlined, InfoCircleOutlined } from "@ant-design/icons";

const CONFIDENCE_TONE = {
  High:   { color: "green",  text: "Best fit" },
  Medium: { color: "blue",   text: "Strong candidate" },
  Low:    { color: "orange", text: "Closest case" },
};

const SEVERITY_TONE = {
  P1: "red",
  P2: "orange",
  P3: "blue",
  P4: "default",
};

// Sprint 9.3 — replace the harsh "Unknown — verify" warning chip with a
// subtle inline (i) glyph beside the field value. Confidence/colour
// signal stays at the top of the card; per-field reassurance is now
// quiet enough not to dominate the layout.
function MutedInfoIcon() {
  return (
    <Tooltip title="Not found in historical catalog — verify before submitting.">
      <InfoCircleOutlined
        style={{
          color: "#8c8c8c",
          marginLeft: 6,
          fontSize: 12,
          cursor: "help",
        }}
      />
    </Tooltip>
  );
}

// Renders a single field value. Three modes:
//   matched → plain value, NO icon, NO tooltip (clean = trusted).
//   unknown → plain value + muted (i) icon with fixed tooltip.
//   absent  → em-dash, no icon.
function FieldValue({ status, value }) {
  if (status === "absent" || value == null || value === "") {
    return <span>—</span>;
  }
  return (
    <span>
      {value}
      {status === "unknown" && <MutedInfoIcon />}
    </span>
  );
}

/**
 * Sprint 9 — SuggestionCard
 *
 * One card representing a single ValidatedCandidate from /intake/extract.
 * The "Use this interpretation" CTA bubbles up via onUse(card).
 *
 * Sprint 9.1 — onUse forwards canonical_form values at the top of the
 * payload so the parent's prefill code receives indexed corpus terms
 * (same byte form Alert mode types directly).
 *
 * Sprint 9.3 — visual softening: replace per-field "Unknown — verify"
 * Tag with a single muted gray InfoCircleOutlined glyph + tooltip.
 * Render the field value using canonical_form (which already falls
 * back to the raw LLM string for unknown fields per Sprint 9.1). The
 * "Use this interpretation" button behaviour, confidence rendering,
 * and band thresholds are unchanged.
 */
export default function SuggestionCard({ card, onUse }) {
  if (!card) return null;
  const tone = CONFIDENCE_TONE[card.confidence] || CONFIDENCE_TONE.Low;
  const v = card.validation || {};
  const cf = card.canonical_form || {};

  // Sprint 9.3 — display values prefer canonical_form (catalog-mapped
  // when matched, raw LLM string when unknown). Keeps the form prefill
  // and the visual rendering perfectly aligned.
  const assetDisplay =
    cf.asset_name != null ? cf.asset_name : card.asset_name || "";
  const alertDisplay =
    cf.alert_type != null ? cf.alert_type : card.alert_type || "";
  const customerDisplay =
    cf.customer != null ? cf.customer : card.customer || "";

  const handleUseClick = () => {
    if (!onUse) return;
    const cfPrefill = card.canonical_form;
    // If the backend included canonical_form (Sprint 9.1+), promote
    // those values to top-level so the parent's existing prefill
    // mapping picks them up. Otherwise pass the card as-is.
    const formReady = cfPrefill
      ? {
          ...card,
          severity: (cfPrefill.severity != null ? cfPrefill.severity : card.severity) || null,
          asset_name:
            cfPrefill.asset_name != null ? cfPrefill.asset_name : (card.asset_name || ""),
          alert_type:
            cfPrefill.alert_type != null ? cfPrefill.alert_type : (card.alert_type || ""),
          customer:
            cfPrefill.customer != null ? cfPrefill.customer : (card.customer || ""),
          location:
            cfPrefill.location != null ? cfPrefill.location : (card.location || ""),
        }
      : card;
    onUse(formReady);
  };

  return (
    <Card
      bodyStyle={{ padding: 18 }}
      style={{
        backgroundColor: "var(--bg-secondary)",
        borderColor: "var(--border-color)",
        borderRadius: 12,
      }}
    >
      <div className="flex justify-between items-center mb-3">
        <Space>
          <Tag color={tone.color} icon={<CheckCircleOutlined />} style={{ fontWeight: 600 }}>
            {tone.text}
          </Tag>
          <Tag color={SEVERITY_TONE[card.severity] || "default"}>
            {card.severity || "Severity?"}
          </Tag>
        </Space>
      </div>

      <div className="mb-2 t-text text-sm">
        <span className="t-text-muted" style={{ fontSize: 11 }}>Asset</span>
        <div style={{ marginTop: 2 }}>
          <FieldValue status={v.asset_status} value={assetDisplay} />
        </div>
      </div>

      <div className="mb-2 t-text text-sm">
        <span className="t-text-muted" style={{ fontSize: 11 }}>Alert type</span>
        <div style={{ marginTop: 2 }}>
          <FieldValue status={v.alert_type_status} value={alertDisplay} />
        </div>
      </div>

      {/* Sprint 13.27 — Customer row removed from the suggestion card.
          Engineers describe the alert type they're seeing — not their
          customer's name — when they ping. So the reactive card now
          mirrors the proactive form's required-fields focus
          (severity · asset · alert_type), keeping the engineer's
          mental model consistent across the two modes. The customer
          field STILL flows through to the form prefill when the
          extractor surfaces it (it's an optional field in the
          downstream Tier1IntakeForm), so no data is lost — just not
          competing for visual real estate on the card. Reinstate by
          un-commenting the block below if customer needs to be
          surfaced on the card again. */}
      {/*
      {(customerDisplay || v.customer_status === "unknown") && (
        <div className="mb-2 t-text text-sm">
          <span className="t-text-muted" style={{ fontSize: 11 }}>Customer</span>
          <div style={{ marginTop: 2 }}>
            <FieldValue status={v.customer_status} value={customerDisplay} />
          </div>
        </div>
      )}
      */}

      {(card.location || card.users_impacted_count) && (
        <div className="mb-2 t-text-muted" style={{ fontSize: 12 }}>
          {card.location && <div>Location: {card.location}</div>}
          {card.users_impacted_count != null && (
            <div>Users impacted: {card.users_impacted_count}</div>
          )}
        </div>
      )}

      <div className="mt-3 flex justify-end">
        <Button
          type="primary"
          onClick={handleUseClick}
          style={{ backgroundColor: "var(--acadia-primary)", borderColor: "var(--acadia-primary)" }}
        >
          Use this interpretation
        </Button>
      </div>
    </Card>
  );
}
