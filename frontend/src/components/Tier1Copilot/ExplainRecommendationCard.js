import React from "react";
import { Button, Card, Progress, Tag } from "antd";
import { CheckCircleFilled, CloseCircleFilled, MinusCircleFilled } from "@ant-design/icons";
import useAutoScrollIntoView from "../../hooks/useAutoScrollIntoView";
import { useTier1Theme } from "../../theme/ThemeProvider";

/**
 * Sprint 7 — ExplainRecommendationCard
 *
 * Shows WHY a ticket ranked on top: field-match checklist + weighted
 * score breakdown + historical success rate. No LLM content.
 */
export default function ExplainRecommendationCard({ explain, onClose }) {
  const { tokens, isModern } = useTier1Theme();
  const scrollRef = useAutoScrollIntoView(!!explain);
  if (!explain) return null;
  const b = explain.score_breakdown || {};
  const fields = Array.isArray(explain.matched_fields)
    ? explain.matched_fields
    : [];
  const hist = explain.historical_success || {};

  const bars = [
    ["Alert type", b.alert_type_match],
    ["Asset", b.asset_match],
    ["Fingerprint", b.fingerprint_match],
    ["Technology", b.technology_match],
    ["Vector similarity", b.vector_similarity],
    ["Resolution quality", b.resolution_quality],
    ["Recency", b.recency],
    ["Success frequency", b.success_frequency],
    ["Same customer", b.same_customer_boost],
    ["Same asset family", b.same_asset_family_boost],
  ];

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
          <span>Why this recommendation?</span>
          {explain.matched_incident && (
            <Tag color="blue">{explain.matched_incident}</Tag>
          )}
        </div>
      }
      extra={<Button onClick={onClose}>Close</Button>}
      bodyStyle={{ padding: 24 }}
      style={cardStyle}
    >
      {fields.length > 0 && (
        <div className="mb-4">
          <div className="t-text font-semibold text-sm mb-2">
            Fields that matched
          </div>
          <ul className="t-text text-sm space-y-1">
            {fields.map((f, i) => (
              <li key={i} className="flex items-center gap-2">
                <MatchIcon status={f.match} />
                <span>
                  <strong>{f.field}:</strong> {f.your_value || "—"}{" "}
                  <span className="t-text-muted">vs ticket</span>{" "}
                  {f.ticket_value || "—"}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="mb-4">
        <div className="t-text font-semibold text-sm mb-2">
          Score breakdown
        </div>
        {bars.map(([label, v]) => (
          <div key={label} className="mb-2">
            <div className="flex justify-between t-text text-xs">
              <span>{label}</span>
              <span>{fmt(v)}</span>
            </div>
            <Progress
              percent={Math.round(Math.max(0, Math.min(1, v || 0)) * 100)}
              showInfo={false}
              size="small"
            />
          </div>
        ))}
        <div className="flex justify-between items-center mt-3">
          <span className="t-text font-semibold">Final score</span>
          <Tag color={b.final_score >= 0.85 ? "green" : b.final_score >= 0.6 ? "orange" : "red"}>
            {fmt(b.final_score)}
          </Tag>
        </div>
      </div>

      {hist.total_similar > 0 && (
        <div className="mt-3 t-text text-sm">
          <strong>{hist.succeeded_count}</strong> of{" "}
          <strong>{hist.total_similar}</strong> similar cases resolved by{" "}
          <em>{hist.primary_fix || "this fix"}</em> (
          {hist.success_rate_percent}%).
        </div>
      )}
    </Card>
  );
}

function MatchIcon({ status }) {
  if (status === true) {
    return <CheckCircleFilled style={{ color: "#0A7A3F" }} />;
  }
  if (status === "partial") {
    return <MinusCircleFilled style={{ color: "#C9870B" }} />;
  }
  return <CloseCircleFilled style={{ color: "#B03A2E" }} />;
}

function fmt(v) {
  if (v === null || v === undefined) return "0.00";
  return (Math.round((v || 0) * 100) / 100).toFixed(2);
}
