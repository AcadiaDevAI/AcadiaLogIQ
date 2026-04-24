import React from "react";
import { Card, Skeleton } from "antd";
import { useTier1Theme } from "../../theme/ThemeProvider";

/**
 * Sprint 8 — SkeletonCard
 *
 * Three variants:
 *   "answer"       — Tier1AnswerCard shape
 *   "diagnostics"  — DeeperDiagnosticsCard shape
 *   "escalation"   — EscalationPackageCard / ExplainRecommendationCard shape
 *
 * Pure presentational. No data dependencies. Uses antd Skeleton
 * primitives (already installed) so we inherit the shimmer animation
 * for free.
 */
export default function SkeletonCard({ variant = "answer" }) {
  const { tokens, isModern } = useTier1Theme();
  const style = {
    backgroundColor: tokens.surfaceBase,
    borderColor: isModern ? tokens.surfaceElevated : "var(--border-color)",
    borderRadius: tokens.radiusMd,
    boxShadow: isModern ? tokens.shadowSm : undefined,
  };

  const bodyStyle = { padding: 24 };

  if (variant === "diagnostics") {
    return (
      <Card bodyStyle={bodyStyle} style={style}>
        <Skeleton.Input active size="small" style={{ width: 160, marginBottom: 12 }} />
        <Skeleton active paragraph={{ rows: 3 }} />
        <div style={{ marginTop: 16, display: "flex", gap: 8 }}>
          <Skeleton.Button active size="small" />
          <Skeleton.Button active size="small" />
          <Skeleton.Button active size="small" />
        </div>
      </Card>
    );
  }

  if (variant === "escalation") {
    return (
      <Card bodyStyle={bodyStyle} style={style}>
        <Skeleton active title paragraph={{ rows: 4 }} />
        <Skeleton.Input active style={{ width: "100%", marginTop: 16, height: 140 }} />
      </Card>
    );
  }

  // default: answer variant
  return (
    <Card bodyStyle={bodyStyle} style={style}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginBottom: 16,
        }}
      >
        <Skeleton.Input active size="small" style={{ width: 160 }} />
        <Skeleton.Input active size="small" style={{ width: 120 }} />
      </div>
      <Skeleton active title paragraph={{ rows: 5 }} />
    </Card>
  );
}
