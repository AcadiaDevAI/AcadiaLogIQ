// Sprint 10 Stage 1A — Smoking Gun.
//
// Per spec §3.2 + §9: pivot signal + bypass instruction +
// recommended action. Frequency badge shows cohort percentage.
// [Helpful] [Historical Matches & Possible Causes ▶]
//
// Empty case (data.empty=true): renders the spec's spare copy.

import React from "react";
import { Card, Tag, Typography, Space } from "antd";
import { ThunderboltOutlined } from "@ant-design/icons";

import HelpfulButton from "./HelpfulButton";
import NextStageButton from "./NextStageButton";
import { STAGE_LABELS } from "../tier1Constants";

const { Title, Paragraph, Text } = Typography;


// Sprint 10.1 — title + caption vary by data.derived_from. When the
// cohort had populated Knowledge_Base.the_mental_pivot fields, render
// the standard "Smoking Gun" title. When we fell back to Primary_Fix
// distillation, re-title to "Best Historical Fix" with a small caption
// telling the engineer this isn't a pivot signal.
const TITLE_BY_DERIVATION = {
  mental_pivot_aggregate: "Smoking Gun",
  primary_fix_fallback:   "Best Historical Fix",
  empty:                  "Smoking Gun",
};

const CAPTION_BY_DERIVATION = {
  mental_pivot_aggregate: null,
  primary_fix_fallback:
    "(distilled from highest-rated past resolution — pivot signal data not available for this cohort)",
  empty:                  null,
};


export default function Stage1aSmokingGun({
  data,
  sessionId,
  onMarkedHelpful,
  onStartNewTicket,
  onReveal,           // (toStage) → fetches + flips reveal[stage_2]=true
  helpfulMarked,
}) {
  if (!data) return null;

  // Sprint 10.1 — title/caption vary by data.derived_from. When the
  // backend fell back to Primary_Fix distillation, the panel re-titles
  // to "Best Historical Fix" and shows a small italic caption.
  const derivation = data.derived_from || (data.empty ? "empty" : "mental_pivot_aggregate");
  const title = TITLE_BY_DERIVATION[derivation] || "Smoking Gun";
  const caption = CAPTION_BY_DERIVATION[derivation];

  if (data.empty) {
    return (
      <Card style={{ marginBottom: 16, borderLeft: "4px solid #6B6B6B" }}>
        <Title level={5} style={{ marginTop: 0 }}>
          <ThunderboltOutlined style={{ marginRight: 8 }} />
          {title}
        </Title>
        <Paragraph type="secondary">
          No single pivot signal dominates this cohort. See historical
          matches below for case-by-case patterns.
        </Paragraph>
        <Footer
          sessionId={sessionId}
          onMarkedHelpful={onMarkedHelpful}
          onStartNewTicket={onStartNewTicket}
          onReveal={onReveal}
          helpfulMarked={helpfulMarked}
        />
      </Card>
    );
  }

  return (
    <Card style={{ marginBottom: 16, borderLeft: "4px solid #C9870B" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 8 }}>
        <div>
          <Title level={5} style={{ marginTop: 0, marginBottom: caption ? 2 : undefined }}>
            <ThunderboltOutlined style={{ marginRight: 8 }} />
            {title}
          </Title>
          {caption ? (
            <Text italic type="secondary" style={{ fontSize: 12 }}>{caption}</Text>
          ) : null}
        </div>
        {derivation === "mental_pivot_aggregate" && data.frequency_in_cohort_percent ? (
          <Tag color="orange">{data.frequency_in_cohort_percent}% of cohort</Tag>
        ) : null}
      </div>

      <Paragraph style={{ marginBottom: 8 }}>
        <Text strong>Pivot signal: </Text>
        <Text>{data.pivot_signal}</Text>
      </Paragraph>

      {data.bypass_instruction ? (
        <Paragraph style={{ marginBottom: 8 }}>
          <Text strong>Bypass: </Text>
          <Text>{data.bypass_instruction}</Text>
        </Paragraph>
      ) : null}

      {data.recommended_action ? (
        <Paragraph style={{ marginBottom: 8 }}>
          <Text strong>Recommended action: </Text>
          <Text code>{data.recommended_action}</Text>
        </Paragraph>
      ) : null}

      {data.seen_in_incidents && data.seen_in_incidents.length > 0 ? (
        <Paragraph style={{ marginBottom: 12 }}>
          <Text type="secondary" style={{ fontSize: 12 }}>
            Seen in:{" "}
            {data.seen_in_incidents.map((inc) => (
              <Tag key={inc} style={{ marginRight: 4, marginBottom: 4 }}>{inc}</Tag>
            ))}
          </Text>
        </Paragraph>
      ) : null}

      <Footer
        sessionId={sessionId}
        onMarkedHelpful={onMarkedHelpful}
        onStartNewTicket={onStartNewTicket}
        onReveal={onReveal}
        helpfulMarked={helpfulMarked}
      />
    </Card>
  );
}


function Footer({ sessionId, onMarkedHelpful, onStartNewTicket, onReveal, helpfulMarked }) {
  return (
    <div
      style={{
        marginTop: 16,
        paddingTop: 12,
        borderTop: "1px solid var(--border-color, #f0f0f0)",
        display: "flex",
        flexWrap: "wrap",
        gap: 12,
        justifyContent: "space-between",
        alignItems: "center",
      }}
    >
      <HelpfulButton
        sessionId={sessionId}
        stage="stage_1a"
        onMarkedHelpful={onMarkedHelpful}
        onStartNewTicket={onStartNewTicket}
        disabled={helpfulMarked}
      />
      <NextStageButton
        sessionId={sessionId}
        fromStage="stage_1a"
        toStage="stage_2"
        label={`${STAGE_LABELS.stage_2}`}
        onReveal={onReveal}
      />
    </div>
  );
}
