// Sprint 12.4 — Environment Context & Tech Component Profile.
//
// Lead-in panel rendered ABOVE Stage 0. Aggregates the cohort's
// technology landscape into a single deduplicated profile so the
// engineer sees the full blast radius before drilling into any
// single ticket. Source data comes from the backend
// `EnvironmentProfile` schema (built by stage0_environment.py),
// which already deduped (case-insensitive) and ordered the values
// by first-seen across the retrieval-similarity-ranked cohort.
//
// Footer matches the journey's standard pattern:
//   HelpfulButton  +  EscalateButton  +  NextStageButton (→ Stage 0)
//
// Empty state (data.empty=true): a small spare card that admits the
// cohort had no environment data populated. Better than hiding the
// panel — engineer needs to know the gap exists so they can flag
// the upstream curation team.

import React from "react";
import { Card, Tag, Typography, Space } from "antd";
import { ApartmentOutlined } from "@ant-design/icons";

import HelpfulButton from "./HelpfulButton";
import NextStageButton from "./NextStageButton";
import EscalateButton from "./EscalateButton";
import { STAGE_LABELS } from "../tier1Constants";

const { Title, Paragraph, Text } = Typography;


function TagSection({ label, items, color }) {
  if (!items || items.length === 0) return null;
  return (
    <Paragraph style={{ marginBottom: 8 }}>
      <Text strong>{label}: </Text>
      <Space size={[4, 4]} wrap>
        {items.map((s) => (
          <Tag key={s} color={color} style={{ marginRight: 0 }}>
            {s}
          </Tag>
        ))}
      </Space>
    </Paragraph>
  );
}


export default function EnvironmentContextPanel({
  data,
  sessionId,
  onMarkedHelpful,
  onStartNewTicket,
  onReveal,
  helpfulMarked,
}) {
  if (!data) return null;

  // Empty case — spare card so the gap is visible (don't silently
  // hide). Same stance as Stage 1A/1B's empty branches.
  if (data.empty) {
    return (
      <Card style={{ marginBottom: 16, borderLeft: "4px solid #6B6B6B" }}>
        <Title level={5} style={{ marginTop: 0 }}>
          <ApartmentOutlined style={{ marginRight: 8 }} />
          {STAGE_LABELS.environment_context}
        </Title>
        <Paragraph type="secondary" style={{ marginBottom: 0 }}>
          No environment data available for this cohort. Skipping
          straight to the historical match below.
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
    <Card style={{ marginBottom: 16, borderLeft: "4px solid #1F6FEB" }}>
      <Title level={5} style={{ marginTop: 0, marginBottom: 12 }}>
        <ApartmentOutlined style={{ marginRight: 8 }} />
        {STAGE_LABELS.environment_context}
      </Title>

      <TagSection
        label="Domains"
        items={data.domain_types}
        color="blue"
      />
      <TagSection
        label="Component categories"
        items={data.component_categories}
        color="geekblue"
      />
      <TagSection
        label="Synaptic clusters"
        items={data.synaptic_cluster_ids}
        color="purple"
      />
      <TagSection
        label="Products & vendors"
        items={data.products_involved}
        color="cyan"
      />
      <TagSection
        label="Technical entities"
        items={data.technical_entities}
        color="default"
      />

      {data.cohort_size ? (
        <Paragraph style={{ marginBottom: 0, marginTop: 8 }}>
          <Text type="secondary" style={{ fontSize: 12 }}>
            Aggregated from {data.tickets_with_data} of{" "}
            {data.cohort_size} cohort ticket
            {data.cohort_size === 1 ? "" : "s"}.
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


function Footer({
  sessionId,
  onMarkedHelpful,
  onStartNewTicket,
  onReveal,
  helpfulMarked,
}) {
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
        stage="environment_context"
        onMarkedHelpful={onMarkedHelpful}
        onStartNewTicket={onStartNewTicket}
        disabled={helpfulMarked}
      />
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <EscalateButton
          sessionId={sessionId}
          fromStage="environment_context"
          onReveal={onReveal}
        />
        {/* Sprint 12.4.1 — "Best Historical Match & Recommended
            Resolution" NextStageButton suppressed: Stage 0 already
            renders directly below this panel (auto-revealed in
            /initial), so a CTA pointing at it would be a no-op for
            the engineer. Escalate-to-Tier-2 stays as the only
            forward action on this lead-in panel. Reinstate by
            un-commenting if Stage 0 is ever moved behind a reveal
            gate. */}
        {/*
        <NextStageButton
          sessionId={sessionId}
          fromStage="environment_context"
          toStage="stage_0"
          label={STAGE_LABELS.stage_0}
          onReveal={onReveal}
        />
        */}
      </div>
    </div>
  );
}
