// Sprint 10 Stage 1B — Do Not Chase.
//
// Per spec §3.3 + §9: up to 8 entries (misleading_signal + rule_out_logic).
// Same next-stage destination as 1A — both lead to Stage 2.
//
// Empty case copy is the spec's exact string:
//   "No recurring red herrings identified — proceed with normal
//    differential diagnosis."

import React from "react";
import { Card, List, Tag, Typography } from "antd";
import { CloseCircleOutlined } from "@ant-design/icons";

import HelpfulButton from "./HelpfulButton";
import NextStageButton from "./NextStageButton";
import { STAGE_LABELS } from "../tier1Constants";

const { Title, Paragraph, Text } = Typography;


// Sprint 10.1 — branch the empty-state copy on data.reason so the
// engineer sees an accurate message:
//   no_data         → cohort lacked anti-waste annotations entirely
//   no_recurring    → entries existed but every ticket was unique
//   below_threshold → some signals repeated but didn't clear min_count
const EMPTY_COPY_BY_REASON = {
  no_data:
    "Source data lacks anti-waste annotations for this cohort.",
  no_recurring:
    "No recurring red herrings — each past ticket explored a distinct false path.",
  below_threshold:
    "Possible red herrings detected, but not enough recurrence to flag confidently.",
};

const FALLBACK_EMPTY_COPY =
  "No recurring red herrings identified — proceed with normal differential diagnosis.";


export default function Stage1bDoNotChase({
  data,
  sessionId,
  onMarkedHelpful,
  onStartNewTicket,
  onReveal,
  helpfulMarked,
}) {
  if (!data) return null;

  const isEmpty = data.empty || !Array.isArray(data.entries) || data.entries.length === 0;

  // Sprint 10.1 — pick the empty-state copy by reason. When the
  // reason key isn't recognised (older backend, unknown taxonomy
  // value) fall through to the generic copy to stay safe.
  const emptyCopy = isEmpty
    ? (EMPTY_COPY_BY_REASON[data.reason] || FALLBACK_EMPTY_COPY)
    : null;

  return (
    <Card style={{ marginBottom: 16, borderLeft: "4px solid #B03A2E" }}>
      <Title level={5} style={{ marginTop: 0 }}>
        <CloseCircleOutlined style={{ marginRight: 8 }} />
        Stage 1B — Do Not Chase
      </Title>

      {isEmpty ? (
        <Paragraph type="secondary">{emptyCopy}</Paragraph>
      ) : (
        <List
          itemLayout="vertical"
          size="small"
          dataSource={data.entries}
          renderItem={(entry, idx) => (
            <List.Item key={`${entry.misleading_signal}-${idx}`} style={{ paddingBottom: 8 }}>
              <div>
                <Text strong>{entry.misleading_signal}</Text>
                <Tag style={{ marginLeft: 8 }}>seen {entry.occurrence_count}×</Tag>
              </div>
              {entry.rule_out_logic ? (
                <Paragraph style={{ marginTop: 4, marginBottom: 4 }}>
                  <Text type="secondary">Why: </Text>
                  <Text>{entry.rule_out_logic}</Text>
                </Paragraph>
              ) : null}
              {entry.seen_in_incidents && entry.seen_in_incidents.length > 0 ? (
                <Text type="secondary" style={{ fontSize: 12 }}>
                  {entry.seen_in_incidents.map((inc) => (
                    <Tag key={inc} style={{ marginRight: 4 }}>{inc}</Tag>
                  ))}
                </Text>
              ) : null}
            </List.Item>
          )}
        />
      )}

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
          stage="stage_1b"
          onMarkedHelpful={onMarkedHelpful}
          onStartNewTicket={onStartNewTicket}
          disabled={helpfulMarked}
        />
        <NextStageButton
          sessionId={sessionId}
          fromStage="stage_1b"
          toStage="stage_2"
          label={`${STAGE_LABELS.stage_2}`}
          onReveal={onReveal}
        />
      </div>
    </Card>
  );
}
