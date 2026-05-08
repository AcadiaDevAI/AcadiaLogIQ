// Sprint 10.2 Stage 0 — Best-Ticket Distillation banner.
// Sprint 10.3 — added evidence_strength branching; CorpusStatsTail
// extracted to its own file with a default export.
//
// Renders the cohort's highest-quality past resolution. The headline
// copy varies by evidence_strength so a score-3 ticket isn't labelled
// "best-rated" (which implies high confidence).
//
// Sparse fallback (data.sparse=true) renders the spec's spare copy.
// No buttons — Stage 0 is informational lead-in only.

import React from "react";
import { Button, Card, List, Tag, Tooltip, Typography } from "antd";
import { MessageOutlined } from "@ant-design/icons";

import CorpusStatsTail from "./CorpusStatsTail";
import DislikeButton from "./DislikeButton";
import EscalateButton from "./EscalateButton";
import HelpfulButton from "./HelpfulButton";
// Sprint 13.10 — Stage 0 now offers a direct shortcut to Stage 3
// (Guided Troubleshooting Workflow) alongside Escalate, so the
// engineer can skip the intermediate Pivot Insights / Stage 2
// panels when they want to go straight to the playbook.
import NextStageButton from "./NextStageButton";
import { STAGE_LABELS } from "../tier1Constants";
import { stripLeadingNumber } from "./stepText";
import useChatHandoff from "./useChatHandoff";

const { Title, Text, Paragraph } = Typography;


function formatMinutes(mins) {
  if (mins == null) return null;
  if (mins < 60) return `${mins}m`;
  const h = Math.floor(mins / 60);
  const m = mins % 60;
  return m > 0 ? `${h}h ${m}m` : `${h}h`;
}


function ringColor(cleanPct) {
  if (cleanPct >= 80) return "#0A7A3F"; // green
  if (cleanPct >= 50) return "#C9870B"; // amber
  return "#B03A2E";                     // red
}


// Sprint 12.1 — Each "How they did it" bullet ends with " - INC-XXX"
// (suffix appended by backend `_resolution_steps`). Extract that
// trailing incident id so the per-bullet "Ask in chat" handoff can
// scope the resulting chat session to that source ticket only.
//
// Returns the incident id string (e.g. "INC-PHOENIX-402") or null
// when no recognisable suffix is found — a null scope just means the
// chat opens unscoped (global Search-in-KB behaviour), which is a
// safe fall-back. Tolerant matcher: accepts a hyphenated alphanumeric
// id with at least one dash, matching the project's INC-* / TKT-*
// patterns without hard-coding a specific prefix.
function extractTrailingIncidentId(stepText) {
  if (typeof stepText !== "string") return null;
  // Pattern: " - <UPPERCASE-TOKEN-WITH-DASH>" anchored at end of string,
  // optional trailing whitespace tolerated.
  const m = stepText.match(/\s-\s([A-Z][A-Z0-9]+(?:-[A-Z0-9]+)+)\s*$/);
  return m ? m[1] : null;
}


// Sprint 13.7 — headline simplified. The previous evidence-strength
// branching appended "The best fix came from INC-XXX (rated N/5)" /
// "Closest match: INC-XXX" tails — those have been removed at the
// request of the user. The headline now reads only the cohort-count
// preamble; the per-ticket detail rendered below the headline carries
// the source incident IDs directly. `formatMinutes` and the unused
// branches are left in place via this single-line builder so the
// imports / signatures don't need a follow-up cleanup.
function buildHeadline(data) {
  const n = data.cohort_size || 0;
  return `We found ${n} similar instances for this issue${n === 1 ? "" : "s"}.`;
}


export default function Stage0BestTicketDistillation({
  data,
  sessionId,
  onReveal,
  // Sprint 13.2 — Stage 0 now carries Helpful + Dislike alongside
  // the existing Escalate. Parent passes the same onMarkedHelpful /
  // onStartNewTicket / helpfulMarked plumbing it already supplies
  // to every other stage panel.
  onMarkedHelpful,
  onStartNewTicket,
  helpfulMarked,
}) {
  // Sprint 11 — per-step "Ask in chat" links. Hook is a no-op when
  // sessionId is missing (defensive — Stage 0 should always have one).
  const { busy: handoffBusy, askInChat } = useChatHandoff(sessionId);
  if (!data) return null;

  // ── Sparse case (cohort empty) ──
  if (data.sparse) {
    return (
      <Card style={{ marginBottom: 16, borderLeft: "4px solid #6B6B6B" }}>
        <Title level={5} style={{ marginTop: 0 }}>Best Historical Match & Recommended Resolution</Title>
        {/* Sprint 13.8 — profile_match line suppressed in the sparse
            case too, for parity with the main panel above. */}
        {/*
        {data.profile_match ? (
          <Paragraph style={{ marginBottom: 8 }}>
            <Text type="secondary">{data.profile_match}</Text>
          </Paragraph>
        ) : null}
        */}
        <Paragraph style={{ marginBottom: 0 }}>
          {/* Sprint 13.7 — softer no-data copy. */}
          No relevant historical tickets were found for this issue.
          Please review the recommended actions below.
        </Paragraph>
      </Card>
    );
  }

  const color = ringColor(data.clean_resolution_percent || 0);

  return (
    <Card style={{ marginBottom: 16, borderLeft: `4px solid ${color}` }}>
      {/* Sprint 13.8 — marginBottom bumped 4 → 16 so there is one
          line of breathing room between the title and the headline.
          The original 4px was set when `profile_match` rendered as a
          tag directly below the title (now commented out); without
          that intermediate element the title and headline collide
          visually. */}
      <Title level={5} style={{ marginTop: 0, marginBottom: 16 }}>
        Best Historical Match & Recommended Resolution
      </Title>

      {/* Sprint 13.8 — profile_match tag suppressed at the user's
          request. The line read e.g.
          "BGP Flap (BFD Down). · Network / Fast Convergence · ny4-core-rtr"
          and surfaced ticket-internal taxonomy that the engineer
          shouldn't see at the Stage 0 level. Backend still computes
          `data.profile_match`; only the render is commented.
          Reinstate by un-commenting the JSX block. */}
      {/*
      {data.profile_match ? (
        <Tag style={{ marginBottom: 8 }}>{data.profile_match}</Tag>
      ) : null}
      */}

      <Paragraph style={{ marginBottom: 12 }}>
        <Text strong>{buildHeadline(data)}</Text>
      </Paragraph>

      {/* {data.what_worked ? (
        <Paragraph style={{ marginBottom: 8 }}>
          <Text strong>What worked: </Text>
          {data.what_worked}
        </Paragraph>
      ) : null} */}

      {/* Sprint 13.7 — bullet source switched from `how_they_did_it`
          (Resolution_Steps) to `top5_incident_summaries`
          (Incident_Summary.INCIDENT). Heading renamed "How they did
          it" → "Possible details are". Per-bullet button label
          renamed "Ask in chat" → "Discuss with LogIQ". The
          ` - INC-XXX` suffix shape is preserved on the new field so
          `extractTrailingIncidentId` keeps scoping the per-bullet
          chat handoff to the right source ticket. The legacy
          `how_they_did_it` field stays on the schema for any other
          consumer; only the render source changed. */}
      {data.top5_incident_summaries && data.top5_incident_summaries.length > 0 ? (
        <div style={{ marginBottom: 8 }}>
          <Text strong>Possible details are:</Text>
          <List
            size="small"
            dataSource={data.top5_incident_summaries}
            renderItem={(s, i) => {
              const cleaned = stripLeadingNumber(s);
              const bulletIncident =
                extractTrailingIncidentId(cleaned) || data.best_incident || null;
              return (
                <List.Item
                  key={i}
                  style={{
                    paddingLeft: 8,
                    display: "flex",
                    alignItems: "flex-start",
                    gap: 8,
                  }}
                >
                  <span style={{ flex: 1 }}>
                    {i + 1}. {cleaned}
                  </span>
                  {sessionId && cleaned ? (
                    <Tooltip
                      title={
                        bulletIncident
                          ? `Discuss this with Logic — the chat will be scoped to ${bulletIncident}`
                          : "Discuss this with Logic"
                      }
                    >
                      <Button
                        type="link"
                        size="small"
                        icon={<MessageOutlined />}
                        loading={handoffBusy}
                        onClick={() => askInChat(cleaned, bulletIncident)}
                        style={{ paddingLeft: 0, paddingRight: 0 }}
                      >
                        Discuss with LogIQ
                      </Button>
                    </Tooltip>
                  ) : null}
                </List.Item>
              );
            }}
          />
        </div>
      ) : null}

      {data.critical_intervention ? (
        <Paragraph style={{ marginBottom: 8 }}>
          <Text strong>Critical intervention: </Text>
          {data.critical_intervention}
        </Paragraph>
      ) : null}

      {/* Sprint 10.3 — CorpusStatsTail is its own component now. */}
      <CorpusStatsTail data={data} />

      {/* Sprint 11 — Escalate button on every stage. Stage 0 has no
          "next stage" CTA of its own (informational lead-in), so this
          sits alone at the bottom right. Click → Stage 5 with a
          traversal log that records "viewed Stage 0, advanced to
          stage_5 at HH:MM UTC". */}
      {sessionId && typeof onReveal === "function" ? (
        <div
          style={{
            marginTop: 12,
            paddingTop: 12,
            borderTop: "1px solid var(--border-color, #f0f0f0)",
            display: "flex",
            flexWrap: "wrap",
            gap: 12,
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          {/* Sprint 13.2 — Helpful + Dislike pair, mirrors every
              other stage's footer. Helpful opens the positive-feedback
              modal (chat parity); Dislike opens the negative one. */}
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <HelpfulButton
              sessionId={sessionId}
              stage="stage_0"
              onMarkedHelpful={onMarkedHelpful}
              onStartNewTicket={onStartNewTicket}
              disabled={helpfulMarked}
            />
            <DislikeButton
              sessionId={sessionId}
              stage="stage_0"
            />
          </div>
          {/* Sprint 13.10 — Escalate to Tier 2 + Guided Troubleshooting
              Workflow shortcut grouped on the right. The latter
              advances directly to Stage 3, bypassing Pivot Insights
              and Stage 2. NextStageButton itself fires the
              `next_stage_clicked` telemetry event and calls
              `onReveal("stage_3")` which mounts the panel and posts
              `stage_advanced` so /resume-state lands the engineer
              at Stage 3 on next remount. */}
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <EscalateButton
              sessionId={sessionId}
              fromStage="stage_0"
              onReveal={onReveal}
            />
            <NextStageButton
              sessionId={sessionId}
              fromStage="stage_0"
              toStage="stage_3"
              label={STAGE_LABELS.stage_3}
              onReveal={onReveal}
            />
          </div>
        </div>
      ) : null}
    </Card>
  );
}
