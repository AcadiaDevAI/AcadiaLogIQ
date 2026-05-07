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
import EscalateButton from "./EscalateButton";
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


// Sprint 10.4 §2.2 — plain-English headlines by evidence_strength.
// "We found N similar past tickets" replaces the old "We've seen this
// issue N times" preamble. Score-3 ticket gets honest framing without
// the architect-speak phrase "adequate, not exemplary".
function buildHeadline(data) {
  const n = data.cohort_size || 0;
  const inc = data.best_incident;
  const score = data.best_quality_score;
  const ttr = formatMinutes(data.best_time_minutes);
  const ttrTail = ttr ? `, fixed in ${ttr}` : "";
  const found = `We found ${n} similar past ticket${n === 1 ? "" : "s"}`;

  switch (data.evidence_strength) {
    case "strong":
      if (!inc) return `${found}.`;
      return `${found}. The best fix came from ${inc} (rated ${score}/5${ttrTail}).`;
    case "adequate":
      if (!inc) return `${found}.`;
      return (
        `${found}. The closest fix came from ${inc} ` +
        `(rated ${score}/5 — this worked, though the write-ups weren't detailed${ttrTail}).`
      );
    case "weak":
      if (!inc) return `${found}, but none scored well.`;
      return (
        `${found}, but the best one was only rated ${score}/5. ` +
        `Use the suggested fix as a starting point — verify before acting.`
      );
    case "none":
    default:
      if (!inc) return (
        `${found}, but none had a quality rating. ` +
        `Review the matches below case-by-case.`
      );
      return `${found}. Closest match: ${inc}${ttrTail}.`;
  }
}


export default function Stage0BestTicketDistillation({ data, sessionId, onReveal }) {
  // Sprint 11 — per-step "Ask in chat" links. Hook is a no-op when
  // sessionId is missing (defensive — Stage 0 should always have one).
  const { busy: handoffBusy, askInChat } = useChatHandoff(sessionId);
  if (!data) return null;

  // ── Sparse case (cohort empty) ──
  if (data.sparse) {
    return (
      <Card style={{ marginBottom: 16, borderLeft: "4px solid #6B6B6B" }}>
        <Title level={5} style={{ marginTop: 0 }}>Best Historical Match & Recommended Resolution</Title>
        {data.profile_match ? (
          <Paragraph style={{ marginBottom: 8 }}>
            <Text type="secondary">{data.profile_match}</Text>
          </Paragraph>
        ) : null}
        <Paragraph style={{ marginBottom: 0 }}>
          {/* Sprint 10.4 §2.1 — plain-English sparse copy. */}
          No similar past tickets found. Use the playbook below.
        </Paragraph>
      </Card>
    );
  }

  const color = ringColor(data.clean_resolution_percent || 0);

  return (
    <Card style={{ marginBottom: 16, borderLeft: `4px solid ${color}` }}>
      <Title level={5} style={{ marginTop: 0, marginBottom: 4 }}>
        Best Historical Match & Recommended Resolution
      </Title>

      {data.profile_match ? (
        <Tag style={{ marginBottom: 8 }}>{data.profile_match}</Tag>
      ) : null}

      <Paragraph style={{ marginBottom: 12 }}>
        <Text strong>{buildHeadline(data)}</Text>
      </Paragraph>

      {data.what_worked ? (
        <Paragraph style={{ marginBottom: 8 }}>
          <Text strong>What worked: </Text>
          {data.what_worked}
        </Paragraph>
      ) : null}

      {data.how_they_did_it && data.how_they_did_it.length > 0 ? (
        <div style={{ marginBottom: 8 }}>
          <Text strong>How they did it:</Text>
          <List
            size="small"
            dataSource={data.how_they_did_it}
            renderItem={(s, i) => {
              // Sprint 11 — Strip any pre-existing "1.", "1)", "1 -", etc.
              // baked into the source string. Otherwise the React index
              // prefix below produces "1. 1. Incident..." double-numbering.
              const cleaned = stripLeadingNumber(s);
              // Sprint 12.1 — Pull the trailing " - INC-XXX" off the
              // bullet so we can (a) scope this bullet's "Ask in chat"
              // handoff to that source ticket, and (b) tell the user
              // explicitly which past ticket the step came from.
              // Falls back to data.best_incident when the suffix is
              // missing, so legacy bullets (or fields without the
              // " - INC-XXX" tail) still get a sensible scope.
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
                  {/* Sprint 11 — per-step "Ask in chat" link.
                      Sprint 12.1 — also passes the bullet's source
                      Incident_Number so the chat session is scoped to
                      that one ticket (chat answers only from that
                      ticket's chunks). When the bullet has no parsable
                      source, we fall back to data.best_incident so the
                      user never lands in an unscoped chat from a
                      Stage 0 bullet click. */}
                  {sessionId && cleaned ? (
                    <Tooltip
                      title={
                        bulletIncident
                          ? `Ask this step in a new chat — answers will be scoped to ${bulletIncident}`
                          : "Ask this step in a new chat"
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
                        Ask in chat
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
            justifyContent: "flex-end",
          }}
        >
          <EscalateButton
            sessionId={sessionId}
            fromStage="stage_0"
            onReveal={onReveal}
          />
        </div>
      ) : null}
    </Card>
  );
}
