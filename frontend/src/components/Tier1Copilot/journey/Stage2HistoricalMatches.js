// Sprint 10 Stage 2 — Historical Matches & Possible Causes.
//
// Per spec §3.4 + §9: accordion with first card expanded by default,
// the rest collapsed. AntD Collapse v5 items API.
// Technical Snapshot (Sprint 11) truncated to 400 chars with "Show more"
// toggle — frontend-only; backend always sends the full string.
//
// [Helpful] [The Troubleshooting Approach ▶]

import React, { useState } from "react";
import { Button, Card, Collapse, Typography } from "antd";

import EscalateButton from "./EscalateButton";
import HelpfulButton from "./HelpfulButton";
import NextStageButton from "./NextStageButton";
import { STAGE_LABELS } from "../tier1Constants";

const { Title, Paragraph, Text } = Typography;

const SNAPSHOT_TRUNCATE_AT = 400;


function TruncatedSnapshot({ text }) {
  // Sprint 11 — restored. The corpus stores Technical_Snapshot as a
  // numbered narrative ("1. … 2. … 3. …") that runs long; collapse to
  // 400 chars with a Show more / Show less toggle. Backend sends full
  // text; truncation is frontend-only.
  const [expanded, setExpanded] = useState(false);
  if (!text) return null;
  const isLong = text.length > SNAPSHOT_TRUNCATE_AT;
  const display = !isLong || expanded
    ? text
    : `${text.slice(0, SNAPSHOT_TRUNCATE_AT).trimEnd()}…`;
  return (
    <Paragraph style={{ marginBottom: 8 }}>
      <Text strong>Technical Snapshot: </Text>
      <span style={{ whiteSpace: "pre-wrap" }}>{display}</span>
      {isLong ? (
        <Button
          type="link"
          size="small"
          style={{ paddingLeft: 6 }}
          onClick={() => setExpanded((v) => !v)}
        >
          {expanded ? "Show less" : "Show more"}
        </Button>
      ) : null}
    </Paragraph>
  );
}


function MatchCardBody({ match }) {
  // Sprint 10.8 §2.6 — render the conditional rows in spec order.
  // Backend prefers the new merged surfaces (incident_summary,
  // resolution_approach); falls back to the Sprint 10.0 fields when
  // the merged versions are absent (sparse cohorts, legacy payloads).
  const incidentSummary = match.incident_summary || match.summary;
  const resolutionLine = (
    match.resolution_approach
    || (match.resolution && match.resolution.length > 0
        ? match.resolution.join("; ")
        : null)
  );

  return (
    <div>
      {incidentSummary ? (
        <Paragraph style={{ marginBottom: 8 }}>
          <Text strong>Incident Summary: </Text>{incidentSummary}
        </Paragraph>
      ) : null}
      {/* Sprint 11 — Technical Snapshot row reinstated. Populated in
          72/180 reachable tickets (file3 + Hypothetical_goldschema).
          Omitted when missing — no dishonest empty rows. */}
      <TruncatedSnapshot text={match.technical_snapshot} />
      {match.symptoms ? (
        <Paragraph style={{ marginBottom: 8 }}>
          <Text strong>Symptoms: </Text>{match.symptoms}
        </Paragraph>
      ) : null}
      {match.root_cause ? (
        <Paragraph style={{ marginBottom: 8 }}>
          <Text strong>Root Cause: </Text>{match.root_cause}
        </Paragraph>
      ) : null}
      {resolutionLine ? (
        <Paragraph style={{ marginBottom: 8 }}>
          <Text strong>Resolution Approach: </Text>{resolutionLine}
        </Paragraph>
      ) : null}
      <FootnoteRow match={match} />
    </div>
  );
}


function FootnoteRow({ match }) {
  const parts = [];
  if (match.customer) parts.push(`Customer: ${match.customer}`);
  if (match.time_to_resolve_minutes != null) {
    parts.push(`Resolved in ${match.time_to_resolve_minutes} min`);
  }
  if (match.closed_without_recurrence) {
    parts.push("Closed without recurrence");
  }
  if (parts.length === 0) return null;
  return (
    <Text type="secondary" style={{ fontSize: 12 }}>
      {parts.join(" · ")}
    </Text>
  );
}


export default function Stage2HistoricalMatches({
  data,
  sessionId,
  onMarkedHelpful,
  onStartNewTicket,
  onReveal,
  helpfulMarked,
}) {
  // Sprint 11 — Backend (build_stage2) returns ALL useful cards;
  // empty / placeholder-only cards are already filtered out. We cap
  // the visible count to data.max_matches_shown (default 5) and
  // expose the rest behind a "View more matches" reveal so engineers
  // can opt in when the cohort is rich. State is purely client-side
  // — no extra fetch needed.
  const maxShown = (
    data && typeof data.max_matches_shown === "number" && data.max_matches_shown > 0
      ? data.max_matches_shown
      : 5
  );
  const [showAll, setShowAll] = React.useState(false);

  if (!data || !Array.isArray(data.matches) || data.matches.length === 0) {
    return (
      <Card style={{ marginBottom: 16, borderLeft: "4px solid #6B6B6B" }}>
        <Title level={5} style={{ marginTop: 0 }}>
          Related Incidents & Probable Causes
        </Title>
        <Paragraph type="secondary">
          No matching historical tickets found for this profile.
        </Paragraph>
      </Card>
    );
  }

  const total = data.matches.length;
  const visible = showAll ? data.matches : data.matches.slice(0, maxShown);
  const hidden = total - visible.length;

  // First card expanded by default, rest collapsed.
  const items = visible.map((m, idx) => ({
    key: `match-${idx}`,
    label: (
      <span>
        <Text strong>MATCH {m.rank} OF {total}</Text>
        {m.incident_number ? <Text> — {m.incident_number}</Text> : null}
        {m.headline ? (
          <Text type="secondary" style={{ marginLeft: 8 }}>· {m.headline}</Text>
        ) : null}
      </span>
    ),
    children: <MatchCardBody match={m} />,
  }));

  return (
    <Card style={{ marginBottom: 16, borderLeft: "4px solid #2563eb" }}>
      <Title level={5} style={{ marginTop: 0 }}>
        Related Incidents & Probable Causes
      </Title>

      <Collapse defaultActiveKey={["match-0"]} items={items} />

      {/* Sprint 11 — "View more matches" reveal. Only shown when the
          backend returned more useful cards than the default visible
          cap (5). Pure client-side toggle — no extra fetch. */}
      {hidden > 0 ? (
        <div style={{ marginTop: 12, textAlign: "center" }}>
          <Button type="link" onClick={() => setShowAll(true)}>
            Would you like to see {hidden} more matched ticket{hidden === 1 ? "" : "s"}?
          </Button>
        </div>
      ) : null}
      {showAll && total > maxShown ? (
        <div style={{ marginTop: 4, textAlign: "center" }}>
          <Button type="link" onClick={() => setShowAll(false)}>
            Show fewer
          </Button>
        </div>
      ) : null}

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
          stage="stage_2"
          onMarkedHelpful={onMarkedHelpful}
          onStartNewTicket={onStartNewTicket}
          disabled={helpfulMarked}
        />
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          {/* Sprint 11 — Escalate from Stage 2. Click → traversal log
              records "Stage 2 — viewed, advanced at HH:MM UTC" then
              jumps to Stage 5. Stages 3 and 4 won't appear in the
              escalation package because they were never seen. */}
          <EscalateButton
            sessionId={sessionId}
            fromStage="stage_2"
            onReveal={onReveal}
          />
          <NextStageButton
            sessionId={sessionId}
            fromStage="stage_2"
            toStage="stage_3"
            label={`${STAGE_LABELS.stage_3}`}
            onReveal={onReveal}
          />
        </div>
      </div>
    </Card>
  );
}
