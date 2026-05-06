// Sprint 10 Stage 5 — Escalation Package.
//
// Per spec §3.7 + §9: reuse existing Sprint 7 EscalationPackageCard
// component unchanged. Sprint 10 just supplies it the richer
// what_was_tried list (which already includes the journey traversal
// log appended by build_journey_escalation_package on the backend).
//
// Sprint 10.7 §4.2 — `autoExpand` prop. When the engineer arrives at
// Stage 5 directly (chat-Escalate jump), the package opens visible
// immediately. On a normal walked-through journey the prop defaults
// to true so the existing UX is preserved — the package always shows
// inline like it did pre-10.7. The prop's real role is documenting
// the resume-direct path; future variants that want a collapsed-by-
// default Stage 5 can pass autoExpand={false}.

import React, { useState } from "react";
import { Button, Card, Typography } from "antd";

import EscalationPackageCard from "../EscalationPackageCard";
import HelpfulButton from "./HelpfulButton";

const { Title, Paragraph } = Typography;


export default function Stage5EscalationPackage({
  data,
  sessionId,
  onMarkedHelpful,
  onStartNewTicket,   // resets journey + intake form
  helpfulMarked,
  // Sprint 10.7 §4.2 — defaults to true. ResolutionJourney passes
  // true explicitly when the engineer was resumed at stage_5 (e.g.
  // via chat-Escalate). Tests assert both branches by passing the
  // prop explicitly.
  autoExpand = true,
}) {
  const [expanded, setExpanded] = useState(autoExpand);

  if (!data) {
    return (
      <Card style={{ marginBottom: 16, borderLeft: "4px solid #6B6B6B" }}>
        <Title level={5} style={{ marginTop: 0 }}>
          Operational Handoff
        </Title>
        <Paragraph type="secondary">
          Could not assemble the escalation package. The Sprint 7 generator
          may have failed; check backend logs.
        </Paragraph>
      </Card>
    );
  }

  return (
    <Card style={{ marginBottom: 16, borderLeft: "4px solid #B03A2E" }}>
      <Title level={5} style={{ marginTop: 0 }}>
        Operational Handoff
      </Title>

      {/* Sprint 10.7 — package render gated on `expanded`. With the
          default autoExpand=true the package is visible on first
          paint (preserving the pre-10.7 walked-through UX); a future
          variant can pass autoExpand={false} and the engineer reveals
          the package via the toggle below.

          Sprint 11 — wire `onClose` so the Close button inside
          EscalationPackageCard actually fires. Mirrors Tier1Workspace's
          pattern (subCardProps.onClose collapses the active sub-card).
          Collapsing back to the prompt lets the engineer re-open via
          "Show escalation package" without losing journey state. */}
      {expanded ? (
        <EscalationPackageCard pkg={data} onClose={() => setExpanded(false)} />
      ) : (
        <Paragraph type="secondary" style={{ marginBottom: 12 }}>
          The escalation package is ready. Open it to copy the
          Tier-2 handoff text.
        </Paragraph>
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
          stage="stage_5"
          onMarkedHelpful={onMarkedHelpful}
          onStartNewTicket={onStartNewTicket}
          disabled={helpfulMarked}
        />
        <div style={{ display: "flex", gap: 8 }}>
          {!expanded ? (
            <Button type="primary" onClick={() => setExpanded(true)}>
              Show escalation package
            </Button>
          ) : null}
          {typeof onStartNewTicket === "function" ? (
            <Button onClick={onStartNewTicket}>New Ticket</Button>
          ) : null}
        </div>
      </div>
    </Card>
  );
}
