// Sprint 10 Stage 3 — The Troubleshooting Approach.
//
// Sprint 13 — replaces the merged-ledger flat list with one Collapse
// panel per cohort ticket (`guided_workflows[]`). Match 1 is open by
// default; Match 2+ are collapsed. Each step inside a panel uses
// LLM-synthesised Intent + Pivot prose; the action text stays
// verbatim. The `Per-ticket troubleshooting detail` accordion
// underneath is preserved unchanged from Sprint 11.
//
// Layout:
//   Card title: "Guided Troubleshooting Workflow"   ← single heading
//     ┌─ Collapse panel: MATCH 1 — INC-XXX — <synthesised header>
//     │    Step 1: <action>     <command>?
//     │      Why: <intent>
//     │      Outcome: <pivot prose>
//     │    ...
//     └─ Collapse panel: MATCH 2 — INC-YYY — ... (collapsed)
//   Per-ticket troubleshooting detail (N)            ← Sprint 11
//   [Helpful] [Search KB / SOP ▶]

import React, { useState } from "react";
import { Button, Card, Collapse, List, Tooltip, Typography } from "antd";
import { MessageOutlined } from "@ant-design/icons";

import DislikeButton from "./DislikeButton";
import EscalateButton from "./EscalateButton";
import HelpfulButton from "./HelpfulButton";
import NextStageButton from "./NextStageButton";
import { stripLeadingNumber } from "./stepText";
import useChatHandoff from "./useChatHandoff";
import { STAGE_LABELS } from "../tier1Constants";

const { Title, Paragraph, Text } = Typography;

const SNAPSHOT_TRUNCATE_AT = 400;


// Sprint 13 — `branchColor` from the Sprint 10.8 merged-ledger render
// is dead now (no branch_label / Primary / Fallback pills in the new
// per-ticket schema). Kept removed; reinstate alongside the legacy
// flat list if the merged ledger ever returns.


// Sprint 11 — local Show more / Show less toggle for long narrative
// fields (Technical_Snapshot, Critical_Intervention, Hero_Action when
// they run long). Backend always sends the full string; truncation is
// purely a frontend convenience. Kept inline (not imported from Stage 2)
// so the component stays self-contained.
function TruncatedText({ text }) {
  const [expanded, setExpanded] = useState(false);
  if (!text) return null;
  const isLong = text.length > SNAPSHOT_TRUNCATE_AT;
  const display = !isLong || expanded
    ? text
    : `${text.slice(0, SNAPSHOT_TRUNCATE_AT).trimEnd()}…`;
  return (
    <span>
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
    </span>
  );
}


// ────────────────────────────────────────────────────────────
// Sprint 13 — Guided Workflow rendering (per-ticket grouped).
// One Collapse panel per cohort ticket; each panel contains that
// ticket's full step playbook with LLM-synthesised Intent + Pivot.
// ────────────────────────────────────────────────────────────
function GuidedWorkflowStepItem({ step }) {
  return (
    <li
      key={step.step_number}
      style={{ marginBottom: 12 }}
    >
      <div>
        <Text strong>Step {step.step_number}: </Text>
        <Text>{step.action}</Text>
      </div>
      {step.command ? (
        <div style={{ marginTop: 4 }}>
          <Text code>{step.command}</Text>
        </div>
      ) : null}
      {step.intent ? (
        <Paragraph style={{ marginTop: 4, marginBottom: 2 }}>
          <Text type="secondary">Why: </Text>{step.intent}
        </Paragraph>
      ) : null}
      {step.pivot ? (
        <Paragraph style={{ marginTop: 0, marginBottom: 0 }}>
          <Text type="secondary">Outcome: </Text>{step.pivot}
        </Paragraph>
      ) : null}
    </li>
  );
}


function GuidedWorkflowPanelBody({ workflow }) {
  return (
    <div>
      {workflow.technical_snapshot ? (
        <Paragraph
          type="secondary"
          style={{ marginTop: 0, marginBottom: 12, fontStyle: "italic" }}
        >
          <TruncatedText text={workflow.technical_snapshot} />
        </Paragraph>
      ) : null}

      <ol style={{ paddingLeft: 20, marginBottom: 0, listStyleType: "none" }}>
        {workflow.steps.map((s) => (
          <GuidedWorkflowStepItem key={s.step_number} step={s} />
        ))}
      </ol>

      {workflow.synthesis_skipped ? (
        <Paragraph
          type="secondary"
          style={{ marginTop: 8, marginBottom: 0, fontSize: 12 }}
        >
          (Showing verbatim source text — LLM synthesis unavailable for this ticket.)
        </Paragraph>
      ) : null}
    </div>
  );
}


function GuidedWorkflows({ workflows }) {
  if (!workflows || workflows.length === 0) return null;

  // Build one Collapse panel per workflow. Default-active key = the
  // workflow flagged `expanded_by_default` (always Match 1 today).
  const items = workflows.map((wf) => ({
    key: `match-${wf.match_rank}`,
    label: (
      <span>
        <Text strong>MATCH {wf.match_rank}</Text>
        <Text> — {wf.incident_number}</Text>
        {wf.header && wf.header !== wf.incident_number ? (
          <Text> — {wf.header}</Text>
        ) : null}
      </span>
    ),
    children: <GuidedWorkflowPanelBody workflow={wf} />,
  }));

  const defaultActiveKeys = workflows
    .filter((wf) => wf.expanded_by_default)
    .map((wf) => `match-${wf.match_rank}`);

  return (
    <Collapse
      // `accordion` deliberately off — engineers may want to compare
      // two tickets' playbooks side-by-side after expanding both.
      defaultActiveKey={defaultActiveKeys}
      items={items}
      style={{ marginBottom: 16 }}
    />
  );
}


// Sprint 11 — per-ticket raw detail card. Renders one section per
// source path on the spec list, in the order an engineer reads them:
// Technical Snapshot → Resolution Steps → Diagnostic Logic → Timeline
// → Critical Intervention → Hero Action → Diagnostic Tests Executed.
// Missing sections are omitted entirely.
function TicketDetailBody({ detail, askInChat, askInChatBusy }) {
  const hasAnything = (
    detail.technical_snapshot
    || (detail.resolution_steps && detail.resolution_steps.length > 0)
    || (detail.diagnostic_logic && detail.diagnostic_logic.length > 0)
    || (detail.timeline && detail.timeline.length > 0)
    || detail.critical_intervention
    || detail.hero_action
    || (detail.diagnostic_tests_executed && detail.diagnostic_tests_executed.length > 0)
  );
  if (!hasAnything) {
    return (
      <Paragraph type="secondary" style={{ marginBottom: 0 }}>
        No troubleshooting detail recorded for this ticket.
      </Paragraph>
    );
  }
  return (
    <div>
      {detail.technical_snapshot ? (
        <Paragraph style={{ marginBottom: 12 }}>
          <Text strong>Technical Snapshot: </Text>
          <TruncatedText text={detail.technical_snapshot} />
        </Paragraph>
      ) : null}

      {detail.resolution_steps && detail.resolution_steps.length > 0 ? (
        <div style={{ marginBottom: 12 }}>
          <Text strong>Resolution Steps:</Text>
          <ol style={{ marginTop: 4, marginBottom: 0, paddingLeft: 20 }}>
            {detail.resolution_steps.map((s, i) => {
              // Sprint 11 — strip pre-existing "1." etc. baked into the
              // source string so the <ol>'s own numbering doesn't double.
              const cleaned = stripLeadingNumber(s);
              return (
                <li
                  key={i}
                  style={{
                    marginBottom: 2,
                    display: "flex",
                    alignItems: "flex-start",
                    gap: 8,
                  }}
                >
                  <span style={{ flex: 1 }}>{cleaned}</span>
                  {askInChat && cleaned ? (
                    <Tooltip title="Ask this step in a new chat">
                      <Button
                        type="link"
                        size="small"
                        icon={<MessageOutlined />}
                        loading={askInChatBusy}
                        onClick={() => askInChat(cleaned)}
                        style={{ paddingLeft: 0, paddingRight: 0 }}
                      >
                        Ask
                      </Button>
                    </Tooltip>
                  ) : null}
                </li>
              );
            })}
          </ol>
        </div>
      ) : null}

      {detail.diagnostic_logic && detail.diagnostic_logic.length > 0 ? (
        <div style={{ marginBottom: 12 }}>
          <Text strong>Diagnostic Logic:</Text>
          <List
            size="small"
            split={false}
            dataSource={detail.diagnostic_logic}
            style={{ marginTop: 4 }}
            renderItem={(dl, i) => (
              <List.Item style={{ paddingLeft: 0, paddingRight: 0, paddingTop: 4, paddingBottom: 4 }} key={i}>
                <div style={{ width: "100%" }}>
                  {dl.action ? (
                    <div><Text strong>Action: </Text>{dl.action}</div>
                  ) : null}
                  {dl.command ? (
                    <div style={{ marginTop: 2 }}><Text code>{dl.command}</Text></div>
                  ) : null}
                  {dl.intent ? (
                    <div style={{ marginTop: 2 }}>
                      <Text type="secondary">Intent: </Text>{dl.intent}
                    </div>
                  ) : null}
                  {dl.pivot ? (
                    <div style={{ marginTop: 2 }}>
                      <Text type="secondary">Pivot: </Text>{dl.pivot}
                    </div>
                  ) : null}
                </div>
              </List.Item>
            )}
          />
        </div>
      ) : null}

      {detail.timeline && detail.timeline.length > 0 ? (
        <div style={{ marginBottom: 12 }}>
          <Text strong>Timeline:</Text>
          <ul style={{ marginTop: 4, marginBottom: 0, paddingLeft: 20, listStyleType: "none" }}>
            {detail.timeline.map((t, i) => (
              <li key={i} style={{ marginBottom: 2 }}>
                {t.time ? <Text code style={{ marginRight: 8 }}>{t.time}</Text> : null}
                {t.action || ""}
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {detail.critical_intervention ? (
        <Paragraph style={{ marginBottom: 12 }}>
          <Text strong>Critical Intervention: </Text>
          <TruncatedText text={detail.critical_intervention} />
        </Paragraph>
      ) : null}

      {detail.hero_action ? (
        <Paragraph style={{ marginBottom: 12 }}>
          <Text strong>Hero Action: </Text>
          <TruncatedText text={detail.hero_action} />
        </Paragraph>
      ) : null}

      {detail.diagnostic_tests_executed && detail.diagnostic_tests_executed.length > 0 ? (
        <div style={{ marginBottom: 0 }}>
          <Text strong>Diagnostic Tests Executed:</Text>
          <ul style={{ marginTop: 4, marginBottom: 0, paddingLeft: 20 }}>
            {detail.diagnostic_tests_executed.map((t, i) => (
              <li key={i} style={{ marginBottom: 2 }}>{t}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}


function PerTicketDetails({ details, maxShown, askInChat, askInChatBusy }) {
  // Sprint 11 — same reveal pattern as Stage 2's "View more matches".
  // Backend returns ALL useful details (already filtered for empties);
  // frontend caps display at `maxShown` and surfaces a reveal button
  // for any surplus. State is purely client-side — no extra fetch.
  const [showAll, setShowAll] = React.useState(false);

  if (!details || details.length === 0) return null;

  const cap = (typeof maxShown === "number" && maxShown > 0) ? maxShown : 5;
  const total = details.length;
  const visible = showAll ? details : details.slice(0, cap);
  const hidden = total - visible.length;

  // First card expanded by default, rest collapsed — matches Stage 2 UX.
  const items = visible.map((d) => ({
    key: `detail-${d.rank}`,
    label: (
      <span>
        <Text strong>MATCH {d.rank}</Text>
        {d.incident_number ? <Text> — {d.incident_number}</Text> : null}
      </span>
    ),
    children: (
      <TicketDetailBody
        detail={d}
        askInChat={askInChat}
        askInChatBusy={askInChatBusy}
      />
    ),
  }));

  return (
    <div style={{ marginTop: 16 }}>
      <Title level={5} style={{ marginBottom: 8 }}>
        Per-ticket troubleshooting detail ({total})
      </Title>
      <Collapse defaultActiveKey={["detail-1"]} items={items} />

      {/* Sprint 11 — "View more" reveal. Only shown when the backend
          returned more useful details than the visible cap. */}
      {hidden > 0 ? (
        <div style={{ marginTop: 12, textAlign: "center" }}>
          <Button type="link" onClick={() => setShowAll(true)}>
            Would you like to see {hidden} more ticket detail{hidden === 1 ? "" : "s"}?
          </Button>
        </div>
      ) : null}
      {showAll && total > cap ? (
        <div style={{ marginTop: 4, textAlign: "center" }}>
          <Button type="link" onClick={() => setShowAll(false)}>
            Show fewer
          </Button>
        </div>
      ) : null}
    </div>
  );
}


export default function Stage3TroubleshootingApproach({
  data,
  sessionId,
  onMarkedHelpful,
  onStartNewTicket,
  onReveal,
  helpfulMarked,
}) {
  // Sprint 11 — single chat-handoff hook for the whole stage.
  // Shared between every per-ticket Resolution Step "Ask" link so
  // the busy state covers all of them at once.
  const { busy: handoffBusy, askInChat } = useChatHandoff(sessionId);

  // Sprint 13 — guided_workflows is the new primary surface.
  // Per-ticket details (Sprint 11) is preserved unchanged.
  const hasGuidedWorkflows = (
    data && Array.isArray(data.guided_workflows) && data.guided_workflows.length > 0
  );
  const hasDetails = (
    data && Array.isArray(data.per_ticket_details)
    && data.per_ticket_details.length > 0
    && data.per_ticket_details.some((d) => (
      d.technical_snapshot
      || (d.resolution_steps && d.resolution_steps.length > 0)
      || (d.diagnostic_logic && d.diagnostic_logic.length > 0)
      || (d.timeline && d.timeline.length > 0)
      || d.critical_intervention
      || d.hero_action
      || (d.diagnostic_tests_executed && d.diagnostic_tests_executed.length > 0)
    ))
  );

  if (!hasGuidedWorkflows && !hasDetails) {
    return (
      <Card style={{ marginBottom: 16, borderLeft: "4px solid #6B6B6B" }}>
        <Title level={5} style={{ marginTop: 0 }}>
          Guided Troubleshooting Workflow
        </Title>
        <Paragraph type="secondary">
          No troubleshooting workflows available for these similar tickets.
        </Paragraph>
      </Card>
    );
  }

  return (
    <Card style={{ marginBottom: 16, borderLeft: "4px solid #2563eb" }}>
      <Title level={5} style={{ marginTop: 0 }}>
        Guided Troubleshooting Workflow
      </Title>

      {hasGuidedWorkflows ? (
        <GuidedWorkflows workflows={data.guided_workflows} />
      ) : null}

      {hasDetails ? (
        <PerTicketDetails
          details={data.per_ticket_details}
          maxShown={data.max_details_shown}
          askInChat={askInChat}
          askInChatBusy={handoffBusy}
        />
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
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <HelpfulButton
            sessionId={sessionId}
            stage="stage_3"
            onMarkedHelpful={onMarkedHelpful}
            onStartNewTicket={onStartNewTicket}
            disabled={helpfulMarked}
          />
          <DislikeButton
            sessionId={sessionId}
            stage="stage_3"
          />
        </div>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          {/* Sprint 11 — Escalate from Stage 3. Click → traversal log
              records "Stage 3 — viewed, advanced at HH:MM UTC" then
              jumps to Stage 5. Stage 4 won't appear in the
              escalation package because it was never seen. */}
          <EscalateButton
            sessionId={sessionId}
            fromStage="stage_3"
            onReveal={onReveal}
          />
          <NextStageButton
            sessionId={sessionId}
            fromStage="stage_3"
            toStage="stage_4"
            label={`${STAGE_LABELS.stage_4}`}
            onReveal={onReveal}
          />
        </div>
      </div>
    </Card>
  );
}
