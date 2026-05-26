// ─────────────────────────────────────────────────────────────
// Block 02 — Guided Troubleshooting Workflow
// Scoped 7-step blue scale + 9 type roles for the consolidated
// step ledger ONLY. All other panels in this file (TicketDetailBody,
// GuidedWorkflows, PerTicketDetails) remain visually unchanged.
// Roles:
//   eyebrow  — uppercase mono chip above the heading
//   display  — large headline ("Guided troubleshooting workflow")
//   accent   — gradient-painted accent word inside the display
//   subhead  — instruction line (when/why to tick boxes)
//   rowIndex — 01 / 02 / 03 … tabular-num counter on the left
//   rowBody  — main action text inside the row
//   rowSource— bare ticket id (e.g. INC-PHOENIX-402) — NO "From:" prefix
//   rowLink  — secondary text actions like "Why:" / "Outcome:" labels
//   codeChip — command pill (tabular nums, no mid-command wrap)
// Behaviour rules baked into CSS:
//   * Checked rows do NOT get strikethrough or dimming — Tier-2 must
//     still be able to read the action text after the engineer ticks
//     it. Only the checkbox itself reflects the state.
//   * codeChip uses font-feature-settings:'tnum' 1 and white-space:
//     nowrap so a long CLI never breaks mid-command.
// ─────────────────────────────────────────────────────────────
const B02_BLUE = {
  50:  "#EFF6FF",
  100: "#DBEAFE",
  200: "#BFDBFE",
  400: "#60A5FA",
  500: "#3B82F6",
  600: "#2563EB",
  700: "#1D4ED8",
};
const B02_CSS = `
.b02-root { color: #0F172A; }
.b02-eyebrow {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 4px 12px; border-radius: 9999px;
  background: ${B02_BLUE[50]};
  border: 1px solid ${B02_BLUE[200]};
  color: ${B02_BLUE[700]};
  font-family: var(--font-mono, 'Geist Mono', 'JetBrains Mono', Consolas, monospace);
  font-size: 10.5px; font-weight: 500; letter-spacing: 0.14em;
  text-transform: uppercase;
  margin-bottom: 10px;
}
.b02-eyebrow__dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: ${B02_BLUE[500]};
  box-shadow: 0 0 8px ${B02_BLUE[400]};
}
.b02-display {
  font-family: var(--font-display, 'Instrument Serif', Georgia, serif);
  font-size: clamp(24px, 2.8vw, 32px);
  font-weight: 400; line-height: 1.12; letter-spacing: -0.018em;
  margin: 0 0 6px 0; color: #0F172A;
}
.b02-accent {
  font-style: italic; font-weight: 400;
  background: linear-gradient(135deg, ${B02_BLUE[400]} 0%, ${B02_BLUE[600]} 60%, ${B02_BLUE[700]} 100%);
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent; color: transparent;
}
.b02-subhead {
  font-family: var(--font-body, 'Geist', system-ui, sans-serif);
  font-size: 12.5px; line-height: 1.5; color: #475569;
  margin: 0 0 14px 0;
}
.b02-rows { list-style: none; padding: 0; margin: 0; }
.b02-row {
  display: grid;
  grid-template-columns: 28px 36px 1fr;
  column-gap: 12px;
  row-gap: 4px;
  align-items: start;
  padding: 12px 0;
  border-top: 1px solid ${B02_BLUE[100]};
}
.b02-row:first-child { border-top: none; padding-top: 4px; }
/* Checkbox cell — Ant Design checkbox renders inside this column;
   no extra rules needed because the existing .acadia-attempted-
   checkbox class still controls its appearance. */
.b02-row__check { padding-top: 2px; }
.b02-rowIndex {
  font-family: var(--font-mono, 'Geist Mono', monospace);
  font-feature-settings: 'tnum' 1;
  font-size: 13px; font-weight: 600; color: ${B02_BLUE[600]};
  line-height: 1.55; padding-top: 2px;
}
.b02-rowBody {
  font-family: var(--font-body, 'Geist', system-ui, sans-serif);
  font-size: 14px; line-height: 1.55; color: #0F172A;
}
/* Completed rows: NO strikethrough, NO opacity dim. Tier-2 still
   needs to read everything an engineer ticked. The checked state is
   communicated solely through the checkbox glyph. */
.b02-row--checked .b02-rowBody,
.b02-row--checked .b02-rowSource,
.b02-row--checked .b02-codeChip { text-decoration: none; opacity: 1; }

.b02-rowSource {
  display: inline-block;
  font-family: var(--font-mono, 'Geist Mono', monospace);
  font-feature-settings: 'tnum' 1;
  font-size: 12.5px; font-weight: 600;
  color: ${B02_BLUE[700]};
  background: ${B02_BLUE[50]};
  border: 1px solid ${B02_BLUE[200]};
  padding: 1px 8px; border-radius: 6px;
  margin-right: 8px;
}
.b02-rowLink {
  font-family: var(--font-body, 'Geist', system-ui, sans-serif);
  font-size: 12.5px; line-height: 1.55; color: #475569;
  margin-top: 4px;
}
.b02-rowLink__label {
  font-weight: 600; color: ${B02_BLUE[700]}; margin-right: 4px;
}
.b02-codeChip {
  display: inline-block; margin-top: 6px;
  font-family: var(--font-mono, 'Geist Mono', 'JetBrains Mono', Consolas, monospace);
  font-feature-settings: 'tnum' 1;
  white-space: nowrap;
  font-size: 12.5px; color: #0F172A;
  background: ${B02_BLUE[50]};
  border: 1px solid ${B02_BLUE[200]};
  border-radius: 8px;
  padding: 4px 10px;
  max-width: 100%;
  overflow-x: auto;
}
`;
let _b02StylesInjected = false;
function _ensureB02Styles() {
  if (typeof document === "undefined" || _b02StylesInjected) return;
  const style = document.createElement("style");
  style.setAttribute("data-acadia-block02", "1");
  style.textContent = B02_CSS;
  document.head.appendChild(style);
  _b02StylesInjected = true;
}


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
//     │    INC-XXX: <action>     <command>?
//     │      Why: <intent>
//     │      Outcome: <pivot prose>
//     │    ...
//     └─ Collapse panel: MATCH 2 — INC-YYY — ... (collapsed)
//   Per-ticket troubleshooting detail (N)            ← Sprint 11
//   [Helpful] [Search KB / SOP ▶]

import React, { useState } from "react";
import { Button, Card, Checkbox, Collapse, List, Tooltip, Typography } from "antd";
import { MessageOutlined } from "@ant-design/icons";

import DislikeButton from "./DislikeButton";
import EscalateButton from "./EscalateButton";
import HelpfulButton from "./HelpfulButton";
import NextStageButton from "./NextStageButton";
import CardWatermark from "./CardWatermark";
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
// Sprint 13.12 — Consolidated 5-step Tier-1 read-only ledger.
// Single flat numbered list replacing the per-ticket Guided
// Workflows view. LLM filters for safe show/read-only actions and
// caps at 5 steps; backend handles synthesis + safety enforcement.
// ────────────────────────────────────────────────────────────
// Sprint 13.14.1 — gray border + gray checked fill so the box is
// visible against the white card background. AntD's default border
// is near-invisible on light themes; we override the inner element's
// border + background via a scoped class injected once below. The
// strikethrough on the action text was removed at the user's
// request — checked rows now stay full-strength so the engineer
// can still re-read the step content.
const ATTEMPTED_CHECKBOX_CSS = `
.acadia-attempted-checkbox .ant-checkbox-inner {
  border-color: #6b7280;
  background-color: #ffffff;
}
.acadia-attempted-checkbox:hover .ant-checkbox-inner,
.acadia-attempted-checkbox .ant-checkbox-checked .ant-checkbox-inner {
  border-color: #4b5563;
}
.acadia-attempted-checkbox .ant-checkbox-checked .ant-checkbox-inner {
  background-color: #6b7280;
}
.acadia-attempted-checkbox .ant-checkbox-checked .ant-checkbox-inner::after {
  border-color: #ffffff;
}
`;


function ConsolidatedSteps({ steps, attemptedSteps, onToggleAttempt }) {
  // Sprint 13.14 — per-step "attempted" tracking. State + checkbox
  // toggle logic is UNCHANGED. Block 02 redesign only re-typesets
  // the rendered markup; the props, the controlled-state contract,
  // and the gray-tone .acadia-attempted-checkbox styles are all
  // preserved.
  const safeMap = attemptedSteps || {};
  const handleToggle = (stepNumber) => {
    if (typeof onToggleAttempt === "function") {
      onToggleAttempt(stepNumber);
    }
  };

  // Block 02 — inject scoped CSS once on first mount. Idempotent.
  React.useEffect(() => { _ensureB02Styles(); }, []);

  if (!steps || steps.length === 0) return null;
  return (
    <>
      {/* Sprint 13.14.1 — scoped gray-tone styles for the attempted
          checkboxes. Kept exactly as before so the checkbox glyph
          itself doesn't change appearance. */}
      <style>{ATTEMPTED_CHECKBOX_CSS}</style>

      {/* ─── Block 02 — Guided Troubleshooting Workflow (new render) ─
          OLD markup (Sprint 13.14.x) preserved for reference below.
          Reinstate by uncommenting the OLD block and removing the
          new b02-* block.

          ── OLD ──
          <Paragraph type="secondary" style={{ fontSize: 12, marginTop: 4, marginBottom: 8 }}>
            Each activity below is consolidated from one matching historical
            ticket — the source incident number is shown next to each label.
            Tick any activity you have already tried that did not resolve
            the issue; your selections flow into the Tier-2 handoff note
            when the ticket is escalated.
          </Paragraph>
          <ol style={{ paddingLeft: 20, marginBottom: 16, marginTop: 8, listStyleType: "none" }}>
            {steps.map((s) => {
              const checked = !!safeMap[s.step_number];
              return (
                <li key={s.step_number} style={{ marginBottom: 14 }}>
                  <div style={{ display: "flex", alignItems: "flex-start", gap: 8 }}>
                    <Tooltip title={checked ? "Marked attempted …" : "Mark this activity as attempted"}>
                      <Checkbox
                        checked={checked}
                        onChange={() => handleToggle(s.step_number)}
                        className="acadia-attempted-checkbox"
                        style={{ marginTop: 4 }}
                      />
                    </Tooltip>
                    <div style={{ flex: 1 }}>
                      <div>
                        {s.incident_number ? <Text strong>{s.incident_number}: </Text> : null}
                        <Text>{s.action}</Text>
                      </div>
                      {s.command ? <div style={{ marginTop: 4 }}><Text code>{s.command}</Text></div> : null}
                      {s.intent ? <Paragraph><Text type="secondary">Why: </Text>{s.intent}</Paragraph> : null}
                      {s.pivot ? <Paragraph><Text type="secondary">Outcome: </Text>{s.pivot}</Paragraph> : null}
                    </div>
                  </div>
                </li>
              );
            })}
          </ol>
          ── /OLD ── */}
      <div className="b02-root">
        {/* Role: subhead — engineer guidance, kept as plain copy
            because the panel's own card-level heading
            "Guided Troubleshooting Workflow" already carries the
            display+accent treatment (added below in the wrapper). */}
        <p className="b02-subhead">
          Each activity below is consolidated from one matching historical
          ticket. Tick the activities you have already tried — your
          selections flow into the Tier-2 handoff note when the ticket
          is escalated.
        </p>

        <ol className="b02-rows">
          {steps.map((s) => {
            const checked = !!safeMap[s.step_number];
            const idx = String(s.step_number).padStart(2, "0");
            return (
              <li
                key={s.step_number}
                className={`b02-row${checked ? " b02-row--checked" : ""}`}
              >
                {/* Column 1 — checkbox (uses existing gray-tone class) */}
                <span className="b02-row__check">
                  <Tooltip
                    title={
                      checked
                        ? "Marked attempted — will be included in the Tier-2 escalation handoff."
                        : "Mark this activity as attempted"
                    }
                  >
                    <Checkbox
                      checked={checked}
                      onChange={() => handleToggle(s.step_number)}
                      className="acadia-attempted-checkbox"
                      aria-label={`Mark activity from ${s.incident_number || `row ${s.step_number}`} as attempted`}
                    />
                  </Tooltip>
                </span>

                {/* Column 2 — rowIndex (01 / 02 / 03 …) */}
                <span className="b02-rowIndex">{idx}</span>

                {/* Column 3 — body + bare-id source + codeChip + Why/Outcome */}
                <div>
                  <div className="b02-rowBody">
                    {/* rowSource = bare ticket id, NO "From:" prefix */}
                    {s.incident_number ? (
                      <span className="b02-rowSource">{s.incident_number}</span>
                    ) : null}
                    {s.action}
                  </div>

                  {/* codeChip — tabular numerals + nowrap so a long
                      CLI never wraps mid-command. */}
                  {s.command ? (
                    <div>
                      <span className="b02-codeChip">{s.command}</span>
                    </div>
                  ) : null}

                  {s.intent ? (
                    <div className="b02-rowLink">
                      <span className="b02-rowLink__label">Why:</span>
                      {s.intent}
                    </div>
                  ) : null}
                  {s.pivot ? (
                    <div className="b02-rowLink">
                      <span className="b02-rowLink__label">Outcome:</span>
                      {s.pivot}
                    </div>
                  ) : null}
                </div>
              </li>
            );
          })}
        </ol>
      </div>
    </>
  );
}


// ────────────────────────────────────────────────────────────
// Sprint 13 — Guided Workflow rendering (per-ticket grouped).
// One Collapse panel per cohort ticket; each panel contains that
// ticket's full step playbook with LLM-synthesised Intent + Pivot.
// Sprint 13.12 — kept on disk but no longer rendered. Reinstate
// the JSX call in the main component if the per-ticket view is
// ever wanted again.
// ────────────────────────────────────────────────────────────
function GuidedWorkflowStepItem({ step }) {
  return (
    <li
      key={step.step_number}
      style={{ marginBottom: 12 }}
    >
      <div>
        <Text strong>Relevant Troubleshooting Activity {step.step_number}: </Text>
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
  // Sprint 13.19 — checkbox state lifted to ResolutionJourney so
  // Stage 5's handoff-note POST can read it. Stage 3 is now a
  // controlled passthrough: receives the map, forwards toggles
  // back via the callback. Defaults make it backward-compatible
  // for any callsite that doesn't pass these props yet.
  attemptedStepsByStep3 = {},
  onToggleAttemptedStep,
}) {
  // Sprint 11 — single chat-handoff hook for the whole stage.
  // Shared between every per-ticket Resolution Step "Ask" link so
  // the busy state covers all of them at once.
  const { busy: handoffBusy, askInChat } = useChatHandoff(sessionId);

  // Sprint 13.12 — `consolidated_steps` is the new primary surface:
  // a single LLM-merged 5-step Tier-1 read-only ledger. The Sprint 13
  // per-ticket `guided_workflows` view has been retired from the UI
  // (kept on the schema for back-compat). Per-ticket detail
  // accordion (Sprint 11) is preserved unchanged.
  const hasConsolidated = (
    data && Array.isArray(data.consolidated_steps) && data.consolidated_steps.length > 0
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

  if (!hasConsolidated && !hasDetails) {
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
    <Card
      style={{
        marginBottom: 16,
        borderLeft: "4px solid #2563eb",
        // Premium revamp — per-card Acadia watermark.
        position: "relative",
        overflow: "hidden",
      }}
    >
      <CardWatermark />
      {/* ─── Block 02 — premium heading replacement (heading only) ───
          OLD heading preserved for reference; the new b02-display +
          accent treatment lives below. Watermark + Card wrapper +
          footer (Helpful/Dislike/Escalate/NextStage) UNCHANGED.

          ── OLD ──
          <Title level={5} style={{ marginTop: 0, position: "relative", zIndex: 1 }}>
            Guided Troubleshooting Workflow
          </Title>
          ── /OLD ── */}
      <div className="b02-root" style={{ position: "relative", zIndex: 1, marginTop: 0, marginBottom: 8 }}>
        <div className="b02-eyebrow">
          <span aria-hidden className="b02-eyebrow__dot" />
          Guided Workflow
        </div>
        <h2 className="b02-display">
          Guided troubleshooting <em className="b02-accent">workflow</em>
        </h2>
      </div>

      {/* Sprint 13.12 — single 5-step read-only ledger replaces the
          per-ticket Guided Workflows render. LLM-synthesised; safe
          for a Tier-1 engineer with read-only access. When the
          backend ships an empty list (LLM down or no candidates),
          a small footnote replaces the list and the per-ticket
          detail accordion below still renders. */}
      {hasConsolidated ? (
        <ConsolidatedSteps
          steps={data.consolidated_steps}
          attemptedSteps={attemptedStepsByStep3}
          onToggleAttempt={onToggleAttemptedStep}
        />
      ) : data?.consolidated_synthesis_skipped ? (
        // Sprint 13.13 — "see per-ticket detail below" reference
        // dropped because the per-ticket detail accordion is now
        // commented out. Engineer sees a generic "unavailable" line.
        <Paragraph type="secondary" style={{ fontSize: 12, marginBottom: 12 }}>
          (Consolidated playbook unavailable — please retry or escalate.)
        </Paragraph>
      ) : null}

      {/* Sprint 13.13 — "Per-ticket troubleshooting detail (N)"
          accordion suppressed at the user's request. The Sprint 11
          PerTicketDetails component, the data builder
          `_build_per_ticket_details`, and the `per_ticket_details[]`
          schema field all remain on disk so reinstating is a single
          comment-toggle. Backend still computes and ships the data
          on /stage-3; only the JSX render is commented. */}
      {/*
      {hasDetails ? (
        <PerTicketDetails
          details={data.per_ticket_details}
          maxShown={data.max_details_shown}
          askInChat={askInChat}
          askInChatBusy={handoffBusy}
        />
      ) : null}
      */}

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
