// Sprint 10 Stage 5 — Operational Handoff (Escalate to Tier 2 view).
//
// Sprint 12.7 — adds the Escalation Routing & Vendor/OEM Engagement
// section (deduped Resolution_Groups + Team_Paths + recommended
// Tier-2 entry candidates) and the "Generate Tier 2 Escalation
// Handoff" action.
//
// Sprint 12.9 — the Sprint 7 EscalationPackageCard render is now
// suppressed. Clicking "Escalate to Tier 2" from any stage lands
// the engineer here, and "here" is now exactly two things:
//   1. Escalation Routing & Vendor/OEM Engagement (data view)
//   2. Generate Tier 2 Escalation Handoff (LLM-backed action)
// The Sprint 7 package render + its `expanded`/`autoExpand` toggle
// + the EscalationPackageCard import are preserved as commented
// blocks; reinstate by un-commenting all three together.

import React, { useEffect, useState } from "react";
import { Alert, Button, Card, Space, Spin, Tag, Typography, message } from "antd";
import { CopyOutlined, LoadingOutlined } from "@ant-design/icons";

// Sprint 12.9 — Sprint 7's full EscalationPackageCard render is
// suppressed. Clicking "Escalate to Tier 2" now lands the engineer
// on the Escalation Routing & Vendor/OEM Engagement view + the
// Generate Tier 2 Escalation Handoff action. Reinstate the import
// (and the JSX block below) if the Sprint 7 package view is ever
// brought back.
// import EscalationPackageCard from "../EscalationPackageCard";
import DislikeButton from "./DislikeButton";
import HelpfulButton from "./HelpfulButton";
import {
  fetchEscalationRouting,
  generateEscalationHandoffNote,
} from "./journeyApi";

const { Title, Paragraph, Text } = Typography;


// ────────────────────────────────────────────────────────────
// Routing section — deduped groups, paths, and Tier-2 candidates.
// Hidden when the cohort carries no routing data at all so the
// engineer never sees an empty stub.
// ────────────────────────────────────────────────────────────
function EscalationRoutingSection({ routing }) {
  if (!routing || routing.empty) return null;

  return (
    <div style={{ marginTop: 16 }}>
      <Title level={5} style={{ marginTop: 0, marginBottom: 8 }}>
        Escalation Routing & Vendor/OEM Engagement
      </Title>

      {routing.recommended_tier2_teams && routing.recommended_tier2_teams.length > 0 ? (
        <div style={{ marginBottom: 12 }}>
          <Text strong>Recommended Tier 2 entry: </Text>
          <Space size={[4, 4]} wrap style={{ marginTop: 4 }}>
            {routing.recommended_tier2_teams.map((t) => (
              <Tag
                key={t.team}
                color={t.occurrence_count > 1 ? "red" : "volcano"}
                title={
                  t.example_path
                    ? `Seen in ${t.occurrence_count} ticket${t.occurrence_count === 1 ? "" : "s"} via path: ${t.example_path}`
                    : undefined
                }
              >
                {t.team}
                {t.occurrence_count > 1 ? ` ×${t.occurrence_count}` : ""}
              </Tag>
            ))}
          </Space>
        </div>
      ) : null}

      {routing.team_paths && routing.team_paths.length > 0 ? (
        <div style={{ marginBottom: 12 }}>
          <Text strong>Historical team paths:</Text>
          <ul style={{ marginTop: 4, marginBottom: 0, paddingLeft: 20 }}>
            {routing.team_paths.map((p) => (
              <li key={p}>
                <Text code>{p}</Text>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {routing.resolution_groups && routing.resolution_groups.length > 0 ? (
        <div style={{ marginBottom: 12 }}>
          <Text strong>Resolution groups across cohort:</Text>{" "}
          <Space size={[4, 4]} wrap style={{ marginTop: 4 }}>
            {routing.resolution_groups.map((g) => (
              <Tag key={g} color="geekblue">
                {g}
              </Tag>
            ))}
          </Space>
        </div>
      ) : null}

      {/* Vendor / OEM block — wired forward-compat. The corpus
          today carries 0 Vendor_OEM_Engagement records. When
          ingestion populates the field, this block lights up
          automatically. Honest empty-state copy meanwhile. */}
      {routing.forensic_data_required && routing.forensic_data_required.length > 0 ? (
        <div style={{ marginBottom: 12 }}>
          <Text strong>Forensic data required before vendor/OEM engagement:</Text>
          <ul style={{ marginTop: 4, marginBottom: 0, paddingLeft: 20 }}>
            {routing.forensic_data_required.map((f) => (
              <li key={f}>{f}</li>
            ))}
          </ul>
        </div>
      ) : routing.vendor_records && routing.vendor_records.length === 0 ? (
        // Sprint 13.23 — promoted from gray secondary text to bold
        // black so the Tier-2 reader's eye lands on it. This line
        // was previously also duplicated as a bold residual at the
        // bottom of the copyable note; the duplicate has been
        // removed in favour of this single canonical placement.
        <Paragraph style={{ marginBottom: 12 }}>
          <Text strong style={{ color: "var(--text-primary, #000)" }}>
            No vendor/OEM engagement records found in this cohort.
          </Text>
        </Paragraph>
      ) : null}

      {routing.cohort_size ? (
        <Paragraph style={{ marginBottom: 0, marginTop: 4 }}>
          <Text type="secondary" style={{ fontSize: 12 }}>
            Aggregated from {routing.tickets_with_data} of{" "}
            {routing.cohort_size} cohort ticket
            {routing.cohort_size === 1 ? "" : "s"}.
          </Text>
        </Paragraph>
      ) : null}
    </div>
  );
}


// ────────────────────────────────────────────────────────────
// Sprint 13.17 — Auto-fetched Tier-2 handoff note. Engineer arrives
// at the Operational Handoff panel (Stage 5) and the note loads
// automatically — no Generate button. UX rationale: when an
// engineer is escalating in a crisis, the report should be ready
// the moment they land here, not behind another click.
//
// The backend assembles the note from:
//   - Cohort tickets (incident-number list + executive summaries)
//   - Stage 3 consolidated read-only steps Tier-1 saw
//   - Traversal log (which stages Tier-1 actually visited)
//   - Stage 5 routing aggregation (Resolution_Groups + Team_Path +
//     Vendor_OEM forensic data)
//
// Failure-open: backend always returns a usable note. When
// `used_fallback=true`, an Alert tells the engineer the diagnostic
// sentence is heuristic and offers Regenerate.
// ────────────────────────────────────────────────────────────
function HandoffNoteAction({ sessionId, attemptedStage3Steps }) {
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState(null);
  const [usedFallback, setUsedFallback] = useState(false);

  // Sprint 13.19 — extract the ticked step numbers from the lifted
  // map. JSON.stringify on the dep array keeps useCallback stable
  // across re-renders that don't actually change the tick state.
  const attemptedStepNumbers = React.useMemo(() => {
    if (!attemptedStage3Steps) return [];
    return Object.entries(attemptedStage3Steps)
      .filter(([, v]) => !!v)
      .map(([k]) => Number(k))
      .filter((n) => Number.isFinite(n) && n > 0);
  }, [attemptedStage3Steps]);

  const fetchNote = React.useCallback(async (opts = {}) => {
    if (!sessionId) return;
    setBusy(true);
    try {
      // Sprint 13.24 PERF — auto-fetches use the cache (fast);
      // Regenerate button passes force=true to bust it server-side.
      const data = await generateEscalationHandoffNote(
        sessionId,
        attemptedStepNumbers,
        { force: !!opts.force },
      );
      setNote(data?.note || "");
      setUsedFallback(!!data?.used_fallback);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error("[journey.handoff_note] generate failed", err);
      message.error("Could not generate the escalation note. Please try again.");
    } finally {
      setBusy(false);
    }
  }, [sessionId, attemptedStepNumbers]);

  // Auto-fetch on mount + whenever sessionId or the ticked-step set
  // changes. Re-fetching on tick changes means engineers can toggle
  // boxes after landing on Stage 5 and Regenerate to refresh.
  React.useEffect(() => {
    fetchNote();
  }, [fetchNote]);

  const handleCopy = async () => {
    if (!note) return;
    try {
      await navigator.clipboard.writeText(note);
      message.success("Escalation note copied.");
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn("[journey.handoff_note] clipboard write failed", err);
      message.warning("Copy failed — select the text manually.");
    }
  };

  return (
    <div style={{ marginTop: 16 }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: 8,
          gap: 8,
          flexWrap: "wrap",
        }}
      >
        {/* Sprint 13.21 — header relabeled at user's request. */}
        <Text strong>Operational Handoff</Text>
        <Space>
          <Button
            size="small"
            icon={<CopyOutlined />}
            onClick={handleCopy}
            disabled={busy || !note}
          >
            Copy
          </Button>
          {/* Sprint 13.24 — Regenerate restored. Click bypasses the
              session-keyed consolidated-ledger cache (force=true)
              and triggers a fresh LLM call. Useful when the engineer
              wants the note re-synthesised after the Haiku model
              produced an off-target diagnostic phrasing. */}
          <Button
            size="small"
            onClick={() => fetchNote({ force: true })}
            disabled={busy}
            icon={busy ? <LoadingOutlined /> : null}
          >
            Regenerate
          </Button>
        </Space>
      </div>

      {busy && !note ? (
        <div style={{ padding: 24, textAlign: "center" }}>
          <Spin />
          <div
            style={{
              marginTop: 8,
              fontSize: 12,
              color: "var(--text-muted, #6b7280)",
            }}
          >
            Composing the handoff note from Tier-1's journey…
          </div>
        </div>
      ) : null}

      {usedFallback ? (
        <Alert
          type="warning"
          showIcon
          message="LLM unavailable — used template fallback. The diagnostic-summary sentence is heuristic; click Regenerate to retry."
          style={{ marginBottom: 8 }}
        />
      ) : null}

      {note ? (
        <pre
          style={{
            background: "var(--surface-2, #fafafa)",
            border: "1px solid var(--border-color, #f0f0f0)",
            borderRadius: 4,
            padding: 12,
            whiteSpace: "pre-wrap",
            wordBreak: "break-word",
            fontFamily:
              "var(--font-monospace, ui-monospace, SFMono-Regular, Menlo, monospace)",
            fontSize: 13,
            lineHeight: 1.45,
            maxHeight: 560,
            overflow: "auto",
          }}
        >
          {note}
        </pre>
      ) : null}
    </div>
  );
}


export default function Stage5EscalationPackage({
  // Sprint 12.9 — `data` (Sprint 7 Tier1EscalationPackage) and
  // `autoExpand` are kept in the destructure for callsite
  // back-compat, but no longer drive any rendering. The panel now
  // shows only Escalation Routing & Vendor/OEM Engagement + the
  // Generate Tier 2 Escalation Handoff action.
  data,            // eslint-disable-line no-unused-vars
  sessionId,
  onMarkedHelpful,
  onStartNewTicket,
  helpfulMarked,
  autoExpand = true,  // eslint-disable-line no-unused-vars
  // Sprint 13.19 — Stage 3 checkbox state lifted from
  // ResolutionJourney. HandoffNoteAction reads ticked step
  // numbers from this map and posts them with the note request.
  attemptedStage3Steps,
}) {
  // Sprint 12.9 — `expanded` / `autoExpand` toggle suppressed along
  // with the Sprint 7 EscalationPackageCard render. Re-enable if
  // that view is ever reinstated.
  // const [expanded, setExpanded] = useState(autoExpand);

  // Sprint 12.7 — Escalation Routing fetched on mount. Failure here
  // only hides the routing block; the Generate-Handoff button still
  // works. Fire-and-forget; no spinner blocking the panel.
  const [routing, setRouting] = useState(null);
  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;
    (async () => {
      try {
        const r = await fetchEscalationRouting(sessionId);
        if (!cancelled) setRouting(r);
      } catch (err) {
        // eslint-disable-next-line no-console
        console.warn("[journey.escalation_routing] fetch failed", err);
      }
    })();
    return () => { cancelled = true; };
  }, [sessionId]);

  // Sprint 12.9 — `if (!data)` early-out removed. The new view does
  // not depend on the Sprint 7 package payload, so a /stage-5 fetch
  // failure no longer blanks the card. Reinstate alongside the
  // EscalationPackageCard if Sprint 7 view returns.

  return (
    <Card style={{ marginBottom: 16, borderLeft: "4px solid #B03A2E" }}>
      <Title level={5} style={{ marginTop: 0 }}>
        Operational Handoff
      </Title>

      {/* Sprint 12.9 — Sprint 7 EscalationPackageCard render
          suppressed. Clicking "Escalate to Tier 2" now lands the
          engineer on the Escalation Routing & Vendor/OEM Engagement
          view (below) + the Generate Tier 2 Escalation Handoff
          action. The full Sprint 7 package view is preserved on
          disk; uncomment the block below + the EscalationPackageCard
          import + the `expanded`/`autoExpand` state above to
          reinstate. */}
      {/*
      {expanded ? (
        <EscalationPackageCard pkg={data} onClose={() => setExpanded(false)} />
      ) : (
        <Paragraph type="secondary" style={{ marginBottom: 12 }}>
          The escalation package is ready. Open it to copy the
          Tier-2 handoff text.
        </Paragraph>
      )}
      */}

      {/* Sprint 12.7 — Routing section. Hidden when empty.
          Sprint 12.9 — primary content of the Escalate-to-Tier-2 view. */}
      <EscalationRoutingSection routing={routing} />

      {/* Sprint 13.17 — Tier-2 handoff note. Auto-fetches on mount;
          no manual Generate button. The note text is what Tier-2
          will paste into ServiceNow / their ticket system.
          Sprint 13.19 — receives the lifted Stage 3 checkbox state
          so the note's Diagnostic Summary reflects ONLY the steps
          the engineer actually ticked. Re-fetches when ticks change. */}
      <HandoffNoteAction
        sessionId={sessionId}
        attemptedStage3Steps={attemptedStage3Steps}
      />

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
            stage="stage_5"
            onMarkedHelpful={onMarkedHelpful}
            onStartNewTicket={onStartNewTicket}
            disabled={helpfulMarked}
          />
          <DislikeButton
            sessionId={sessionId}
            stage="stage_5"
          />
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {/* Sprint 12.7.1 — "Show escalation package" CTA suppressed:
              with autoExpand=true (default) the package is already
              visible on first paint; the button only appeared after the
              engineer manually collapsed via EscalationPackageCard's
              Close affordance, which is rare on this card. Reinstate
              by un-commenting if the journey is ever switched to a
              collapsed-by-default Stage 5 (autoExpand={false}). */}
          {/*
          {!expanded ? (
            <Button type="primary" onClick={() => setExpanded(true)}>
              Show escalation package
            </Button>
          ) : null}
          */}
          {typeof onStartNewTicket === "function" ? (
            <Button onClick={onStartNewTicket}>New Ticket</Button>
          ) : null}
        </div>
      </div>
    </Card>
  );
}
