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
import { Alert, Button, Card, Modal, Space, Spin, Tag, Tooltip, Typography, message } from "antd";
import { CopyOutlined, LoadingOutlined, ReloadOutlined } from "@ant-design/icons";

import { useTheme } from "../../../hooks/ThemeContext";

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
  getActivityVersion,
  subscribeActivityVersion,
} from "./journeyApi";

// Sprint 13.30 — stale-note pulse animation. Injected once on first
// import via a module-level guard so the keyframes exist for the
// entire app lifetime, not re-injected per render. The
// `prefers-reduced-motion: reduce` media query disables the
// animation for users who've opted out (WCAG 2.3.3 + browser
// vestibular-trigger settings) — the dot still appears as a static
// indicator, just without the visual pulse.
let _acadiaPulseStylesInjected = false;
function _ensurePulseStyles() {
  if (typeof document === "undefined" || _acadiaPulseStylesInjected) return;
  const style = document.createElement("style");
  style.setAttribute("data-acadia-pulse", "1");
  style.textContent = `
    @keyframes acadia-pulse-dot {
      0%, 100% { transform: scale(1);   opacity: 1;    box-shadow: 0 0 0 0 rgba(11, 49, 92, 0.55); }
      50%      { transform: scale(1.15); opacity: 0.85; box-shadow: 0 0 0 6px rgba(11, 49, 92, 0); }
    }
    .acadia-stale-dot {
      animation: acadia-pulse-dot 1.4s ease-in-out infinite;
    }
    @media (prefers-reduced-motion: reduce) {
      .acadia-stale-dot { animation: none !important; }
    }
  `;
  document.head.appendChild(style);
  _acadiaPulseStylesInjected = true;
}

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
function HandoffNoteAction({ sessionId, attemptedStage3Steps, onRefreshRouting }) {
  // Sprint 13.32.13 — dark-theme fix. The handoff-note <pre> below
  // had `background: var(--surface-2, #fafafa)`; `--surface-2` is
  // undefined in this app's theme system so the fallback (#fafafa)
  // applied in every theme, leaving the Operational Handoff content
  // block solid white in dark mode while text colour swapped to
  // light. We now derive bg + text from the live theme; light theme
  // keeps the original #fafafa exactly.
  const { isDark } = useTheme();
  const notePreBg = isDark ? "#16161d" : "#fafafa";
  const notePreText = isDark ? "#e2e8f0" : "inherit";
  const notePreBorder = isDark ? "#2a2a3d" : "#f0f0f0";

  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState(null);
  const [usedFallback, setUsedFallback] = useState(false);

  // Sprint 13.30 — stale-note detection. `noteVersion` snapshots the
  // journey activity counter at fetch time; `currentActivityVersion`
  // tracks the live counter via subscription. When they diverge,
  // the note no longer reflects the engineer's latest journey state.
  // Initial null on noteVersion → first fetch hasn't completed yet →
  // never stale (don't false-flag a still-loading note).
  const [noteVersion, setNoteVersion] = useState(null);
  const [currentActivityVersion, setCurrentActivityVersion] = useState(
    () => getActivityVersion(),
  );
  React.useEffect(() => {
    _ensurePulseStyles();
    const unsubscribe = subscribeActivityVersion((v) => {
      setCurrentActivityVersion(v);
    });
    return unsubscribe;
  }, []);
  const isStale =
    noteVersion !== null
    && currentActivityVersion > noteVersion
    && !!note
    && !busy;

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
    if (!sessionId) return "";
    setBusy(true);
    try {
      // Sprint 13.24 PERF — auto-fetches use the cache (fast);
      // Regenerate button passes force=true to bust it server-side.
      const data = await generateEscalationHandoffNote(
        sessionId,
        attemptedStepNumbers,
        { force: !!opts.force },
      );
      const fresh = data?.note || "";
      setNote(fresh);
      setUsedFallback(!!data?.used_fallback);
      // Sprint 13.30 — stamp the version AT successful fetch
      // completion. Any bump that lands after this stamps a higher
      // version → next render flips isStale=true. Stamping at
      // completion (vs. start) intentionally treats events that
      // arrived *during* the fetch as "after" the note — those
      // weren't reflected in this run, so they should still trigger
      // the stale signal.
      setNoteVersion(getActivityVersion());
      return fresh;
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error("[journey.handoff_note] generate failed", err);
      message.error("Could not generate the escalation note. Please try again.");
      return "";
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

  // Sprint 13.30 — Regenerate is the hard-reload path: bust caches
  // server-side, refetch routing, and refetch the note. Used both by
  // the Regenerate button and by the Copy-while-stale modal's
  // "Regenerate first" CTA. Returns the fresh note string so the
  // caller (e.g. Copy-while-stale) can chain a clipboard write
  // without racing setState.
  const regenerate = React.useCallback(async () => {
    if (typeof onRefreshRouting === "function") {
      onRefreshRouting();
    }
    return fetchNote({ force: true });
  }, [fetchNote, onRefreshRouting]);

  const doCopy = async (text) => {
    const target = (typeof text === "string" ? text : note) || "";
    if (!target) return;
    try {
      await navigator.clipboard.writeText(target);
      message.success("Escalation note copied.");
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn("[journey.handoff_note] clipboard write failed", err);
      message.warning("Copy failed — select the text manually.");
    }
  };

  // Sprint 13.30 — Copy guardrail. When stale, show a confirm modal
  // with two CTAs:
  //   - "Regenerate & copy" (default, primary, Acadia color) →
  //     regen + auto-copy the freshly returned note.
  //   - "Copy current" (secondary) → copy the existing stale note
  //     as-is. Engineer's choice; we don't block them outright.
  // Catches the failure mode where the engineer's eyes miss the
  // pulsing dot and they paste a stale note into the Tier-2 ticket.
  const handleCopy = () => {
    if (!note) return;
    if (!isStale) {
      doCopy();
      return;
    }
    Modal.confirm({
      title: "Updates available",
      content:
        "Activity has occurred in this journey since this note was generated. "
        + "Regenerate to refresh, or copy the current note as-is.",
      okText: "Regenerate & copy",
      cancelText: "Copy current",
      okButtonProps: {
        type: "primary",
        style: {
          background: "var(--acadia-primary)",
          borderColor: "var(--acadia-primary)",
        },
      },
      onOk: async () => {
        const fresh = await regenerate();
        await doCopy(fresh);
      },
      onCancel: () => doCopy(),
    });
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
              and triggers a fresh LLM call.
              Sprint 13.30 — Regenerate is a hard reload of the
              entire Stage 5 view (note + routing + cohort cache)
              and gains a stale-state affordance: when journey
              activity has occurred since the last successful note
              fetch (`isStale === true`), the button switches from
              the default outlined style to an Acadia-primary fill,
              shows a pulsing dot indicator (CSS keyframes; honors
              prefers-reduced-motion), and surfaces a tooltip
              explaining what to do. The non-stale (steady) state
              is the original Sprint 13.24 button — visually
              identical to before, so engineers who never trigger
              activity past Stage 5 see no behavioural change. */}
          <Tooltip
            title={
              isStale
                ? "Activity has occurred in this journey since this note was generated. Click to refresh."
                : "Regenerate the handoff note (bypasses cache)."
            }
          >
            <Button
              size="small"
              onClick={regenerate}
              disabled={busy}
              icon={busy ? <LoadingOutlined /> : <ReloadOutlined />}
              type={isStale ? "primary" : "default"}
              style={
                isStale
                  ? {
                      background: "var(--acadia-primary)",
                      borderColor: "var(--acadia-primary)",
                      color: "#fff",
                      position: "relative",
                      paddingRight: 18,
                    }
                  : { position: "relative" }
              }
            >
              {isStale ? "Refresh" : "Regenerate"}
              {isStale ? (
                <span
                  className="acadia-stale-dot"
                  aria-hidden="true"
                  style={{
                    position: "absolute",
                    top: -3,
                    right: -3,
                    width: 9,
                    height: 9,
                    borderRadius: "50%",
                    background: "var(--acadia-primary, #0b315c)",
                    border: "2px solid var(--bg-secondary, #fff)",
                    pointerEvents: "none",
                  }}
                />
              ) : null}
            </Button>
          </Tooltip>
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
            background: notePreBg,
            color: notePreText,
            border: `1px solid ${notePreBorder}`,
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
  // Sprint 13.30 — fetcher lifted to a useCallback so the child
  // HandoffNoteAction's Regenerate button can invoke it (alongside
  // the note refetch) for a hard-reload of the whole Stage 5 view.
  // The auto-fetch on mount is preserved verbatim — only Regenerate
  // gains an extra trigger path.
  const [routing, setRouting] = useState(null);
  const fetchRouting = React.useCallback(async () => {
    if (!sessionId) return;
    try {
      const r = await fetchEscalationRouting(sessionId);
      setRouting(r);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn("[journey.escalation_routing] fetch failed", err);
    }
  }, [sessionId]);
  useEffect(() => {
    fetchRouting();
  }, [fetchRouting]);

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
        onRefreshRouting={fetchRouting}
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
