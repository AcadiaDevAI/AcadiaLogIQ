// LandingPage — premium hero (uichanges.md Prompt 02).
//
// LOGIC PRESERVED VERBATIM from the pre-revamp version:
//   - useChat() state + dispatch
//   - setSessionMode API call when state.sessionId exists
//   - dispatch SET_MODE { selectedMode, subMode }
//   - 4 main mode options + Troubleshooting sub-mode
//   - submitting state + error toast on failure
//
// EVERYTHING ELSE is presentation. The new look matches the
// "Operational Intelligence Platform" hero artboard:
//   - Eyebrow pill (mono caps, aurora tint)
//   - Instrument Serif headline with italic aurora-gradient accent
//   - Sub-line in Geist
//   - Glass "command card" replacing the two Radio.Group cards
//   - Mode pills across the top, sub-mode selector inline below
//   - Primary CTA "Continue" inherits the global aurora button
//   - Footer ticker strip (4 stats — visual filler, no live data)

import React, { useEffect, useState } from "react";
import { Button, message } from "antd";
import {
  ToolOutlined,
  FileTextOutlined,
  RiseOutlined,
  ApiOutlined,
  ArrowRightOutlined,
  ExperimentOutlined,
  ProfileOutlined,
  FileSearchOutlined,
  ThunderboltOutlined,
  AlertOutlined,
} from "@ant-design/icons";
import { useChat } from "../hooks/ChatContext";
import { setSessionMode } from "../services/api";
import QuickActionsBar from "./QuickActionsBar";


// ─── Logic — unchanged from before the revamp ────────────────────────
const MODE_OPTIONS = [
  {
    value: "troubleshooting",
    label: "Troubleshoot",
    sub: "Resolve incidents using historical data",
    icon: <ToolOutlined />,
    accent: "var(--aurora-1)",
  },
  {
    value: "ticket_handling",
    label: "Ticket handling",
    sub: "Process and route tickets",
    icon: <FileTextOutlined />,
    accent: "var(--aurora-2)",
  },
  {
    value: "escalation",
    label: "Escalate",
    sub: "Package and hand off to Tier 2",
    icon: <RiseOutlined />,
    accent: "var(--p2)",
  },
  {
    value: "vendor_oem",
    label: "Vendor / OEM",
    sub: "Engage external support",
    icon: <ApiOutlined />,
    accent: "var(--aurora-3)",
  },
];

const TROUBLESHOOTING_SUB_OPTIONS = [
  { value: "customer_specific",   label: "Customer-specific" },
  { value: "technology_specific", label: "Technology-specific" },
];


// Entry-selector tiles for the pre-landing view. RCA / Gap / Filter /
// SNOW reuse the same handlers the QuickActionsBar pills fire (so
// picking RCA and clicking Continue opens the same popup the top-pill
// RCA button opens). "Proactive / Reactive" is the lone entry that
// advances to the existing Troubleshoot / Ticket-handling / Escalate /
// Vendor-OEM mode-picker view.
const ENTRY_OPTIONS = [
  {
    value: "rca",
    label: "Root Cause Analysis",
    sub: "Walk through the post-incident RCA workflow",
    icon: <ExperimentOutlined />,
    accent: "var(--aurora-2)",
    handlerProp: "onOpenRca",
  },
  {
    value: "gap",
    label: "Gap Analysis",
    sub: "Compare against best-practice runbooks",
    icon: <ProfileOutlined />,
    accent: "var(--aurora-3)",
    handlerProp: "onOpenGapAnalysis",
  },
  {
    value: "escalation_procedure",
    label: "Escalation Procedure",
    sub: "Review the Tier-1 to Tier-2 escalation workflow",
    icon: <AlertOutlined />,
    accent: "var(--p2)",
    handlerProp: "onOpenEscalationProcedure",
  },
  // Sprint 13.36 — Ticket Filter and Connect to ServiceNow tiles
  // suppressed at the user's request. Code preserved here so the
  // entries can be reinstated by un-commenting this block.
  // {
  //   value: "ticket_filter",
  //   label: "Ticket Filter",
  //   sub: "Search the historical ticket corpus",
  //   icon: <FileSearchOutlined />,
  //   accent: "var(--aurora-1)",
  //   handlerProp: "onOpenTicketFilter",
  // },
  // {
  //   value: "snow",
  //   label: "Connect to ServiceNow",
  //   sub: "Pull a ticket directly from ServiceNow",
  //   icon: <ApiOutlined />,
  //   accent: "var(--p2)",
  //   handlerProp: "onOpenServiceNow",
  // },
  {
    value: "proactive_reactive",
    label: "Proactive / Reactive",
    sub: "Begin a Tier-1 triage with the mode picker",
    icon: <ThunderboltOutlined />,
    accent: "var(--aurora-2)",
    handlerProp: null,   // null → advances to the mode-picker view
  },
];


// ─── Style fragments (declared once, reused) ─────────────────────────
const eyebrowStyle = {
  display: "inline-flex",
  alignItems: "center",
  gap: 8,
  padding: "6px 14px",
  borderRadius: 999,
  background: "rgba(30, 79, 175, 0.10)",
  border: "1px solid rgba(30, 79, 175, 0.32)",
  color: "var(--acadia-primary)",
  fontFamily: "var(--font-mono)",
  fontSize: 11,
  fontWeight: 500,
  letterSpacing: "0.12em",
  textTransform: "uppercase",
};

const dotStyle = {
  width: 6, height: 6, borderRadius: "50%",
  background: "var(--acadia-primary)",
  boxShadow: "0 0 8px var(--acadia-primary)",
};

const headlineStyle = {
  fontFamily: "var(--font-display)",
  fontSize: "clamp(40px, 6.5vw, 76px)",
  lineHeight: 1.02,
  letterSpacing: "-0.025em",
  color: "var(--text)",
  margin: "20px 0 16px",
  fontWeight: 400,
  textAlign: "center",
};

const subStyle = {
  fontFamily: "var(--font-body)",
  fontSize: 16.5,
  lineHeight: 1.55,
  color: "var(--text-muted)",
  margin: "0 auto 40px",
  maxWidth: 620,
  textAlign: "center",
};

const commandCardStyle = {
  position: "relative",
  background: "linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02))",
  border: "1px solid var(--border-strong)",
  borderRadius: 20,
  backdropFilter: "blur(24px) saturate(140%)",
  WebkitBackdropFilter: "blur(24px) saturate(140%)",
  boxShadow: "var(--shadow-lg)",
  padding: 28,
};

// The signature aurora glow that sits behind the command card.
const haloStyle = {
  position: "absolute",
  inset: -2,
  background: "var(--aurora)",
  opacity: 0.35,
  borderRadius: 22,
  filter: "blur(20px)",
  zIndex: -1,
  pointerEvents: "none",
};


// ─── Mode pill component ─────────────────────────────────────────────
// `large` boosts the font + padding for the entry-view tiles (RCA /
// Gap / Proactive · Reactive) so they read as primary CTAs. Default
// sizing is preserved for the mode-picker view pills.
function ModePill({ option, active, onClick, large = false }) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: large ? 10 : 8,
        padding: large ? "14px 26px" : "10px 16px",
        borderRadius: 999,
        background: active
          ? `linear-gradient(180deg, ${option.accent}1A, ${option.accent}08)`
          : "rgba(255, 255, 255, 0.03)",
        border: `1px solid ${active ? `${option.accent}55` : "rgba(255,255,255,0.10)"}`,
        color: active ? option.accent : "var(--text-muted)",
        fontFamily: "var(--font-body)",
        fontSize: large ? 16 : 13,
        fontWeight: large ? 600 : 500,
        cursor: "pointer",
        transition: "all 150ms var(--ease-out)",
      }}
      onMouseEnter={(e) => {
        if (!active) {
          e.currentTarget.style.background = "rgba(255, 255, 255, 0.06)";
          e.currentTarget.style.color = "var(--text)";
        }
      }}
      onMouseLeave={(e) => {
        if (!active) {
          e.currentTarget.style.background = "rgba(255, 255, 255, 0.03)";
          e.currentTarget.style.color = "var(--text-muted)";
        }
      }}
    >
      <span
        style={{
          display: "inline-flex",
          alignItems: "center",
          fontSize: large ? 18 : "inherit",
        }}
      >
        {option.icon}
      </span>
      <span>{option.label}</span>
    </button>
  );
}


// ─── Sub-mode pill ───────────────────────────────────────────────────
function SubPill({ option, active, onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        padding: "8px 14px",
        borderRadius: 8,
        background: active
          ? "rgba(124, 237, 229, 0.10)"
          : "rgba(255, 255, 255, 0.025)",
        border: `1px solid ${active ? "rgba(124, 237, 229, 0.40)" : "rgba(255,255,255,0.08)"}`,
        color: active ? "var(--aurora-1)" : "var(--text-muted)",
        fontFamily: "var(--font-body)",
        fontSize: 12.5,
        cursor: "pointer",
        transition: "all 150ms var(--ease-out)",
      }}
    >
      {option.label}
    </button>
  );
}


// ─── Ticker strip — visual filler (no live data wired by design) ─────
const TICKER_STATS = [
  { value: "99.98%", label: "Uptime · 24h" },
  { value: "1.4k",   label: "Tickets resolved" },
  { value: "92%",    label: "First-contact rate" },
  { value: "<4s",    label: "Avg. AI response" },
];


function TickerStrip() {
  return (
    <div
      style={{
        position: "absolute",
        bottom: 24,
        left: "50%",
        transform: "translateX(-50%)",
        display: "flex",
        gap: 0,
        background: "rgba(11, 15, 30, 0.50)",
        border: "1px solid var(--border)",
        borderRadius: 16,
        backdropFilter: "blur(18px) saturate(140%)",
        WebkitBackdropFilter: "blur(18px) saturate(140%)",
        padding: "14px 4px",
        boxShadow: "var(--shadow-md)",
      }}
    >
      {TICKER_STATS.map((stat, i) => (
        <React.Fragment key={stat.label}>
          <div style={{ padding: "0 24px", textAlign: "center" }}>
            <div
              style={{
                fontFamily: "var(--font-display)",
                fontSize: 26,
                lineHeight: 1,
                color: "var(--text)",
                letterSpacing: "-0.01em",
              }}
            >
              {stat.value}
            </div>
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 10.5,
                color: "var(--text-dim)",
                marginTop: 4,
                textTransform: "uppercase",
                letterSpacing: "0.08em",
              }}
            >
              {stat.label}
            </div>
          </div>
          {i < TICKER_STATS.length - 1 && (
            <div
              style={{
                width: 1,
                background: "var(--border)",
                margin: "4px 0",
              }}
            />
          )}
        </React.Fragment>
      ))}
    </div>
  );
}


// ─── Component ───────────────────────────────────────────────────────
export default function LandingPage({
  // Top-pill-bar handlers. Passed in from AppLayout via LandingRouter.
  // Each is optional — pill renders only when its handler is provided.
  onOpenRca,
  onOpenGapAnalysis,
  onOpenEscalationProcedure,
  onOpenTicketFilter,
  onOpenServiceNow,
  // Wired from LandingRouter — fires when the user picks the
  // "Proactive / Reactive" entry tile and clicks Continue. The
  // router switches to its Tier1IntakeForm screen (the intake form
  // where the engineer chooses proactive alert vs. reactive channel).
  // Optional: when missing, the entry falls back to the in-page
  // mode-picker view as a safety net.
  onProactiveReactive,
} = {}) {
  // LOGIC PRESERVED BYTE-FOR-BYTE
  const { state, dispatch } = useChat();

  // Collapse the left sidebar on mount so the landing page renders
  // with the 56-px rail instead of the full 280-px panel. Fires once
  // per landing-page mount; the engineer can still hover-peek or
  // click to expand. Cleanup is intentionally omitted — expanding
  // the sidebar is the user's explicit action and should persist
  // after this component unmounts.
  useEffect(() => {
    dispatch({ type: "SET_SIDEBAR", payload: false });
  }, [dispatch]);

  // Two-step landing. The new "entry" view is the pre-landing tile
  // grid (RCA / Gap / Proactive-Reactive). Clicking a tile fires its
  // action immediately — no Continue button. Proactive/Reactive
  // routes to the Tier-1 intake form; the other tiles open their
  // corresponding popup (the same popup the top QuickActionsBar pill
  // would have opened).
  const [view, setView] = useState("entry");

  const [mode, setMode] = useState(null);
  const [subMode, setSubMode] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  const needsSubMode = mode === "troubleshooting";
  const canContinue = !!mode && (!needsSubMode || !!subMode);

  const entryHandlers = {
    onOpenRca,
    onOpenGapAnalysis,
    onOpenEscalationProcedure,
    onOpenTicketFilter,
    onOpenServiceNow,
  };
  const handleEntrySelect = (opt) => {
    if (!opt) return;
    if (opt.value === "proactive_reactive") {
      // Hand off to the router so it can swap to the Tier-1 intake
      // form (the Proactive / Reactive screen). Fall back to the
      // in-page mode-picker view only when the router didn't wire
      // the callback — keeps the panel functional in legacy mounts.
      if (typeof onProactiveReactive === "function") {
        onProactiveReactive();
      } else {
        setView("mode");
      }
      return;
    }
    const fn = opt.handlerProp ? entryHandlers[opt.handlerProp] : null;
    if (typeof fn === "function") {
      fn();
    } else {
      message.warning("This action is not available right now.");
    }
  };

  const handleContinue = async () => {
    if (!canContinue) return;
    setSubmitting(true);
    try {
      if (state.sessionId) {
        await setSessionMode(state.sessionId, {
          selectedMode: mode,
          subMode: subMode || null,
        });
      }
      dispatch({
        type: "SET_MODE",
        payload: { selectedMode: mode, subMode: subMode || null },
      });
    } catch (err) {
      message.error("Could not save your selection. Please try again.");
      setSubmitting(false);
      return;
    }
    setSubmitting(false);
  };

  // ─── Pre-landing entry view ─────────────────────────────────────────
  // Shown first. Five selectable tiles + Continue. The QuickActionsBar
  // is intentionally OMITTED here — the same four actions are in the
  // tile grid below, so showing them at the top too would be
  // duplicate. All other screens still get the QuickActionsBar.
  if (view === "entry") {
    return (
      <div
        style={{
          position: "relative",
          flex: 1,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          padding: "48px 8px 120px",
          minHeight: 0,
          overflow: "auto",
        }}
      >
        <div style={{ width: "100%", maxWidth: 1200, position: "relative" }}>
          {/* Acadia watermark — sits above the eyebrow as the brand
              mark for this landing surface. */}
          <div style={{ display: "flex", justifyContent: "center", marginBottom: 18 }}>
            <img
              src="/logo.png"
              alt="Acadia"
              style={{
                width: 110,
                height: "auto",
                opacity: 0.92,
                userSelect: "none",
                display: "block",
              }}
              draggable={false}
            />
          </div>

          {/* Eyebrow */}
          <div style={{ display: "flex", justifyContent: "center" }}>
            <div style={eyebrowStyle}>
              <span style={dotStyle} />
              Operational Intelligence Platform
            </div>
          </div>

          {/* Headline */}
          <h1 style={headlineStyle}>
            Resolve incidents like{" "}
            <em
              className="aurora-text"
              style={{ fontStyle: "italic", fontWeight: 400 }}
            >
              your best engineer
            </em>{" "}
            on best day.
          </h1>

          {/* Sub-line */}
          <p style={subStyle}>
            Pick an entry point. RCA, and Gap Analysis open as focused popups.
            <br />
            Proactive / Reactive advances to the full Tier-1 mode picker.
          </p>

          {/* Glass command card — same chrome as the mode-picker view */}
          <div style={{ position: "relative" }}>
            <div style={haloStyle} aria-hidden />

            <div style={commandCardStyle}>
              {/* Entry tiles — centred and large so the three CTAs
                  read as the primary affordance on the page. */}
              <div
                style={{
                  display: "flex",
                  gap: 12,
                  flexWrap: "wrap",
                  justifyContent: "center",
                  marginBottom: 24,
                }}
              >
                {ENTRY_OPTIONS.map((opt) => (
                  <ModePill
                    key={opt.value}
                    option={opt}
                    active={false}
                    onClick={() => handleEntrySelect(opt)}
                    large
                  />
                ))}
              </div>
            </div>
          </div>

          {/* Footer hint */}
          <p
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: 10.5,
              color: "var(--text-dim)",
              textAlign: "center",
              marginTop: 24,
              textTransform: "uppercase",
              letterSpacing: "0.1em",
            }}
          >
            Your selection sets the working context for this session.
          </p>
        </div>

        {/* Ticker strip suppressed at the user's request.
            Reinstate by un-commenting the line below.
            <TickerStrip /> */}
      </div>
    );
  }

  // ─── Existing mode-picker view (reached only via Proactive/Reactive) ─
  return (
    <div
      style={{
        position: "relative",
        flex: 1,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        padding: "48px 8px 120px",
        minHeight: 0,
        overflow: "auto",
      }}
    >
      <div style={{ width: "100%", maxWidth: 1200, position: "relative" }}>
        {/* Horizontal quick-action pills — RCA / Gap / Filter /
            ServiceNow. Lives above the eyebrow, tinted to match the
            atmospheric watercolor blobs (iris / violet / teal / amber). */}
        <QuickActionsBar
          onOpenRca={onOpenRca}
          onOpenGapAnalysis={onOpenGapAnalysis}
          onOpenEscalationProcedure={onOpenEscalationProcedure}
          onOpenTicketFilter={onOpenTicketFilter}
          onOpenServiceNow={onOpenServiceNow}
        />

        {/* Acadia watermark — sits above the eyebrow as the brand mark
            for this landing surface. */}
        <div style={{ display: "flex", justifyContent: "center", marginBottom: 18 }}>
          <img
            src="/logo.png"
            alt="Acadia"
            style={{
              width: 110,
              height: "auto",
              opacity: 0.92,
              userSelect: "none",
              display: "block",
            }}
            draggable={false}
          />
        </div>

        {/* Eyebrow */}
        <div style={{ display: "flex", justifyContent: "center" }}>
          <div style={eyebrowStyle}>
            <span style={dotStyle} />
            Operational Intelligence Platform
          </div>
        </div>

        {/* Headline — Instrument Serif with italic-aurora accent */}
        <h1 style={headlineStyle}>
          Resolve incidents like{" "}
          <em
            className="aurora-text"
            style={{ fontStyle: "italic", fontWeight: 400 }}
          >
            your best engineer
          </em>{" "}
          on best day.
        </h1>

        {/* Sub-line */}
        <p style={subStyle}>
          LogIQ pairs structured incident memory with an AI that
          watches every Tier-1 step. Pick how you want to work. The rest is
          decided in seconds, not minutes.
        </p>

        {/* The glass command card — aurora halo behind it */}
        <div style={{ position: "relative" }}>
          <div style={haloStyle} aria-hidden />

          <div style={commandCardStyle}>
            {/* Mode pills row */}
            <div
              style={{
                display: "flex",
                gap: 8,
                flexWrap: "wrap",
                marginBottom: needsSubMode ? 16 : 24,
              }}
            >
              {MODE_OPTIONS.map((opt) => (
                <ModePill
                  key={opt.value}
                  option={opt}
                  active={mode === opt.value}
                  onClick={() => {
                    setMode(opt.value);
                    setSubMode(null);
                  }}
                />
              ))}
            </div>

            {/* Sub-mode strip — only when troubleshooting */}
            {needsSubMode && (
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  marginBottom: 20,
                  paddingBottom: 16,
                  borderBottom: "1px solid var(--border)",
                }}
              >
                <span
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: 10.5,
                    color: "var(--text-dim)",
                    textTransform: "uppercase",
                    letterSpacing: "0.1em",
                    marginRight: 4,
                  }}
                >
                  Context
                </span>
                {TROUBLESHOOTING_SUB_OPTIONS.map((opt) => (
                  <SubPill
                    key={opt.value}
                    option={opt}
                    active={subMode === opt.value}
                    onClick={() => setSubMode(opt.value)}
                  />
                ))}
              </div>
            )}

            {/* Description of what's currently selected */}
            {mode && (
              <p
                style={{
                  fontFamily: "var(--font-body)",
                  fontSize: 13.5,
                  color: "var(--text-muted)",
                  margin: "0 0 18px",
                  lineHeight: 1.5,
                }}
              >
                {MODE_OPTIONS.find((m) => m.value === mode)?.sub}
                {needsSubMode && subMode && (
                  <>
                    {" · "}
                    <span style={{ color: "var(--aurora-1)" }}>
                      {
                        TROUBLESHOOTING_SUB_OPTIONS.find((s) => s.value === subMode)
                          ?.label
                      }
                    </span>
                  </>
                )}
              </p>
            )}

            {/* CTA row */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                gap: 12,
                paddingTop: 4,
                borderTop: "1px solid var(--border)",
                marginTop: needsSubMode ? 4 : 12,
                paddingTopShim: 16,
              }}
            >
              <div
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: 11,
                  color: "var(--text-dim)",
                  textTransform: "uppercase",
                  letterSpacing: "0.1em",
                  paddingTop: 16,
                }}
              >
                {canContinue ? "Ready" : "Pick a mode to continue"}
              </div>
              <div style={{ paddingTop: 12 }}>
                <Button
                  type="primary"
                  size="large"
                  onClick={handleContinue}
                  disabled={!canContinue || submitting}
                  loading={submitting}
                  icon={!submitting ? <ArrowRightOutlined /> : null}
                  iconPosition="end"
                >
                  {submitting ? "Saving…" : "Continue"}
                </Button>
              </div>
            </div>
          </div>
        </div>

        {/* Footer hint */}
        <p
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 10.5,
            color: "var(--text-dim)",
            textAlign: "center",
            marginTop: 24,
            textTransform: "uppercase",
            letterSpacing: "0.1em",
          }}
        >
          Your selection sets the working context for this session.
        </p>
      </div>

      {/* Ticker strip suppressed at the user's request.
          Reinstate by un-commenting the line below.
          <TickerStrip /> */}
    </div>
  );
}
