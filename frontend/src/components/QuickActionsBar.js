// QuickActionsBar — horizontal row of vibrant aurora-tinted pill
// buttons. Used on every "landing" surface (LandingPage modes-picker,
// Tier1IntakeForm split view, and any future entry point) so the
// engineer always has one-click access to RCA / Gap Analysis /
// Ticket Filter / Connect to ServiceNow.
//
// Design:
//   - Spherical (border-radius 9999px) — never feels boxy
//   - Watercolor fill: soft gradient in the action's accent color,
//     mirrors the atmospheric mesh on the page backdrop
//   - Each pill tinted to a different aurora hue:
//       RCA            → iris    #5B8DEF
//       Gap Analysis   → violet  #A78BFA
//       Ticket Filter  → teal    #7CEDE5 / #0BA89F
//       Connect to SN  → amber   #FFB347 / #D88A1A
//   - Hover: gradient intensifies, ring brightens, soft colored glow,
//     1px lift
//
// Each pill renders only when its matching handler prop is provided —
// callers that haven't wired a handler get a graceful no-op (the pill
// is simply skipped) rather than a crash.
//
// Pure presentation: no business logic, no API calls, no state.

import React from "react";
import {
  ExperimentOutlined,
  ProfileOutlined,
  FileSearchOutlined,
  ApiOutlined,
} from "@ant-design/icons";


const QUICK_ACTIONS = [
  {
    key: "rca",
    label: "Root Cause Analysis",
    icon: <ExperimentOutlined />,
    accent: {
      c: "#5B8DEF",
      soft: "rgba(91, 141, 239, 0.14)",
      hoverFill: "rgba(91, 141, 239, 0.28)",
      ring: "rgba(91, 141, 239, 0.45)",
    },
    handlerProp: "onOpenRca",
  },
  {
    key: "gap",
    label: "Gap Analysis",
    icon: <ProfileOutlined />,
    accent: {
      c: "#A78BFA",
      soft: "rgba(167, 139, 250, 0.14)",
      hoverFill: "rgba(167, 139, 250, 0.28)",
      ring: "rgba(167, 139, 250, 0.45)",
    },
    handlerProp: "onOpenGapAnalysis",
  },
  // Sprint 13.36 — Ticket Filter and Connect to ServiceNow pills
  // suppressed at the user's request. Code preserved so the entries
  // can be reinstated by un-commenting this block.
  // {
  //   key: "filter",
  //   label: "Ticket Filter",
  //   icon: <FileSearchOutlined />,
  //   accent: {
  //     c: "#0BA89F",
  //     soft: "rgba(124, 237, 229, 0.18)",
  //     hoverFill: "rgba(124, 237, 229, 0.34)",
  //     ring: "rgba(11, 168, 159, 0.45)",
  //   },
  //   handlerProp: "onOpenTicketFilter",
  // },
  // {
  //   key: "snow",
  //   label: "Connect to ServiceNow",
  //   icon: <ApiOutlined />,
  //   accent: {
  //     c: "#D88A1A",
  //     soft: "rgba(255, 179, 71, 0.18)",
  //     hoverFill: "rgba(255, 179, 71, 0.34)",
  //     ring: "rgba(216, 138, 26, 0.45)",
  //   },
  //   handlerProp: "onOpenServiceNow",
  // },
];


function QuickActionPill({ action, onClick }) {
  const a = action.accent;
  const idleBg = `linear-gradient(135deg, ${a.soft}, rgba(255,255,255,0.40))`;
  const hoverBg = `linear-gradient(135deg, ${a.hoverFill}, ${a.soft})`;

  return (
    <button
      type="button"
      onClick={onClick}
      onMouseEnter={(e) => {
        e.currentTarget.style.transform = "translateY(-1px)";
        e.currentTarget.style.background = hoverBg;
        e.currentTarget.style.borderColor = a.ring;
        e.currentTarget.style.boxShadow =
          `0 8px 22px -8px ${a.ring}, 0 0 0 1px ${a.ring} inset`;
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.transform = "translateY(0)";
        e.currentTarget.style.background = idleBg;
        e.currentTarget.style.borderColor = `${a.c}40`;
        e.currentTarget.style.boxShadow = "0 1px 3px rgba(10, 16, 24, 0.06)";
      }}
      style={{
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        gap: 8,
        padding: "10px 18px",
        height: 40,
        // Uniform pill width — sized to fit the longest label
        // ("Connect to ServiceNow") so every pill in the bar reads
        // as the same size regardless of label length.
        minWidth: 220,
        borderRadius: 9999,
        background: idleBg,
        backdropFilter: "blur(10px) saturate(150%)",
        WebkitBackdropFilter: "blur(10px) saturate(150%)",
        border: `1px solid ${a.c}40`,
        color: a.c,
        fontFamily: "var(--font-body, 'Geist', 'Inter', system-ui, sans-serif)",
        fontSize: 13.5,
        fontWeight: 600,
        letterSpacing: "0.01em",
        cursor: "pointer",
        whiteSpace: "nowrap",
        boxShadow: "0 1px 3px rgba(10, 16, 24, 0.06)",
        transition:
          "transform 180ms cubic-bezier(0.16, 1, 0.3, 1), " +
          "background 220ms cubic-bezier(0.16, 1, 0.3, 1), " +
          "border-color 180ms cubic-bezier(0.16, 1, 0.3, 1), " +
          "box-shadow 220ms cubic-bezier(0.16, 1, 0.3, 1)",
      }}
    >
      <span style={{ display: "inline-flex", alignItems: "center", fontSize: 15 }}>
        {action.icon}
      </span>
      <span>{action.label}</span>
    </button>
  );
}


/**
 * QuickActionsBar
 *
 * Props:
 *   onOpenRca, onOpenGapAnalysis, onOpenTicketFilter, onOpenServiceNow
 *     — optional click handlers. Each pill renders only if its
 *       matching handler is a function.
 *   align — "center" (default) | "start" | "end"
 *   marginBottom — number; defaults to 32px (below the bar to the
 *       next content row)
 */
export default function QuickActionsBar({
  align = "center",
  marginBottom = 32,
  ...handlers
}) {
  const items = QUICK_ACTIONS.filter(
    (a) => typeof handlers[a.handlerProp] === "function",
  );
  if (items.length === 0) return null;

  const justify =
    align === "start" ? "flex-start" :
    align === "end" ? "flex-end" : "center";

  return (
    <div
      style={{
        display: "flex",
        flexWrap: "wrap",
        gap: 12,
        justifyContent: justify,
        marginBottom,
      }}
      aria-label="Quick actions"
    >
      {items.map((action) => (
        <QuickActionPill
          key={action.key}
          action={action}
          onClick={handlers[action.handlerProp]}
        />
      ))}
    </div>
  );
}
