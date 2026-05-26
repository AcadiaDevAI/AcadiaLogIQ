// BackArrowButton — attractive top-left back/return affordance.
//
// Rendered by the Ticket Filter and Connect-to-ServiceNow flows at
// the top-left of their content area. Clicking it triggers the same
// "go to previous screen" path the bottom-right "Return to Stages"
// button uses (App.js' handleReturnFromTicketFilter /
// handleReturnFromServiceNow), so the engineer has two equally good
// exits — bottom-right for a deliberate completion, top-left for a
// quick "this was a wrong turn" bail-out.
//
// Visual language
// ---------------
// Spherical pill matches the four landing quick-action pills
// (RCA/Gap/Filter/ServiceNow). Default tint is iris (#5B8DEF) so
// it reads as a "navigation accent" rather than a destructive
// action. Hover lifts the pill ~1 px with a soft glow shadow and
// slides the arrow left a hair, telegraphing the direction of
// travel. On press the lift collapses, giving a tactile click
// feel. CSS-only — no JS state, no framer-motion dependency.

import React from "react";
import { ArrowLeftOutlined } from "@ant-design/icons";


export default function BackArrowButton({
  onClick,
  // Accessible name only (rendered into aria-label) — no visible
  // text. The icon-only design matches the user's intent for a
  // clean, minimal affordance.
  label = "Go back",
  // Optional override colour. The aurora-iris default is the
  // intended look; callers can pass a different aurora token if a
  // specific screen wants to colour-match its own accent.
  accent = "var(--aurora-2, #5B8DEF)",
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="back-arrow-pill back-arrow-pill--icon-only"
      aria-label={label}
      title={label}
      // Inline styles for the parts that read from CSS variables —
      // the rest (hover lift, glow, micro-translate) lives in the
      // companion `.back-arrow-pill` rules in index.css so hover/
      // active pseudo-classes can target them.
      style={{
        // Translucent aurora wash on a glass surface. Light enough
        // that the screen content behind it reads cleanly, saturated
        // enough that the pill registers as a real button.
        background: `radial-gradient(circle at 30% 30%, ${accent}26, ${accent}0F 75%)`,
        borderColor: `${accent}66`,
        color: accent,
      }}
    >
      <ArrowLeftOutlined className="back-arrow-pill__icon" />
    </button>
  );
}
