// CardWatermark — Acadia logo watermark for journey stage cards.
//
// Drop inside any positioned-relative card to add a per-card Acadia
// mark. Used on the four "data" stages:
//   - Best Historical Match & Recommended Resolution (Stage 2)
//   - Guided Troubleshooting Workflow (Stage 3)
//   - Knowledge Base & SOP Reference (Stage 4)
//   - Operational Handoff (Stage 5)
//
// PLACEMENT:
//   Pinned to the BOTTOM-RIGHT corner of the host card by default.
//   This is the only spot guaranteed to stay visible — Stage 2 packs
//   the body with AntD <Collapse> panels whose own white background
//   would otherwise mask a centered watermark.
//
// USAGE
//   1. Add `style={{ position: "relative", overflow: "hidden" }}`
//      to the host <Card>.
//   2. Render <CardWatermark /> as the FIRST child inside the body.
//   3. Title / content can sit at `zIndex: 1` if they share the same
//      stacking context, but the watermark already lives at
//      `zIndex: 0` with `pointer-events: none` so it never blocks
//      interaction.
//
// The image is purely decorative; an empty `alt` keeps it out of
// screen-reader output.

import React from "react";


export default function CardWatermark({
  // Subtle brand mark — visible enough to register, never competes
  // with content. Stage 2 also needs Collapse panels in `ghost` mode
  // (transparent) for the watermark to show through; otherwise the
  // panel's white surface masks it regardless of opacity.
  opacity = 0.10,
  size = 400,
  position = "center",
}) {
  // Position presets. Default = center so the watermark sits behind
  // the card content like a brand backdrop. Corner placements stay
  // available for callers that want a subtler mark.
  const POSITIONS = {
    "center":       { top: "50%", left: "50%", transform: "translate(-50%, -50%)" },
    "bottom-right": { bottom: 14, right: 18 },
    "bottom-left":  { bottom: 14, left: 18 },
    "top-right":    { top: 14, right: 18 },
    "top-left":     { top: 14, left: 18 },
  };
  const placement = POSITIONS[position] || POSITIONS["center"];

  return (
    <div
      aria-hidden="true"
      style={{
        position: "absolute",
        pointerEvents: "none",
        opacity,
        zIndex: 0,
        ...placement,
      }}
    >
      <img
        src="/logo.png"
        alt=""
        style={{
          width: size,
          height: "auto",
          userSelect: "none",
          display: "block",
        }}
        draggable={false}
      />
    </div>
  );
}
