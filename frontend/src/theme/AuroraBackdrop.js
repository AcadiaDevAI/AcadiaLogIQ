// AuroraBackdrop — reusable atmospheric backdrop.
//
// The body of the premium theme already paints a fixed gradient mesh
// via `.theme-premium::before` (see premium.css), so this component
// is only needed in places where the body backdrop is blocked by a
// solid container above it (modals, drawers, opaque side-panes).
//
// Layered composition (per uichanges.md Prompt 02):
//   - Three soft radial blobs (teal / iris / violet), large + blurred
//   - A faint 64px grid masked with radial fade
//   - A noise grain at 4% opacity, blend-mode overlay
//
// All layers are pointer-events: none so the backdrop never eats
// clicks. The component renders at the start of its parent, so
// position the parent `relative` and ensure content uses z-index > 0.

import React from "react";


const blob = (left, top, color, alpha, size = 520, blur = 60) => ({
  position: "absolute",
  left,
  top,
  width: size,
  height: size,
  background: `radial-gradient(circle, ${color}${Math.round(alpha * 255).toString(16).padStart(2, "0")} 0%, transparent 70%)`,
  filter: `blur(${blur}px)`,
  pointerEvents: "none",
});

const gridDataUri =
  "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='64' height='64'><path d='M64 0H0V64' fill='none' stroke='rgba(255,255,255,0.04)' stroke-width='1'/></svg>";

const noiseDataUri =
  "data:image/svg+xml;utf8,<svg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/></filter><rect width='100%25' height='100%25' filter='url(%23n)' opacity='0.55'/></svg>";


export default function AuroraBackdrop({ children }) {
  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        overflow: "hidden",
        pointerEvents: "none",
        zIndex: 0,
      }}
    >
      {/* Three soft radial blobs */}
      <div style={blob("-120px", "-100px", "#7CEDE5", 0.18, 600, 60)} />
      <div style={blob("calc(100% - 480px)", "-60px", "#A78BFA", 0.16, 540, 56)} />
      <div style={blob("calc(50% - 280px)", "calc(100% - 300px)", "#5B8DEF", 0.14, 560, 60)} />

      {/* Faint grid, radial-faded at edges */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          backgroundImage: `url("${gridDataUri}")`,
          backgroundSize: "64px 64px",
          maskImage:
            "radial-gradient(ellipse at center, rgba(0,0,0,0.85) 0%, transparent 70%)",
          WebkitMaskImage:
            "radial-gradient(ellipse at center, rgba(0,0,0,0.85) 0%, transparent 70%)",
        }}
      />

      {/* Noise grain */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          backgroundImage: `url("${noiseDataUri}")`,
          backgroundSize: "220px 220px",
          opacity: 0.04,
          mixBlendMode: "overlay",
        }}
      />

      {children}
    </div>
  );
}
