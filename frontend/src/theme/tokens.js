// ─────────────────────────────────────────────────────────────────────
// Acadia LogIQ — design tokens (premium theme source of truth)
//
// JS module mirror of the CSS variables defined in
// `src/theme/premium.css`. Components that need a token at runtime
// (inline style, framer-motion, CSS-in-JS) import from here; CSS-only
// consumers read the same value via `var(--token-name)`.
//
// Keep this file in sync with premium.css. The CSS file is the
// authoritative source for what the browser paints — this file is
// just the JS façade.
//
// To extend: add the token in BOTH places. Resist the urge to compute
// derived colors at runtime — predictability beats cleverness here.
// ─────────────────────────────────────────────────────────────────────


// ─── Color ───────────────────────────────────────────────────────────
// Premium LIGHT — white base, near-black text, aurora accents.
export const color = {
  bg:           "#F7F8FB",
  bgRaised:     "#FFFFFF",
  border:       "rgba(10, 16, 24, 0.08)",
  borderStrong: "rgba(10, 16, 24, 0.14)",
  text:         "#0A1018",
  textMuted:    "rgba(10, 16, 24, 0.62)",
  textDim:      "rgba(10, 16, 24, 0.40)",
};

// ─── Aurora accent ──────────────────────────────────────────────────
export const aurora = {
  c1: "#7CEDE5",  // teal
  c2: "#5B8DEF",  // iris
  c3: "#A78BFA",  // violet
  gradient:     "linear-gradient(135deg, #7CEDE5 0%, #5B8DEF 55%, #A78BFA 100%)",
  gradientSoft:
    "linear-gradient(135deg, rgba(124,237,229,0.18) 0%, rgba(91,141,239,0.18) 55%, rgba(167,139,250,0.18) 100%)",
};

// ─── Severity ───────────────────────────────────────────────────────
export const severity = {
  p1:   "#FF5E7A",
  p2:   "#FFB347",
  p3:   "#7CEDE5",
  p4:   "#9BA3B4",
  good: "#46D69A",
};

// ─── Radii ──────────────────────────────────────────────────────────
export const radius = {
  6: "6px", 8: "8px", 10: "10px", 12: "12px", 16: "16px", 20: "20px",
};

// ─── Spacing scale (4-base) ─────────────────────────────────────────
export const spacing = {
  4: "4px", 8: "8px", 12: "12px", 16: "16px",
  24: "24px", 32: "32px", 48: "48px", 64: "64px",
};

// ─── Shadows ─ softer for light theme; aurora glow stays iris-tinted ─
export const shadow = {
  sm: "0 1px 2px rgba(10,16,24,0.06), 0 1px 4px rgba(10,16,24,0.04)",
  md: "0 4px 12px rgba(10,16,24,0.08), 0 2px 4px rgba(10,16,24,0.05)",
  lg: "0 14px 32px -8px rgba(91,141,239,0.22), 0 4px 12px rgba(10,16,24,0.06)",
  // Signature aurora glow — primary button + glass input bar use this.
  auroraGlow:
    "0 14px 32px -8px rgba(91,141,239,0.45), 0 0 0 1px rgba(124,237,229,0.35) inset",
};

// ─── Motion (cohesive easing across the app) ────────────────────────
export const motion = {
  easeOut: "cubic-bezier(0.16, 1, 0.3, 1)",
  easeInOut: "cubic-bezier(0.4, 0, 0.2, 1)",
  spring: "cubic-bezier(0.34, 1.56, 0.64, 1)",
  fast: "150ms",
  base: "220ms",
  slow: "420ms",
};

// ─── Typography ─────────────────────────────────────────────────────
export const font = {
  display: "'Instrument Serif', Georgia, 'Times New Roman', serif",
  body:
    "'Geist', 'Inter', 'Poppins', system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
  mono: "'Geist Mono', 'JetBrains Mono', 'SF Mono', Consolas, Monaco, monospace",
};


// ─── Default export — convenience bag ────────────────────────────────
const tokens = { color, aurora, severity, radius, spacing, shadow, motion, font };
export default tokens;
