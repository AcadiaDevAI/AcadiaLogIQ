// Sprint 8 — Acadia Tier-1 modern theme tokens.
// Consumed by ThemeProvider (theme/ThemeProvider.js) and exposed via
// the useAcadiaTier1Theme() hook. No runtime env lookups here; this
// module is pure data + a null-fallback "classic" bag that preserves
// the Sprint 6/7 visual style when the modern-theme flag is off.

export const MODERN_TOKENS = {
  // Acadia brand navy — derived from logo
  primary:         "#1E3A8A",
  primaryHover:    "#1E40AF",
  primaryLight:    "#3B82F6",

  // Gradient accents
  gradientStart:   "#1E3A8A",
  gradientEnd:     "#4F46E5",
  gradientAccent:  "linear-gradient(135deg, #1E3A8A 0%, #4F46E5 100%)",

  // Surface
  surfaceBase:     "#FFFFFF",
  surfaceElevated: "#F8FAFC",
  surfaceHover:    "#F1F5F9",

  // Text
  textPrimary:     "#0F172A",
  textSecondary:   "#475569",
  textMuted:       "#94A3B8",

  // Feedback surfaces
  successBg:       "#ECFDF5",
  successText:     "#065F46",
  warningBg:       "#FFFBEB",
  warningText:     "#92400E",
  infoBg:          "#EFF6FF",
  infoText:        "#1E40AF",

  // Confidence band visuals (applied to the Tier1AnswerCard banner)
  confidenceHigh:   { bg: "#ECFDF5", text: "#065F46", ring: "#10B981" },
  confidenceMedium: { bg: "#EFF6FF", text: "#1E40AF", ring: "#3B82F6" },
  confidenceLow:    { bg: "#FEFCE8", text: "#854D0E", ring: "#CA8A04" },
  confidenceNone:   { bg: "#F8FAFC", text: "#64748B", ring: "#CBD5E1" },

  // Shape + elevation
  radiusSm: "8px",
  radiusMd: "12px",
  radiusLg: "16px",
  shadowSm: "0 1px 2px rgba(15,23,42,0.05)",
  shadowMd: "0 4px 6px -1px rgba(15,23,42,0.10), 0 2px 4px -2px rgba(15,23,42,0.10)",
  shadowLg: "0 10px 15px -3px rgba(15,23,42,0.10), 0 4px 6px -4px rgba(15,23,42,0.10)",

  // Motion
  transitionFast: "120ms cubic-bezier(0.4, 0, 0.2, 1)",
  transitionMed:  "200ms cubic-bezier(0.4, 0, 0.2, 1)",

  // Flag marker — components can branch without reading window flags
  isModern: true,
};

// The "classic" bag leaves styling to Ant Design's defaults + CSS vars
// already in the stylesheet. Returning `isModern: false` lets
// components skip applying any modern-specific style overrides.
export const CLASSIC_TOKENS = {
  isModern: false,
  // A minimal surface so shared components can still colour-code
  // without branching on isModern everywhere.
  primary:         "#0A3F63",          // the legacy Acadia CTA colour
  surfaceBase:     "var(--bg-secondary, #ffffff)",
  surfaceElevated: "var(--bg-primary, #f4f4f5)",
  textPrimary:     "var(--text-primary, #0f172a)",
  textMuted:       "var(--text-muted, #6b6b6b)",
  confidenceHigh:   { bg: "#e6f4ea", text: "#0A7A3F", ring: "#0A7A3F" },
  confidenceMedium: { bg: "#fef3c7", text: "#C9870B", ring: "#C9870B" },
  confidenceLow:    { bg: "#fee2e2", text: "#B03A2E", ring: "#B03A2E" },
  confidenceNone:   { bg: "#f1f5f9", text: "#6B6B6B", ring: "#CBD5E1" },
  radiusSm: "4px",
  radiusMd: "6px",
  radiusLg: "8px",
};
