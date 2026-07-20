// ─────────────────────────────────────────────────────────────────────
// US Pharma — THEME PALETTE (single source of truth for the org's red)
// ─────────────────────────────────────────────────────────────────────
//
// US Pharma repaints the shared "premium" (Aurora Operations) surface
// from its default pale-blue / iris accent to a Walgreens-style RED.
// Every red value the org uses lives HERE so there is exactly one place
// to retune the brand.
//
// Consumers:
//   • uspharma.css          — rebinds the premium CSS variables
//                             (--brand-accent, --aurora, …) under the
//                             `.org-uspharma` scope, using these hexes.
//   • UsPharmaThemeScope.js  — feeds `usPharmaAntdTheme` into a nested
//                             AntD ConfigProvider so AntD-computed colors
//                             (focus rings, Select, Tabs, Spin…) turn red.
//   • Stage0BestTicketDistillation.js — imports `USP_RED_SCALE` for its
//                             locally-injected <style> block.
//
// NOTHING outside src/orgs/uspharma imports this file. Acadia is never
// touched.
// ─────────────────────────────────────────────────────────────────────

// Walgreens brand red and its interaction states (progressively darker,
// mirroring the pattern the premium blue used: base → hover → active).
export const WALGREENS_RED = "#E31837";

export const USP_RED = {
  primary:       "#E31837", // Walgreens red — primary accent / CTA
  primaryHover:  "#C4132F",
  primaryActive: "#A50E27",
  light:         "#F26D7D", // soft red for muted accents / disabled
  // Brighter tone that stays legible on the dark (Acadia-navy) surface.
  primaryDark:   "#FF4D67",
};

// Tinted red ramp (50 → 700) that mirrors the blue scale it replaces in
// Stage0. Pale at the top for backgrounds/borders, saturated at 500
// (Walgreens red), deep at 700 for text on light surfaces.
export const USP_RED_SCALE = {
  50:  "#FEF2F3",
  100: "#FDE0E4",
  200: "#FBC5CC",
  400: "#F26D7D",
  500: "#E31837", // Walgreens red
  600: "#C4132F",
  700: "#A50E27",
};

// rgba tint helper — matches the low-alpha washes the org uses inline
// (e.g. the IntakeForm eyebrow pill background/border).
export const uspRedAlpha = (a) => `rgba(227, 24, 55, ${a})`;

// ── AntD token overrides for a NESTED ConfigProvider ─────────────────
// Nested ConfigProviders inherit (inherit=true) the global premium theme
// — algorithm, radii, fonts, all component tokens — so we only need to
// state the handful of accent tokens that must flip from iris to red.
export const usPharmaAntdTheme = {
  token: {
    colorPrimary:       USP_RED.primary,
    colorPrimaryHover:  USP_RED.primaryHover,
    colorPrimaryActive: USP_RED.primaryActive,
    colorInfo:          USP_RED.primary,
    colorLink:          USP_RED.primary,
    colorLinkHover:     USP_RED.primaryHover,
    colorLinkActive:    USP_RED.primaryActive,
  },
  components: {
    Button: {
      defaultHoverColor:       USP_RED.primary,
      defaultHoverBorderColor: uspRedAlpha(0.55),
    },
    Tabs: {
      itemSelectedColor: USP_RED.primary,
      inkBarColor:       USP_RED.primary,
    },
    Select: {
      optionSelectedBg: uspRedAlpha(0.10),
    },
    Input: {
      activeBorderColor: USP_RED.primary,
      hoverBorderColor:  uspRedAlpha(0.55),
    },
    Spin:    { colorPrimary: USP_RED.primary },
    Tooltip: {},
  },
};

export default usPharmaAntdTheme;
