// ─────────────────────────────────────────────────────────────────────
// Acadia LogIQ — THEME CONFIGURATION (single source of truth)
// ─────────────────────────────────────────────────────────────────────
//
// HOW TO SWITCH THEMES
// ────────────────────
//   Edit the constant below. That's the only change you need.
//
//     ACTIVE_THEME = "premium"  → Aurora Operations (dark-only,
//                                 atmospheric backdrop, glass panels,
//                                 Instrument Serif + Geist + Geist Mono)
//                                 — currently the DEFAULT.
//
//     ACTIVE_THEME = "legacy"   → original light/dark toggle UI
//                                 (Poppins + Inter, indigo accents)
//                                 — preserved byte-for-byte for rollback.
//
// Restart the dev server (or wait for hot-reload) and the entire
// surface repaints. Zero component edits required — both themes use
// the same CSS variable names; only the values differ.
//
//
// ARCHITECTURE
// ────────────
// • Legacy theme  → `src/index.css` (.theme-light / .theme-dark)
// • Premium theme → `src/theme/premium.css` (.theme-premium)
// • Shared design tokens (JS façade) → `src/theme/tokens.js`
//
// Both themes define the SAME CSS variable names (`--bg-primary`,
// `--text-primary`, `--brand-accent`, etc.). The premium theme adds
// NEW tokens (glass, gradients, aurora, severity scale, motion) that
// the legacy theme doesn't have — components that opt into those
// tokens degrade gracefully on legacy (the var is undefined; CSS
// falls through to whatever default is set on the property).
//
//
// HARD CONSTRAINTS
// ────────────────
// • Never touch backend / API / auth / ChatContext from this file or
//   from any theme stylesheet. Presentation only.
// • Components must stay theme-agnostic. If a component uses
//   `var(--bg-primary)` it works under any theme automatically.
// ─────────────────────────────────────────────────────────────────────


/**
 * The single switch. Change this one word to swap the entire UI.
 * @type {"premium" | "legacy"}
 */
export const ACTIVE_THEME = "premium";


/**
 * Convenience helper — true when premium is active. Useful for the
 * tiny number of places we want to conditionally render premium-only
 * chrome (aurora backdrops, gradient text) without sprinkling string
 * comparisons everywhere.
 */
export const isPremiumTheme = () => ACTIVE_THEME === "premium";


/**
 * Returns the CSS class(es) to apply on the root <div>.
 *
 * Premium light = `theme-premium`
 * Premium dark  = `theme-premium theme-premium-dark`
 *   (companion class layers Acadia-navy variable rebinds on top of
 *    the base premium rules — see src/theme/premium-dark.css)
 * Legacy       = `theme-light` or `theme-dark`
 *
 * @param {boolean} isDark — current legacy light/dark toggle value
 * @returns {string}
 */
export function resolveThemeClass(isDark) {
  if (ACTIVE_THEME === "premium") {
    return isDark ? "theme-premium theme-premium-dark" : "theme-premium";
  }
  return isDark ? "theme-dark" : "theme-light";
}


/**
 * AntD ConfigProvider theme tokens — premium or legacy.
 *
 * Centralised here so App.js stays a thin shell. Every AntD-driven
 * component (Button, Card, Modal, Select, Tabs, Alert, Tag, …) picks
 * up the right colors / radii / font from these tokens automatically.
 *
 * @param {boolean} isDark — current legacy toggle value
 * @param {{ darkAlgorithm: any, defaultAlgorithm: any }} algorithms
 * @returns {object} AntD theme config
 */
export function resolveAntdTheme(isDark, algorithms) {
  if (ACTIVE_THEME === "premium" && isDark) {
    // Aurora Operations — premium DARK. Acadia-navy backdrop with
    // the same iris/sky-blue accents. CSS-level overrides in
    // premium-dark.css do the heavy lifting; these tokens keep
    // AntD's internal calculations (focus rings, hover algorithms,
    // contrast hints) consistent with the dark surface.
    return {
      algorithm: algorithms.darkAlgorithm,
      token: {
        // Acadia sky-blue from the logo reads as primary on navy.
        colorPrimary:       "#3DA2E0",
        colorPrimaryHover:  "#5BB8ED",
        colorPrimaryActive: "#2B8FCC",

        colorBgContainer:   "#0F2240",
        colorBgElevated:    "#142D52",
        colorBgLayout:      "transparent",
        colorBorder:        "rgba(91, 141, 239, 0.22)",
        colorBorderSecondary: "rgba(91, 141, 239, 0.14)",

        colorText:          "#E8F1FB",
        colorTextSecondary: "#9EB2CC",
        colorTextTertiary:  "#6B7E9C",

        colorSuccess:       "#46D69A",
        colorWarning:       "#FFB347",
        colorError:         "#FF5E7A",
        colorInfo:          "#5B8DEF",

        borderRadius:       10,
        borderRadiusLG:     16,
        borderRadiusSM:     8,

        fontFamily:
          "'Geist', 'Inter', 'Poppins', system-ui, -apple-system, sans-serif",
        fontSize: 14,
        controlHeight: 38,
        controlHeightLG: 46,
        controlHeightSM: 30,

        wireframe: false,
      },
      components: {
        Button: {
          primaryShadow: "none",
          defaultBg: "#0F2240",
          defaultBorderColor: "rgba(91, 141, 239, 0.36)",
          defaultColor: "#E8F1FB",
          defaultHoverBg: "#142D52",
          defaultHoverBorderColor: "rgba(91, 141, 239, 0.55)",
          defaultHoverColor: "#5B8DEF",
          dangerShadow: "none",
        },
        Card: {
          colorBgContainer: "#0F2240",
          colorBorderSecondary: "rgba(91, 141, 239, 0.22)",
          headerBg: "transparent",
          actionsBg: "transparent",
        },
        Modal: {
          contentBg: "rgba(15, 34, 64, 0.92)",
          headerBg: "transparent",
          colorBgMask: "rgba(0, 8, 20, 0.62)",
        },
        Collapse: {
          headerBg: "transparent",
          contentBg: "transparent",
          colorBorder: "rgba(91, 141, 239, 0.22)",
        },
        Tabs: {
          colorBgContainer: "transparent",
          itemColor: "#9EB2CC",
          itemSelectedColor: "#5B8DEF",
          itemHoverColor: "#E8F1FB",
          inkBarColor: "#5B8DEF",
        },
        Select: {
          colorBgContainer: "#0F2240",
          colorBgElevated: "#142D52",
          optionSelectedBg: "rgba(91, 141, 239, 0.22)",
          optionActiveBg: "rgba(91, 141, 239, 0.12)",
        },
        Input: {
          colorBgContainer: "#0F2240",
          activeBorderColor: "#5B8DEF",
          hoverBorderColor: "rgba(91, 141, 239, 0.55)",
        },
        Tag: {
          defaultBg: "rgba(91, 141, 239, 0.12)",
          defaultColor: "#E8F1FB",
        },
        Alert: {
          colorInfoBg:    "rgba(91, 141, 239, 0.14)",
          colorInfoBorder:"rgba(91, 141, 239, 0.36)",
          colorWarningBg: "rgba(255, 179, 71, 0.14)",
          colorWarningBorder: "rgba(232, 154, 42, 0.40)",
          colorErrorBg:   "rgba(255, 94, 122, 0.14)",
          colorErrorBorder: "rgba(224, 69, 95, 0.40)",
          colorSuccessBg: "rgba(70, 214, 154, 0.14)",
          colorSuccessBorder: "rgba(31, 176, 127, 0.40)",
        },
        Empty:   { colorTextDisabled: "#6B7E9C" },
        Spin:    { colorPrimary: "#5B8DEF" },
        Tooltip: { colorBgSpotlight: "#15294A" },
      },
    };
  }

  if (ACTIVE_THEME === "premium") {
    // Aurora Operations — premium LIGHT. Aurora identity preserved
    // (teal → iris → violet accents) on a white-and-black surface.
    // CSS-level overrides in premium.css do the heavy lifting; the
    // tokens here keep AntD's internal calculations consistent.
    return {
      algorithm: algorithms.defaultAlgorithm,
      token: {
        // Iris reads as primary on white. The actual gradient fill
        // is applied in premium.css via .ant-btn-primary override.
        colorPrimary:       "#5B8DEF",
        colorPrimaryHover:  "#4A75D9",
        colorPrimaryActive: "#3D5DC9",

        colorBgContainer:   "#FFFFFF",
        colorBgElevated:    "#FFFFFF",
        colorBgLayout:      "transparent",
        colorBorder:        "rgba(10, 16, 24, 0.08)",
        colorBorderSecondary: "rgba(10, 16, 24, 0.04)",

        colorText:          "#0A1018",
        colorTextSecondary: "rgba(10, 16, 24, 0.62)",
        colorTextTertiary:  "rgba(10, 16, 24, 0.40)",

        colorSuccess:       "#1FB07F",
        colorWarning:       "#E89A2A",
        colorError:         "#E0455F",
        colorInfo:          "#5B8DEF",

        borderRadius:       10,
        borderRadiusLG:     16,
        borderRadiusSM:     8,

        fontFamily:
          "'Geist', 'Inter', 'Poppins', system-ui, -apple-system, sans-serif",
        fontSize: 14,
        controlHeight: 38,
        controlHeightLG: 46,
        controlHeightSM: 30,

        wireframe: false,
      },
      components: {
        Button: {
          // Aurora gradient lives on a CSS class override in
          // premium.css — AntD can't accept gradients as tokens.
          primaryShadow: "none",
          defaultBg: "#FFFFFF",
          defaultBorderColor: "rgba(10, 16, 24, 0.14)",
          defaultColor: "#0A1018",
          defaultHoverBg: "#FFFFFF",
          defaultHoverBorderColor: "rgba(91, 141, 239, 0.55)",
          defaultHoverColor: "#5B8DEF",
          dangerShadow: "none",
        },
        Card: {
          colorBgContainer: "#FFFFFF",
          colorBorderSecondary: "rgba(10, 16, 24, 0.08)",
          headerBg: "transparent",
          actionsBg: "transparent",
        },
        Modal: {
          contentBg: "rgba(255, 255, 255, 0.94)",
          headerBg: "transparent",
          colorBgMask: "rgba(10, 16, 24, 0.42)",
        },
        Collapse: {
          headerBg: "transparent",
          contentBg: "transparent",
          colorBorder: "rgba(10, 16, 24, 0.08)",
        },
        Tabs: {
          colorBgContainer: "transparent",
          itemColor: "rgba(10, 16, 24, 0.62)",
          itemSelectedColor: "#5B8DEF",
          itemHoverColor: "#0A1018",
          inkBarColor: "#5B8DEF",
        },
        Select: {
          colorBgContainer: "#FFFFFF",
          colorBgElevated: "#FFFFFF",
          optionSelectedBg: "rgba(91, 141, 239, 0.10)",
          optionActiveBg: "rgba(10, 16, 24, 0.04)",
        },
        Input: {
          colorBgContainer: "#FFFFFF",
          activeBorderColor: "#5B8DEF",
          hoverBorderColor: "rgba(91, 141, 239, 0.55)",
        },
        Tag: {
          defaultBg: "rgba(10, 16, 24, 0.04)",
          defaultColor: "#0A1018",
        },
        Alert: {
          colorInfoBg:    "rgba(91, 141, 239, 0.08)",
          colorInfoBorder:"rgba(91, 141, 239, 0.25)",
          colorWarningBg: "rgba(255, 179, 71, 0.10)",
          colorWarningBorder: "rgba(232, 154, 42, 0.30)",
          colorErrorBg:   "rgba(255, 94, 122, 0.08)",
          colorErrorBorder: "rgba(224, 69, 95, 0.30)",
          colorSuccessBg: "rgba(70, 214, 154, 0.10)",
          colorSuccessBorder: "rgba(31, 176, 127, 0.30)",
        },
        Empty:   { colorTextDisabled: "rgba(10, 16, 24, 0.40)" },
        Spin:    { colorPrimary: "#5B8DEF" },
        Tooltip: { colorBgSpotlight: "rgba(10, 16, 24, 0.92)" },
      },
    };
  }

  // ── Legacy themes — unchanged from before this revamp. ────────────
  return isDark
    ? {
        algorithm: algorithms.darkAlgorithm,
        token: {
          colorPrimary: "#6366f1",
          colorBgContainer: "#16161d",
          colorBgElevated: "#1e1e28",
          colorBorder: "#2a2a3d",
          colorText: "#e2e8f0",
          colorTextSecondary: "#94a3b8",
          borderRadius: 8,
          fontFamily: "'Poppins', 'Inter', system-ui, sans-serif",
        },
        components: {
          Button:   { primaryShadow: "none" },
          Collapse: { headerBg: "transparent", contentBg: "transparent" },
          Tabs: {
            colorBgContainer: "transparent",
            itemColor: "#94a3b8",
            itemSelectedColor: "#a5b4fc",
            inkBarColor: "#6366f1",
          },
        },
      }
    : {
        algorithm: algorithms.defaultAlgorithm,
        token: {
          colorPrimary: "#4f46e5",
          colorBgContainer: "#ffffff",
          colorBgElevated: "#f8f9fb",
          colorBorder: "#dee2e6",
          colorText: "#1a1a2e",
          colorTextSecondary: "#495057",
          borderRadius: 8,
          fontFamily: "'Poppins', 'Inter', system-ui, sans-serif",
        },
        components: {
          Button:   { primaryShadow: "none" },
          Collapse: { headerBg: "transparent", contentBg: "transparent" },
          Tabs: {
            colorBgContainer: "transparent",
            itemColor: "#495057",
            itemSelectedColor: "#4f46e5",
            inkBarColor: "#4f46e5",
          },
        },
      };
}
