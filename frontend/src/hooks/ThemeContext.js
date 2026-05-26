import React, { createContext, useContext, useState, useCallback } from "react";
import { ACTIVE_THEME, resolveThemeClass } from "../theme/theme-config";

// ─────────────────────────────────────────────────────────────────────
// Global theme context
//
// Two themes live in the codebase:
//
//   • "premium" (DEFAULT) → Aurora Operations with a light surface by
//     default and an Acadia-navy DARK companion. Toggling the
//     header sun/moon icon flips between them:
//       light → root class is `theme-premium`
//       dark  → root class is `theme-premium theme-premium-dark`
//     The dark companion stylesheet
//     (src/theme/premium-dark.css) only paints when both classes are
//     present, so light premium is unaffected.
//
//   • "legacy" → original light/dark toggle. The `mode` state flips
//     between "light" and "dark", and the root <div> gets
//     `theme-light` or `theme-dark` accordingly.
//
// `isDark` reflects the toggle state under both legacy and premium
// so AntD's dark algorithm is picked when appropriate and components
// that read `isDark` (e.g. the Stage 5 handoff <pre> background)
// recolor correctly.
// ─────────────────────────────────────────────────────────────────────

const ThemeContext = createContext();

export function ThemeProvider({ children }) {
  const [mode, setMode] = useState("light"); // "light" | "dark"

  const toggleTheme = useCallback(() => {
    setMode((prev) => (prev === "light" ? "dark" : "light"));
  }, []);

  // Premium ships with a default light surface AND an Acadia-navy
  // dark companion (src/theme/premium-dark.css + the dark branch of
  // resolveAntdTheme). `isDark` now reflects the actual toggle state
  // under both legacy and premium so:
  //   • AntD picks the dark algorithm when the user toggles dark.
  //   • Components that read `isDark` (e.g. the Stage 5 handoff
  //     <pre> background) recolor correctly.
  // Light premium behaviour is identical to before — when mode is
  // "light", `isDark === false`, the root class is `theme-premium`
  // alone, and premium-dark.css selectors never fire.
  const isPremium = ACTIVE_THEME === "premium";
  const isDark = mode === "dark";

  const themeClass = resolveThemeClass(mode === "dark");

  const value = React.useMemo(
    () => ({ mode, isDark, isPremium, toggleTheme, themeClass }),
    [mode, isDark, isPremium, toggleTheme, themeClass]
  );

  return (
    <ThemeContext.Provider value={value}>
      <div className={themeClass}>{children}</div>
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used within ThemeProvider");
  return ctx;
}
