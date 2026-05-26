import React, { createContext, useContext, useState, useCallback } from "react";
import { ACTIVE_THEME, resolveThemeClass } from "../theme/theme-config";

// ─────────────────────────────────────────────────────────────────────
// Global theme context
//
// Two themes live in the codebase:
//
//   • "premium" (DEFAULT) → Aurora Operations dark-only theme. When
//     this is the active theme (set in src/theme/theme-config.js), the
//     legacy light/dark toggle is ignored visually and the root <div>
//     always gets the class `theme-premium`.
//
//   • "legacy"            → original light/dark toggle. The `mode`
//     state below still flips between "light" and "dark", and the
//     root <div> gets `theme-light` or `theme-dark` accordingly.
//
// `isDark` is preserved as a derived value so consumers (App.js's
// AntD ConfigProvider, MobileHeader's theme toggle button) keep
// working unchanged. Under premium it always returns true so AntD's
// dark algorithm is selected.
// ─────────────────────────────────────────────────────────────────────

const ThemeContext = createContext();

export function ThemeProvider({ children }) {
  const [mode, setMode] = useState("light"); // "light" | "dark"

  const toggleTheme = useCallback(() => {
    setMode((prev) => (prev === "light" ? "dark" : "light"));
  }, []);

  // Under premium, the legacy mode flag is ignored visually but kept
  // intact so toggling it (e.g. the header sun/moon icon) does no
  // harm. Premium is now a LIGHT theme (white-and-black with aurora
  // accents), so `isDark` reports false to make AntD pick the light
  // algorithm in resolveAntdTheme.
  const isPremium = ACTIVE_THEME === "premium";
  const isDark = isPremium ? false : mode === "dark";

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
