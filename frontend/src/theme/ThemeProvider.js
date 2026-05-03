import React, { createContext, useContext, useMemo } from "react";
import { CLASSIC_TOKENS, MODERN_TOKENS } from "./acadiaTheme";

// Sprint 8 — Tier-1 scoped theme.
// Intentionally NOT named `ThemeProvider` at the call site: the
// application already has `hooks/ThemeContext.ThemeProvider` for the
// chat-wide light/dark mode. This one wraps the Tier-1 tree only, so
// its effects never leak outside the copilot.
//
// Sprint 11 — User-facing Classic/Modern toggle removed from the UI.
// The active theme is now driven purely by REACT_APP_LOGIQ_TIER1_MODERN_THEME:
//   true  → modern tokens always
//   false → classic tokens always
// The localStorage preference, setPreference, togglePreference, and the
// previous useLocalStorage hook are gone — they only existed to back
// the toggle button. `useTier1Theme()` still returns the same shape
// (stub setters keep call sites that may have referenced them safe);
// `flagOn` still reflects the env so any remaining gated render falls
// through correctly.

const MODERN_THEME_FLAG_ON =
  process.env.REACT_APP_LOGIQ_TIER1_MODERN_THEME === "true";

const Tier1ThemeContext = createContext({
  tokens: MODERN_THEME_FLAG_ON ? MODERN_TOKENS : CLASSIC_TOKENS,
  isModern: MODERN_THEME_FLAG_ON,
  preference: MODERN_THEME_FLAG_ON ? "modern" : "classic",
  setPreference: () => {},
  togglePreference: () => {},
  flagOn: MODERN_THEME_FLAG_ON,
});


export default function Tier1ThemeProvider({ children }) {
  const isModern = MODERN_THEME_FLAG_ON;
  const tokens = isModern ? MODERN_TOKENS : CLASSIC_TOKENS;

  const value = useMemo(
    () => ({
      tokens,
      isModern,
      preference: isModern ? "modern" : "classic",
      // Stubs — the toggle was removed, so any stale call to setPreference
      // / togglePreference is a no-op rather than a crash.
      setPreference: () => {},
      togglePreference: () => {},
      flagOn: MODERN_THEME_FLAG_ON,
    }),
    [tokens, isModern],
  );

  return (
    <Tier1ThemeContext.Provider value={value}>
      {children}
    </Tier1ThemeContext.Provider>
  );
}

export function useTier1Theme() {
  return useContext(Tier1ThemeContext);
}
