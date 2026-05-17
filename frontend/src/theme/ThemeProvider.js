import React, { createContext, useContext, useMemo } from "react";
import { MODERN_TOKENS } from "./acadiaTheme";

// Sprint 8 — Tier-1 scoped theme.
// Intentionally NOT named `ThemeProvider` at the call site: the
// application already has `hooks/ThemeContext.ThemeProvider` for the
// chat-wide light/dark mode. This one wraps the Tier-1 tree only, so
// its effects never leak outside the copilot.
//
// Sprint 11 — User-facing Classic/Modern toggle removed from the UI.
// Modern tokens are now used unconditionally. `useTier1Theme()` still
// returns the same shape (stub setters keep call sites that may have
// referenced them safe).

const Tier1ThemeContext = createContext({
  tokens: MODERN_TOKENS,
  isModern: true,
  preference: "modern",
  setPreference: () => {},
  togglePreference: () => {},
  flagOn: true,
});


export default function Tier1ThemeProvider({ children }) {
  const value = useMemo(
    () => ({
      tokens: MODERN_TOKENS,
      isModern: true,
      preference: "modern",
      // Stubs — the toggle was removed, so any stale call to setPreference
      // / togglePreference is a no-op rather than a crash.
      setPreference: () => {},
      togglePreference: () => {},
      flagOn: true,
    }),
    [],
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
