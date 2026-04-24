import React, {
  createContext,
  useCallback,
  useContext,
  useMemo,
} from "react";
import useLocalStorage from "../hooks/useLocalStorage";
import { CLASSIC_TOKENS, MODERN_TOKENS } from "./acadiaTheme";

// Sprint 8 — Tier-1 scoped theme.
// Intentionally NOT named `ThemeProvider` at the call site: the
// application already has `hooks/ThemeContext.ThemeProvider` for the
// chat-wide light/dark mode. This one wraps the Tier-1 tree only, so
// its effects never leak outside the copilot.

const Tier1ThemeContext = createContext({
  tokens: CLASSIC_TOKENS,
  isModern: false,
  preference: "classic",
  setPreference: () => {},
  flagOn: false,
});

const MODERN_THEME_FLAG_ON =
  process.env.REACT_APP_LOGIQ_TIER1_MODERN_THEME === "true";

const STORAGE_KEY = "acadia_tier1_theme_preference";
const DEFAULT_PREF = "modern"; // only honoured when the flag is on

export default function Tier1ThemeProvider({ children }) {
  const [preference, setPreference] = useLocalStorage(
    STORAGE_KEY, DEFAULT_PREF,
  );

  const activeIsModern =
    MODERN_THEME_FLAG_ON && preference === "modern";
  const tokens = activeIsModern ? MODERN_TOKENS : CLASSIC_TOKENS;

  const togglePreference = useCallback(() => {
    setPreference((prev) => (prev === "modern" ? "classic" : "modern"));
  }, [setPreference]);

  const value = useMemo(
    () => ({
      tokens,
      isModern: activeIsModern,
      preference,
      setPreference,
      togglePreference,
      flagOn: MODERN_THEME_FLAG_ON,
    }),
    [tokens, activeIsModern, preference, setPreference, togglePreference],
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
