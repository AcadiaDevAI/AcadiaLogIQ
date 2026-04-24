import { useCallback, useEffect, useState } from "react";

/**
 * Minimal useLocalStorage hook (Sprint 8).
 *
 * - Reads the stored JSON value on mount; returns `initialValue` if
 *   nothing is stored or parse fails.
 * - Writes to localStorage on every setter call.
 * - Listens for the `storage` event so sibling tabs stay in sync.
 * - SSR-safe: if `window` is undefined (shouldn't happen in CRA but
 *   cheap insurance) the hook behaves like a plain useState.
 */
export default function useLocalStorage(key, initialValue) {
  const read = useCallback(() => {
    if (typeof window === "undefined") return initialValue;
    try {
      const raw = window.localStorage.getItem(key);
      if (raw === null || raw === undefined) return initialValue;
      return JSON.parse(raw);
    } catch {
      return initialValue;
    }
  }, [key, initialValue]);

  const [value, setValue] = useState(read);

  const write = useCallback(
    (next) => {
      setValue((prev) => {
        const resolved = typeof next === "function" ? next(prev) : next;
        try {
          if (typeof window !== "undefined") {
            window.localStorage.setItem(key, JSON.stringify(resolved));
          }
        } catch {
          /* quota / private mode — ignore */
        }
        return resolved;
      });
    },
    [key],
  );

  useEffect(() => {
    if (typeof window === "undefined") return undefined;
    const handler = (e) => {
      if (e.key !== key) return;
      try {
        setValue(e.newValue === null ? initialValue : JSON.parse(e.newValue));
      } catch {
        setValue(initialValue);
      }
    };
    window.addEventListener("storage", handler);
    return () => window.removeEventListener("storage", handler);
  }, [key, initialValue]);

  return [value, write];
}
