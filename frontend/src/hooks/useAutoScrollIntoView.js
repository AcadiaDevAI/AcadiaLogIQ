import { useEffect, useRef } from "react";

/**
 * Sprint 8 — useAutoScrollIntoView
 *
 * Returns a ref. When `enabled` flips truthy (or the `resetKey` changes),
 * the referenced element is scrolled into the viewport with smooth
 * behavior on the next tick.
 */
export default function useAutoScrollIntoView(enabled, resetKey = null) {
  const ref = useRef(null);

  useEffect(() => {
    if (!enabled || !ref.current) return undefined;
    const el = ref.current;
    const timer = setTimeout(() => {
      try {
        el.scrollIntoView({ behavior: "smooth", block: "start" });
      } catch {
        /* older browsers: no-op */
      }
    }, 120);
    return () => clearTimeout(timer);
  }, [enabled, resetKey]);

  return ref;
}