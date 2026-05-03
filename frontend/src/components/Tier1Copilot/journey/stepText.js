// Sprint 11 — shared step-text helpers for Stage 0 / Stage 3 / etc.
//
// Source-corpus Resolution_Steps strings are sometimes pre-numbered
// ("1. Incident detection..."). When the React render wraps them in
// <ol> or prepends "{i+1}. ", the number doubles to "1. 1. Incident
// detection..." Strip any leading "N." / "N)" / "N:" / "N -" prefix
// before render so the visible numbering stays clean regardless of
// the source shape.
//
// Detection is deliberately conservative: we only strip when the
// match is at the very start of the trimmed string and the number
// is followed by a separator + whitespace. So "Step 1: Check line"
// or "v1.2 release notes" are NOT mistaken for numbered steps.

const LEADING_NUMBER_RE = /^\s*(\d{1,3})\s*[.\):\-]\s+/;


export function stripLeadingNumber(text) {
  if (typeof text !== "string") return text;
  return text.replace(LEADING_NUMBER_RE, "").trim();
}


// Re-export for tests (intentionally exported separately so a future
// regex tweak can be unit-tested without exporting any internals).
export const _LEADING_NUMBER_RE = LEADING_NUMBER_RE;
