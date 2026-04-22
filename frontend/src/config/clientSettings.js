/**
 * Client-side feature flag settings.
 *
 * Flags default to "enabled" in production but can be disabled via
 * REACT_APP_* environment variables for instant rollback without
 * requiring a code change.
 *
 * Usage in components:
 *   import { settings } from "../config/clientSettings";
 *   if (settings.RICH_FORMATTING_ENABLED) { ... }
 */

const parseBool = (value, defaultValue) => {
  if (value === undefined || value === null || value === "") return defaultValue;
  const normalized = String(value).toLowerCase().trim();
  if (["false", "0", "no", "off", "disabled"].includes(normalized)) return false;
  if (["true", "1", "yes", "on", "enabled"].includes(normalized)) return true;
  return defaultValue;
};

export const settings = {
  // Master flag for rich response formatting (markdown tables, syntax highlighting,
  // code copy buttons, callout styling). When disabled, responses render as plain
  // markdown via ReactMarkdown without any plugins — exactly pre-feature behavior.
  RICH_FORMATTING_ENABLED: parseBool(
    process.env.REACT_APP_RICH_FORMATTING_ENABLED,
    true,
  ),

  // Rich Formatting Polish — Fix 1: robust inline vs block code detection.
  // When True (default), CodeBlock.js uses a multi-signal heuristic
  // (language class absence + no newlines + short length) to identify
  // inline code, working around react-markdown v9 dropping the `inline`
  // prop. When False, reverts to the bare `inline === true` check which
  // fails silently on v9+ and causes every `INC-10037` to render as a
  // dark code block.
  CODE_INLINE_DETECTION_ENABLED: parseBool(
    process.env.REACT_APP_CODE_INLINE_DETECTION_ENABLED,
    true,
  ),

  // Guided workflow — when true, LandingPage gates entry to ChatArea.
  // Must match backend GUIDED_WORKFLOW_ENABLED for consistent UX.
  GUIDED_WORKFLOW_ENABLED: parseBool(
    process.env.REACT_APP_GUIDED_WORKFLOW_ENABLED,
    false,
  ),

  // Sprint 2 — forms, modal, pattern card rendering.
  // Must match backend LOGIQ_SPRINT2_BACKEND for consistent UX.
  LOGIQ_SPRINT2_FRONTEND: parseBool(
    process.env.REACT_APP_LOGIQ_SPRINT2_FRONTEND,
    false,
  ),
};
