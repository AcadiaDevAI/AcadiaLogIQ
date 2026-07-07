import React from "react";

/**
 * Org-module registry — resolves the active org's slug to that org's screens.
 *
 * Each org gets a folder under src/orgs/<org>/ exporting the screens it may
 * override (IntakeForm, Workspace/Journey, ...). Screens are code-split via
 * React.lazy so an org's bundle is only downloaded when that org is active.
 *
 * An org that overrides nothing re-exports Acadia's screens, so it inherits
 * shared behavior verbatim (see backend/orgs for the mirror on the API side).
 * Unknown / missing slug → Acadia (fail-safe default).
 */
const REGISTRY = {
  acadia: {
    IntakeForm: React.lazy(() => import("./acadia/IntakeForm")),
    Workspace: React.lazy(() => import("./acadia/Workspace")),
  },
  uspharma: {
    IntakeForm: React.lazy(() => import("./uspharma/IntakeForm")),
    Workspace: React.lazy(() => import("./uspharma/Workspace")),
  },
};

// Collapse slug variants to a registry key. Clerk appends a long numeric
// org-id suffix (e.g. "us-pharma-1782167031742315025"), so strip a trailing
// "-<6+ digits>" first, then keep alphanumerics:
//   "us-pharma-1782167031742315025" -> "uspharma"
//   "US Pharma" / "us_pharma"        -> "uspharma"
// Anything that isn't a known non-Acadia org → "acadia".
function normKey(slug) {
  if (!slug) return "acadia";
  const base = String(slug).toLowerCase().trim().replace(/-\d{6,}$/, "");
  const n = base.replace(/[^a-z0-9]/g, "");
  if (n === "uspharma") return "uspharma";
  return "acadia"; // acadia-consultants + every unknown org fall back here
}

/**
 * useOrgModule(slug) → { IntakeForm, Workspace } for the given org.
 * The components are lazy — render them inside a <Suspense> boundary.
 */
export function useOrgModule(slug) {
  const key = normKey(slug);
  return REGISTRY[key] || REGISTRY.acadia;
}

/**
 * isUSPharma(slug) — true when the active org resolves to US Pharma.
 * Used to gate US-Pharma-only UI (e.g. swapping Guided Troubleshooting for a
 * KB / SOP button) inside otherwise-shared components. Acadia → false.
 */
export function isUSPharma(slug) {
  return normKey(slug) === "uspharma";
}

export default useOrgModule;
