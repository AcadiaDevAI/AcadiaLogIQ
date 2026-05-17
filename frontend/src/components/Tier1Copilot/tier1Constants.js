// Sprint 6 — Tier-1 Copilot dropdown options.
// Single source of truth so the intake form and answer card agree.

export const SEVERITY_OPTIONS = [
  { value: "P1", label: "P1 — Critical" },
  { value: "P2", label: "P2 — High" },
  { value: "P3", label: "P3 — Medium" },
  { value: "P4", label: "P4 — Low" },
];

export const TECHNOLOGY_OPTIONS = [
  { value: "", label: "— Select (optional) —" },
  { value: "Citrix", label: "Citrix / VDI" },
  { value: "BGP", label: "BGP / Routing" },
  { value: "OSPF", label: "OSPF" },
  { value: "DNS", label: "DNS" },
  { value: "DHCP", label: "DHCP" },
  { value: "VPN", label: "VPN" },
  { value: "SDWAN", label: "SD-WAN" },
  { value: "Firewall", label: "Firewall" },
  { value: "LoadBalancer", label: "Load Balancer" },
  { value: "Storage", label: "Storage" },
  { value: "Database", label: "Database" },
  { value: "Email", label: "Email / Exchange" },
  { value: "Other", label: "Other" },
];

export const FOLLOWUP_ACTIONS = [
  { key: "next_best_solution", label: "Show next-best solution" },
  { key: "deeper_diagnostics", label: "Deeper diagnostics" },
  { key: "escalation_note", label: "Generate escalation note" },
  { key: "search_kb_sop", label: "Search KB / SOP" },
  { key: "explain_recommendation", label: "Explain this recommendation" },
];

export const CONFIDENCE_COLOR = {
  High: "#0A7A3F",
  Medium: "#C9870B",
  Low: "#B03A2E",
  None: "#6B6B6B",
};

// Sprint 7 — trust-calibrated labels (legacy strings kept for Sprint 7
// callers that pass `progressive` but not `uxFixes`).
export const CONFIDENCE_LABEL = {
  High: "Strong match",
  Medium: "Partial match — verify before acting",
  Low: "Weak match — use as reference only",
  None: "No historical evidence",
};

// Sprint 7 — LandingRouter mounts the Tier1Workspace container in
// place of the Sprint 6 single-card flow.
export const TIER1_PROGRESSIVE_ON = true;

// ─────────────────────────────────────────────────────────────
// Sprint 8 — UX polish + rewritten confidence copy
// Track A: confidence labels, wired buttons, skeletons, arrow
// pagination fetch, intake redesign.
// Track B: modern Acadia-navy theme.
// ─────────────────────────────────────────────────────────────
export const TIER1_UX_FIXES_ON = true;

export const TIER1_MODERN_THEME_ON = true;

// Sprint 8.1 demo — download buttons for UAT review.
export const TIER1_DOWNLOAD_DEMO_ON = true;

// ─────────────────────────────────────────────────────────────
// Sprint 9 — Universal Intake (Email/Phone/Portal/Chat/Note).
// ─────────────────────────────────────────────────────────────
export const UNIVERSAL_INTAKE_ON = true;

export const INTAKE_SOURCES = [
  { value: "alert", label: "Alert" },
  { value: "email", label: "Email" },
  { value: "phone", label: "Phone" },
  { value: "portal", label: "Portal" },
  { value: "chat", label: "Chat" },
  { value: "note", label: "Note" },
];

// Sprint 11 — Top-level intake mode. Replaces the 6-button source
// picker (Alert/Email/Phone/Portal/Chat/Note) with a 2-button choice:
//   proactive → Alert (machine-generated monitoring signal)
//   reactive  → Email / Phone / Portal / Chat / Note (human-reported)
// The original INTAKE_SOURCES list is preserved as the sub-channel
// catalog inside the Reactive panel — backend extraction templates
// still key off the channel string, so we keep the signal.
export const INTAKE_MODES = [
  { value: "proactive", label: "Proactive", source: "alert" },
  { value: "reactive", label: "Reactive", defaultSource: "email" },
];

// Reactive sub-channel options — every INTAKE_SOURCES entry except
// "alert" (which is the Proactive mode itself).
export const REACTIVE_SUB_CHANNELS = INTAKE_SOURCES.filter(
  (s) => s.value !== "alert",
);

export const INTAKE_MAX_RAW_CHARS = 10000;

// ─────────────────────────────────────────────────────────────
// Sprint 10 — Tier-1 Resolution Journey. ResolutionJourney replaces
// the answer card after Analyze.
// ─────────────────────────────────────────────────────────────
export const TIER1_JOURNEY_ON = true;

export const STAGE_LABELS = {
  environment_context: "Environment Context & Tech Component Profile",
  stage_0: "Best Historical Match & Recommended Resolution",
  stage_1a: "Smoking Gun",
  stage_1b: "Do Not Chase",
  stage_2: "Related Incidents & Probable Causes",
  stage_3: "Guided Troubleshooting Workflow",
  stage_4: "Knowledge Base & SOP Reference",
  stage_5: "Operational Handoff",
};

// Neutralised confidence copy (§3). Primary + subline so the banner
// can show calibration without the harsh "Weak" language that eroded
// engineer trust in Sprint 7.
export const CONFIDENCE_LABEL_V2 = {
  High: {
    primary: "Best match",
    subline: "High similarity to your alert",
    tone: "positive",
  },
  Medium: {
    primary: "Strong candidate",
    subline: "Closely related — review and apply carefully",
    tone: "positive",
  },
  Low: {
    primary: "Closest historical case",
    subline: "Limited overlap — use for reference",
    tone: "neutral",
  },
  None: {
    primary: "No close match found",
    subline: "No historical incident matches this alert pattern",
    tone: "empty",
  },
};
