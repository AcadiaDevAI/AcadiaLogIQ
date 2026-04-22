import React from "react";
import { Tag } from "antd";

/**
 * ModeBadge — compact visual for the locked mode + sub-mode.
 * Rendered by ChatArea once selectedMode is set.
 * Kept intentionally minimal so later sprints can restyle without churn.
 */
const MODE_LABELS = {
  troubleshooting: "Troubleshooting",
  ticket_handling: "Ticket Handling",
  escalation: "Escalation",
  vendor_oem: "Vendor / OEM",
};

const SUB_MODE_LABELS = {
  customer_specific: "Customer",
  technology_specific: "Technology",
  ticket_create: "Create",
  ticket_update: "Update",
  ticket_close: "Close",
  ticket_validate: "Validate",
};

export default function ModeBadge({ mode, subMode }) {
  if (!mode) return null;
  const main = MODE_LABELS[mode] || mode;
  const sub = subMode ? SUB_MODE_LABELS[subMode] || subMode : null;
  return (
    <Tag
      color="blue"
      className="text-[11px] font-medium rounded-md px-2 py-0.5"
      style={{ backgroundColor: "var(--brand-light)", color: "var(--brand-accent)", borderColor: "var(--brand-accent)" }}
    >
      Mode: {main}{sub ? ` · ${sub}` : ""}
    </Tag>
  );
}
