import React from "react";
import { useTier1Theme } from "../../theme/ThemeProvider";

/**
 * Sprint 8 — SeverityChipSelector
 *
 * Four P1..P4 buttons with a one-click select. Renders as a grid that
 * wraps to 2x2 on narrow widths. Visual priority colours are muted
 * tints in the modern theme (per §6.2 — no raw red/yellow).
 */
const TONE = {
  P1: { bg: "#FDECEC", text: "#B03A2E", ring: "#B03A2E" },
  P2: { bg: "#FEF6E6", text: "#C9870B", ring: "#C9870B" },
  P3: { bg: "#E8F0FE", text: "#1E40AF", ring: "#3B82F6" },
  P4: { bg: "#ECFDF5", text: "#065F46", ring: "#10B981" },
};

const LABELS = [
  { value: "P1", label: "P1 — Critical" },
  { value: "P2", label: "P2 — High" },
  { value: "P3", label: "P3 — Medium" },
  { value: "P4", label: "P4 — Low" },
];

export default function SeverityChipSelector({ value, onChange, disabled }) {
  const { tokens, isModern } = useTier1Theme();

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))",
        gap: 8,
      }}
      role="radiogroup"
      aria-label="Severity"
    >
      {LABELS.map(({ value: v, label }) => {
        const selected = value === v;
        const tone = TONE[v] || TONE.P3;
        const pad = isModern ? "10px 12px" : "8px 10px";
        const radius = tokens.radiusMd || "12px";
        return (
          <button
            key={v}
            type="button"
            role="radio"
            aria-checked={selected}
            disabled={disabled}
            onClick={() => !disabled && onChange(v)}
            style={{
              padding: pad,
              borderRadius: radius,
              cursor: disabled ? "not-allowed" : "pointer",
              opacity: disabled ? 0.6 : 1,
              background: selected
                ? isModern
                  ? tokens.gradientAccent
                  : tone.bg
                : isModern
                ? tokens.surfaceElevated
                : tone.bg,
              color: selected
                ? isModern
                  ? "#FFFFFF"
                  : tone.text
                : tone.text,
              border: `2px solid ${selected ? tone.ring : "transparent"}`,
              fontWeight: 600,
              fontSize: 14,
              transition: tokens.transitionFast || "150ms ease",
              boxShadow:
                selected && isModern ? tokens.shadowSm : "none",
              textAlign: "center",
            }}
          >
            {label}
          </button>
        );
      })}
    </div>
  );
}
