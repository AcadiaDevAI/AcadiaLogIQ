import React from "react";
import { Button, Tooltip } from "antd";
import { BgColorsOutlined } from "@ant-design/icons";
import { useTier1Theme } from "../../theme/ThemeProvider";

/**
 * Sprint 8 — ThemeToggle
 *
 * Renders ONLY when REACT_APP_LOGIQ_TIER1_MODERN_THEME=true. Clicking
 * flips the per-user localStorage preference between "modern" and
 * "classic"; sibling tabs stay in sync via the `storage` event the
 * useLocalStorage hook listens to.
 */
export default function ThemeToggle() {
  const { flagOn, preference, togglePreference } = useTier1Theme();
  if (!flagOn) return null;
  const label =
    preference === "modern" ? "Switch to classic" : "Switch to modern";
  return (
    <Tooltip title={label}>
      <Button
        size="small"
        icon={<BgColorsOutlined />}
        onClick={togglePreference}
        aria-label={label}
      >
        {preference === "modern" ? "Classic" : "Modern"}
      </Button>
    </Tooltip>
  );
}
