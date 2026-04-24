import React from "react";
import { Tag } from "antd";
import { ClockCircleOutlined } from "@ant-design/icons";

/**
 * Sprint 7 — SessionTimer
 * Pure presentational — renders elapsed seconds as "Xm Ys".
 */
export default function SessionTimer({ elapsedSeconds = 0 }) {
  const s = Math.max(0, Math.floor(elapsedSeconds));
  const min = Math.floor(s / 60);
  const sec = s % 60;
  const label = min > 0 ? `${min}m ${sec}s` : `${sec}s`;
  return (
    <Tag icon={<ClockCircleOutlined />} color="blue" style={{ fontWeight: 500 }}>
      {label}
    </Tag>
  );
}
