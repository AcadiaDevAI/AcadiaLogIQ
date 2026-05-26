// Button showcase — uichanges.md Prompt 01 deliverable.
//
// Eyeball verification of the 4-tier button system across sizes and
// states. Mount it temporarily wherever you want (no router wired by
// default — the app uses imperative right-pane takeover, not URL
// routing — so this is a drop-in component for diagnostic viewing).
//
// Usage:
//   import ButtonShowcase from "./theme/ButtonShowcase";
//   // ... render <ButtonShowcase /> in your dev environment

import React from "react";
import { Button, Space, Typography } from "antd";
import {
  PlusOutlined, DownloadOutlined, ReloadOutlined, DeleteOutlined,
} from "@ant-design/icons";

const { Title, Text } = Typography;


const HEADER_STYLE = {
  fontFamily: "var(--font-display)",
  fontSize: 28,
  color: "var(--text)",
  marginBottom: 8,
  marginTop: 32,
  letterSpacing: "-0.01em",
};

const SUB_STYLE = {
  fontFamily: "var(--font-mono)",
  fontSize: 11,
  color: "var(--text-dim)",
  textTransform: "uppercase",
  letterSpacing: "0.1em",
  marginBottom: 16,
};

const TIER_LABEL = {
  fontFamily: "var(--font-mono)",
  fontSize: 11,
  color: "var(--text-muted)",
  textTransform: "uppercase",
  letterSpacing: "0.08em",
  marginBottom: 8,
};


function Row({ label, children }) {
  return (
    <div style={{ marginBottom: 18 }}>
      <div style={TIER_LABEL}>{label}</div>
      <Space size="middle" wrap>{children}</Space>
    </div>
  );
}


export default function ButtonShowcase() {
  return (
    <div style={{ padding: 48, maxWidth: 1100, margin: "0 auto" }}>
      <Title style={HEADER_STYLE} level={1}>
        Acadia button system
      </Title>
      <div style={SUB_STYLE}>4 tiers × 3 sizes × 5 states</div>

      {/* ─ PRIMARY ────────────────────────────── */}
      <Row label="01 · Primary — aurora gradient">
        <Button type="primary" size="small">Small</Button>
        <Button type="primary">Default (38px)</Button>
        <Button type="primary" size="large" icon={<PlusOutlined />}>
          Large with icon
        </Button>
        <Button type="primary" loading>Loading</Button>
        <Button type="primary" disabled>Disabled</Button>
      </Row>

      {/* ─ SECONDARY ──────────────────────────── */}
      <Row label="02 · Secondary — glass">
        <Button size="small">Small</Button>
        <Button icon={<DownloadOutlined />}>Default</Button>
        <Button size="large" icon={<ReloadOutlined />}>Large with icon</Button>
        <Button loading>Loading</Button>
        <Button disabled>Disabled</Button>
      </Row>

      {/* ─ GHOST ──────────────────────────────── */}
      <Row label="03 · Ghost — tertiary">
        <Button type="text" size="small">Small</Button>
        <Button type="text">Default</Button>
        <Button type="text" size="large">Large</Button>
        <Button type="text" loading>Loading</Button>
        <Button type="text" disabled>Disabled</Button>
      </Row>

      {/* ─ DESTRUCTIVE ────────────────────────── */}
      <Row label="04 · Destructive — soft coral">
        <Button danger size="small">Small</Button>
        <Button danger icon={<DeleteOutlined />}>Default</Button>
        <Button danger size="large" icon={<DeleteOutlined />}>
          Large with icon
        </Button>
        <Button danger loading>Loading</Button>
        <Button danger disabled>Disabled</Button>
      </Row>

      <hr className="premium-divider" style={{ margin: "40px 0 24px" }} />

      <Title style={HEADER_STYLE} level={2}>
        State matrix
      </Title>
      <div style={SUB_STYLE}>idle · hover · active · focus · loading · disabled</div>
      <Text style={{ color: "var(--text-muted)", fontSize: 13 }}>
        Hover, click, tab-focus, and observe — every tier scales / glows
        identically. Focus ring is aurora teal (no default blue).
      </Text>
    </div>
  );
}
