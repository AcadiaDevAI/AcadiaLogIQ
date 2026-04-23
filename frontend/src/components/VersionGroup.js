import React, { useState } from "react";
import { Button, Tooltip, Popconfirm } from "antd";
import {
  FileOutlined,
  DeleteOutlined,
  CaretRightOutlined,
  CaretDownOutlined,
} from "@ant-design/icons";

// Sprint 3-PREP-B — doc_kind badge (mirrors Sidebar.js; intentionally
// duplicated to avoid a cross-file export cycle since Sidebar imports
// this component). Flag-gated so flag-off renders nothing.
const KIND_LABELS = {
  ticket: "tickets",
  sop: "sop",
  kb: "kb",
  contact_customer: "cust",
  contact_vendor: "vnd",
  vendor_case: "case",
};
const KIND_COLORS = {
  ticket: "#3b82f6",
  sop: "#10b981",
  kb: "#8b5cf6",
  contact_customer: "#f59e0b",
  contact_vendor: "#ef4444",
  vendor_case: "#ec4899",
};
const BULK_INGEST_ENABLED =
  (process.env.REACT_APP_LOGIQ_BULK_INGEST_FRONTEND || "false").toLowerCase() === "true";

function DocKindBadge({ kind }) {
  if (!BULK_INGEST_ENABLED) return null;
  const label = KIND_LABELS[kind];
  if (!label) return null;
  return (
    <span
      className="text-[9px] px-1.5 py-0.5 rounded flex-shrink-0"
      style={{ backgroundColor: KIND_COLORS[kind] || "#64748b", color: "#fff" }}
      title={`doc_kind: ${kind}`}
    >
      {label}
    </span>
  );
}

/**
 * Groups multiple versions of the same document family under a single
 * collapsible row. Expects `versions` to be sorted descending by version_rank
 * (newest first). The first entry is rendered as the visible "latest" row;
 * the rest appear when the group is expanded.
 */
export default function VersionGroup({ versions, isAdmin, onDelete }) {
  const [expanded, setExpanded] = useState(false);

  if (!versions || versions.length === 0) return null;

  const latest = versions[0];
  const olderCount = versions.length - 1;

  const formatDate = (iso) => {
    if (!iso) return "";
    try {
      const d = new Date(iso);
      if (isNaN(d.getTime())) return "";
      return d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
    } catch {
      return "";
    }
  };

  const renderDelete = (f) =>
    isAdmin && (
      <Popconfirm
        title={`Delete "${f.name}"?`}
        description="This will remove the file and all its indexed data."
        onConfirm={() => onDelete(f.id, f.name)}
        okText="Delete"
        cancelText="Cancel"
        okButtonProps={{ danger: true }}
      >
        <Tooltip title="Delete file">
          <Button
            type="text"
            size="small"
            icon={<DeleteOutlined />}
            className="opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0"
            style={{ color: "var(--text-muted)" }}
            danger
          />
        </Tooltip>
      </Popconfirm>
    );

  return (
    <div
      className="flex flex-col rounded-lg t-bg-tertiary border"
      style={{ borderColor: "var(--border-color)" }}
    >
      {/* Header row — latest version + expand toggle */}
      <div
        className="group flex items-center justify-between px-3 py-2 cursor-pointer"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-2 min-w-0 flex-1">
          {expanded ? (
            <CaretDownOutlined style={{ color: "var(--text-muted)", fontSize: 10 }} />
          ) : (
            <CaretRightOutlined style={{ color: "var(--text-muted)", fontSize: 10 }} />
          )}
          <FileOutlined style={{ color: "#6366f1" }} />
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-1.5">
              <p className="text-xs t-text truncate max-w-[120px]">{latest.name}</p>
              <DocKindBadge kind={latest.doc_kind} />
              <span
                className="text-[9px] px-1.5 py-0.5 rounded"
                style={{ backgroundColor: "#10b981", color: "#fff" }}
              >
                Latest
              </span>
              {olderCount > 0 && (
                <span
                  className="text-[9px] px-1.5 py-0.5 rounded"
                  style={{ backgroundColor: "var(--border-color)", color: "var(--text-muted)" }}
                >
                  +{olderCount} older
                </span>
              )}
            </div>
            <p className="text-[10px] t-text-muted">
              {latest.size_mb != null ? `${latest.size_mb.toFixed(1)}MB · ` : ""}
              <span
                style={{
                  color:
                    latest.status === "indexed"
                      ? "#10b981"
                      : latest.status === "failed"
                      ? "#ef4444"
                      : "#f59e0b",
                }}
              >
                {latest.status}
              </span>
            </p>
          </div>
        </div>
        <div onClick={(e) => e.stopPropagation()}>{renderDelete(latest)}</div>
      </div>

      {/* Expanded version list */}
      {expanded && (
        <div
          className="flex flex-col border-t"
          style={{ borderColor: "var(--border-color)" }}
        >
          {versions.map((f, idx) => {
            const rank = f.version_rank != null ? f.version_rank : versions.length - idx;
            const isActive = idx === 0;
            return (
              <div
                key={f.id}
                className="group flex items-center justify-between px-3 py-1.5 pl-8"
                style={{
                  backgroundColor: isActive ? "var(--brand-light, rgba(10,63,99,0.06))" : "transparent",
                }}
              >
                <div className="flex items-center gap-2 min-w-0 flex-1">
                  <span className="text-[10px] t-text-muted font-mono">v{rank}</span>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5">
                      <p className="text-[11px] t-text truncate max-w-[120px]">{f.name}</p>
                      {isActive && (
                        <span
                          className="text-[9px] px-1 py-0.5 rounded"
                          style={{ backgroundColor: "#0A3F63", color: "#fff" }}
                        >
                          Active
                        </span>
                      )}
                    </div>
                    <p className="text-[10px] t-text-muted">
                      {formatDate(f.created_at)}
                    </p>
                  </div>
                </div>
                {renderDelete(f)}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
