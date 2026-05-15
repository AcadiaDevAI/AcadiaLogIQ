// Sprint 13.32 — RCA-from-incident-number right-pane flow.
//
// Sidebar's "RCA" button flips AppLayout's rcaOpen flag, and this
// component takes over the right pane. The engineer enters a ticket
// number, submits, and the backend runs both LLM calls in parallel
// returning two Markdown blobs. We render them in a two-panel
// AntD Collapse. Bottom-right button returns to the prior surface
// (whatever AppLayout was rendering before — chat, landing, or a
// live Tier-1 journey).
//
// Failure paths:
//   * 404 (ticket not found) → inline form-level error, retry stays
//     open with the input prefilled so the engineer can fix a typo.
//   * 500 / network error    → full-pane error state + Try again.
//   * Per-panel LLM failure  → the OTHER panel still renders; the
//     failed panel shows a small error banner inside its body so
//     the engineer can re-submit just the form to try again.

import React, { useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github.css";
import {
  Alert,
  Button,
  Card,
  Collapse,
  Empty,
  Input,
  Space,
  Spin,
  Typography,
  message,
} from "antd";
import {
  ArrowLeftOutlined,
  CopyOutlined,
  FilePdfOutlined,
  FileSearchOutlined,
  FileWordOutlined,
  LoadingOutlined,
} from "@ant-design/icons";

import { generateRCA } from "./rcaApi";
import { exportPdf, exportWord } from "./rcaExport";


const { Title, Paragraph, Text } = Typography;


// ─────────────────────────────────────────────────────────────
// PDF-style document CSS, scoped to .rca-markdown so nothing
// leaks to the chat pipeline. Injected once via a module guard.
//
// Design intent:
//   * Section headers (h2) read like a printed report — heavier
//     weight, comfortable top margin, a hairline underline that
//     visually separates each section.
//   * h1 acts as the document title (the prompt's header block).
//   * Tables get a real header-row fill + cell borders so they
//     read as data, not prose.
//   * Bullet/numbered lists have breathing room between items.
//   * Blockquote is a callout (left accent bar + tinted bg) so
//     the Internal RCA's "Definitive Root Cause" pops.
//   * Code-blocks (Internal RCA's CLI / Ansible snippets) get a
//     dark monospace card with comfortable padding.
//   * Dark-mode-aware via CSS variables already in the app shell.
// ─────────────────────────────────────────────────────────────
let _rcaStylesInjected = false;
function _ensureRcaStyles() {
  if (typeof document === "undefined" || _rcaStylesInjected) return;
  const style = document.createElement("style");
  style.setAttribute("data-acadia-rca", "1");
  style.textContent = `
    .rca-paper {
      background: var(--bg-elevated, #ffffff);
      color: var(--text-primary, #1f2937);
      padding: 40px 48px;
      border-radius: 6px;
      box-shadow: 0 1px 2px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
      max-width: 900px;
      margin: 0 auto;
    }
    @media (max-width: 720px) {
      .rca-paper { padding: 24px 20px; }
    }
    .rca-markdown {
      font-family: 'Poppins', 'Inter', system-ui, -apple-system, Segoe UI, sans-serif;
      line-height: 1.65;
      font-size: 14px;
      color: inherit;
    }
    .rca-markdown h1 {
      font-size: 24px;
      font-weight: 700;
      margin: 0 0 4px 0;
      letter-spacing: -0.01em;
      color: inherit;
    }
    .rca-markdown h1 + p,
    .rca-markdown h1 + p strong {
      font-size: 15px;
      color: var(--text-secondary, #475569);
    }
    .rca-markdown h2 {
      font-size: 17px;
      font-weight: 700;
      margin: 32px 0 12px 0;
      padding-bottom: 6px;
      border-bottom: 1px solid var(--border-color, #e5e7eb);
      letter-spacing: -0.005em;
      color: inherit;
    }
    .rca-markdown h2:first-child { margin-top: 0; }
    .rca-markdown h3 {
      font-size: 14px;
      font-weight: 600;
      margin: 20px 0 8px 0;
      color: inherit;
    }
    .rca-markdown p {
      margin: 0 0 12px 0;
    }
    .rca-markdown strong { font-weight: 600; }
    .rca-markdown em { font-style: italic; }
    .rca-markdown ul,
    .rca-markdown ol {
      margin: 0 0 14px 0;
      padding-left: 24px;
    }
    .rca-markdown li {
      margin-bottom: 6px;
    }
    .rca-markdown li > p {
      margin-bottom: 4px;
    }
    .rca-markdown hr {
      border: none;
      border-top: 1px solid var(--border-color, #e5e7eb);
      margin: 24px 0;
    }
    /* Tables — PDF-style: header-row fill, cell borders, comfortable padding */
    .rca-markdown table {
      width: 100%;
      border-collapse: collapse;
      margin: 12px 0 18px 0;
      font-size: 13.5px;
      border: 1px solid var(--border-color, #e5e7eb);
    }
    .rca-markdown thead {
      background: var(--brand-light, #f3f4f6);
    }
    .rca-markdown th,
    .rca-markdown td {
      padding: 10px 12px;
      border: 1px solid var(--border-color, #e5e7eb);
      text-align: left;
      vertical-align: top;
    }
    .rca-markdown th {
      font-weight: 600;
      color: var(--text-primary, #1f2937);
    }
    .rca-markdown tbody tr:nth-child(even) td {
      background: var(--bg-subtle, #fafafa);
    }
    /* Blockquote — definitive root cause callout */
    .rca-markdown blockquote {
      margin: 14px 0;
      padding: 12px 16px;
      border-left: 4px solid var(--brand-accent, #100d4b);
      background: var(--brand-light, #f5f5fa);
      border-radius: 0 4px 4px 0;
      color: inherit;
    }
    .rca-markdown blockquote p:last-child { margin-bottom: 0; }
    /* Inline code */
    .rca-markdown code {
      font-family: 'JetBrains Mono', 'Fira Code', Consolas, Menlo, monospace;
      font-size: 12.5px;
      padding: 1px 6px;
      background: var(--bg-subtle, #f1f5f9);
      border-radius: 3px;
      color: var(--text-primary, #0f172a);
    }
    /* Fenced code blocks — CLI / Ansible / yaml snippets */
    .rca-markdown pre {
      background: #0f172a;
      color: #e2e8f0;
      padding: 14px 16px;
      border-radius: 6px;
      overflow-x: auto;
      margin: 12px 0 18px 0;
      font-size: 12.5px;
      line-height: 1.55;
    }
    .rca-markdown pre code {
      background: transparent;
      padding: 0;
      color: inherit;
      font-size: inherit;
    }
    /* Dark-mode adjustments */
    [data-theme="dark"] .rca-paper,
    .dark .rca-paper {
      background: #1e1e28;
      color: #e2e8f0;
      box-shadow: 0 1px 2px rgba(0,0,0,0.3), 0 4px 16px rgba(0,0,0,0.2);
    }
    [data-theme="dark"] .rca-markdown h2,
    .dark .rca-markdown h2 { border-bottom-color: #2a2a3d; }
    [data-theme="dark"] .rca-markdown table,
    .dark .rca-markdown table,
    [data-theme="dark"] .rca-markdown th,
    .dark .rca-markdown th,
    [data-theme="dark"] .rca-markdown td,
    .dark .rca-markdown td { border-color: #2a2a3d; }
    [data-theme="dark"] .rca-markdown thead,
    .dark .rca-markdown thead { background: #2a2a3d; }
    [data-theme="dark"] .rca-markdown tbody tr:nth-child(even) td,
    .dark .rca-markdown tbody tr:nth-child(even) td { background: #16161d; }
    [data-theme="dark"] .rca-markdown blockquote,
    .dark .rca-markdown blockquote {
      background: rgba(99, 102, 241, 0.08);
      border-left-color: #6366f1;
    }
    [data-theme="dark"] .rca-markdown code,
    .dark .rca-markdown code { background: #16161d; color: #e2e8f0; }
  `;
  document.head.appendChild(style);
  _rcaStylesInjected = true;
}


// Local markdownComponents — kept minimal because the heavy lifting
// is done in scoped CSS above. We only intercept `code` to suppress
// the chat pipeline's CopyButton wrapper (not appropriate inside a
// document body), and `a` to force same-window navigation for any
// inline links the LLM emits.
const rcaMarkdownComponents = {
  a: ({ href, children, ...rest }) => (
    <a href={href} target="_blank" rel="noopener noreferrer" {...rest}>
      {children}
    </a>
  ),
};


// Render wrapper — wraps the markdown in a "paper" container so each
// panel reads like a printed report rather than chat bubble prose.
// Sprint 13.32.10 — forwarded ref points at the .rca-paper element so
// the Export PDF / Export Word buttons can capture the exact DOM the
// engineer sees on screen (no separate render path, no style drift).
const RCAMarkdown = React.forwardRef(function RCAMarkdown({ source }, ref) {
  if (!source) return null;
  return (
    <div className="rca-paper" ref={ref}>
      <div className="rca-markdown">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[rehypeHighlight]}
          components={rcaMarkdownComponents}
        >
          {source}
        </ReactMarkdown>
      </div>
    </div>
  );
});


// Small "copy to clipboard" affordance — useful since the engineer
// usually wants to paste the customer-facing RCA into email or
// the internal one into Confluence / a wiki.
function CopyMarkdownButton({ source, label }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = async () => {
    if (!source) return;
    try {
      await navigator.clipboard.writeText(source);
      setCopied(true);
      message.success(`${label} copied`);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      message.error("Copy failed — your browser may have blocked clipboard access.");
    }
  };
  return (
    <Button
      size="small"
      icon={<CopyOutlined />}
      onClick={(e) => {
        e.stopPropagation();
        handleCopy();
      }}
      disabled={!source}
    >
      {copied ? "Copied" : "Copy"}
    </Button>
  );
}


// Sprint 13.32.10 — Export to PDF / Word.
//
// Both buttons operate on the live DOM node the panel rendered
// (passed in via `getNode`), so the export captures the exact same
// PDF-style layout the engineer sees on screen. PDF generation is
// async (html2pdf returns a promise) and busies the button while
// running. Word generation is synchronous from the caller's POV.
function ExportButtons({ getNode, filename, source }) {
  const [busyPdf, setBusyPdf] = useState(false);

  const handleExportPdf = async (e) => {
    e.stopPropagation();
    const node = typeof getNode === "function" ? getNode() : null;
    if (!node) {
      message.error("Nothing to export yet — wait for the report to render.");
      return;
    }
    setBusyPdf(true);
    try {
      await exportPdf(node, filename);
      message.success("PDF download started.");
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error("[rca.export] PDF failed", err);
      message.error("PDF export failed — please try again.");
    } finally {
      setBusyPdf(false);
    }
  };

  const handleExportWord = (e) => {
    e.stopPropagation();
    const node = typeof getNode === "function" ? getNode() : null;
    if (!node) {
      message.error("Nothing to export yet — wait for the report to render.");
      return;
    }
    try {
      exportWord(node, filename);
      message.success("Word download started.");
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error("[rca.export] Word failed", err);
      message.error("Word export failed — please try again.");
    }
  };

  return (
    <>
      <Button
        size="small"
        icon={busyPdf ? <LoadingOutlined /> : <FilePdfOutlined />}
        onClick={handleExportPdf}
        disabled={!source || busyPdf}
      >
        PDF
      </Button>
      <Button
        size="small"
        icon={<FileWordOutlined />}
        onClick={handleExportWord}
        disabled={!source}
      >
        Word
      </Button>
    </>
  );
}


export default function RCAFlow({ onReturnToStages, initialPayload = null }) {
  // Sprint 13.32 — inject the PDF-style document CSS on first mount.
  // Idempotent; subsequent mounts hit the module guard and no-op.
  React.useEffect(() => {
    _ensureRcaStyles();
  }, []);

  // Sprint 13.32.10 — refs to each panel's .rca-paper node so the
  // PDF / Word export buttons can capture the rendered DOM directly.
  // Refs (vs. re-rendering through a memo) means the export pipeline
  // ships exactly what's on screen — no parallel render, no drift.
  const customerPaperRef = useRef(null);
  const internalPaperRef = useRef(null);

  const [incidentNumber, setIncidentNumber] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  // Form-level error (404 / network) is distinct from per-panel
  // LLM errors carried inside `result`.
  const [formError, setFormError] = useState(null);

  // Sprint 13.32.6 — panel-visibility flags from the modal. Default
  // to both visible so the legacy "open RCAFlow directly with a
  // ticket number in the form" path still produces both panels.
  // When the modal hands us a payload, we honour the engineer's
  // exact selection (one, two, or both — modal enforces ≥1).
  const [showCustomer, setShowCustomer] = useState(true);
  const [showInternal, setShowInternal] = useState(true);
  // File-upload placeholder. We keep the file on state so the engineer
  // sees what they picked, but the backend parsing path is intentionally
  // stubbed in this iteration — the modal said "feature in progress."
  const [uploadedFile, setUploadedFile] = useState(null);

  const canSubmit = !busy && !!incidentNumber.trim();

  const handleSubmit = async (opts = {}) => {
    const inc = (opts.incidentNumber ?? incidentNumber).trim();
    if (!inc || busy) return;
    setBusy(true);
    setFormError(null);
    setResult(null);
    try {
      const data = await generateRCA(inc);
      setResult(data);
    } catch (err) {
      const status = err?.response?.status;
      if (status === 404) {
        setFormError(
          `No ticket found with incident number "${inc}". `
          + "Check the number and try again.",
        );
      } else if (status === 400) {
        setFormError("Please enter a ticket number.");
      } else {
        // eslint-disable-next-line no-console
        console.error("[RCAFlow] generate failed", err);
        setFormError(
          "Something went wrong generating the RCA. "
          + "Please try again in a moment.",
        );
      }
    } finally {
      setBusy(false);
    }
  };

  // Sprint 13.32.6 — when AppLayout hands us an `initialPayload`
  // (from the entry modal), seed the form state + panel filters
  // and auto-run the request. `initialPayload.file` is captured for
  // the UX scaffold but the backend Excel-parse pipeline is pending,
  // so we surface a friendly "feature in progress" notice instead
  // of pretending the upload was processed.
  React.useEffect(() => {
    if (!initialPayload) return;
    if (initialPayload.panels) {
      setShowCustomer(!!initialPayload.panels.external);
      setShowInternal(!!initialPayload.panels.internal);
    }
    if (initialPayload.file) {
      setUploadedFile(initialPayload.file);
    }
    if (initialPayload.incidentNumber) {
      setIncidentNumber(initialPayload.incidentNumber);
      handleSubmit({ incidentNumber: initialPayload.incidentNumber });
    }
    // Intentionally one-shot — the prop only changes when the modal
    // re-opens, which won't happen while RCAFlow is mounted.
  }, [initialPayload]);

  const handleStartOver = () => {
    setResult(null);
    setFormError(null);
    setIncidentNumber("");
    setUploadedFile(null);
  };

  // Convenience: only-file submission with no incident number ⇒
  // show the placeholder card instead of a request error.
  const fileOnlyMode = !!uploadedFile && !incidentNumber.trim() && !result;

  // Collapse items — Customer-Facing first because it's the shorter
  // executive view; Internal is the long-form technical follow-up.
  // Sprint 13.32.6 — each panel is only included when the engineer
  // ticked it in the entry modal (legacy direct-form usage defaults
  // to both visible). Prompts, request, and rendering are unchanged
  // — we just drop panels the engineer didn't ask for.
  const allCollapseItems = result
    ? [
        {
          key: "customer",
          label: (
            <Space>
              <Text strong style={{ fontSize: 15 }}>
                Customer-Facing External RCA
              </Text>
              <Text type="secondary" style={{ fontSize: 12 }}>
                — Approved for external distribution
              </Text>
            </Space>
          ),
          extra: (
            <Space size={6} onClick={(e) => e.stopPropagation()}>
              <ExportButtons
                getNode={() => customerPaperRef.current}
                filename={`${result.incident_number || "RCA"}_Customer-Facing-RCA`}
                source={result.customer_facing_md}
              />
              <CopyMarkdownButton
                source={result.customer_facing_md}
                label="Customer-Facing RCA"
              />
            </Space>
          ),
          children: (
            <>
              {result.customer_facing_error ? (
                <Alert
                  type="warning"
                  showIcon
                  message={result.customer_facing_error}
                  style={{ marginBottom: 12 }}
                />
              ) : null}
              {result.customer_facing_md ? (
                <RCAMarkdown source={result.customer_facing_md} ref={customerPaperRef} />
              ) : !result.customer_facing_error ? (
                <Empty
                  image={Empty.PRESENTED_IMAGE_SIMPLE}
                  description="No Customer-Facing RCA returned."
                />
              ) : null}
            </>
          ),
        },
        {
          key: "internal",
          label: (
            <Space>
              <Text strong style={{ fontSize: 15 }}>
                Internal Incident RCA
              </Text>
              <Text type="secondary" style={{ fontSize: 12 }}>
                — Confidential, internal use only
              </Text>
            </Space>
          ),
          extra: (
            <Space size={6} onClick={(e) => e.stopPropagation()}>
              <ExportButtons
                getNode={() => internalPaperRef.current}
                filename={`${result.incident_number || "RCA"}_Internal-RCA`}
                source={result.internal_md}
              />
              <CopyMarkdownButton
                source={result.internal_md}
                label="Internal RCA"
              />
            </Space>
          ),
          children: (
            <>
              {result.internal_error ? (
                <Alert
                  type="warning"
                  showIcon
                  message={result.internal_error}
                  style={{ marginBottom: 12 }}
                />
              ) : null}
              {result.internal_md ? (
                <RCAMarkdown source={result.internal_md} ref={internalPaperRef} />
              ) : !result.internal_error ? (
                <Empty
                  image={Empty.PRESENTED_IMAGE_SIMPLE}
                  description="No Internal RCA returned."
                />
              ) : null}
            </>
          ),
        },
      ]
    : [];

  const collapseItems = allCollapseItems.filter((it) => {
    if (it.key === "customer") return showCustomer;
    if (it.key === "internal") return showInternal;
    return true;
  });

  return (
    <div className="flex flex-col h-full w-full t-bg-primary">
      {/* Scrollable body */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="w-full max-w-5xl mx-auto">
          {/* Header */}
          <div style={{ marginBottom: 16 }}>
            <Title level={3} style={{ marginBottom: 4 }}>
              RCA Report Generator
            </Title>
            <Paragraph type="secondary" style={{ marginBottom: 0 }}>
              Enter a historical incident number to generate two RCA
              documents in parallel — one for customer/stakeholder
              distribution and one for internal engineering review.
            </Paragraph>
          </div>

          {/* Ticket-number form */}
          <Card style={{ marginBottom: 16 }} bodyStyle={{ padding: 16 }}>
            <Space.Compact style={{ width: "100%" }}>
              <Input
                placeholder="e.g. INC-LAN-88902"
                value={incidentNumber}
                onChange={(e) => setIncidentNumber(e.target.value)}
                onPressEnter={handleSubmit}
                disabled={busy}
                size="large"
                prefix={<FileSearchOutlined style={{ color: "#94a3b8" }} />}
                aria-label="Incident number"
              />
              <Button
                type="primary"
                size="large"
                onClick={handleSubmit}
                disabled={!canSubmit}
                icon={busy ? <LoadingOutlined /> : null}
              >
                {busy ? "Generating…" : "Generate RCA"}
              </Button>
              {result || formError ? (
                <Button
                  size="large"
                  onClick={handleStartOver}
                  disabled={busy}
                >
                  Clear
                </Button>
              ) : null}
            </Space.Compact>

            {formError ? (
              <Alert
                type="error"
                showIcon
                message={formError}
                style={{ marginTop: 12 }}
              />
            ) : null}
          </Card>

          {/* Loading state */}
          {busy && !result ? (
            <Card>
              <div style={{ textAlign: "center", padding: 32 }}>
                <Spin size="large" />
                <Paragraph type="secondary" style={{ marginTop: 12, marginBottom: 0 }}>
                  Generating selected RCA documents in parallel — this
                  usually takes 15-30 seconds.
                </Paragraph>
              </div>
            </Card>
          ) : null}

          {/* Sprint 13.32.6 — Excel/CSV-only path placeholder.
              When the engineer uploaded a file but didn't enter a
              ticket number, the backend Excel-parsing pipeline isn't
              wired yet; surface that honestly instead of pretending
              the upload was processed. */}
          {fileOnlyMode && !busy ? (
            <Alert
              type="info"
              showIcon
              message="Spreadsheet upload received"
              description={
                <span>
                  <strong>{uploadedFile?.name}</strong> is queued for processing.
                  The Excel/CSV-to-RCA pipeline is being built — for now,
                  please provide an incident number above to generate the
                  selected RCA document(s).
                </span>
              }
              style={{ marginBottom: 16 }}
            />
          ) : null}

          {/* Results */}
          {result ? (
            <Collapse
              items={collapseItems}
              defaultActiveKey={["customer", "internal"]}
              bordered
            />
          ) : null}
        </div>
      </div>

      {/* Fixed bottom-right "Return to Stages" button */}
      <div
        style={{
          padding: "12px 16px",
          borderTop: "1px solid var(--border-color, #e5e7eb)",
          display: "flex",
          justifyContent: "flex-end",
          flexShrink: 0,
        }}
      >
        <Button
          type="default"
          icon={<ArrowLeftOutlined />}
          onClick={() => {
            if (typeof onReturnToStages === "function") {
              onReturnToStages();
            }
          }}
        >
          Return to Stages
        </Button>
      </div>
    </div>
  );
}
