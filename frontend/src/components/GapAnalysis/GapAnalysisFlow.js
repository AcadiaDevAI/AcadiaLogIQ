// Gap Analysis — right-pane flow.
//
// Mirrors the RCAFlow surface intentionally so the engineer's
// muscle memory carries across (same input affordance, same
// collapsible-panels layout, same "return" semantics) but is a
// fully independent component tree. No CSS bleed, no shared state,
// no shared import paths into the RCA folder.
//
// User journey
// ------------
// 1. Sidebar's "Gap Analysis" button opens GapAnalysisEntryModal.
// 2. Modal collects an incident number, calls onSubmit.
// 3. AppLayout sets `gapPayload` + flips `gapOpen=true`.
// 4. This component mounts, reads `initialPayload`, kicks off the
//    backend POST, then renders the two reports in an AntD Collapse.
// 5. "Return" button at the bottom unmounts the flow (AppLayout
//    flips `gapOpen=false`).
//
// Failure modes
// -------------
// * 404 (ticket not found) → inline form-level error; retry stays
//   open with the prior number prefilled.
// * 500 / network error    → form-level error; "Try again" available.
// * Per-panel LLM failure  → the *other* panel still renders; the
//   failed panel surfaces a small inline warning so the engineer
//   can re-submit just the form to retry.

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
  Tag,
  Tooltip,
  Typography,
  message,
} from "antd";
import {
  ArrowLeftOutlined,
  CopyOutlined,
  DislikeOutlined,
  DislikeFilled,
  FilePdfOutlined,
  FileSearchOutlined,
  FileWordOutlined,
  LikeOutlined,
  LikeFilled,
  LoadingOutlined,
  ReloadOutlined,
  SafetyOutlined,
  AuditOutlined,
} from "@ant-design/icons";

import { generateGapAnalysis, recordGapAnalysisFeedback } from "./gapAnalysisApi";
import { exportPdf, exportWord } from "./gapAnalysisExport";


const { Title, Paragraph, Text } = Typography;


// ─────────────────────────────────────────────────────────────
// Scoped CSS — wraps the two report panels in a "paper" look so
// each reads as a printed document rather than chat bubble prose.
//
// Class names are namespaced `.gap-paper` / `.gap-markdown` so this
// style sheet cannot bleed into RCAFlow's `.rca-paper` styles even
// if both flows are mounted in rapid succession during navigation.
//
// Injected exactly once per session via a module-level guard.
// ─────────────────────────────────────────────────────────────
let _gapStylesInjected = false;
function _ensureGapStyles() {
  if (typeof document === "undefined" || _gapStylesInjected) return;
  const style = document.createElement("style");
  style.setAttribute("data-acadia-gap-analysis", "1");
  style.textContent = `
    .gap-paper {
      background: var(--bg-elevated, #ffffff);
      color: var(--text-primary, #1f2937);
      padding: 40px 48px;
      border-radius: 6px;
      box-shadow: 0 1px 2px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
      max-width: 960px;
      margin: 0 auto;
    }
    @media (max-width: 720px) {
      .gap-paper { padding: 24px 20px; }
    }
    .gap-markdown {
      font-family: 'Poppins', 'Inter', system-ui, -apple-system, Segoe UI, sans-serif;
      line-height: 1.65;
      font-size: 14px;
      color: inherit;
    }
    .gap-markdown h1 {
      font-size: 24px;
      font-weight: 700;
      margin: 0 0 16px;
      padding-bottom: 8px;
      border-bottom: 2px solid #1f2937;
    }
    .gap-markdown h2 {
      font-size: 18px;
      font-weight: 700;
      margin: 28px 0 12px;
      padding-bottom: 6px;
      border-bottom: 1px solid #e5e7eb;
      color: #1f2937;
    }
    .gap-markdown h3 {
      font-size: 15px;
      font-weight: 600;
      margin: 22px 0 8px;
      color: #1f2937;
    }
    .gap-markdown p,
    .gap-markdown li,
    .gap-markdown strong,
    .gap-markdown em {
      color: #1f2937;
    }
    .gap-markdown ul, .gap-markdown ol {
      padding-left: 22px;
      margin: 8px 0 14px;
    }
    .gap-markdown li { margin: 4px 0; }
    .gap-markdown table {
      width: 100%;
      border-collapse: collapse;
      margin: 14px 0 18px;
      font-size: 13px;
    }
    .gap-markdown thead {
      background: #eef2ff;
    }
    .gap-markdown thead th {
      font-weight: 600;
      text-align: left;
      padding: 8px 10px;
      border: 1px solid #e5e7eb;
      color: #1f2937;
    }
    .gap-markdown tbody td {
      padding: 8px 10px;
      border: 1px solid #e5e7eb;
      vertical-align: top;
    }
    .gap-markdown tbody tr:nth-child(even) td {
      background: #f9fafb;
    }
    .gap-markdown blockquote {
      border-left: 4px solid #6366f1;
      background: rgba(99, 102, 241, 0.06);
      margin: 14px 0;
      padding: 10px 14px;
      color: #1f2937;
    }
    .gap-markdown code {
      background: #f3f4f6;
      color: #1f2937;
      padding: 1px 5px;
      border-radius: 3px;
      font-size: 13px;
      font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
    }
    .gap-markdown pre {
      background: #0f172a;
      color: #e2e8f0;
      padding: 14px 16px;
      border-radius: 6px;
      overflow-x: auto;
      font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
      font-size: 12.5px;
      line-height: 1.55;
      margin: 14px 0;
    }
    .gap-markdown pre code {
      background: transparent;
      color: inherit;
      padding: 0;
      border-radius: 0;
    }
    .gap-markdown hr {
      border: none;
      border-top: 1px solid #e5e7eb;
      margin: 22px 0;
    }

    /* Dark-mode parity — class hook supplied by the app shell. */
    .theme-dark .gap-paper {
      background: #1e1e28;
      color: #e2e8f0;
      box-shadow: 0 1px 2px rgba(0,0,0,0.4), 0 4px 16px rgba(0,0,0,0.4);
    }
    .theme-dark .gap-markdown h1 { color: #f1f5f9; border-bottom-color: #e2e8f0; }
    .theme-dark .gap-markdown h2 {
      color: #e2e8f0;
      border-bottom-color: #2a2a3d;
    }
    .theme-dark .gap-markdown h3 { color: #e2e8f0; }
    .theme-dark .gap-markdown p,
    .theme-dark .gap-markdown li,
    .theme-dark .gap-markdown strong,
    .theme-dark .gap-markdown em { color: #e2e8f0; }
    .theme-dark .gap-markdown hr { border-top-color: #2a2a3d; }
    .theme-dark .gap-markdown table,
    .theme-dark .gap-markdown th,
    .theme-dark .gap-markdown td { border-color: #2a2a3d; color: #e2e8f0; }
    .theme-dark .gap-markdown thead { background: #2a2a3d; }
    .theme-dark .gap-markdown thead th { color: #f1f5f9; }
    .theme-dark .gap-markdown tbody tr:nth-child(even) td { background: #16161d; }
    .theme-dark .gap-markdown blockquote {
      background: rgba(99, 102, 241, 0.12);
      border-left-color: #6366f1;
      color: #e2e8f0;
    }
    .theme-dark .gap-markdown code {
      background: #16161d;
      color: #e2e8f0;
    }
  `;
  document.head.appendChild(style);
  _gapStylesInjected = true;
}


// We only override `a` to force same-window navigation for any
// inline links the LLM emits. Everything else uses ReactMarkdown's
// defaults so the scoped CSS above is the single source of truth.
const gapMarkdownComponents = {
  a: ({ href, children, ...rest }) => (
    <a href={href} target="_blank" rel="noopener noreferrer" {...rest}>
      {children}
    </a>
  ),
};


// Renders a single Markdown blob inside the "paper" container so it
// looks like a printed document. forwardRef keeps the door open for
// future export-to-PDF / Word features without re-rendering.
const GapMarkdown = React.forwardRef(function GapMarkdown({ source }, ref) {
  if (!source) return null;
  return (
    <div className="gap-paper" ref={ref}>
      <div className="gap-markdown">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[rehypeHighlight]}
          components={gapMarkdownComponents}
        >
          {source}
        </ReactMarkdown>
      </div>
    </div>
  );
});


// ─────────────────────────────────────────────────────────────
// ExportButtons — PDF + Word download per panel.
//
// Operates on the live DOM node the panel rendered (passed in via
// `getNode`), so the export captures the same paper-styled view
// the engineer sees on screen — no parallel render, no style drift.
// PDF generation is async (html2pdf returns a promise) and busies
// the button while running. Word export is effectively synchronous.
// ─────────────────────────────────────────────────────────────
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
      console.error("[gap.export] PDF failed", err);
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
      console.error("[gap.export] Word failed", err);
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


// ─────────────────────────────────────────────────────────────
// PanelFeedback — 👍 / 👎 row beneath each report body.
//
// Click 👍 → green Like icon + toast. Cache stays intact.
// Click 👎 → red Dislike icon + toast + cache invalidated server-
//            side. The parent is notified via `onDisliked` so it
//            can re-run the panel through the LLM (the next
//            "Regenerate" or auto-rerun gets fresh output).
// State is local — once an engineer clicks 👍 or 👎 the icon stays
// filled until the panel re-renders (which happens on regenerate
// or full reload).
// ─────────────────────────────────────────────────────────────
function PanelFeedback({ incidentNumber, panel, label, onDisliked, disabled }) {
  const [selection, setSelection] = useState(null); // 'like' | 'dislike' | null
  const [busy, setBusy] = useState(false);

  const send = async (kind) => {
    if (!incidentNumber || busy) return;
    setBusy(true);
    try {
      const res = await recordGapAnalysisFeedback(incidentNumber, panel, kind);
      setSelection(kind);
      if (kind === "like") {
        message.success(`Thanks — marked ${label} as helpful.`);
      } else {
        message.success(
          `Thanks — ${label} flagged for regeneration.`
          + (res?.invalidated ? " Cache cleared." : ""),
        );
        if (typeof onDisliked === "function") onDisliked();
      }
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error("[gap.feedback]", err);
      message.error("Couldn't record feedback — please try again.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <Space size={6} style={{ marginTop: 12 }}>
      <Text type="secondary" style={{ fontSize: 12 }}>
        Was this {label} helpful?
      </Text>
      <Tooltip title="Helpful — keeps the cached result for everyone">
        <Button
          size="small"
          icon={
            selection === "like"
              ? <LikeFilled style={{ color: "#10b981" }} />
              : <LikeOutlined />
          }
          onClick={() => send("like")}
          disabled={busy || disabled}
        >
          Helpful
        </Button>
      </Tooltip>
      <Tooltip title="Dislike — clears the cached result so the next Generate produces a fresh report">
        <Button
          size="small"
          icon={
            selection === "dislike"
              ? <DislikeFilled style={{ color: "#ef4444" }} />
              : <DislikeOutlined />
          }
          onClick={() => send("dislike")}
          disabled={busy || disabled}
        >
          Dislike
        </Button>
      </Tooltip>
    </Space>
  );
}


// Copy-to-clipboard affordance — engineers usually want to paste
// the Gap Analysis into a tracking ticket or the post-mortem into
// Confluence / Notion.
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


export default function GapAnalysisFlow({ onReturnToStages, initialPayload = null }) {
  // Inject scoped CSS on first mount; the module guard makes
  // subsequent mounts a no-op.
  React.useEffect(() => {
    _ensureGapStyles();
  }, []);

  // Refs are forwarded into the rendered panels so future export
  // features can capture the live DOM. Unused today but cheap.
  const gapPaperRef = useRef(null);
  const pmPaperRef = useRef(null);

  // In-flight de-dupe. React StrictMode double-fires useEffect on
  // mount in dev, and `busy`/setBusy is async (next render only), so
  // a state guard alone can't block the second fire. This ref is
  // mutated synchronously so the second fire short-circuits before
  // it ever calls the API — preventing a wasted parallel LLM run
  // that would also race the cache write. Also guards against rapid
  // double-clicks from the user. Tracks the incident currently
  // being fetched ("" / null when idle).
  const inFlightIncRef = useRef(null);

  const [incidentNumber, setIncidentNumber] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  // Form-level errors (404 / network) are distinct from per-panel
  // LLM errors carried inside `result`.
  const [formError, setFormError] = useState(null);

  // Panel visibility flags — populated by the entry modal's payload.
  // Default to both visible so any path that mounts this flow without
  // a payload (manual form entry, future deep-linking, etc.) still
  // produces a full report set rather than a blank screen.
  const [showGap, setShowGap] = useState(true);
  const [showPostMortem, setShowPostMortem] = useState(true);

  // Track which panel is being regenerated so its header can show
  // a small spinner without freezing the whole flow.
  const [regeneratingPanel, setRegeneratingPanel] = useState(null); // 'gap_analysis' | 'post_mortem' | null

  const canSubmit = !busy && !!incidentNumber.trim();

  const handleSubmit = async (opts = {}) => {
    const inc = (opts.incidentNumber ?? incidentNumber).trim();
    if (!inc || busy) return;
    // StrictMode dev-mode re-fires this effect; without a ref guard
    // the second fire kicks off a parallel POST that races the cache
    // write and forces an extra LLM run. The ref is mutated
    // synchronously so the second fire short-circuits immediately.
    if (inFlightIncRef.current === inc) return;
    inFlightIncRef.current = inc;
    setBusy(true);
    setFormError(null);
    setResult(null);
    try {
      const data = await generateGapAnalysis(inc, {
        // Initial submit honours the cache. Regenerate is a separate
        // explicit button that toggles these flags per panel below.
        regenerateGapAnalysis: !!opts.regenerateGapAnalysis,
        regeneratePostMortem: !!opts.regeneratePostMortem,
      });
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
        console.error("[GapAnalysisFlow] generate failed", err);
        setFormError(
          "Something went wrong generating the Gap Analysis. "
          + "Please try again in a moment.",
        );
      }
    } finally {
      setBusy(false);
      inFlightIncRef.current = null;
    }
  };

  // Auto-run when AppLayout hands us an `initialPayload` (the entry
  // modal's submitted incident number + panel selection). One-shot
  // — the prop only changes when the modal re-opens, which can't
  // happen while this flow is mounted.
  React.useEffect(() => {
    if (!initialPayload) return;
    if (initialPayload.panels) {
      // Boolean coercion guards against legacy callers passing
      // missing fields (treat absent as "show this panel").
      setShowGap(initialPayload.panels.gapAnalysis !== false);
      setShowPostMortem(initialPayload.panels.postMortem !== false);
    }
    if (initialPayload.incidentNumber) {
      setIncidentNumber(initialPayload.incidentNumber);
      handleSubmit({ incidentNumber: initialPayload.incidentNumber });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialPayload]);

  const handleStartOver = () => {
    setResult(null);
    setFormError(null);
    setIncidentNumber("");
  };

  // Per-panel "Regenerate" — bypasses cache for just the named
  // panel and re-runs the LLM. The other panel keeps whatever
  // state it currently has (cached or freshly generated). The
  // backend's `regenerate_*` flags handle this; we send only the
  // one panel's flag set to true.
  const handleRegeneratePanel = async (panelKey) => {
    const inc = (result?.incident_number || incidentNumber).trim();
    if (!inc) return;
    setRegeneratingPanel(panelKey);
    try {
      const data = await generateGapAnalysis(inc, {
        regenerateGapAnalysis: panelKey === "gap_analysis",
        regeneratePostMortem: panelKey === "post_mortem",
      });
      setResult(data);
      message.success(
        panelKey === "gap_analysis"
          ? "Gap Analysis regenerated."
          : "Blameless Post-Mortem regenerated.",
      );
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error("[gap.regenerate]", err);
      message.error("Regenerate failed — please try again.");
    } finally {
      setRegeneratingPanel(null);
    }
  };

  // Collapse items — Gap Analysis first (it's the structured master
  // report and the user's primary target), Post-Mortem second.
  // Both panels are expanded by default so the engineer doesn't
  // miss either one. They're freely collapsible thereafter.
  //
  // The `allCollapseItems` list always carries both panels; the
  // `collapseItems` filter below honours the engineer's selection
  // from the entry modal. Prompts, request, and rendering are
  // unchanged — we just drop panels the engineer didn't ask for.
  const allCollapseItems = result
    ? [
        {
          key: "gap_analysis",
          label: (
            <Space wrap>
              <SafetyOutlined style={{ color: "#6366f1" }} />
              <Text strong style={{ fontSize: 15 }}>
                Gap Analysis Report
              </Text>
              <Text type="secondary" style={{ fontSize: 12 }}>
                — Technical, Process & Communication / Silo
              </Text>
              {regeneratingPanel === "gap_analysis" ? (
                <Tag icon={<LoadingOutlined spin />} color="warning">
                  Regenerating…
                </Tag>
              ) : null}
            </Space>
          ),
          extra: (
            <Space size={6} onClick={(e) => e.stopPropagation()}>
              <Tooltip title="Regenerate this panel — bypasses cache and re-runs the LLM">
                <Button
                  size="small"
                  icon={<ReloadOutlined />}
                  onClick={() => handleRegeneratePanel("gap_analysis")}
                  disabled={!!regeneratingPanel}
                >
                  Regenerate
                </Button>
              </Tooltip>
              <ExportButtons
                getNode={() => gapPaperRef.current}
                filename={`${result.incident_number || "GapAnalysis"}_Gap-Analysis`}
                source={result.gap_analysis_md}
              />
              <CopyMarkdownButton
                source={result.gap_analysis_md}
                label="Gap Analysis"
              />
            </Space>
          ),
          children: (
            <>
              {result.gap_analysis_error ? (
                <Alert
                  type="warning"
                  showIcon
                  message={result.gap_analysis_error}
                  style={{ marginBottom: 12 }}
                />
              ) : null}
              {result.gap_analysis_md ? (
                <>
                  <GapMarkdown source={result.gap_analysis_md} ref={gapPaperRef} />
                  <PanelFeedback
                    incidentNumber={result.incident_number}
                    panel="gap_analysis"
                    label="Gap Analysis"
                    disabled={!!regeneratingPanel}
                    onDisliked={() => handleRegeneratePanel("gap_analysis")}
                  />
                </>
              ) : !result.gap_analysis_error ? (
                <Empty
                  image={Empty.PRESENTED_IMAGE_SIMPLE}
                  description="No Gap Analysis returned."
                />
              ) : null}
            </>
          ),
        },
        {
          key: "post_mortem",
          label: (
            <Space wrap>
              <AuditOutlined style={{ color: "#10b981" }} />
              <Text strong style={{ fontSize: 15 }}>
                Blameless Post-Mortem
              </Text>
              <Text type="secondary" style={{ fontSize: 12 }}>
                — SRE 13-Section Report
              </Text>
              {regeneratingPanel === "post_mortem" ? (
                <Tag icon={<LoadingOutlined spin />} color="warning">
                  Regenerating…
                </Tag>
              ) : null}
            </Space>
          ),
          extra: (
            <Space size={6} onClick={(e) => e.stopPropagation()}>
              <Tooltip title="Regenerate this panel — bypasses cache and re-runs the LLM">
                <Button
                  size="small"
                  icon={<ReloadOutlined />}
                  onClick={() => handleRegeneratePanel("post_mortem")}
                  disabled={!!regeneratingPanel}
                >
                  Regenerate
                </Button>
              </Tooltip>
              <ExportButtons
                getNode={() => pmPaperRef.current}
                filename={`${result.incident_number || "GapAnalysis"}_Blameless-Post-Mortem`}
                source={result.post_mortem_md}
              />
              <CopyMarkdownButton
                source={result.post_mortem_md}
                label="Post-Mortem"
              />
            </Space>
          ),
          children: (
            <>
              {result.post_mortem_error ? (
                <Alert
                  type="warning"
                  showIcon
                  message={result.post_mortem_error}
                  style={{ marginBottom: 12 }}
                />
              ) : null}
              {result.post_mortem_md ? (
                <>
                  <GapMarkdown source={result.post_mortem_md} ref={pmPaperRef} />
                  <PanelFeedback
                    incidentNumber={result.incident_number}
                    panel="post_mortem"
                    label="Post-Mortem"
                    disabled={!!regeneratingPanel}
                    onDisliked={() => handleRegeneratePanel("post_mortem")}
                  />
                </>
              ) : !result.post_mortem_error ? (
                <Empty
                  image={Empty.PRESENTED_IMAGE_SIMPLE}
                  description="No Post-Mortem returned."
                />
              ) : null}
            </>
          ),
        },
      ]
    : [];

  // Apply the engineer's panel selection. When `result` is null the
  // form is still being filled in, so this list is naturally empty
  // and the Collapse render below is skipped.
  const collapseItems = allCollapseItems.filter((it) => {
    if (it.key === "gap_analysis") return showGap;
    if (it.key === "post_mortem") return showPostMortem;
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
              Gap Analysis Report Generator
            </Title>
            <Paragraph type="secondary" style={{ marginBottom: 0 }}>
              Enter a historical incident number to produce two
              long-form reports in parallel — the LogIQ Gap Analysis
              (technical, process, silo) and a blameless 13-section
              post-mortem suitable for engineering, operations, and
              leadership review.
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
                {busy ? "Generating…" : "Generate Gap Analysis"}
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

          {/* Loading state — Gap Analysis prompts are longer than
              RCA's so we set the expectation higher. */}
          {busy && !result ? (
            <Card>
              <div style={{ textAlign: "center", padding: 32 }}>
                <Spin size="large" />
                <Paragraph type="secondary" style={{ marginTop: 12, marginBottom: 0 }}>
                  Generating Gap Analysis and Blameless Post-Mortem in
                  parallel — this usually takes 30-60 seconds for a
                  rich incident.
                </Paragraph>
              </div>
            </Card>
          ) : null}

          {/* Results — collapsible panels, both expanded by default */}
          {result ? (
            <Collapse
              items={collapseItems}
              defaultActiveKey={["gap_analysis", "post_mortem"]}
              bordered
            />
          ) : null}
        </div>
      </div>

      {/* Fixed bottom-right "Return" button */}
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
