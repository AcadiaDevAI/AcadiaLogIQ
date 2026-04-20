import React, { useState, useCallback, useRef } from "react";
import { Button, Tooltip, message } from "antd";
import { CopyOutlined, CheckOutlined } from "@ant-design/icons";
import { settings } from "../config/clientSettings";

/**
 * CodeBlock
 *
 * Custom renderer for markdown code blocks. Works in two modes:
 *
 *   1. Inline code (backtick-wrapped): renders as <code> with .markdown-body styling
 *   2. Block code (triple-backtick-wrapped): renders as <pre><code> with
 *      syntax highlighting (via rehype-highlight) and a copy button overlay.
 *
 * Design:
 *   - No hardcoded language list — relies on rehype-highlight auto-detection
 *   - Copy button appears only on hover (desktop) or always (mobile)
 *   - Uses AntD message toast for copy confirmation (consistent with rest of app)
 *   - Theme-aware via CSS variables (no hardcoded colors)
 *   - Falls back gracefully if navigator.clipboard unavailable
 */
export default function CodeBlock({ node, inline, className, children, ...props }) {
  const [copied, setCopied] = useState(false);
  const codeRef = useRef(null);

  // ── Robust inline vs block detection ──────────────────
  // React-markdown v9 no longer reliably passes `inline` prop. We detect
  // inline code via multiple signals that work across versions:
  //
  //   1. `inline` prop if explicitly passed (older react-markdown)
  //   2. Absence of `language-*` className (block fences always have language class)
  //   3. Content is short single-line string (inline code never contains \n)
  //
  // Feature flag allows rollback to legacy behavior if needed.
  const rawText = String(children || "");
  const hasLanguageClass = /language-/.test(className || "");
  const hasNewline = rawText.includes("\n");

  const isInline = settings.CODE_INLINE_DETECTION_ENABLED
    ? (inline === true || (!hasLanguageClass && !hasNewline && rawText.length < 200))
    : (inline === true);

  if (isInline) {
    return (
      <code className={`inline-code ${className || ""}`.trim()} {...props}>
        {children}
      </code>
    );
  }

  // Block code — extract language from className (format: "language-bash", "language-python", etc.)
  const languageMatch = /language-(\w+)/.exec(className || "");
  const language = languageMatch ? languageMatch[1] : null;

  // Resolve code text from children for copy.
  // React-markdown passes nested structure; we extract text content reliably.
  const resolveText = () => {
    if (codeRef.current) {
      return codeRef.current.innerText || codeRef.current.textContent || "";
    }
    return String(children || "").trim();
  };

  const handleCopy = useCallback(async () => {
    const text = resolveText();
    if (!text) return;

    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
      } else {
        // Fallback for non-HTTPS contexts or older browsers
        const textArea = document.createElement("textarea");
        textArea.value = text;
        textArea.style.position = "fixed";
        textArea.style.opacity = "0";
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand("copy");
        document.body.removeChild(textArea);
      }
      setCopied(true);
      message.success({ content: "Copied!", duration: 1.5, key: "codecopy" });
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      message.error({ content: "Copy failed", duration: 2, key: "codecopy" });
    }
  }, []);

  return (
    <div className="code-block-wrapper">
      {language && (
        <span className="code-block-lang" aria-hidden="true">
          {language}
        </span>
      )}
      <Tooltip title={copied ? "Copied!" : "Copy"} placement="left">
        <Button
          type="text"
          size="small"
          icon={copied ? <CheckOutlined /> : <CopyOutlined />}
          onClick={handleCopy}
          className="code-block-copy-btn"
          style={{ color: copied ? "#10b981" : "var(--text-faint)" }}
          aria-label="Copy code"
        />
      </Tooltip>
      <pre {...props}>
        <code ref={codeRef} className={className}>
          {children}
        </code>
      </pre>
    </div>
  );
}
