// Gap Analysis — client-side export helpers (PDF + Word).
//
// Deliberate fork of `RCA/rcaExport.js` so the two flows stay
// independently maintainable. Functionally identical — both panels
// render a `.gap-paper` container styled identically to RCA's
// `.rca-paper`, so the same DOM-to-PDF / DOM-to-DOC capture path
// works without modification.

import html2pdf from "html2pdf.js";


// Sanitise a string into a filesystem-safe filename fragment.
// Used to build downloads like `INC-LAN-88902_Gap-Analysis.pdf`.
function _safeFilename(s) {
  if (!s) return "gap-analysis";
  return String(s).replace(/[^A-Za-z0-9._-]+/g, "_").slice(0, 80);
}


// ─────────────────────────────────────────────────────────────
// PDF export
// ─────────────────────────────────────────────────────────────
//
// `element` is the DOM node to render — pass the `.gap-paper`
// wrapper so the print captures the same visual styling the
// engineer sees on screen.
// `filename` is the base name; ".pdf" is appended automatically.
export async function exportPdf(element, filename) {
  if (!element) {
    throw new Error("exportPdf: element is required");
  }
  const opts = {
    margin: [0.5, 0.5, 0.5, 0.5],
    filename: `${_safeFilename(filename)}.pdf`,
    image: { type: "jpeg", quality: 0.98 },
    html2canvas: {
      scale: 2,
      useCORS: true,
      backgroundColor: "#ffffff",
      logging: false,
    },
    jsPDF: {
      unit: "in",
      format: "letter",
      orientation: "portrait",
      compress: true,
    },
    pagebreak: { mode: ["avoid-all", "css", "legacy"] },
  };
  return html2pdf().set(opts).from(element).save();
}


// ─────────────────────────────────────────────────────────────
// Word export
// ─────────────────────────────────────────────────────────────
//
// Same .doc-as-HTML trick used by RCA: wrap the rendered HTML in
// an MS-Office-namespaced shell with a minimal Word-friendly CSS
// subset. Word opens the .doc blob natively with formatting.
const _WORD_DOC_STYLES = `
  @page WordSection1 {
    size: Letter;
    margin: 0.75in;
    mso-page-orientation: portrait;
  }
  div.WordSection1 { page: WordSection1; }
  body {
    font-family: 'Calibri', 'Segoe UI', sans-serif;
    font-size: 11pt;
    line-height: 1.5;
    color: #1f2937;
  }
  h1 { font-size: 22pt; font-weight: 700; margin: 0 0 6pt 0; color: #0b3158; }
  h2 {
    font-size: 14pt; font-weight: 700; margin: 18pt 0 8pt 0;
    padding-bottom: 4pt; border-bottom: 1pt solid #d1d5db; color: #0b3158;
  }
  h3 { font-size: 12pt; font-weight: 600; margin: 12pt 0 6pt 0; }
  p { margin: 0 0 8pt 0; }
  ul, ol { margin: 0 0 10pt 0; padding-left: 28pt; }
  li { margin-bottom: 4pt; }
  table {
    border-collapse: collapse; width: 100%; margin: 8pt 0 12pt 0;
    border: 0.5pt solid #d1d5db;
  }
  th, td {
    border: 0.5pt solid #d1d5db; padding: 6pt 8pt;
    text-align: left; vertical-align: top; font-size: 10.5pt;
  }
  th { background: #f3f4f6; font-weight: 600; }
  blockquote {
    margin: 8pt 0; padding: 8pt 12pt;
    border-left: 3pt solid #0b3158; background: #f5f5fa;
  }
  code {
    font-family: 'Consolas', 'Courier New', monospace; font-size: 10pt;
    background: #f1f5f9; padding: 1pt 4pt;
  }
  pre {
    font-family: 'Consolas', 'Courier New', monospace; font-size: 10pt;
    background: #0f172a; color: #e2e8f0; padding: 10pt 12pt;
    border-radius: 4pt; white-space: pre-wrap;
  }
  pre code { background: transparent; color: inherit; padding: 0; }
`;


// `htmlSource` is either a string (innerHTML) or a DOM element
// (we pull its outerHTML). `filename` is the base name; ".doc" is
// appended automatically.
export function exportWord(htmlSource, filename) {
  if (!htmlSource) {
    throw new Error("exportWord: htmlSource is required");
  }
  const innerHtml = typeof htmlSource === "string"
    ? htmlSource
    : (htmlSource.outerHTML || htmlSource.innerHTML || "");
  if (!innerHtml) {
    throw new Error("exportWord: htmlSource produced empty HTML");
  }
  const doc = `<!DOCTYPE html>
<html xmlns:o="urn:schemas-microsoft-com:office:office"
      xmlns:w="urn:schemas-microsoft-com:office:word"
      xmlns="http://www.w3.org/TR/REC-html40">
<head>
  <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
  <meta name="ProgId" content="Word.Document">
  <meta name="Generator" content="Microsoft Word 15">
  <title>${_safeFilename(filename)}</title>
  <style>${_WORD_DOC_STYLES}</style>
</head>
<body>
  <div class="WordSection1">
    ${innerHtml}
  </div>
</body>
</html>`;

  // BOM at the start is what makes Word's "open" path treat the
  // file as Unicode-encoded HTML rather than raw text.
  const blob = new Blob(["﻿", doc], {
    type: "application/msword",
  });
  const url = URL.createObjectURL(blob);
  try {
    const a = document.createElement("a");
    a.href = url;
    a.download = `${_safeFilename(filename)}.doc`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  } finally {
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
}
