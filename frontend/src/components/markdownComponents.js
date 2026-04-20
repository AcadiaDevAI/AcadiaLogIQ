import React from "react";
import CodeBlock from "./CodeBlock";

/**
 * Shared component map passed to react-markdown's `components` prop.
 *
 * Currently overrides:
 *   - code/pre  → delegates to CodeBlock (syntax highlighting + copy button)
 *   - a         → external links open in new tab with security rel
 *
 * Future extensions (when needed):
 *   - img    → AntD Image with lazy-load + preview modal
 *   - table  → AntD Table wrapper for sortable/filterable output
 *   - blockquote → custom Callout component with icon variants
 *
 * Keep this mapping narrow. Don't override elements unless there's a
 * clear UX win — unnecessary overrides slow rendering and add bugs.
 */
export const markdownComponents = {
  // Code blocks — custom renderer with syntax highlighting + copy button.
  // The `pre` override is needed because react-markdown wraps `code` in `pre`
  // by default, and we want CodeBlock to own the entire block rendering.
  code: CodeBlock,
  pre: ({ children }) => <>{children}</>,

  // External links — open in new tab, add rel for security
  a: ({ href, children, ...props }) => (
    <a
      href={href}
      target={href?.startsWith("http") ? "_blank" : undefined}
      rel={href?.startsWith("http") ? "noopener noreferrer" : undefined}
      {...props}
    >
      {children}
    </a>
  ),
};
