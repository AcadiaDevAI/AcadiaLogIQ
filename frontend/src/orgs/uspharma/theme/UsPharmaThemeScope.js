import React from "react";
import { ConfigProvider } from "antd";

// Importing the stylesheet here (module-load side effect) ships the
// `.org-uspharma` variable rebinds with the app bundle. The rules are
// scoped to `.org-uspharma`, so they are inert for every other org.
import "./uspharma.css";
import { usPharmaAntdTheme } from "./palette";

/**
 * UsPharmaThemeScope — flips AntD's design tokens to Walgreens red for
 * the US Pharma subtree.
 *
 * The global AntD <ConfigProvider> in App.js lives ABOVE OrgContextProvider,
 * so it can't know the active org. This nested ConfigProvider sits INSIDE
 * the org-aware tree and inherits (inherit=true) the premium theme, only
 * overriding the accent tokens — so AntD-computed colors (focus rings,
 * Select selection, Tabs ink bar, Spin, links) turn red alongside the
 * CSS-variable repaint carried by the `.org-uspharma` class.
 *
 * When `enabled` is false this renders children untouched — Acadia and
 * every other org keep the shared iris theme.
 *
 * @param {{ enabled: boolean, children: React.ReactNode }} props
 */
export default function UsPharmaThemeScope({ enabled, children }) {
  if (!enabled) return children;
  return <ConfigProvider theme={usPharmaAntdTheme}>{children}</ConfigProvider>;
}
