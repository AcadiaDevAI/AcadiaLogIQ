import React, { useState } from "react";
import { Button, Empty, Space } from "antd";
import { LeftOutlined, RightOutlined } from "@ant-design/icons";
import SuggestionCard from "./SuggestionCard";

/**
 * Sprint 9 — SuggestionCarousel
 *
 * Horizontal navigation through up to 4 ValidatedCandidate cards
 * (◀ N/M ▶). Stays AntD-only — no Carousel auto-scroll, just an
 * index-driven render so keyboard nav and arrow buttons are explicit.
 */
export default function SuggestionCarousel({ candidates, onUse }) {
  const [idx, setIdx] = useState(0);

  if (!candidates || candidates.length === 0) {
    return (
      <Empty description="No suggestions yet — paste content above and click Extract." />
    );
  }

  const total = candidates.length;
  const safeIdx = Math.max(0, Math.min(idx, total - 1));
  const card = candidates[safeIdx];

  const goPrev = () => setIdx(Math.max(0, safeIdx - 1));
  const goNext = () => setIdx(Math.min(total - 1, safeIdx + 1));

  return (
    <div>
      <div className="flex justify-between items-center mb-2">
        <span className="t-text font-semibold">Suggested interpretations</span>
        <Space>
          <Button
            size="small"
            icon={<LeftOutlined />}
            onClick={goPrev}
            disabled={safeIdx <= 0}
            aria-label="Previous suggestion"
          />
          <span className="t-text-muted text-xs">
            {safeIdx + 1} / {total}
          </span>
          <Button
            size="small"
            icon={<RightOutlined />}
            onClick={goNext}
            disabled={safeIdx >= total - 1}
            aria-label="Next suggestion"
          />
        </Space>
      </div>
      <SuggestionCard card={card} onUse={onUse} />
    </div>
  );
}
