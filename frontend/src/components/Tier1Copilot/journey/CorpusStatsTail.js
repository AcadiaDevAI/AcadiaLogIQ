// Sprint 10.3 — CorpusStatsTail extracted as its own file with a
// proper default export. Antd 5.29.3.
//
// Sprint 10.4 §2.3 — plain-English labels:
//   - Trigger label: "Show details" (was "Show full corpus stats")
//   - "Cohort size" → "Similar tickets found"
//   - "Cleanly resolved" → "Top-quality resolutions"
//   - When clean_resolution_count === 0, contextualise with
//     "(most scored 3/5 — adequate)" instead of broadcasting "0%"
//   - "0% resolved cleanly" line is GONE entirely
//   - "Average time to resolve" + "(platform average: ... min)"
//   - Bottom muted line uses "past tickets" not "corpus"
//
// Sprint 10.3 carry-over: items array is built in useMemo so children
// resolve to a single Fragment with no `&&` shortcuts that could
// produce undefined children. `Statistic.Group` is NOT a real antd
// 5.x subcomponent — we use plain divs.

import React, { useMemo } from "react";
import { Collapse, Typography } from "antd";

const { Text } = Typography;


export default function CorpusStatsTail({ data }) {
  const items = useMemo(() => {
    if (!data) return [];

    // Sprint 13.32.3 — align every cohort-size reference with what
    // the engineer actually sees in the bullet list above. The
    // backend's `cohort_size` counts every dict in the cohort even
    // when one of them lacks an Incident_Summary to render; the
    // visible list (`top5_incident_summaries`) is the curated set.
    // Clamping cleanCount to the visible count avoids the nonsensical
    // "5 of 4" denominator that emerged once the visible count
    // dropped below cohort_size.
    const rawCohortSize = data.cohort_size || 0;
    const visibleSize = Array.isArray(data.top5_incident_summaries)
      ? data.top5_incident_summaries.length
      : 0;
    const cohortSize = visibleSize || rawCohortSize;
    const rawCleanCount = data.clean_resolution_count || 0;
    const cleanCount = Math.min(rawCleanCount, cohortSize);
    const cleanPct = cohortSize
      ? Math.round((cleanCount / cohortSize) * 100)
      : 0;
    const avgMinCohort = data.avg_minutes_to_resolve_cohort;
    const platformMedian = data.platform_median_minutes;
    const corpusSize = data.corpus_size || 0;

    // Sprint 10.4 §2.3 — "Top-quality resolutions" suffix. When zero
    // are top-quality, contextualise instead of showing a scary "0%".
    let topQualitySuffix = "";
    if (cleanCount === 0 && cohortSize > 0) {
      topQualitySuffix = " (most scored 3/5 — adequate)";
    } else if (cleanPct) {
      topQualitySuffix = ` (${cleanPct}%)`;
    }

    // Pre-resolve every conditional row to either a JSX node or null —
    // never leave an `&&` shortcut that can yield `undefined`.
    const avgRow = avgMinCohort
      ? (
        <div>
          <strong>Average time to resolve:</strong> {avgMinCohort} minutes
          {platformMedian
            ? ` (platform average: ${platformMedian} min)`
            : ""}
        </div>
      )
      : null;

    const corpusFooter = corpusSize > 0
      ? (
        <div style={{ color: "var(--text-muted, #888)", marginTop: 6 }}>
          {cohortSize} of {corpusSize} total past tickets matched this profile.
        </div>
      )
      : null;

    const body = (
      <div style={{ fontSize: 13, lineHeight: 1.7 }}>
        <div><strong>Similar tickets found:</strong> {cohortSize}</div>
        <div>
          <strong>Top-quality resolutions:</strong> {cleanCount} of {cohortSize}
          {topQualitySuffix}
        </div>
        {avgRow}
        {corpusFooter}
      </div>
    );

    return [
      {
        key: "stats",
        label: <Text type="secondary">Show details</Text>,
        children: body,
      },
    ];
  }, [data]);

  if (!data) return null;
  return <Collapse ghost items={items} style={{ marginTop: 12 }} />;
}
