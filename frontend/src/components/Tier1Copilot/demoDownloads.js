// Sprint 8.1 demo — temporary download helpers for UAT review.
// Safe to delete entirely after demo: the whole file is self-contained.
// Every touchpoint elsewhere is marked with `// Sprint 8.1 demo`.
//
// Flag: REACT_APP_LOGIQ_TIER1_DOWNLOAD_DEMO

function timestamp() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return (
    d.getFullYear() +
    pad(d.getMonth() + 1) +
    pad(d.getDate()) +
    "-" +
    pad(d.getHours()) +
    pad(d.getMinutes()) +
    pad(d.getSeconds())
  );
}

function triggerDownload(filename, content) {
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function formatAnswer(match, rank) {
  const a = (match && match.answer) || {};
  const totalMatches = (match && match.total_matches) || 5;
  const lines = [
    "========================================",
    `  MATCH ${rank} of ${totalMatches}`,
    "========================================",
    "",
    `Matched Incident: ${(match && match.matched_incident) || "—"}`,
    `Confidence: ${(match && match.confidence) || "—"}`,
    `Response ID: ${(match && match.response_id) || "—"}`,
    `Cache Hit: ${match && match.cache_hit ? "yes" : "no"}`,
    "",
    "--- Issue Understanding ---",
    a.issue_understanding || "",
    "",
    "--- Historical Match ---",
    a.historical_match || "",
    "",
    "--- Most Likely Cause ---",
    a.most_likely_cause || "",
    "",
    "--- Recommended First Checks ---",
    ...(a.recommended_first_checks || []).map((s, i) => `${i + 1}. ${s}`),
    "",
    "--- Most Likely Fix ---",
    a.most_likely_fix || "",
    "",
    "--- Validation ---",
    a.validation || "",
    "",
    "--- Escalate If ---",
    a.escalate_if || "",
    "",
    "--- Follow-up Question ---",
    a.follow_up_question || "",
    "",
    "",
  ];
  return lines.join("\n");
}

export async function downloadAllMatches(
  sessionId,
  fetchMatchByIndex,
  alertPayload,
) {
  try {
    if (!sessionId || typeof fetchMatchByIndex !== "function") {
      // eslint-disable-next-line no-alert
      alert("Download unavailable — session or fetcher missing.");
      return;
    }
    const indexes = [0, 1, 2, 3, 4];
    const matches = await Promise.all(
      indexes.map(async (i) => {
        try {
          return await fetchMatchByIndex(sessionId, i);
        } catch {
          return null;
        }
      }),
    );

    const header = [
      "ACADIA LOG IQ — TIER-1 ALERT COPILOT",
      "Top 5 Historical Matches",
      `Generated: ${new Date().toISOString()}`,
      "",
      "Alert Input:",
      `  Severity:    ${(alertPayload && alertPayload.severity) || "—"}`,
      `  Asset:       ${(alertPayload && alertPayload.asset_name) || "—"}`,
      `  Alert Type:  ${(alertPayload && alertPayload.alert_type) || "—"}`,
      `  Customer:    ${(alertPayload && alertPayload.customer) || "—"}`,
      `  Technology:  ${(alertPayload && alertPayload.technology) || "—"}`,
      `  Error Code:  ${(alertPayload && alertPayload.error_code) || "—"}`,
      `  Notes:       ${(alertPayload && alertPayload.notes) || "—"}`,
      "",
      "========================================",
      "",
    ].join("\n");

    const body = matches
      .map((m, idx) =>
        m ? formatAnswer(m, idx + 1) : `MATCH ${idx + 1}: (not available)\n\n`,
      )
      .join("\n");

    triggerDownload(`tier1-all-matches-${timestamp()}.txt`, header + body);
  } catch (err) {
    // eslint-disable-next-line no-console
    console.error("[demo-download] all-matches failed:", err);
    // eslint-disable-next-line no-alert
    alert("Download failed. Check console for details.");
  }
}

export function downloadDiagnostics(
  diagnosticsData,
  whatTried,
  matchedIncident,
) {
  try {
    if (!diagnosticsData) {
      // eslint-disable-next-line no-alert
      alert("No diagnostics to download yet.");
      return;
    }
    const steps = (diagnosticsData && diagnosticsData.steps) || [];
    const lines = [
      "ACADIA LOG IQ — TIER-1 DEEPER DIAGNOSTICS",
      `Generated: ${new Date().toISOString()}`,
      `Based on: ${matchedIncident || "—"}`,
      "",
      "========================================",
      "  DIAGNOSTIC GOAL",
      "========================================",
      (diagnosticsData && diagnosticsData.goal) || "",
      "",
    ];

    steps.forEach((step) => {
      lines.push("========================================");
      lines.push(`  STEP ${step.step_number}: ${step.title || ""}`);
      lines.push("========================================");
      lines.push("");
      lines.push("What to check:");
      lines.push(`  ${step.what_to_check || ""}`);
      lines.push("");
      lines.push("Why:");
      lines.push(`  ${step.why || ""}`);
      lines.push("");
      if (step.command) {
        lines.push("Command:");
        lines.push(`  ${step.command}`);
        lines.push("");
      }
      lines.push("Expected result:");
      lines.push(`  ${step.expected_result || ""}`);
      lines.push("");
      lines.push("If abnormal:");
      lines.push(`  ${step.next_action_if_abnormal || ""}`);
      lines.push("");
      lines.push("If normal:");
      lines.push(`  ${step.next_action_if_normal || ""}`);
      lines.push("");
    });

    if (diagnosticsData && diagnosticsData.validation) {
      lines.push("========================================");
      lines.push("  VALIDATION");
      lines.push("========================================");
      lines.push(diagnosticsData.validation);
      lines.push("");
    }

    if (diagnosticsData && diagnosticsData.next_question) {
      lines.push("========================================");
      lines.push("  NEXT QUESTION");
      lines.push("========================================");
      lines.push(diagnosticsData.next_question.prompt || "");
      lines.push("Options:");
      (diagnosticsData.next_question.options || []).forEach((o) =>
        lines.push(`  - ${o}`),
      );
      lines.push("");
    }

    if (whatTried && whatTried.length > 0) {
      lines.push("========================================");
      lines.push("  SESSION LOG — WHAT WAS TRIED");
      lines.push("========================================");
      whatTried.forEach((entry, i) => {
        lines.push(
          `${i + 1}. Step: ${entry.step || entry.step_id || "—"}`,
        );
        lines.push(
          `   Outcome: ${entry.outcome || entry.result || "—"}`,
        );
        if (entry.answer) lines.push(`   Answer: ${entry.answer}`);
        if (entry.note) lines.push(`   Note: ${entry.note}`);
        lines.push("");
      });
    }

    triggerDownload(
      `tier1-diagnostics-${timestamp()}.txt`,
      lines.join("\n"),
    );
  } catch (err) {
    // eslint-disable-next-line no-console
    console.error("[demo-download] diagnostics failed:", err);
    // eslint-disable-next-line no-alert
    alert("Download failed. Check console for details.");
  }
}
