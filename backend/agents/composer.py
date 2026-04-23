"""
Response Composer Agent — synthesizes analysis findings into a final answer.
Takes the per-step findings from the Analysis Agent and composes a coherent,
grounded, bullet-point answer. Uses Haiku by default. Enforces grounding
rules so the final answer stays document-faithful.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

from backend.config import settings
from backend.agents.base import AgentStepResult, TokenBudget, invoke_llm

logger = logging.getLogger("acadia-log-iq")


# ─────────────────────────────────────────────────────────────
# Default composer voice (pre-Sprint-3A baseline)
#
# Lifted to module scope so _select_composer_voice() can return
# `_composer_rules` BY IDENTITY when Sprint 3A is flag-off, when no session
# mode is set, or when the active mode is not in _VOICE_BY_MODE. Identity-
# equality is the formal flag-off guarantee — see §4.1 of the revised
# sprint brief and the offline regression test.
# ─────────────────────────────────────────────────────────────
_composer_rules = """You are a senior operations engineer synthesizing findings from a multi-step analysis into a single conversational answer for a trainee.

The analysis findings below were produced by sub-agents reading the source documents. Use them as your factual basis but answer the user's question in your own words, like a human expert talking to a colleague.

Rules:
- Match response length to the question. Short question → short answer. Broad question → fuller answer.
- Use natural prose. Bullets only when the question asks for a list or comparison.
- Do not add section headers unless the user asked for a structured breakdown.
- Do not say "based on the findings" or "according to the analysis". Just answer.
- If the findings don't fully cover the question, say what's missing in one plain sentence and give the best partial answer you can."""


# ─────────────────────────────────────────────────────────────
# Sprint 3A — mode-tuned composer voices
#
# Only Troubleshooting is PRD-aligned in this iteration (Sprint 3A-REVISED).
# The other three voice strings remain defined so Sprints 3C (escalation),
# 3D (ticket_handling), and 3E (vendor_oem) can wire them into the active
# dispatch table when their respective corpora (SOPs, contact directories,
# past vendor cases) are ingested. Until then they are intentionally NOT
# in _VOICE_BY_MODE — firing them on ticket-history data would be a PRD
# violation.
# ─────────────────────────────────────────────────────────────

_VOICE_TROUBLESHOOTING = """You are a senior on-call operations engineer walking a trainee through a live troubleshooting incident. The analysis findings below were produced by sub-agents reading the source ticket history. Use them as your factual basis.

Structure the answer with these exact headings (as bold labels on their own line), in this order:

**PROBABLE ROOT CAUSE** — one or two sentences stating the most likely root cause in plain language, grounded in the strongest evidence in the findings.

**EVIDENCE** — a short bulleted list (2–5 items) of the specific observations, ticket IDs, timestamps, or log excerpts from the findings that point to that root cause. Bold ticket IDs and identifiers.

**VERIFICATION STEPS** — a numbered list of concrete commands or checks the trainee can run NOW to confirm the root cause before escalating. Favor non-destructive reads first. If a step involves a literal command, wrap it in `backticks`.

**ESCALATION TRIGGER** — one sentence stating the condition under which the trainee should stop troubleshooting and escalate (e.g., "If step 3 returns X, page the network vendor TAC immediately").

Rules:
- Lead with PROBABLE ROOT CAUSE as the very first content after any preamble the renderer adds. Do not add a separate introductory paragraph before it.
- If the findings don't fully support a confident root cause, say so in the PROBABLE ROOT CAUSE block in one sentence and still provide VERIFICATION STEPS that would disambiguate.
- Do not say "based on the findings" or "according to the analysis". Just answer."""


_VOICE_TICKET_HANDLING = """You are a senior service-desk lead coaching a trainee on how to handle an incoming ticket correctly. The analysis findings below were produced by sub-agents reading the ticket and related standards.

Answer conversationally but lead with what the trainee should DO next on this ticket. When the question asks about procedure or standards compliance, cite the specific field, status, or SLA clause by name.

Rules:
- Match response length to the question.
- Prefer concrete next-step verbs ("Update the affected-CI field to...", "Set priority to P2 because...") over abstract advice.
- Call out any missing required fields or SLA risks explicitly.
- Do not say "based on the findings" or "according to the analysis". Just answer."""


# Sprint 3D — 4 sub-mode voices for Ticket Handling. Each assumes the
# findings were retrieved with doc_kinds=["sop"] (wired in the retrieval
# layer via resolve_mode_doc_kinds). These override the generic
# _VOICE_TICKET_HANDLING above when LOGIQ_SPRINT3D_BACKEND=True and the
# session carries a matching sub_mode.
_VOICE_TH_CREATE = """You are helping a NOC engineer create a new ticket. The findings below are SOP/runbook excerpts describing how tickets should be documented (doc_kind=sop).

Structure the answer as:

**Required information to capture:**
- [checklist drawn from SOP, 5–10 items — use `- [ ] item` format]

**Ticket classification guidance:**
- [from SOP: severity criteria, category list]

**Initial next owner:**
- [from SOP: triage rules]

Rules:
- Use checklist format (`- [ ] item`) for the required-information block.
- If the SOP content is generic, adapt to the likely incident type in the user's query.
- Do NOT invent fields that are not grounded in SOP findings.
- Do not say "based on the findings" or "according to the analysis". Just answer."""


_VOICE_TH_UPDATE = """You are helping a NOC engineer update an active ticket. Findings are SOP/runbook guidance (doc_kind=sop).

Structure the answer as:

**What to add to the ticket:**
- [from SOP: progress-note expectations]

**Status change criteria:**
- [from SOP: when to change status, and to what]

**Customer communication required?** Yes/No — plus one-sentence why.

Rules:
- If the user's query includes current ticket state, tailor the advice to that state.
- Be concise — updates happen under time pressure.
- Do not say "based on the findings" or "according to the analysis". Just answer."""


_VOICE_TH_CLOSE = """You are helping a NOC engineer prepare a ticket for closure. Findings are SOP/runbook closure criteria (doc_kind=sop).

Structure the answer as:

**Closure readiness checklist:**
- [ ] Summary of work performed documented
- [ ] Root cause captured
- [ ] Customer notified / confirmation received
- [ ] Follow-up items logged in a separate ticket
- [ ] Post-mortem triggered if P1
- [... additional SOP-specific items drawn from findings ...]

**Any of these FAIL the closure?** [if findings suggest gaps, ask the user directly]

**Suggested closure note:** [draft 2–4 sentences based on SOP template; the user can edit it]

Rules:
- Every SOP-defined closure criterion in the findings must appear as a checkbox.
- The draft note is editable by the user — do not treat it as final.
- Do not say "based on the findings" or "according to the analysis". Just answer."""


_VOICE_TH_VALIDATE = """You are validating a ticket's quality against SOP standards. The user has pasted ticket content into the query; your job is to critique it. Findings are SOP/runbook quality criteria (doc_kind=sop).

Structure the answer as:

**Strengths of this ticket:**
- [what is well-documented per SOP]

**Gaps identified:**
- [missing fields, unclear language, incorrect classification, etc.]

**Recommended edits:**
- [specific improvements, citing SOP template fields]

**Quality score:** 1–5 (based on how many SOP criteria are met)

Rules:
- Be direct but constructive.
- Cite which SOP criterion each gap violates (by field name or clause).
- Do not say "based on the findings" or "according to the analysis". Just answer."""


# Sprint 3D — (mode, sub_mode) keyed dispatch for Ticket Handling.
# Consulted first inside _select_composer_voice when LOGIQ_SPRINT3D_BACKEND
# is on; falls back to the copy-and-extend _VOICE_BY_MODE path when the
# pair is not registered.
_VOICE_BY_MODE_SUB = {
    ("ticket_handling", "ticket_create"):   _VOICE_TH_CREATE,
    ("ticket_handling", "ticket_update"):   _VOICE_TH_UPDATE,
    ("ticket_handling", "ticket_close"):    _VOICE_TH_CLOSE,
    ("ticket_handling", "ticket_validate"): _VOICE_TH_VALIDATE,
}


_VOICE_ESCALATION = """You are a NOC shift lead producing an escalation reference card. The findings below are contact records from the escalation directory (doc_kind=contact_customer). Your job is to present them as a ready-to-use escalation path — NOT as a narrative paragraph. This is for a 3 AM P1: the engineer needs to see a phone number in seconds.

Structure the answer EXACTLY as follows, using these exact bold labels and markdown headings:

**Customer / Context:** <customer name from the query or from the findings, or "Generic tier directory" if no customer is named>

**Escalation Path:**

### Level 1 — <team name>
- **Name:** <name>
- **Role:** <role>
- **Phone:** <phone_primary>
- **Email:** <email>
- **Hours:** <hours>
- **Notes:** <notes>

### Level 2 — <team name>
(same structure, only if a Level 2 contact is present in the findings)

### Level 3 — <team name>
(only if present)

**After-hours / on-call:** <contact for after-hours path, if any>

**Triggers for escalation:** <pulled from escalation.triggers if present in the findings>

Rules:
- Render ONE primary record per escalation level. If multiple people exist at the same level, list the first as the level header and any additional contacts as sub-bullets ("Alt: Name — Phone — Email") under that level.
- If a field is missing from the record, OMIT that line entirely. Do NOT write "N/A" or "Not documented" — silent omission keeps the card readable at a glance.
- If no contacts are found for the customer named in the query, write exactly one sentence: "No contacts found for <customer>. Check tier-level fallback." — then render whatever generic tier-level contacts do exist from the findings.
- Do not invent phone numbers, emails, or names. If the finding lacks a number, leave the Phone line out.
- Do not add commentary, preamble, or "based on the directory" phrasing."""


_VOICE_VENDOR_OEM = """You are preparing a vendor engagement for a NOC engineer. The findings below include vendor contact records (doc_kind=contact_vendor) AND prior vendor cases (doc_kind=vendor_case) retrieved from the corpus. Produce a three-part response using these EXACT bold section headers and markdown structure.

**Part 1 — Vendor Case Writeup:**

**Product Affected:** <exact model / firmware from findings>
**Issue Summary:** <1–2 sentences>
**Steps to Reproduce:**
1. <step from findings>
2. <...>
**Expected Behavior:** <what should happen>
**Actual Behavior:** <what does happen — VERBATIM error codes and log lines>
**Environment:** <from findings: topology, peers, scale>
**Troubleshooting Performed:** <numbered list of what has already been tried>
**Business Impact:** <1 sentence>

**Part 2 — Vendor Contact:**

**Vendor:** <from contact record>
**TAC Phone:** <tac_phone>
**Support Portal:** <support_portal_url>
**Account Manager:** <account_manager>
**SLA Tier:** <sla_tier>

**Part 3 — Prior Similar Cases (if any):**
- <Case ID> — brief outcome in 1 line
- <Case ID> — ...

Rules:
- Every error code, command output, and log line in Part 1 must be VERBATIM from findings — do not paraphrase.
- If a field is not populated by findings, write exactly "Not captured in current analysis" on that line (do not omit the line — vendor TAC expects the structure).
- If no vendor contact records were retrieved, write "No vendor contact records found — open case via generic vendor portal" under Part 2 and omit the individual fields.
- Keep Part 3 brief — 2–4 prior cases maximum, each one line.
- Do not say "based on the findings" or "according to the analysis". Just produce the case."""


# Sprint 3B — KB/Runbook pivot voice. Triggered by `voice_override="kb_pivot"`
# when the user 👎s a troubleshooting answer and the backend re-runs retrieval
# filtered to doc_kinds=["sop","kb"]. The 5-section format is the PRD §6
# runbook pattern (Validate config → Upstream dep → Alarms → Compare with
# prior incidents → Vendor involvement).
_VOICE_KB_PIVOT = """You are presenting KB/runbook guidance to a trainee whose historical-ticket answer was not helpful. The findings below were produced by retrieving content filtered to SOPs and KB articles only — do NOT reference ticket history.

Your job: extract actionable next steps from the KB/runbook findings below. Structure the answer in this exact order, using these exact bold headings on their own lines:

1. **Validate configuration / service state** — specific commands to run or screens to check.
2. **Check upstream dependency / transport path** — what to verify (BGP peers, provider link, DNS, auth source, etc.).
3. **Verify known alarms / logs / event patterns** — where to look and what signature to match.
4. **Compare with prior similar incidents** — what KB article or pattern this most resembles.
5. **Confirm whether vendor involvement is required** — yes/no plus the criterion (e.g., "Yes if the interface still flaps after step 2").

Rules:
- Use EXACT numbered headings above. Do not rename or reorder them.
- Each step = 1–2 concrete actions pulled from the findings. If the findings do not cover a step, write exactly: "Not documented in current KB — skip".
- Wrap literal commands in `backticks`. Bold any KB article IDs or runbook names (e.g., **KB-PLAT-014**, **RUNBOOK-BGP-FLAP**).
- If no KB guidance exists at all for this issue, say so plainly in one sentence under heading 1 and suggest escalation under heading 5. Do not invent content.
- Do not say "based on the findings" or "according to the analysis". Just answer."""


# Sprint 3A-REVISED — only Troubleshooting is PRD-aligned in this
# iteration. Other voices remain defined (above) for future Sprints
# 3C/3D/3E but are intentionally NOT in the active dispatch table. This
# prevents them from firing on corpora that don't yet exist (SOPs,
# contact directories, vendor cases).
_VOICE_BY_MODE = {
    "troubleshooting": _VOICE_TROUBLESHOOTING,
    # "ticket_handling": _VOICE_TICKET_HANDLING,  # Enabled by Sprint 3D
    # "escalation":      _VOICE_ESCALATION,       # Enabled by Sprint 3C
    # "vendor_oem":      _VOICE_VENDOR_OEM,       # Enabled by Sprint 3E
}


# Sprint 4 — Fingerprint-First Expert Copilot voice.
#
# Triggered when a user enters an exact fingerprint (e.g., BGP-5-ADJCHANGE)
# on the landing screen and the hybrid retriever finds a matching
# gold-standard ticket (quality_score >= FINGERPRINT_MIN_QUALITY_SCORE).
# The findings passed to the Composer in this mode come from ONE richly
# annotated ticket whose metadata includes Symptom_Solution_Mapping,
# Operational_SOP, Knowledge_Base, and remediation_payload.
#
# Structure is fixed: Phase 1 (Forensic Triage) → Phase 2 (Branching
# Diagnostics) → Expert Pivot → Phase 3 (Validated Fix with RaC snippet).
# The Expert Pivot block is the "handoff" moment where the junior engineer
# stops following the script and starts adjudicating evidence.
_VOICE_EXPERT_COPILOT = """You are the "Expert Troubleshooting Copilot." The user typed an exact fingerprint code on the landing screen (e.g., BGP-5-ADJCHANGE) and the retrieval layer found ONE matching gold-standard ticket with a rich Symptom → Solution map, Operational SOP, Knowledge Base refs, and a remediation payload. Your job is to walk a NOC engineer through resolving this specific signature as if you were sitting next to them.

Use this EXACT structure, with these EXACT markdown headings on their own lines, in this order:

# Troubleshooting Guide: <fingerprint code> — <one-line plain-English summary>

## Phase 1: Forensic Triage
One short paragraph stating what this fingerprint means and the immediate symptoms the engineer will see. Then a bulleted "first-look" checklist of 3–5 non-destructive read commands or dashboard checks drawn from the findings. Wrap every literal command in `backticks`.

## Phase 2: Branching Diagnostics
A branching decision tree. Use this exact format:

- **If <observation A>** → likely cause: <cause A>. Next: `<command>` and check `<field>`.
- **If <observation B>** → likely cause: <cause B>. Next: `<command>` and check `<field>`.
- **If neither** → proceed to Expert Pivot.

Keep it 3–5 branches. Pull the observation/cause pairs directly from the ticket's Symptom_Solution_Mapping.

## Expert Pivot
One short paragraph stating the single highest-confidence hypothesis from the gold ticket (the one its actual resolution vindicated). Then a numbered list of 2–4 concrete verification steps that would either confirm or rule out that hypothesis. Bold any KB article IDs, runbook names, or ticket IDs (e.g., **KB-PLAT-014**, **INC-10037**).

## Phase 3: Validated Fix
The exact remediation steps that resolved the gold ticket. Lead with one sentence stating the fix. Then the runbook as a numbered list. If the ticket's remediation_payload includes a Remediation-as-Code (RaC) snippet, include it verbatim in a fenced code block with the correct language hint. Close with a one-line verification command the engineer should run AFTER the fix to confirm the signature cleared.

Rules:
- Use the EXACT headings above. Do not rename, reorder, or add headings.
- Every command and config snippet goes in `backticks` or a fenced code block. Never paraphrase a command.
- Ground everything in the single gold-ticket findings. If a section has no supporting evidence, write exactly: "Not documented for this fingerprint — consult the on-call escalation path." Do not invent steps.
- Do not say "based on the findings" or "according to the analysis". Just answer."""


# Sprint 3B — override-mode voice map. Keyed by explicit `voice_override`
# passed by the caller (not by selected_mode). Sprint 3B seeded this with
# KB pivot; Sprint 4 added the Expert Copilot voice. Kept separate from
# _VOICE_BY_MODE so a normal session-mode lookup never accidentally
# routes into an override voice.
_VOICE_BY_OVERRIDE = {
    "kb_pivot": _VOICE_KB_PIVOT,
    "expert_copilot": _VOICE_EXPERT_COPILOT,  # Sprint 4
}


def _select_composer_voice(
    session_mode: Any,
    voice_override: Optional[str] = None,
    sub_mode: Optional[str] = None,
) -> str:
    """
    Return the composer voice prompt for the given session mode.

    Sprint 3A flag-off guarantee: when LOGIQ_SPRINT3A_BACKEND is False,
    when session_mode is None, or when the mode is not in _VOICE_BY_MODE,
    this function returns `_composer_rules` BY IDENTITY (`is`). Callers
    can therefore assert `_select_composer_voice(m) is _composer_rules`
    to prove the flag-off / deferred-mode path emits a byte-identical
    prompt to the pre-3A baseline.

    Sprint 3B — `voice_override` takes priority over session_mode when
    set AND LOGIQ_SPRINT3B_BACKEND is True. Used today only by the KB
    pivot pipeline (voice_override="kb_pivot"). When Sprint 3B is off,
    the override is ignored and the Sprint 3A path runs unchanged.

    Sprint 3D — when LOGIQ_SPRINT3D_BACKEND is True, a (mode, sub_mode)
    pair registered in `_VOICE_BY_MODE_SUB` wins over _VOICE_BY_MODE.
    Today that only activates Ticket Handling sub-modes
    (ticket_create/update/close/validate). The 3D flag is checked
    AFTER the Sprint 3A gate, so ticket-handling voices never fire
    when 3A is off — preserving the flag-off identity guarantee.

    `session_mode` accepts either:
      - None (no mode selected / flag off / read failed)
      - a SessionMode dataclass (reads `.selected_mode` and `.sub_mode`)
      - a raw string mode name (already normalized)

    `sub_mode` is an explicit override — when given, it beats the
    dataclass's `.sub_mode` attribute. Callers typically leave it None
    and let the dataclass supply it.
    """
    # Sprint 3B / Sprint 4 — voice_override wins when any owning sprint
    # flag is on AND an override voice exists for the key. Each override
    # key is "owned" by whichever sprint introduced it; keeping the gate
    # as an OR across owners lets Sprint 4 (expert_copilot) work when the
    # operator turns on LOGIQ_SPRINT4_BACKEND without also needing 3B on.
    # The caller is the only place that decides WHICH key to pass, so a
    # caller running under Sprint 4 only will never pass "kb_pivot" and
    # vice versa — the OR is safe.
    if voice_override and (
        getattr(settings, "LOGIQ_SPRINT3B_BACKEND", False)
        or getattr(settings, "LOGIQ_SPRINT4_BACKEND", False)
    ):
        override_voice = _VOICE_BY_OVERRIDE.get(voice_override)
        if override_voice is not None:
            return override_voice

    if not getattr(settings, "LOGIQ_SPRINT3A_BACKEND", False):
        return _composer_rules

    if session_mode is None:
        return _composer_rules

    mode_name: Optional[str] = None
    resolved_sub: Optional[str] = sub_mode
    if isinstance(session_mode, str):
        mode_name = session_mode
    else:
        mode_name = getattr(session_mode, "selected_mode", None)
        if resolved_sub is None:
            resolved_sub = getattr(session_mode, "sub_mode", None)

    if not mode_name:
        return _composer_rules

    # Sprint 3D — (mode, sub_mode) dispatch wins over the mode-only table
    # when the 3D flag is on AND a pair is registered. This is consulted
    # BEFORE the Sprint 3C copy-and-extend so a future pair like
    # ("escalation", <sub>) could slot in without changing this block.
    if (
        resolved_sub
        and getattr(settings, "LOGIQ_SPRINT3D_BACKEND", False)
    ):
        pair_voice = _VOICE_BY_MODE_SUB.get((mode_name, resolved_sub))
        if pair_voice is not None:
            return pair_voice

    # Sprint 3C / 3E — copy-and-extend pattern. The module-level
    # _VOICE_BY_MODE stays frozen (only Troubleshooting active by
    # default); each per-call dict adds sprint-specific voices when
    # their flag is on. Mutating the module dict would be order-sensitive
    # across sprints and leak across test cases. Copy keeps each call
    # pure and the flag-off path identity-equal to _composer_rules.
    active_table = dict(_VOICE_BY_MODE)
    if getattr(settings, "LOGIQ_SPRINT3C_BACKEND", False):
        active_table["escalation"] = _VOICE_ESCALATION
    if getattr(settings, "LOGIQ_SPRINT3E_BACKEND", False):
        active_table["vendor_oem"] = _VOICE_VENDOR_OEM

    voice = active_table.get(mode_name)
    if voice is None:
        return _composer_rules

    return voice


# ─────────────────────────────────────────────────────────────
# Composer markdown formatting addendum (Fix 5 — Rich Formatting Polish)
#
# The brief specifies this helper sit in orchestrator.py, but Composer's
# prompt construction is delegated here to run_composer() — which means
# orchestrator.py already imports this module. Placing the helper here
# avoids a circular import while still wrapping the Composer system
# prompt at its actual construction site.
# ─────────────────────────────────────────────────────────────

_COMPOSER_MARKDOWN_ADDENDUM = """

Markdown formatting rules (use only when they genuinely help readability):
- When the user asks to COMPARE specific items (e.g., "Compare INC-10005 and INC-10006"), lead the answer with a markdown table showing key attributes side-by-side (Customer, Priority, Duration, Resolution Approach, Outcome, etc.), then follow with a brief prose paragraph explaining the key insight or lesson. Always include a table for compare queries unless the items have no comparable attributes.
- Ticket IDs, customer names, product names, component names → use **bold** emphasis (e.g., **INC-10037**, **Enterprise-617**, **ADTRAN 908E**). Do NOT wrap identifiers in backticks.
- Inline code (`backticks`) is ONLY for literal commands (`show bgp summary`), file paths (`/etc/config`), or variable names. Never for ticket IDs.
- Step-by-step procedures → use numbered lists with clear action verbs.
- Key recommendations or important takeaways → use > blockquote on its own paragraph.
- Commands, configs, code → fenced code blocks with language hint (```bash, ```python, etc.).
- Default to flowing prose for non-compare questions. Do not force structure when prose is clearer.

CRITICAL — Confident synthesis: When the user asks for lessons, insights, takeaways, recommendations, biggest learnings, or "what should I learn from X", you MUST synthesize a confident answer from the RESOLUTION DETAIL, ROOT CAUSE, ITIL 5-WHY, SOP EXECUTION STEPS, and QA AUDITOR GAPS sections. The lesson is implicit in how the incident was resolved and what went wrong — extract it and present it confidently in your own words. Do NOT refuse by saying "not explicitly stated" or "I could not find this" just because the literal word "lesson" is absent from the document. The documents contain everything needed to answer insight questions through inference.

CRITICAL — Cross-cutting pattern synthesis: When the user asks analytical questions spanning multiple tickets (e.g., "common root causes", "recurring themes", "most frequently recommended improvements"), structure your answer as grouped insights with counts. Format each pattern as:

- **Pattern name** (N tickets) — brief description with representative examples

Example: "Hardware failures (14 tickets) — predominantly Cisco router SPE modules and switch stack failures; examples include INC-10005, INC-10006, INC-10032."

After the grouped patterns, add a brief 1-2 sentence synthesis paragraph explaining the dominant theme or recommendation. Never return raw ticket ID lists for analytical questions."""


def _build_composer_prompt_with_markdown(base_prompt: str) -> str:
    """
    Append markdown formatting guidance to the Composer base prompt.

    Controlled by AGENT_COMPOSER_MARKDOWN_ENABLED flag. When disabled,
    returns base_prompt unchanged — exactly pre-fix behavior, byte-for-byte.
    """
    if not getattr(settings, "AGENT_COMPOSER_MARKDOWN_ENABLED", True):
        return base_prompt
    return base_prompt + _COMPOSER_MARKDOWN_ADDENDUM


def run_composer(
    *,
    query: str,
    findings: List[str],
    source_names: List[str],
    budget: TokenBudget,
    generate_fn: Callable,
    bedrock_client: Any,
    session_mode: Any = None,
    voice_override: Optional[str] = None,
) -> AgentStepResult:
    """
    Composer Agent: synthesizes step-by-step findings into a final answer.

    How it works:
    1. Receives all findings from the Analysis Agent
    2. Asks the composer model to merge them into a coherent answer
    3. Enforces grounding rules: no outside knowledge, bullet-point format
    4. Returns the composed answer as an AgentStepResult

    If the composer fails or budget is exhausted, falls back to
    concatenating the raw findings directly.
    """
    # --- Prepare the findings block ---
    findings_text = "\n\n".join(findings) if findings else "[No analysis findings available]"
    sources_str = ", ".join(sorted(set(source_names))[:6]) if source_names else "uploaded documents"

    # Conversational composer voice — see CONVERSATIONAL_REFACTOR_BRIEF goal 1.2.
    # Length-proportional, bullets-only-when-asked, no meta-phrasing about findings.
    #
    # Fix 5 — the rules section is wrapped with the markdown addendum helper
    # (flag-gated). The content section below the wrap is appended verbatim
    # so that when AGENT_COMPOSER_MARKDOWN_ENABLED=False the final prompt is
    # byte-for-byte identical to the pre-fix composer prompt.
    #
    # Sprint 3A-REVISED — _select_composer_voice() returns `_composer_rules`
    # BY IDENTITY when the Sprint 3A flag is off, when no mode is set, or
    # when the active mode is not in _VOICE_BY_MODE (today that is every
    # mode except "troubleshooting"). The identity guarantee preserves the
    # pre-3A byte-identical prompt for the default path.
    _composer_rules_active = _select_composer_voice(
        session_mode, voice_override=voice_override,
    )
    if voice_override and _composer_rules_active is _VOICE_BY_OVERRIDE.get(voice_override):
        _voice_label = f"override:{voice_override}"
    elif _composer_rules_active is not _composer_rules:
        _voice_label = "mode_aware"
    else:
        _voice_label = "default"
    logger.info("[composer_voice] voice=%s", _voice_label)

    _composer_content = f"""

AVAILABLE SOURCES: {sources_str}

ANALYSIS FINDINGS:
{findings_text}

USER QUESTION: {query}

FINAL ANSWER:"""

    prompt = _build_composer_prompt_with_markdown(_composer_rules_active) + _composer_content

    # Brief 5 / Part 2 — composer always produces analytical synthesis output,
    # regardless of the input query's class. Cap at the analytical tier so we
    # still trim the old 2048 blanket.
    _composer_max_tokens = (
        settings.RESPONSE_TOKENS_ANALYTICAL
        if settings.RESPONSE_TOKEN_CAPS_ENABLED
        else settings.AGENT_COMPOSER_MAX_TOKENS
    )

    # Fix 4 — analytical budget awareness. When the pipeline ran with the
    # elevated analytical ceiling (AGENT_ANALYTICAL_BUDGET) and there is
    # ample headroom remaining, prefer the full analytical response cap so
    # synthesis completes end-to-end instead of truncating. The effective
    # max is still bounded by budget.remaining inside invoke_llm(), so this
    # never over-spends; it just prevents premature truncation when room
    # exists. Flag-gated implicitly via AGENT_ANALYTICAL_BUDGET (0 = off).
    _analytical_floor = getattr(settings, "AGENT_ANALYTICAL_BUDGET", 0)
    if (
        _analytical_floor
        and budget.max_total >= _analytical_floor
        and budget.remaining >= settings.AGENT_BUDGET_COMPOSER_TOKENS
    ):
        _prev_max = _composer_max_tokens
        _composer_max_tokens = max(
            _composer_max_tokens, settings.RESPONSE_TOKENS_ANALYTICAL,
        )
        if _composer_max_tokens != _prev_max:
            logger.info(
                "[composer_budget] analytical floor applied: %d → %d",
                _prev_max, _composer_max_tokens,
            )

    logger.info(
        "[resp_class] composer class=analytical max_tokens=%d", _composer_max_tokens,
    )
    step_result = invoke_llm(
        prompt=prompt,
        model=settings.AGENT_COMPOSER_MODEL,
        max_tokens=_composer_max_tokens,
        budget=budget,
        agent_name="composer",
        generate_fn=generate_fn,
        bedrock_client=bedrock_client,
    )

    # --- Fallback: if composer fails, concatenate raw findings ---
    if not step_result.success or not step_result.output.strip():
        logger.warning("Composer failed, falling back to raw findings")
        fallback = "Based on the analysis of the uploaded documents:\n\n"
        for finding in findings:
            fallback += f"{finding}\n\n"
        step_result.output = fallback.strip()
        step_result.agent_name = "composer (fallback)"

    logger.info(
        "Composer complete: %d chars, model=%s, %dms",
        len(step_result.output), step_result.model_used, step_result.duration_ms,
    )

    return step_result


# ─────────────────────────────────────────────────────────────
# Sprint 5 — Hybrid LLM pipeline for gold-schema JSON tickets
#
# Templates render Phase 2, Phase 3, Header, Fingerprints, KB citations
# deterministically (expert_copilot_template.py, zero LLM). The LLM is
# asked ONLY for the two sections that genuinely need synthesis:
#   - Phase 1: Forensic Triage (narrative + dependency analysis)
#   - Expert Pivot (mental pivot + red herrings + invisible triggers)
#
# Prompt payload shrinks from ~40KB to ~8KB. Latency drops ~5x vs the
# Sprint 4 full-JSON path. Output structure (4 sections) is unchanged
# — the user sees the same Expert Copilot guide they saw in Sprint 4.
# ─────────────────────────────────────────────────────────────
def run_hybrid_expert_pipeline(
    *,
    json_ticket: Dict[str, Any],
    pre_rendered_sections: Dict[str, str],
    budget: TokenBudget,
    generate_fn: Callable,
    bedrock_client: Any,
) -> str:
    """Sprint 5 hybrid pipeline.

    Args:
        json_ticket: the full gold-schema ticket dict (Metadata +
            Symptom_Solution_Mapping + Operational_SOP +
            remediation_payload + Knowledge_Base + Header).
        pre_rendered_sections: dict keyed by "header", "phase_2",
            "phase_3", "kb_citations" (the last is optional). Values
            are markdown strings produced by expert_copilot_template.
        budget: a TokenBudget sized like the Sprint 4 path
            (AGENT_MAX_TOTAL_TOKENS default).
        generate_fn: the shared safe_generate callable.
        bedrock_client: shared Bedrock client.

    Returns the stitched final markdown answer — the string saved to
    chat history AND to the chunk cache.
    """
    ssm = json_ticket.get("Symptom_Solution_Mapping") or {}
    meta = json_ticket.get("Metadata") or {}
    kb_list = json_ticket.get("Knowledge_Base") or []
    kb_excerpt = kb_list[0] if isinstance(kb_list, list) and kb_list else {}

    focused_context = {
        "header": json_ticket.get("Header"),
        "incident_number": meta.get("Incident_Number"),
        "target_service": meta.get("Target_Service"),
        "affected_assets": meta.get("Affected_Assets"),
        "customer_name": meta.get("customer_name") or meta.get("Customer_Name"),
        "detected_symptom": ssm.get("Detected_Symptom") if isinstance(ssm, dict) else None,
        "origin_event": ssm.get("Origin_Event") if isinstance(ssm, dict) else None,
        "fingerprints": meta.get("Fingerprints"),
        "knowledge_base_primary": kb_excerpt,
    }

    prompt = (
        "You are the Expert Troubleshooting Copilot. Generate ONLY two "
        "sections of a troubleshooting guide:\n\n"
        "1. **Phase 1: Forensic Triage** — 3-4 sentences establishing "
        "what the fingerprint(s) technically imply, plus a "
        "'First-look checklist' with 3-5 initial verification commands.\n\n"
        "2. **Expert Pivot** — 2-3 paragraphs capturing the single "
        "highest-confidence hypothesis (the 'mental pivot') and 2-4 "
        "verification steps to confirm it.\n\n"
        "Do NOT generate Phase 2, Phase 3, headers, fingerprint lists, "
        "or remediation code — those are rendered separately.\n\n"
        "Context (JSON):\n"
        f"{json.dumps(focused_context, indent=2, ensure_ascii=False)}\n\n"
        "Output ONLY the two sections requested, with their markdown "
        "headers. Nothing else."
    )

    step_result = invoke_llm(
        prompt=prompt,
        model=settings.AGENT_COMPOSER_MODEL,
        max_tokens=1200,
        budget=budget,
        agent_name="expert_copilot_hybrid",
        generate_fn=generate_fn,
        bedrock_client=bedrock_client,
    )

    llm_output = (step_result.output or "").strip() if step_result.success else ""

    parts = [
        pre_rendered_sections.get("header", ""),
        "",
        "---",
        "",
        llm_output,
        "",
        "---",
        "",
        pre_rendered_sections.get("phase_2", ""),
        "",
        "---",
        "",
        pre_rendered_sections.get("phase_3", ""),
    ]
    kb_cite = pre_rendered_sections.get("kb_citations")
    if kb_cite:
        parts.extend(["", "---", "", kb_cite])

    stitched = "\n".join(p for p in parts if p is not None)
    logger.info(
        "[sprint5_hybrid] llm_chars=%d final_chars=%d model=%s",
        len(llm_output), len(stitched), step_result.model_used,
    )
    return stitched
