# Tier-1 Alert Copilot — Developer Guide

## What this does
Form-driven troubleshooting copilot for NOC Tier-1 engineers. The
engineer fills 3 required + 6 optional fields, the backend retrieves
the closest historical incident from the gold-schema JSON ticket
corpus, and returns a fixed 8-section troubleshooting answer with one
follow-up question.

## Why it exists
Tier-1 engineers operate under cognitive pressure. They don't want
chat or free-form Q&A — they want a guided form → ONE answer → ONE
next step. This module delivers that without coupling to the existing
Acadia chat / fingerprint / mode flows.

## Request lifecycle
```
POST /tier1/analyze
    ↓ Tier1AnalyzeRequest validation (schemas.py)
    ↓ normalize_alert()          — signature + search terms
    ↓ get_cached_answer(hash)    — fast path (<300ms)
    ↓ retrieve_top_matches()     — stage 1 exact → stage 2 hybrid
    ↓ extract_compact_context()  — 11 fields only
    ↓ build_prompt()             — strict 8-section system prompt
    ↓ Bedrock Haiku (max=600, temp=0.2 via invoke_llm)
    ↓ parse_answer() + 1 retry + template_fallback
    ↓ set_cached_answer()
    → Tier1AnalyzeResponse
```

## Module ownership
| File | Owns |
|---|---|
| `schemas.py` | API contract (request / response / answer section) |
| `normalizer.py` | Signature + search-term derivation |
| `alias_dictionary.py` | In-memory term expansion, rebuilt at startup |
| `retrieval.py` | Two-stage match + weighted reranking |
| `context_extractor.py` | 11-field compact LLM context |
| `prompt_builder.py` | Strict prompt assembly + output parser + template fallback |
| `cache.py` | Signature-keyed answer cache (tier1_answer_cache) |
| `feedback.py` | 👍/👎 telemetry + 5 follow-up action dispatch |
| `aggregator.py` | Startup hook: build alias dictionary once |
| `routes.py` | FastAPI endpoints (/tier1/analyze, /tier1/feedback, /tier1/health) |

## Input → Processing → Output
- **Input:** structured alert (3 required + 6 optional fields) + session_id
- **Processing:** normalize → signature → cache lookup → retrieval → context → LLM → parse → cache
- **Output:** 8-section answer + confidence badge + 1 follow-up question

## How to debug
- Log prefix: `[tier1_copilot]`
- Last analyze rows:
  ```sql
  SELECT signature_hash, alert_signature, confidence, created_at
    FROM tier1_answer_cache
    ORDER BY created_at DESC LIMIT 5;
  ```
- Recent feedback:
  ```sql
  SELECT response_id, helpful, follow_up_action, created_at
    FROM tier1_feedback
    ORDER BY created_at DESC LIMIT 20;
  ```
- Force a cache miss for testing:
  ```sql
  DELETE FROM tier1_answer_cache WHERE signature_hash = '<hash>';
  ```
- Hard-disable the module: set `LOGIQ_TIER1_COPILOT_BACKEND=false`
  in `backend/.env` and restart — the /tier1/* routes vanish.

## Example request
```http
POST /tier1/analyze
Content-Type: application/json

{
  "severity": "P2",
  "asset_name": "V-Desktop Environment",
  "alert_type": "Desktop Slowness",
  "customer": "Aetheris Corp",
  "session_id": "abc123"
}
```

## Example response
```json
{
  "matched_incident": "INC-ALPHA-027",
  "confidence": "High",
  "similar_count": 4,
  "answer": {
    "issue_understanding": "...",
    "historical_match": "...",
    "most_likely_cause": "...",
    "recommended_first_checks": ["...", "..."],
    "most_likely_fix": "...",
    "validation": "...",
    "escalate_if": "...",
    "follow_up_question": "..."
  },
  "cache_hit": false,
  "response_id": "a1b2c3d4..."
}
```

## Flag gating
| Setting | Default | Effect when off |
|---|---|---|
| `LOGIQ_TIER1_COPILOT_BACKEND` | `False` | Router not mounted → /tier1/* returns 404. Sprint 1-5 byte-identical. |
| `TIER1_CACHE_TTL_DAYS` | `7` | Cache rows expire after N days. Stale reads treated as misses. |
| `TIER1_TOP_K` | `5` | Number of ranked candidates returned by retrieval. |
| `TIER1_HIGH_CONFIDENCE_THRESHOLD` | `0.85` | final_score ≥ this → "High". |
| `TIER1_MIN_CONFIDENCE_THRESHOLD` | `0.60` | final_score in [min, high) → "Medium"; [0.40, min) → "Low"; < 0.40 → "None". |

## Frontend flag
`REACT_APP_LOGIQ_TIER1_COPILOT_FRONTEND` (build-time arg) — when
`"true"`, the LandingRouter renders a "Tier-1 Copilot" entry button
alongside the fingerprint and mode-picker screens. When unset or
`"false"` the button is hidden and Sprint 4 UX is byte-identical.

## Future extension points
1. **Pre-embed alert_signature** for sub-100ms vector search instead of
   recomputing the query embedding per request.
2. **Streaming output** (SSE) so the frontend shows sections as they
   arrive rather than the 2-3s Haiku block.
3. **Multi-engineer handoff** — share alert state + transcript across
   shift handover.
4. **Auto-classification of severity** from alert_type + asset-class so
   engineers don't have to select P1..P4 manually.
5. **Cross-customer pattern alerts** — "5 customers reported Citrix
   slowness in the last hour" banners driven off cache table hits.
6. **Feedback-driven re-ranking** — down-weight tickets whose cached
   answers keep getting 👎 from different engineers.

---

## Sprint 7 Flow Extensions

### Request lifecycle (extended)

```
POST /tier1/analyze
    ↓
Session created in tier1_sessions (stores top_5_match_ids + started_at)
    ↓
Client renders Tier1Workspace with ◀ 1/5 ▶ arrows + confidence banner
    ↓
Progressive UI:
  [👍]     → "Start a new alert" (Sprint 6 thumbs-up CTA)
  [👎]     → Reveals 5 chips (thumbs_down_count on session increments)
    ├─ Similar Cases       → arrow pagination (no new LLM call)
    ├─ Deeper Diagnostics  → POST /tier1/deeper-diagnostics
    ├─ Escalation Package  → POST /tier1/escalation-package
    ├─ Search KB/SOP       → placeholder message (redirects to Acadia chat)
    └─ Why Recommended     → POST /tier1/explain

Passive monitoring:
  GET /tier1/session/{id}/status   (polled every 60s)
    → if stuck_nudge=true → StuckDetectionModal
    → one-shot flip via mark_stuck_shown() so it never re-fires
```

### New endpoints (Sprint 7, all gated behind LOGIQ_TIER1_PROGRESSIVE_BACKEND)

| Method | Path | Purpose |
|---|---|---|
| POST | `/tier1/session` | Explicit session create |
| GET  | `/tier1/session/{id}/status` | Poll stuck flag + timer + thumbs_down count |
| POST | `/tier1/session/{id}/match-index` | Arrow pagination server-side |
| POST | `/tier1/session/{id}/action` | Append a what_tried entry |
| POST | `/tier1/deeper-diagnostics` | Structured diagnostic card |
| POST | `/tier1/escalation-package` | Bundle assembly |
| POST | `/tier1/explain` | Score breakdown + match evidence |

### Module ownership (Sprint 7 additions)

| File | Owns |
|---|---|
| diagnostics/deeper_diagnostics.py | Structured SOP → diagnostic card (skeleton + optional LLM format) |
| diagnostics/escalation_package.py | Bundle assembly with optional Sprint 3C contact enrichment |
| diagnostics/explain_recommendation.py | Score breakdown + field-match evidence |
| diagnostics/stuck_detector.py | Pure nudge threshold logic |
| session_state/tier1_session.py | Per-alert session CRUD |

### Ranking formula

Sprint 7 lowers Sprint 6's alert_type / asset / fingerprint / technology /
vector_similarity weights by 0.05 each and adds `recency`,
`success_frequency`, `same_customer_boost`, and `same_asset_family`
(each 0.075) — see `settings.TIER1_RANKING_WEIGHTS`. Sum is exactly
1.00 (enforced by `test_weights_sum_to_one`). When the progressive flag
is off, the Sprint 6 weight dict is used verbatim.

### How to debug

- Log prefix: `[tier1_copilot:sprint7]`
- Inspect session: `SELECT * FROM tier1_sessions ORDER BY created_at DESC LIMIT 5;`
- Inspect stuck events: `SELECT * FROM tier1_feedback WHERE event_type='stuck_nudge';`
- Force stuck modal for testing: set `TIER1_STUCK_THRESHOLD_SECONDS=10` in `.env`, restart.
- Arrow pagination not working: confirm `top_5_match_ids` populated on the session row.
- LLM skipped deeper-diagnostics format: check logs for
  `deeper-diag formatter failed`; the skeleton path still returns a
  valid step list (llm_used=false in the response).

---

## Sprint 8 — UX fixes + modern theme

### What changed (behaviour — Track A, flag `LOGIQ_TIER1_UX_FIXES_BACKEND` + `REACT_APP_LOGIQ_TIER1_UX_FIXES_FRONTEND`)

- **Confidence labels** rewritten from harsh calibration copy to neutral
  trust-calibrated copy: *Best match* / *Strong candidate* /
  *Closest historical case* / *No close match found*. The underlying
  score bands are unchanged.
- **Deeper Diagnostics** Continue / Skip / Skip-to-escalation buttons
  fully wired; Continue on the last step closes the card with a success
  toast; multi-choice `next_question.options` render as clickable
  buttons on the final step.
- **Skeleton loaders + auto-scroll** on every card fetch (Deeper
  Diagnostics, Escalation Package, Explain Recommendation) and on
  arrow pagination.
- **Arrow pagination fetches real content** via the new endpoint
  `GET /tier1/session/{id}/match/{index}` — the answer card body and
  confidence banner update for the rank-N match.
- **Intake form** redesigned as a progressive conversational flow:
  severity chips, 3 required fields prominent, 6 optional fields
  collapsed behind *Add more context*.

### What changed (theme — Track B, flag `REACT_APP_LOGIQ_TIER1_MODERN_THEME`)

- New `Tier1ThemeProvider` scoped to the Tier-1 tree (never touches the
  Acadia chat's existing `hooks/ThemeContext`).
- Acadia-navy primary + indigo→violet gradient accent tokens.
- `ThemeToggle` in the Workspace header flips classic ↔ modern per-user
  via localStorage; the `storage` event keeps sibling tabs in sync.
- Every Tier-1 card reads `useTier1Theme()` to choose between classic
  Ant Design styling and the modern token set.

### New endpoint

| Method | Path | Purpose |
|---|---|---|
| GET | `/tier1/session/{id}/match/{index}` | Rank-N match's Tier1AnalyzeResponse. Gated behind `LOGIQ_TIER1_UX_FIXES_BACKEND`. |

Scores are **recomputed** via `_rerun_retrieval_for_session` rather than
stored on the session — no schema change needed. Cache still benefits
from the Sprint 6 answer cache when the same alert + chunk combination
repeats.

### Module ownership (Sprint 8 additions)

| File | Owns |
|---|---|
| `frontend/src/theme/acadiaTheme.js` | Modern + classic token bags |
| `frontend/src/theme/ThemeProvider.js` | `Tier1ThemeProvider` + `useTier1Theme` |
| `frontend/src/hooks/useLocalStorage.js` | Storage-event-aware persistence |
| `frontend/src/hooks/useAutoScrollIntoView.js` | Smooth-scroll-on-mount |
| `Tier1Copilot/SkeletonCard.js` | 3-variant loader |
| `Tier1Copilot/SeverityChipSelector.js` | P1..P4 chip picker |
| `Tier1Copilot/AssetAutocomplete.js` | AntD-backed suggest (static list in v1) |
| `Tier1Copilot/ThemeToggle.js` | Classic ↔ modern toggle |

### How to debug

- Arrow-pagination returns the same card: confirm
  `REACT_APP_LOGIQ_TIER1_UX_FIXES_FRONTEND=true` and
  `LOGIQ_TIER1_UX_FIXES_BACKEND=true`; check
  `GET /tier1/session/{id}/match/{index}` 200s in the network tab.
- Theme flash on load: `Tier1ThemeProvider` must wrap the entire
  Workspace tree; inspect the DOM to confirm the Provider is the
  outermost element of `Tier1WorkspaceInner`'s parent.
- Intake form still flat: confirm the UX-fixes frontend flag is baked
  into the bundle (`process.env.REACT_APP_LOGIQ_TIER1_UX_FIXES_FRONTEND`
  is inlined at build time, not runtime).

### Rollback

- Theme only: user toggles via the button; or flip
  `REACT_APP_LOGIQ_TIER1_MODERN_THEME=false` and rebuild — theme gone
  system-wide but behaviour fixes stay.
- Everything: flip both frontend flags + `LOGIQ_TIER1_UX_FIXES_BACKEND`
  off and rebuild — Sprint 7 byte-identical.
