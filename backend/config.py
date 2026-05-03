"""
Configuration — Phases 2-6 Complete + Accuracy Fixes + Performance Tuning.
Includes: contextual ingestion, hybrid retrieval, model routing,
multi-agent troubleshooting, answer validation guardrails,
accuracy fixes, and performance optimizations.
"""

import sys
from pathlib import Path
from typing import ClassVar, Dict, List, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


if sys.version_info < (3, 11):
    raise RuntimeError("This application requires Python 3.11 or higher")


BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """
    Global application configuration.
    """

    # ----------------------------------------------------------------
    # Storage
    # ----------------------------------------------------------------

    UPLOAD_DIR: Path = BASE_DIR / "uploads"

    MAX_FILE_SIZE_MB: int = 100
    ALLOWED_FILE_TYPES: List[str] = [
        "log", "txt", "md", "json", "pdf", "docx",
    ]

    # ----------------------------------------------------------------
    # AWS / Bedrock — model IDs
    # ----------------------------------------------------------------

    AWS_REGION: str = "us-east-1"

    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_SESSION_TOKEN: Optional[str] = None

    BEDROCK_EMBED_MODEL: str = "amazon.titan-embed-text-v2:0"
    BEDROCK_LLM_MODEL: str = "mistral.mistral-7b-instruct-v0:2"
    BEDROCK_HAIKU_MODEL: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    BEDROCK_SONNET_MODEL: str = "us.anthropic.claude-sonnet-4-6"

    # ----------------------------------------------------------------
    # Database
    # ----------------------------------------------------------------

    DATABASE_URL: str

    # ----------------------------------------------------------------
    # Phase-1 chunking compatibility
    # ----------------------------------------------------------------

    MAX_CHARS: int = 8000
    OVERLAP: int = 300
    BATCH_SIZE: int = 10

    # ----------------------------------------------------------------
    # Phase-2 contextual chunking (accuracy fix: generous limits)
    # ----------------------------------------------------------------

    CHUNK_MAX_CHARS: int = 6000
    CHUNK_MIN_CHARS: int = 200
    CHUNK_OVERLAP_CHARS: int = 0
    CHUNK_BATCH_SIZE: int = 10          # ← was 6, now 10 chunks per Haiku call (fewer API calls)

    LLM_CHUNK_FALLBACK_PREVIEW_CHARS: int = 8000
    ENABLE_LLM_CHUNK_FALLBACK: bool = True

    MAX_METADATA_INPUT_CHARS: int = 1800
    MAX_CONTEXT_SUMMARY_CHARS: int = 120
    MAX_OPERATIONAL_LABELS: int = 8
    MAX_METADATA_RETRIES: int = 2

    HAIKU_TEMPERATURE: float = 0.0
    HAIKU_MAX_TOKENS: int = 4096

    # ----------------------------------------------------------------
    # Adaptive ingestion batching — sizes Haiku batches by estimated
    # output tokens so verbose PDFs do not truncate JSON mid-response.
    # ----------------------------------------------------------------

    ADAPTIVE_INGESTION_BATCHING_ENABLED: bool = True
    ADAPTIVE_TARGET_OUTPUT_TOKENS: int = 4000
    ADAPTIVE_BATCH_SIZE_MIN: int = 3
    ADAPTIVE_BATCH_SIZE_MAX: int = 25
    ADAPTIVE_MAX_TOKENS_BUFFER: float = 1.3
    ADAPTIVE_MAX_TOKENS_MIN: int = 2000
    ADAPTIVE_MAX_TOKENS_MAX: int = 8000

    # ----------------------------------------------------------------
    # CSV column-aware parsing
    # ----------------------------------------------------------------
    CSV_COLUMN_AWARE_PARSING_ENABLED: bool = True
    CSV_MAX_ROWS_PER_UPLOAD: int = 50000

    # ----------------------------------------------------------------
    # Dynamic per-organization schema inference
    # ----------------------------------------------------------------
    DYNAMIC_SCHEMA_INFERENCE_ENABLED: bool = True
    SCHEMA_CONFIDENCE_THRESHOLD: float = 0.5

    # ----------------------------------------------------------------
    # Retrieval: prefer latest version of a document family
    # ----------------------------------------------------------------
    PREFER_LATEST_VERSION_IN_RETRIEVAL: bool = True

    # ----------------------------------------------------------------
    # Concurrency — PERFORMANCE TUNED
    # ----------------------------------------------------------------

    METADATA_CONCURRENCY: int = 6       # ← was 4, now 6 parallel Haiku calls
    EMBED_CONCURRENCY: int = 12         # ← was 8, now 12 parallel embed calls

    # ----------------------------------------------------------------
    # Phase-3: Hybrid Retrieval Orchestration
    # ----------------------------------------------------------------

    VECTOR_WEIGHT: float = 0.45
    BM25_WEIGHT: float = 0.30
    KEYWORD_WEIGHT: float = 0.25

    VECTOR_CANDIDATES: int = 25
    BM25_CANDIDATES: int = 20
    KEYWORD_CANDIDATES: int = 15

    FTS_MIN_RANK: float = 0.01
    EXACT_TERM_BOOST: float = 2.0
    RRF_K: int = 60

    RERANKER_BACKEND: str = "llm"
    RERANK_TOP_K: int = 10
    RERANK_CANDIDATES: int = 20
    RERANK_SCORE_WEIGHT: float = 0.70
    RERANK_FUSION_WEIGHT: float = 0.30

    KEYWORD_QUERY_THRESHOLD: float = 0.60
    SEMANTIC_QUERY_THRESHOLD: float = 0.60
    ENABLE_QUERY_CLASSIFICATION: bool = True

    ENABLE_METADATA_FILTER: bool = True
    METADATA_FILTER_CANDIDATES: int = 10

    MIN_GROUNDING_SCORE: float = 0.18
    MIN_KEYWORD_OVERLAP: int = 1

    # ----------------------------------------------------------------
    # Phase-4: Model Routing
    # ----------------------------------------------------------------

    ENABLE_MODEL_ROUTING: bool = True
    ROUTING_DEFAULT_MODEL: str = "haiku"

    COMPLEXITY_SIMPLE_THRESHOLD: float = 0.30
    COMPLEXITY_COMPLEX_THRESHOLD: float = 0.70

    SONNET_MAX_TOKENS: int = 4096
    SONNET_TEMPERATURE: float = 0.1
    SONNET_MONTHLY_BUDGET_USD: float = 50.0

    HAIKU_ANSWER_MAX_TOKENS: int = 2048
    HAIKU_ANSWER_TEMPERATURE: float = 0.1

    SESSION_CONTEXT_MAX_MESSAGES: int = 4
    SESSION_CONTEXT_MAX_CHARS: int = 2000
    INCLUDE_METADATA_IN_PROMPT: bool = True
    INCLUDE_CONFIDENCE_IN_PROMPT: bool = True

    CX_WEIGHT_MULTI_STEP: float = 0.30
    CX_WEIGHT_REASONING: float = 0.25
    CX_WEIGHT_CONTEXT_SIZE: float = 0.15
    CX_WEIGHT_LOW_CONFIDENCE: float = 0.20
    CX_WEIGHT_MULTI_DOC: float = 0.10

    # ----------------------------------------------------------------
    # Phase-5: Multi-Agent Troubleshooting
    # ----------------------------------------------------------------

    ENABLE_AGENT_MODE: bool = True
    AGENT_COMPLEXITY_THRESHOLD: float = 0.65
    AGENT_MIN_SOURCES: int = 1
    AGENT_MAX_STEPS: int = 4

    AGENT_PLANNER_MODEL: str = "sonnet"
    AGENT_ANALYSIS_MODEL: str = "haiku"
    AGENT_COMPOSER_MODEL: str = "haiku"

    AGENT_PLANNER_MAX_TOKENS: int = 1024
    AGENT_ANALYSIS_MAX_TOKENS: int = 1500
    AGENT_COMPOSER_MAX_TOKENS: int = 2048

    AGENT_MAX_TOTAL_TOKENS: int = 8000
    AGENT_TIMEOUT_SECONDS: int = 45

    # ----------------------------------------------------------------
    # Goal 4: Dynamic agent token budget
    # ----------------------------------------------------------------
    # Static AGENT_MAX_TOTAL_TOKENS above is a legacy fallback. When
    # ENABLE_DYNAMIC_AGENT_BUDGET is True, the orchestrator computes
    # max_total = CLARIFIER + PLANNER + steps*PER_STEP + COMPOSER after
    # the Planner returns its step list, clamped by HARD_CAP so a runaway
    # plan can't explode cost. Analyst stops early if remaining budget
    # would dip below the Composer reserve.

    AGENT_BUDGET_PER_STEP_TOKENS: int = 3500
    AGENT_BUDGET_PLANNER_TOKENS: int = 1500
    AGENT_BUDGET_COMPOSER_TOKENS: int = 3000
    AGENT_BUDGET_CLARIFIER_TOKENS: int = 500
    AGENT_BUDGET_HARD_CAP_TOKENS: int = 20000
    ENABLE_DYNAMIC_AGENT_BUDGET: bool = True

    ENABLE_CLARIFIER: bool = True
    AGENT_CLARIFIER_MODEL: str = "haiku"
    AGENT_CLARIFIER_MAX_TOKENS: int = 400
    CLARIFIER_MIN_QUERY_LEN: int = 3
    CLARIFIER_MAX_QUESTIONS: int = 2

    # ----------------------------------------------------------------
    # Tier 2 aggregation intent classifier (Brief 2)
    # ----------------------------------------------------------------
    # When regex detect_aggregation_intent misses, fall through to a
    # Haiku classifier that reads one-shot intent off the query text.
    # Runtime-toggleable: flip AGGREGATION_CLASSIFIER_ENABLED=False to
    # fall back to regex-only behavior without a deploy.
    AGGREGATION_CLASSIFIER_ENABLED: bool = True
    AGGREGATION_CLASSIFIER_MODEL: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    AGGREGATION_CLASSIFIER_MAX_TOKENS: int = 300
    AGGREGATION_CLASSIFIER_TEMPERATURE: float = 0.0  # deterministic
    AGGREGATION_CLASSIFIER_MIN_CONFIDENCE: float = 0.6
    AGGREGATION_CLASSIFIER_TIMEOUT_SECONDS: float = 4.0

    # ----------------------------------------------------------------
    # Same-session query rewriter (Brief 3)
    # ----------------------------------------------------------------
    # Resolves pronouns, ordinals, ellipsis, and filter swaps against
    # the last few turns BEFORE aggregation detect / retrieval run, so
    # "what caused it?" after "tell me about INC-10015" becomes
    # "What caused INC-10015?" automatically. Toggle the enabled flag
    # off to revert to pre-Brief-3 pipeline behavior without a deploy.
    QUERY_REWRITER_ENABLED: bool = True
    QUERY_REWRITER_MODEL: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    QUERY_REWRITER_MAX_TOKENS: int = 200
    QUERY_REWRITER_TEMPERATURE: float = 0.0
    QUERY_REWRITER_TIMEOUT_SECONDS: float = 3.5
    QUERY_REWRITER_MAX_HISTORY_TURNS: int = 8
    QUERY_REWRITER_MIN_QUERY_LEN_FOR_REWRITE: int = 2

    # ----------------------------------------------------------------
    # Brief 4 — LLM optimization sprint
    # ----------------------------------------------------------------
    # Opt 1 — Query-rewriter micro-cache
    QUERY_REWRITER_CACHE_ENABLED: bool = True
    QUERY_REWRITER_CACHE_TTL_SECONDS: int = 60
    QUERY_REWRITER_CACHE_MAX_ENTRIES: int = 1000

    # Opt 2 — Parallel rewriter + aggregation classifier
    PARALLEL_REWRITE_AND_CLASSIFY_ENABLED: bool = True

    # Opt 3 — Merged triage classifier (complexity + intent + mode in one call)
    MERGED_TRIAGE_ENABLED: bool = True
    MERGED_TRIAGE_MODEL: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    MERGED_TRIAGE_MAX_TOKENS: int = 150
    MERGED_TRIAGE_TIMEOUT_SECONDS: float = 3.0

    # Opt 4 — Skip reranker on tiny result sets
    SKIP_RERANK_ON_TINY_RESULTS_ENABLED: bool = True
    RERANK_MIN_CHUNKS: int = 3

    # Opt 5 — Context compression for generation
    CONTEXT_COMPRESSION_ENABLED: bool = True
    CONTEXT_COMPRESSION_MIN_CHUNK_CHARS: int = 5000

    # ----------------------------------------------------------------
    # Brief 5 — Cost optimization sprint
    # ----------------------------------------------------------------
    # Part 1 — Semantic Answer Cache
    SEMANTIC_CACHE_ENABLED: bool = True
    SEMANTIC_CACHE_SIMILARITY_THRESHOLD: float = 0.94
    SEMANTIC_CACHE_MIN_CONFIDENCE_TO_CACHE: float = 0.75
    SEMANTIC_CACHE_TTL_DAYS: int = 7
    SEMANTIC_CACHE_MAX_ENTRIES: int = 50000
    SEMANTIC_CACHE_EMBED_MODEL: str = "amazon.titan-embed-text-v2:0"
    SEMANTIC_CACHE_LOG_HITS: bool = True

    # Part 2 — Response Token Caps
    RESPONSE_TOKEN_CAPS_ENABLED: bool = True
    RESPONSE_TOKENS_CLASSIFICATION: int = 30
    RESPONSE_TOKENS_SHORT_FACT: int = 120
    # Sprint 2.7 Bug D — was 350; raised to 600 so multi-section
    # explanations (hardware + application + network) fit in one Haiku
    # call instead of triggering the walkthrough-reclassification retry.
    RESPONSE_TOKENS_EXPLANATION: int = 600
    RESPONSE_TOKENS_WALKTHROUGH: int = 600
    RESPONSE_TOKENS_ANALYTICAL: int = 1200
    RESPONSE_TOKENS_DEFAULT: int = 350

    # Part 3 — Input Guardrails + Output Sanitizer
    INPUT_GUARDRAILS_ENABLED: bool = True
    INPUT_GUARDRAIL_REGEX_BLOCK: bool = True
    INPUT_GUARDRAIL_CLASSIFIER_ENABLED: bool = True
    INPUT_GUARDRAIL_CLASSIFIER_MODEL: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    INPUT_GUARDRAIL_CLASSIFIER_MAX_TOKENS: int = 80
    INPUT_GUARDRAIL_CLASSIFIER_MIN_CONFIDENCE: float = 0.75
    INPUT_GUARDRAIL_CLASSIFIER_TIMEOUT_SECONDS: float = 2.0
    INPUT_GUARDRAIL_SCRUB_PII: bool = True
    OUTPUT_SANITIZER_ENABLED: bool = True
    OUTPUT_SANITIZER_SCRUB_PII: bool = True

    # ----------------------------------------------------------------
    # Two-bug fix brief — feature flags for safe rollback
    # ----------------------------------------------------------------
    # Bug 1: classifier-path group_by_customer now defaults to
    # ranking_direction="highest" (most-first). Flip to False only if
    # a regression surfaces on "fewest/lowest" phrasings.
    GROUP_BY_DEFAULT_DESC_ENABLED: bool = True
    # Bug 2: carry original-query identifiers as a scope-lock across
    # all Analyst sub-steps so per-step retrieval stays grounded in
    # the same tickets the user asked about. Flip to False to revert
    # to step-text-only retrieval.
    ANALYST_IDENTIFIER_LOCK_ENABLED: bool = True
    # Bug 3: aggregation classifier is stateless — filters it returns must
    # be textually present in the query. Strips hallucinated customer /
    # priority / sla / component filters that bleed from prior turns.
    # Flip to False to allow raw classifier output without the guard.
    AGG_CLASSIFIER_STATELESS_ENABLED: bool = True

    # ----------------------------------------------------------------
    # Rewriter + regex fix brief — feature flags for safe rollback
    # ----------------------------------------------------------------
    # Issue 1: defensive guard that rejects rewriter output which injects
    # filters (priority, SLA, customer names) into already-self-contained
    # queries. Flip to False to disable the safety check without a deploy.
    REWRITER_SAFETY_GUARD_ENABLED: bool = True
    # Issue 2: regex aggregation classifier strips P1..P4 priority tokens
    # before running the customer-name regex and rejects non-customer
    # tokens (P1, SLA, SBC, etc.) so "How many P1 tickets?" no longer
    # captures "P1" as a customer name. Flip to False to revert.
    REGEX_PRIORITY_AWARE_PARSING_ENABLED: bool = True

    # ----------------------------------------------------------------
    # Four-bug fix brief — feature flags for safe rollback
    # ----------------------------------------------------------------
    # Bug 1: refuse queries where scrubbed PII was the lookup key (e.g.
    # "Find tickets for SSN 123-45-6789"). Incidental PII still scrubs.
    PII_CENTRAL_REFUSAL_ENABLED: bool = True
    # Bug 2: apply the customer/priority/sla filters to the ENTIRE ranking
    # result set, not just the top row, so "lowest Nebula-Corp quality
    # score" returns Nebula-Corp-only tickets across top-1 + others-near.
    RANKING_CUSTOMER_SCOPE_FIX_ENABLED: bool = True
    # Bug 3: first-class rework_detected filter threaded classifier →
    # AggIntent → SQL. Requires ingestion to populate metadata_json's
    # rework_detected key; the clause is safe when the field is absent.
    REWORK_FILTER_ENABLED: bool = True
    # Bug 4A: reject the aggregation fast-path when the query mentions
    # content-only keywords (brand names, change-request, "involved",
    # etc.) that need chunk-text retrieval instead of metadata SQL.
    AGG_CONTENT_FILTER_REJECT_ENABLED: bool = True
    # Bug 4B: enable avg/sum/min/max numeric aggregation operations over
    # whitelisted metadata_json numeric fields.
    NUMERIC_AGGREGATION_OPS_ENABLED: bool = True

    # ----------------------------------------------------------------
    # Final cleanup brief — feature flags for safe rollback
    # ----------------------------------------------------------------
    # Bug 1: populate metadata_json->>'rework_detected' during gold-ticket
    # ingestion so the rework SQL filter can actually match rows. Flipping
    # to False doesn't un-ingest existing rows, but it stops the field
    # from being written on subsequent uploads.
    INGEST_REWORK_METADATA_ENABLED: bool = True
    # Bug 2: thread priority/sla_met/rework_detected/component filters
    # through _run_group_by_customer, mirroring the ranking fix. Without
    # this, "Among P1 tickets, which customers had the most SLA misses?"
    # returns the unfiltered P1 distribution.
    GROUP_BY_FULL_FILTER_ENABLED: bool = True
    # Bug 3: multi-strategy JSON parsing for Mistral reranker output so
    # "Extra data" edge cases recover the ranking instead of silently
    # dropping to fusion order. Fallback behavior unchanged on total
    # parse failure — we just reduce WHEN that fallback fires.
    RERANKER_ROBUST_PARSE_ENABLED: bool = True

    # ----------------------------------------------------------------
    # Interactive Clarifier (Brief 6) — feature flags
    # ----------------------------------------------------------------
    # Presents 3-4 clickable clarification options when retrieval surfaces
    # multiple distinct candidates with no clear winner. Runs AFTER retrieval,
    # BEFORE agent pipeline / full RAG generation. Falls through safely on
    # any error so existing behavior is preserved when the flag is off.
    INTERACTIVE_CLARIFIER_ENABLED: bool = True
    INTERACTIVE_CLARIFIER_AMBIGUITY_THRESHOLD: float = 0.6
    INTERACTIVE_CLARIFIER_MAX_PER_SESSION: int = 3
    INTERACTIVE_CLARIFIER_MAX_TOKENS: int = 600
    INTERACTIVE_CLARIFIER_TIMEOUT_SECONDS: float = 4.0
    INTERACTIVE_CLARIFIER_MODEL: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

    # ----------------------------------------------------------------
    # Goal 2: Schema-agnostic identifier extraction
    # ----------------------------------------------------------------
    # Config-only extensibility — add new schemas here WITHOUT code changes.
    #
    # IDENTIFIER_PATTERNS maps id_type → regex. The id_type string MUST
    # match what the ingestion schema writes into chunks.metadata_json
    # ['id_type'] (see STRUCTURED_SCHEMAS in contextual_ingestion_service).
    # Each regex must have exactly ONE capture group yielding the raw
    # identifier body (digits or the whole token). Patterns are evaluated
    # in declaration order; earlier matches own their span so later
    # patterns can't re-tag the same token (prevents 'INC-10005' from
    # being misclassified as an issue_key).
    IDENTIFIER_PATTERNS: Dict[str, str] = {
        "ticket_number": r"\bINC[-\s]?(\d{3,})\b",
        "issue_key":     r"\b([A-Za-z][A-Za-z0-9]+-\d+)\b",
    }
    # IDENTIFIER_CANONICAL_FORMAT maps id_type → str.format template. The
    # capture group from the matching regex is passed as positional arg 0.
    # The canonical string is UPPER()-compared against
    # chunks.metadata_json->>'primary_id', so formats must match whatever
    # the ingestion schema wrote at ingest time.
    IDENTIFIER_CANONICAL_FORMAT: Dict[str, str] = {
        "ticket_number": "INC-{0}",
        "issue_key":     "{0}",
    }

    # ----------------------------------------------------------------
    # Phase-6: Validation Guardrails & Confidence Scoring
    # ----------------------------------------------------------------

    ENABLE_ANSWER_VALIDATION: bool = True

    CONF_WEIGHT_RETRIEVAL: float = 0.30
    CONF_WEIGHT_COVERAGE: float = 0.25
    CONF_WEIGHT_GROUNDING: float = 0.25
    CONF_WEIGHT_CONSISTENCY: float = 0.20

    VALIDATION_MIN_CONFIDENCE: float = 0.35
    VALIDATION_MIN_GROUNDING: float = 0.25
    VALIDATION_MIN_COVERAGE: float = 0.20

    VALIDATION_HALLUCINATION_PHRASES: List[str] = [
        "as an AI",
        "I don't have access",
        "based on my training",
        "in general",
        "typically",
        "it is commonly known",
        "from my knowledge",
        "as of my last update",
    ]

    VALIDATION_WARN_SUPERSEDED: bool = True
    VALIDATION_SUPERSEDED_PENALTY: float = 0.20

    VALIDATION_MAX_RETRIES: int = 1
    VALIDATION_RETRY_EXPAND_K: int = 3

    ENABLE_EVAL_LOGGING: bool = True
    EVAL_LOG_FILE: Optional[str] = None

    # ----------------------------------------------------------------
    # Feature flags
    # ----------------------------------------------------------------

    ENABLE_METADATA_EXTRACTION: bool = True
    ENABLE_VERSION_DETECTION: bool = True
    ENABLE_DUPLICATE_CHECK: bool = True
    ENABLE_CHUNK_SUMMARY: bool = True
    ENABLE_TABLE_PARSING: bool = True
    ENABLE_CODE_BLOCK_DETECTION: bool = True
    INCLUDE_OLD_VERSIONS: bool = False

    # ----------------------------------------------------------------
    # API
    # ----------------------------------------------------------------

    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    REQUEST_TIMEOUT: int = 30
    API_KEY: Optional[str] = None
    UI_API_KEY: Optional[str] = None
    API_BASE: str = "http://localhost:8000"

    # ----------------------------------------------------------------
    # Clerk authentication
    # ----------------------------------------------------------------

    CLERK_SECRET_KEY: Optional[str] = None
    CLERK_PUBLISHABLE_KEY: Optional[str] = None
    CLERK_ENABLED: str = "false"

    # ----------------------------------------------------------------
    # SES email
    # ----------------------------------------------------------------

    SES_SENDER_EMAIL: str = "noreply@acadiaconsultants.com"
    SES_FEEDBACK_RECIPIENT: str = "dev@acadiaconsultants.com"
    SES_REGION: Optional[str] = None
    SES_ENABLED: str = "true"

    # ----------------------------------------------------------------
    # Pattern Analytics (Layer 3 selective stats engine)
    # ----------------------------------------------------------------
    PATTERN_ANALYTICS_ENABLED: bool = True
    PATTERN_ANALYTICS_MIN_SIMILAR_TICKETS: int = 3
    PATTERN_ANALYTICS_CONFIDENCE_THRESHOLD: float = 0.7
    PATTERN_ANALYTICS_SIMILARITY_THRESHOLD: float = 0.65
    PATTERN_ANALYTICS_RECENCY_WINDOW_DAYS: int = 30
    PATTERN_ANALYTICS_CACHE_TTL_HOURS: int = 24
    PATTERN_ANALYTICS_TOP_N_ACTIONS: int = 3
    PATTERN_ANALYTICS_FORCE_ENABLE_IN_TROUBLESHOOTING_MODE: bool = True

    # ----------------------------------------------------------------
    # Rich Response Formatting Brief (Approach A — pure markdown)
    # ----------------------------------------------------------------
    # Master flag for rich formatting prompt guidance. When enabled, backend
    # adds a markdown formatting addendum to the Claude system prompt, telling
    # the LLM when to use tables, code blocks, bold, inline code, blockquotes.
    # When False, the prompt is byte-identical to the pre-brief conversational
    # engineer voice (zero behavioral change).
    RICH_FORMATTING_PROMPT_ENABLED: bool = True

    # When True, the grounding rules emitted by context_builder include the
    # "Formatting guidance" block that tells the model when to reach for
    # markdown tables / fenced code / bold. When False, only the original
    # answering guidance is emitted. Independent of the system-prompt flag
    # so the two prompt layers can be A/B tested.
    RICH_FORMATTING_GROUNDING_RULES_ENABLED: bool = True

    # ----------------------------------------------------------------
    # Pattern Analytics Polish Brief — 3 surgical fixes, each flag-gated.
    # ----------------------------------------------------------------
    # Fix 1 — queries.load_similar_tickets_for_topic splits topic into
    # individual terms and OR-matches tsvector + ILIKE across summary /
    # resolution_text / component / chunk content. Flip False to fall
    # back to the legacy ILIKE-OR-on-chunks path preserved under
    # _load_similar_tickets_legacy.
    PATTERN_ANALYTICS_SQL_OR_MATCHING_ENABLED: bool = True

    # Fix 2 — when pattern_context is active, orchestrator caps planner
    # steps, bumps the dynamic budget, and reserves composer tokens so
    # the Composer never gets starved. All other (non-pattern) queries
    # use the legacy dynamic budget exactly as before.
    PATTERN_ANALYTICS_BUDGET_TUNING_ENABLED: bool = True
    PATTERN_ANALYTICS_MAX_PLANNER_STEPS: int = 3
    PATTERN_ANALYTICS_AGENT_BUDGET: int = 25000
    PATTERN_ANALYTICS_COMPOSER_RESERVE: int = 4000

    # Fix 3 — Luhn-validate candidate credit-card digit sequences before
    # redacting. Prevents ticket IDs / long numeric identifiers from
    # being falsely masked as PII. Only affects the credit_card PII type
    # — SSN, email, AWS keys, private keys are untouched.
    PII_CREDIT_CARD_LUHN_VALIDATION_ENABLED: bool = True

    # ────────────────────────────────────────────────
    # Rich Formatting Polish — Fix 5: Agent Composer Markdown
    # ────────────────────────────────────────────────
    #
    # When True, the multi-agent Composer's system prompt receives a
    # markdown formatting addendum instructing it to use tables for
    # compare queries, bold for identifiers, fenced code blocks for
    # commands, etc. When False, Composer behaves exactly as before
    # (prose-heavy synthesis, no markdown instructions) — the prompt
    # is byte-for-byte identical to the pre-fix composer prompt.
    #
    # Rollback: set to False, restart backend. Zero impact on other flags.
    AGENT_COMPOSER_MARKDOWN_ENABLED: bool = True

    # ────────────────────────────────────────────────
    # Cross-Cutting Analytical Router
    # ────────────────────────────────────────────────
    #
    # When True, analytical queries ("common root causes", "recurring
    # patterns", "most frequently recommended improvements") are detected
    # before the SQL aggregation classifier and routed to the agent
    # pipeline for proper content synthesis. When False, all queries flow
    # through existing agg_classifier exactly as before.
    CROSS_CUTTING_ANALYTICAL_ROUTING_ENABLED: bool = True

    # Minimum confidence to route as analytical (0.0-1.0).
    # Higher = fewer false positives, some analytical queries miss.
    # Lower = more analytical routing, some SQL queries misrouted.
    CROSS_CUTTING_DETECTOR_CONFIDENCE_THRESHOLD: float = 0.75

    # Minimum distinct signal categories required (prevents single-word
    # triggers like "list all tickets" with only one scope signal).
    CROSS_CUTTING_MIN_SIGNAL_CATEGORIES: int = 2

    # Fast mode ticket limit (top reranked tickets the analyst reads).
    CROSS_CUTTING_FAST_MODE_TICKET_LIMIT: int = 20

    # Deep mode ticket limit (all matching, capped for cost).
    CROSS_CUTTING_DEEP_MODE_TICKET_LIMIT: int = 50

    # Analytical result cache TTL in seconds (1 hour default).
    CROSS_CUTTING_CACHE_TTL_SECONDS: int = 3600

    # ────────────────────────────────────────────────
    # Agent Pipeline Performance Tuning
    # Additive flags — all existing budget/flag behavior preserved when
    # these four are flipped False.
    # ────────────────────────────────────────────────

    # Budget ceiling for analytical + compare queries (cross-cutting, deep
    # analysis). Separate from PATTERN_ANALYTICS_AGENT_BUDGET (pattern-specific
    # queries). Higher ceiling prevents Composer truncation on complex work.
    AGENT_ANALYTICAL_BUDGET: int = 25000

    # Concurrency limit for parallel analyst step execution.
    # 3 = reasonable balance of speed vs Bedrock rate limits.
    ANALYST_PARALLEL_CONCURRENCY: int = 3

    # Run independent analyst steps in parallel (wave-based).
    # Dependent steps (synthesize/combine/cross-reference) stay sequential.
    ANALYST_PARALLEL_EXECUTION_ENABLED: bool = True

    # Cache retrieval results keyed by identifier list within one pipeline run.
    AGENT_RETRIEVAL_CACHE_ENABLED: bool = True

    # Early-skip clarifier when identifiers/high-confidence retrieval/simple
    # patterns make clarification unnecessary.
    CLARIFIER_EARLY_SKIP_ENABLED: bool = True

    # ────────────────────────────────────────────────
    # Dynamic Agent Budget
    # Replaces static AGENT_ANALYTICAL_BUDGET for analytical queries with a
    # formula: base + (tickets × per_ticket) + (steps × per_step), × deep
    # multiplier when applicable, clamped to [floor, ceiling].
    # AGENT_ANALYTICAL_BUDGET (25000) is preserved as the fallback used
    # when AGENT_DYNAMIC_BUDGET_ENABLED is False.
    # ────────────────────────────────────────────────

    # Master flag — when True, analytical queries use formula-based dynamic
    # budget. When False, falls back to static AGENT_ANALYTICAL_BUDGET for
    # byte-identical pre-brief behavior.
    AGENT_DYNAMIC_BUDGET_ENABLED: bool = True

    # Base synthesis budget (Composer minimum — prevents truncation floor).
    AGENT_DYNAMIC_BUDGET_BASE: int = 8000

    # Per-ticket overhead (synthesis tokens needed per ticket analyzed).
    AGENT_DYNAMIC_BUDGET_PER_TICKET: int = 1500

    # Per-step overhead (planner step contribution to final synthesis).
    AGENT_DYNAMIC_BUDGET_PER_STEP: int = 2000

    # Floor — minimum viable budget regardless of inputs.
    AGENT_DYNAMIC_BUDGET_FLOOR: int = 10000

    # Ceiling — hard cap to prevent runaway cost on huge queries.
    AGENT_DYNAMIC_BUDGET_CEILING: int = 35000

    # Multiplier for deep analysis mode (explicit user request for depth).
    AGENT_DYNAMIC_BUDGET_DEEP_MULTIPLIER: float = 1.5

    # ─────────────────────────────────────────────────────────────
    # Guided Workflow (PRD: User Journey - LogIQ Landing Page Guidance)
    # Sprint 1 — session state foundation + landing page shell.
    # ─────────────────────────────────────────────────────────────

    # Master kill-switch. When False, the backend ignores mode endpoints
    # and the frontend skips the landing page. Flip to True to roll out.
    GUIDED_WORKFLOW_ENABLED: bool = False

    # When True, a session with a locked mode rejects /ask queries that
    # look clearly cross-mode. Sprint 1 keeps this OFF (soft mode); the
    # real enforcement arrives in Sprint 2 once context-break detection
    # is wired. Flag exists now so Sprint 1 is forward-compatible.
    MODE_LOCK_STRICT: bool = False

    # Sprint 2 placeholder — regex-based context-break phrase detection.
    # Declared now so Sprint 1 deploys don't need a config reload later.
    CONTEXT_BREAK_DETECTION_ENABLED: bool = False

    # Sprint 2 placeholder — lets triage_classifier's existing Haiku call
    # emit a context_break hint on ambiguous messages (no new LLM call).
    CONTEXT_BREAK_LLM_HINT_ENABLED: bool = False

    # ─────────────────────────────────────────────────────────────
    # Sprint 2 — Context awareness + pattern response + structured forms.
    # Single master flag for the whole sprint. Soft dependency on
    # Sprint 1 (GUIDED_WORKFLOW_ENABLED) — Sprint 2 code no-ops with
    # a log line when Sprint 1 is off.
    # ─────────────────────────────────────────────────────────────
    LOGIQ_SPRINT2_BACKEND: bool = False

    # ─────────────────────────────────────────────────────────────
    # Sprint 2.5 — Retrieval & Pattern Hotfix.
    # Single master switch for all 9 bug fixes. When False every
    # touched site falls back byte-for-byte to pre-hotfix behavior.
    # The three HOTFIX_* tunables are *defaults*, not independent
    # feature flags — operators may override them in .env if needed.
    # ─────────────────────────────────────────────────────────────
    LOGIQ_HOTFIX_BACKEND: bool = False

    HOTFIX_VOCAB_MIN_OCCURRENCE: int = 1
    HOTFIX_SEMANTIC_CACHE_THRESHOLD: float = 0.985
    HOTFIX_COMPOSER_RESERVE_TOKENS: int = 5000

    # ─────────────────────────────────────────────────────────────
    # Production Retrieval Fix v2 — 5 deeper architectural bugs.
    # All flags default True so the v2 behavior ships on by default
    # once LOGIQ_HOTFIX_BACKEND is True; each flag is also an
    # independent kill-switch for fast rollback.
    # ─────────────────────────────────────────────────────────────

    # Bug #2: JSON-tree-aware vocabulary learning — keys become
    # field_names, values matching identifier pattern become
    # identifiers, short repeated uppercase values become enums.
    VOCABULARY_JSON_STRUCTURAL_PARSING: bool = True
    VOCABULARY_ENUM_MIN_OCCURRENCE: int = 3

    # Bug #1: identifier extractor consults learned_vocabulary and
    # skips tokens classified as 'field_name' so
    # Resolution_Quality_Score isn't treated as a ticket ID.
    IDENTIFIER_VOCAB_TYPE_CHECK: bool = True

    # Bug #5: multi-column exact lookup across incident_number /
    # vector_id / external_incident_id + suffix stripping so
    # INC-ALPHA-001_SEMANTIC_UNIT maps to INC-ALPHA-001.
    IDENTIFIER_MULTI_COLUMN_LOOKUP: bool = True
    IDENTIFIER_SUFFIX_STRIP_ENABLED: bool = True

    # Bug #4: reserve Composer tokens in a separate pool so the
    # Analyst can't drain the budget and starve the synthesis step.
    AGENT_BUDGET_SEPARATE_COMPOSER_POOL: bool = True

    # Bug #3: admin-only ingestion verification + reindex endpoints.
    INGESTION_VERIFICATION_ENABLED: bool = True

    # ─────────────────────────────────────────────────────────────
    # Sprint 2.6 — Numeric Equality Filter
    # Single master flag gating every new code path added by
    # Sprint 2.6 (score = N / quality_score = N equality filter).
    # Default False so production pre-flip behavior is byte-identical.
    # ─────────────────────────────────────────────────────────────
    LOGIQ_NUMERIC_FILTER_BACKEND: bool = False

    # ─────────────────────────────────────────────────────────────
    # Sprint 2.7 — Accuracy Hotfix (5 bugs)
    # Single master flag gating every new code path added by
    # Sprint 2.7. Flag-off = byte-identical pre-2.7 behavior.
    # HOTFIX_EXPLANATION_TOKENS_CAP is a tunable int (not a second
    # on/off flag) so operators can retune without a code change.
    # ─────────────────────────────────────────────────────────────
    LOGIQ_ACCURACY_HOTFIX_BACKEND: bool = False
    HOTFIX_EXPLANATION_TOKENS_CAP: int = 600

    # ─────────────────────────────────────────────────────────────
    # Sprint 2.8 — Dynamic Content Term Extraction
    # Single master flag gating every new code path added by
    # Sprint 2.8 (dynamic stopword+metadata subtraction for compound
    # content+metadata filters, Bug F customer-name variant match,
    # aggregation-intent-aware reranker cap). Flag-off = byte-identical
    # pre-2.8 behavior. RERANK_TOP_K_AGGREGATION is a tunable int (not
    # a second on/off flag) so operators can retune the aggregation
    # reranker ceiling without a code change.
    # ─────────────────────────────────────────────────────────────
    LOGIQ_COMPOUND_FILTER_BACKEND: bool = False
    RERANK_TOP_K_AGGREGATION: int = 40

    # ─────────────────────────────────────────────────────────────
    # Sprint 2.8.1 — Rewriter ellipsis tightening
    # LOGIQ_REWRITER_STRICT_ELLIPSIS gates the TRUE-ellipsis
    # structural requirement (referential pronoun / sentence
    # conjunction / short fragment without named entity).
    # REWRITER_ELLIPSIS_MIN_CONF raises the confidence floor for
    # accepting an ellipsis_expanded rewrite from 0.85 → 0.92 —
    # SAFE to apply even when the master flag is off.
    # ─────────────────────────────────────────────────────────────
    LOGIQ_REWRITER_STRICT_ELLIPSIS: bool = False
    REWRITER_ELLIPSIS_MIN_CONF: float = 0.92

    # ─────────────────────────────────────────────────────────────
    # Sprint 3A — Mode-Aware Prompts + Post-👍 Action Buttons
    # Single master flag gating (1) mode-tuned composer voice selection
    # and (2) post-thumbs-up action chip rendering signal. Flag-off =
    # byte-identical pre-3A behavior — _composer_rules is returned by
    # identity (the default string object), and the frontend chip
    # registry never renders.
    # ─────────────────────────────────────────────────────────────
    LOGIQ_SPRINT3A_BACKEND: bool = False

    # ─────────────────────────────────────────────────────────────
    # Sprint 3B — 👎 KB/Runbook pivot + low-similarity confidence band
    #
    # Flag-off path: /feedback/state skips the pivot branch and /ask
    # omits the confidence_band field — byte-identical post-3A-REVISED.
    # LOW_SIMILARITY_THRESHOLD controls the top-chunk score below which
    # the answer is tagged `confidence_band="low"` so the frontend can
    # render the "⚠️ Low similarity match" banner.
    # ─────────────────────────────────────────────────────────────
    LOGIQ_SPRINT3B_BACKEND: bool = False
    LOW_SIMILARITY_THRESHOLD: float = 0.35

    # ─────────────────────────────────────────────────────────────
    # Sprint 3C — Escalation mode → contact_customer corpus
    # Flag-off path: escalation mode uses default voice + ticket-history
    # retrieval (post-3A-REVISED behavior byte-identical). Flag-on:
    # retrieval filters to doc_kinds=["contact_customer"] and the
    # composer picks _VOICE_ESCALATION via copy-and-extend (the module
    # _VOICE_BY_MODE dict is NEVER mutated).
    # ─────────────────────────────────────────────────────────────
    LOGIQ_SPRINT3C_BACKEND: bool = False

    # ─────────────────────────────────────────────────────────────
    # Sprint 3D — Ticket Handling mode → sop corpus
    # Flag-off path: ticket_handling mode uses default voice +
    # ticket-history retrieval (post-3A-REVISED behavior byte-identical).
    # Flag-on: retrieval filters to doc_kinds=["sop"] and the composer
    # dispatches via (mode, sub_mode) into 4 sub-mode voices:
    # ticket_create / ticket_update / ticket_close / ticket_validate.
    # ─────────────────────────────────────────────────────────────
    LOGIQ_SPRINT3D_BACKEND: bool = False

    # ─────────────────────────────────────────────────────────────
    # Sprint 3E — Vendor/OEM mode → contact_vendor + vendor_case corpora
    # Flag-off path: vendor_oem mode uses default voice + ticket-history
    # retrieval (post-3A-REVISED behavior byte-identical). Flag-on:
    # retrieval filters to doc_kinds=["contact_vendor", "vendor_case"]
    # and the composer picks _VOICE_VENDOR_OEM via copy-and-extend.
    # ─────────────────────────────────────────────────────────────
    LOGIQ_SPRINT3E_BACKEND: bool = False

    # ─────────────────────────────────────────────────────────────
    # Sprint 4 — Fingerprint-First Expert Copilot
    # Flag-off path: fingerprint endpoints 404, LandingRouter falls
    # through to the Sprint 1 LandingPage, GIN indexes sit unused, new
    # chat_sessions columns stay NULL. Flag-on: FingerprintInputScreen
    # is the first screen; /fingerprint/lookup runs GIN-indexed exact-
    # match retrieval on chunks.metadata_json -> Metadata -> Fingerprints
    # and composes via voice_override="expert_copilot".
    #
    # FINGERPRINT_REGEX — relaxed to accept any non-empty string. The
    # original anchored UPPERCASE-with-hyphen pattern was rejecting
    # inputs we now want to pass straight through to the SQL exact-
    # match (e.g., "%BGP-5-ADJCHANGE"), so the gate has been loosened
    # to "any non-empty". The JSONB `?` operator in retrieve_by_fingerprint
    # will simply return no rows on shapes that don't exist in
    # chunks.metadata_json -> Metadata -> Fingerprints, which is the
    # correct behavior for a miss — no need to fail fast on shape.
    # ─────────────────────────────────────────────────────────────
    LOGIQ_SPRINT4_BACKEND: bool = False
    FINGERPRINT_REGEX: ClassVar[str] = r".+"
    FINGERPRINT_MIN_QUALITY_SCORE: int = 3

    # ─────────────────────────────────────────────────────────────
    # Sprint 11 — Journey-aware retrieval context for /ask.
    # When ON, /ask looks up the chat session's journey_session_id,
    # loads the originating ticket's intake (severity / asset_name /
    # alert_type / customer / ...), and prepends a compact context
    # prefix to the BM25 / keyword query so retrieval narrows toward
    # the right corner of the corpus when the chat originated from a
    # Stage 0 / Stage 3 "Ask in chat" link.
    # The LLM prompt + the visible chat message are NOT touched.
    # Flag-off path: /ask behaves bit-for-bit as before.
    # ─────────────────────────────────────────────────────────────
    LOGIQ_JOURNEY_CHAT_RETRIEVAL_CONTEXT: bool = True

    # ─────────────────────────────────────────────────────────────
    # Sprint 5 — Template-First Expert Copilot + Answer Cache
    # Flag-off path: /fingerprint/lookup runs the full Sprint 4 LLM
    # pipeline end-to-end (byte-identical). Flag-on + gold-schema JSON
    # ticket: template-render Phase 2, Phase 3, header, KB citations
    # deterministically; LLM is asked only for Phase 1 narrative +
    # Expert Pivot; final rendered answer is cached on the chunk row
    # so repeat lookups skip the LLM entirely. Non-gold-schema
    # retrievals (PDFs, Word, KBs, contacts, partial tickets) ALWAYS
    # use the full LLM path regardless of this flag. Cache is
    # invalidated automatically because Sprint 2.9 ingestion
    # DELETEs + re-INSERTs the chunk row on re-upload.
    # ─────────────────────────────────────────────────────────────
    LOGIQ_SPRINT5_BACKEND: bool = False
    EXPERT_COPILOT_CACHE_TTL_DAYS: int = 30

    # ─────────────────────────────────────────────────────────────
    # Sprint 6 — Tier-1 Alert Copilot (form-driven triage module)
    # Flag-off path: /tier1/* routes are NOT mounted (the include
    # block in api.py is skipped), so any call returns FastAPI's own
    # 404. The landing page hides its Tier-1 entry button via the
    # frontend build arg REACT_APP_LOGIQ_TIER1_COPILOT_FRONTEND.
    # Sprint 1–5 behavior is byte-identical when this flag is off.
    # ─────────────────────────────────────────────────────────────
    LOGIQ_TIER1_COPILOT_BACKEND: bool = False
    TIER1_CACHE_TTL_DAYS: int = 7
    TIER1_TOP_K: int = 5
    TIER1_HIGH_CONFIDENCE_THRESHOLD: float = 0.85
    TIER1_MIN_CONFIDENCE_THRESHOLD: float = 0.60

    # ─────────────────────────────────────────────────────────────
    # Sprint 7 — Tier-1 Progressive Workflow
    # Flag-off path: new routes (/tier1/session, /tier1/deeper-diagnostics,
    # /tier1/escalation-package, /tier1/explain, /tier1/session/{id}/...)
    # return 404, the Sprint 6 ranking weights are used verbatim, and the
    # new response fields (top_5_match_ids / session_id / started_at) are
    # present on Tier1AnalyzeResponse but remain empty/None so Sprint 6
    # clients stay byte-identical.
    # ─────────────────────────────────────────────────────────────
    LOGIQ_TIER1_PROGRESSIVE_BACKEND: bool = False
    TIER1_STUCK_THRESHOLD_SECONDS: int = 480   # 8 minutes
    TIER1_TOP_N_MATCHES: int = 5

    # ─────────────────────────────────────────────────────────────
    # Sprint 8 — Tier-1 UX polish
    # Gates the single new endpoint that powers arrow pagination:
    #   GET /tier1/session/{session_id}/match/{match_index}
    # Flag-off path: the endpoint returns 404 (defensively, in addition
    # to the router mount requiring Sprint 6's flag). Sprint 6/7
    # response shapes and ranking weights are untouched.
    # ─────────────────────────────────────────────────────────────
    LOGIQ_TIER1_UX_FIXES_BACKEND: bool = False

    # ─────────────────────────────────────────────────────────────
    # Sprint 10 — Tier-1 Resolution Journey
    # Flag-off path: /tier1/journey/* router not mounted (returns 404),
    # Tier1Workspace renders the existing Sprint 6/7/8 answer-card path
    # unchanged. Flag-on path: GET /initial returns Stage 0 + 1A + 1B
    # always-visible; Stages 2-5 lazy-fetched per labeled next-stage
    # button. Helpful clicks log telemetry to tier1_journey_events
    # (migration 040) but never advance the flow.
    # ─────────────────────────────────────────────────────────────
    LOGIQ_TIER1_JOURNEY_BACKEND: bool = False
    TIER1_JOURNEY_TOP_N: int = 5
    TIER1_JOURNEY_LOOKBACK_MONTHS: int = 18
    TIER1_JOURNEY_LLM_POLISH: bool = False
    TIER1_JOURNEY_BUNDLE_CACHE_TTL_SECONDS: int = 600
    TIER1_JOURNEY_STAGE0_DOMINANT_THRESHOLD: float = 0.4

    # ─────────────────────────────────────────────────────────────
    # Sprint 9 — Universal Intake (Email/Phone/Portal/Chat/Note)
    # Flag-off path: the /intake/* router is not mounted, the frontend
    # SourceToggle is hidden via REACT_APP_LOGIQ_UNIVERSAL_INTAKE_FRONTEND,
    # and Sprint 6/7/8 paths are byte-identical. Catalog builder is
    # lazy: first /intake/extract call materialises the in-memory
    # IntakeCatalogs from chunks.metadata_json (no startup cost).
    # ─────────────────────────────────────────────────────────────
    LOGIQ_UNIVERSAL_INTAKE_BACKEND: bool = False
    INTAKE_MAX_RAW_CHARS: int = 10000
    INTAKE_MAX_CANDIDATES: int = 5            # asked of the LLM
    INTAKE_MAX_CARDS: int = 4                 # shown to the engineer
    INTAKE_CATALOG_MAX_TERMS_PER_TYPE: int = 10000
    # Sprint 9 baseline (kept for backward compat — Sprint 9.2 adds
    # per-field thresholds below that take precedence in the validator).
    INTAKE_FUZZY_MATCH_THRESHOLD: float = 0.70
    INTAKE_FUZZY_MATCH_THRESHOLD_CUSTOMER: float = 0.85

    # ─────────────────────────────────────────────────────────────
    # Sprint 9.2 — per-field fuzzy-match thresholds + token bump.
    # The Sprint 9 single-threshold (0.70) was too permissive on short
    # asset names (e.g. NY4-CORE-RTR-01 vs v-bay-core-rtr scored 0.69
    # — a near-miss the validator was rubber-stamping). 9.2 raises the
    # bars per-field so the long-tail asset names that the reproduction
    # tests proved exist in the corpus actually win their lookups.
    # The per-field constants below are the operative thresholds; the
    # legacy INTAKE_FUZZY_MATCH_THRESHOLD constant remains for any
    # external callers but the new validator code never reads it.
    # ─────────────────────────────────────────────────────────────
    INTAKE_FUZZY_THRESHOLD_ASSET: float = 0.85
    INTAKE_FUZZY_THRESHOLD_ALERT: float = 0.80
    INTAKE_FUZZY_THRESHOLD_CUSTOMER: float = 0.85
    # Bumped from Sprint 9's 1500 — evidence-substring fields add tokens.
    INTAKE_EXTRACTION_MAX_TOKENS: int = 2000
    # Weights MUST sum to 1.00. Declared ClassVar so pydantic treats it as
    # a constant, not a settable field (mutable dicts aren't a valid
    # Settings field type and spec §12 explicitly wants a single source of
    # truth callers reuse for the /tier1/explain breakdown).
    TIER1_RANKING_WEIGHTS: ClassVar[Dict[str, float]] = {
        "alert_type_match":      0.25,
        "asset_match":           0.15,
        "fingerprint_match":     0.15,
        "technology_match":      0.05,
        "vector_similarity":     0.05,
        "resolution_quality":    0.10,
        "recency":               0.05,
        "success_frequency":     0.05,
        "same_customer_boost":   0.075,
        "same_asset_family":     0.075,
    }

    # ─────────────────────────────────────────────────────────────
    # Sprint 2.9 — JSON Structure Validator
    # Rejects uploads whose CONTENT looks like JSON (first non-whitespace
    # byte is { or [) but fails strict parse. Flag-off = byte-identical
    # pre-2.9 behavior (silent False in _is_gold_ticket_json on malformed
    # JSON, fall-through to generic text chunking). See
    # SPRINT_2_9_JSON_VALIDATOR.md for runtime acceptance walk.
    # ─────────────────────────────────────────────────────────────
    LOGIQ_JSON_VALIDATOR_BACKEND: bool = False

    # ─────────────────────────────────────────────────────────────
    # Sprint 3-PREP-A — doc_kind multi-corpus tagging
    # Single master flag that gates the mode→corpus retrieval filter.
    # Flag-off = retriever ignores doc_kinds kwarg even when passed;
    # storage/ingestion always writes the doc_kind column (defaults to
    # 'ticket') so data is ready when the flag flips on.
    # VALID_DOC_KINDS is a ClassVar so pydantic treats it as a constant,
    # not a settable field — the set is immutable and referenced via
    # settings.VALID_DOC_KINDS from ingestion for input validation.
    # ─────────────────────────────────────────────────────────────
    LOGIQ_DOC_KIND_BACKEND: bool = False
    VALID_DOC_KINDS: ClassVar[frozenset] = frozenset({
        "ticket",            # JSON gold-ticket history (Troubleshooting)
        "sop",               # Standard operating procedures, runbooks
        "kb",                # Knowledge base articles
        "contact_customer",  # Customer contact directory (Escalation)
        "contact_vendor",    # Vendor contact directory (Vendor-OEM)
        "vendor_case",       # Historical vendor case records
    })

    # ─────────────────────────────────────────────────────────────
    # Sprint 3-PREP-B — Bulk ingestion (folder / S3 → corpus)
    # Gates both the CLI (`backend/scripts/bulk_ingest.py`) and the
    # doc_kind form field surfaced by the /upload endpoint + admin UI
    # dropdown. Flag-off: CLI exits 2, /upload silently coerces any
    # incoming doc_kind to "ticket" (backward-compatible), UI dropdown
    # is disabled via REACT_APP_LOGIQ_BULK_INGEST_FRONTEND so the UX
    # is byte-identical to post-PREP-A.
    # ─────────────────────────────────────────────────────────────
    LOGIQ_BULK_INGEST_BACKEND: bool = False
    BULK_INGEST_MAX_FILES_PER_RUN: int = 10000

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @field_validator("CLERK_ENABLED", "SES_ENABLED", mode="before")
    @classmethod
    def normalize_string_flags(cls, value):
        if isinstance(value, bool):
            return "true" if value else "false"
        return value


settings = Settings()

AWS_REGION = settings.AWS_REGION
BEDROCK_EMBED_MODEL = settings.BEDROCK_EMBED_MODEL
BEDROCK_LLM_MODEL = settings.BEDROCK_LLM_MODEL

settings.UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
