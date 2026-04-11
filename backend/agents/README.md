# Agents Module — Multi-Agent Troubleshooting Pipeline

## Overview

The `backend/agents/` module implements a **multi-agent agentic AI pipeline** (Phase 5 of the project) designed to handle complex, multi-step user queries. It follows the **Planner-Analyst-Composer** pattern — a well-established agentic architecture where specialized AI agents collaborate in a sequential pipeline, each performing a distinct cognitive role.

**Yes, this is Agentic AI.** It exhibits the core characteristics of an agentic system:

| Agentic Property | How This Module Implements It |
|---|---|
| **Autonomy** | The orchestrator autonomously decides whether a query needs multi-agent processing based on complexity scoring and pattern matching |
| **Task Decomposition** | The Planner agent breaks complex queries into concrete sub-steps without human intervention |
| **Sequential Reasoning** | Agents execute in a pipeline (Plan -> Analyze -> Compose), with each stage building on the previous |
| **Self-Monitoring** | Token budget tracker and wall-clock timeout enforce resource limits; agents stop early if budget is exhausted |
| **Graceful Degradation** | If any agent fails, the pipeline falls back automatically (e.g., raw findings instead of composed answer) |

---

## Architecture

```
User Query
    |
    v
+---------------------------+
| Orchestrator              |    Decides: should this query use agents?
| (orchestrator.py)         |    5-gate escalation check
+---------------------------+
    |
    | (only if complex query)
    v
+---------------------------+
| Planner Agent             |    Decomposes query into sub-steps
| (planner.py)              |    Model: Claude Sonnet (via Bedrock)
| Max tokens: 1024          |
+---------------------------+
    |
    | list of steps
    v
+---------------------------+
| Analyst Agent             |    Executes each step against documents
| (analyst.py)              |    Model: Claude Haiku (via Bedrock)
| Max tokens: 1500/step     |    Runs one LLM call per step
+---------------------------+
    |
    | list of findings
    v
+---------------------------+
| Composer Agent            |    Synthesizes findings into final answer
| (composer.py)             |    Model: Claude Haiku (via Bedrock)
| Max tokens: 2048          |
+---------------------------+
    |
    v
Final Answer (returned to /ask endpoint)
```

---

## File-by-File Breakdown

### `__init__.py`
Package entry point. Exposes two public functions:
- `run_agent_pipeline()` — runs the full Planner -> Analyst -> Composer pipeline
- `should_escalate_to_agents()` — decides whether a query is complex enough for agent mode

### `base.py` — Shared Infrastructure
Contains the foundational types and utilities used by all agents:

- **`TokenBudget`** — Tracks cumulative token usage across the entire pipeline run. Enforces a hard ceiling (`AGENT_MAX_TOTAL_TOKENS = 8000`) to control costs. Each agent checks the budget before making an LLM call.
- **`AgentStepResult`** — Data class for a single agent's output (name, output text, model used, tokens consumed, duration, success/error).
- **`AgentPipelineResult`** — Full pipeline result returned to the `/ask` endpoint (final answer, all step results, plan, total tokens/time, internal reasoning summary).
- **`invoke_llm()`** — Unified LLM invoker that routes to the correct model:
  - `"mistral"` -> existing Mistral generate function
  - `"haiku"` -> AWS Bedrock Claude Haiku
  - `"sonnet"` -> AWS Bedrock Claude Sonnet
- **`_invoke_claude()`** — Low-level Bedrock Messages API caller for Claude models.

### `orchestrator.py` — The Decision Maker
The orchestrator serves as the **entry gate** and **pipeline coordinator**.

**Escalation Logic (`should_escalate_to_agents`):**
A query must pass ALL 5 gates to trigger agent mode:

1. **Feature flag** — `ENABLE_AGENT_MODE` must be `True`
2. **Complexity tier** — Must be classified as `"complex"` by Phase 4 classifier
3. **Score threshold** — Complexity score must exceed `0.65`
4. **Pattern match** — Query must match at least one agent-eligible regex pattern:
   - Troubleshooting keywords: *troubleshoot, diagnose, debug, step-by-step*
   - Comparison keywords: *compare, contrast, versus, pros and cons*
   - Multi-document synthesis: *across, all documents, summarize all*
   - Guided remediation: *remediate, fix and verify, resolve then*
   - Root cause analysis: *root cause + recommend, why + how*
   - End-to-end workflows: *end-to-end, complete process, full workflow*
5. **Minimum sources** — At least `AGENT_MIN_SOURCES` (1) documents must be available

**Pipeline Execution (`run_agent_pipeline`):**
Runs Planner -> Analyst -> Composer sequentially with timeout checks between each stage.

### `planner.py` — The Strategist
- **Model:** Claude Sonnet (the most capable model in the pipeline — used for the hardest reasoning task)
- **Purpose:** Takes the user's complex query and a preview of document context, then produces a JSON array of concrete analysis steps (max 4 steps)
- **Grounding:** Steps must be answerable from the documents only — no outside knowledge
- **Fallback:** If parsing fails, produces a single step: "Analyze the documents to answer: {query}"

### `analyst.py` — The Researcher
- **Model:** Claude Haiku (cheaper, used for repeated per-step analysis)
- **Purpose:** Iterates through each step from the Planner and executes it against the full document context
- **Output:** A list of grounded findings (3-6 bullet points per step), with evidence from documents only
- **Cost control:** Checks token budget before each step; stops early if budget is exhausted

### `composer.py` — The Writer
- **Model:** Claude Haiku
- **Purpose:** Takes all findings from the Analyst and synthesizes them into a coherent, well-structured final answer
- **Rules enforced:** Bullet-point format, document-faithful, cites source documents, no repetition
- **Fallback:** If composer fails, concatenates raw findings directly

---

## Cost Control Mechanisms

| Control | Value | Purpose |
|---|---|---|
| `AGENT_MAX_TOTAL_TOKENS` | 8,000 | Hard ceiling across all agents in one pipeline run |
| `AGENT_TIMEOUT_SECONDS` | 45s | Wall-clock timeout for the entire pipeline |
| `AGENT_MAX_STEPS` | 4 | Maximum analysis steps the Planner can produce |
| `AGENT_PLANNER_MAX_TOKENS` | 1,024 | Per-call token cap for Planner |
| `AGENT_ANALYSIS_MAX_TOKENS` | 1,500 | Per-call token cap for each Analyst step |
| `AGENT_COMPOSER_MAX_TOKENS` | 2,048 | Per-call token cap for Composer |
| Model selection | Sonnet for planning, Haiku for analysis/composition | Expensive model only where complex reasoning is needed |

---

## How It Fits Into the Application

```
User query via /ask endpoint
    |
    v
Phase 4: Complexity Classification (simple / moderate / complex)
    |
    +-- simple/moderate --> Standard RAG pipeline (single LLM call)
    |
    +-- complex --> should_escalate_to_agents()
                        |
                        +-- No  --> Standard RAG pipeline
                        +-- Yes --> run_agent_pipeline() --> Multi-agent answer
```

Simple and moderate queries **never** enter the agent pipeline. Only genuinely complex, multi-step queries that match specific patterns are escalated, keeping costs low for the majority of interactions.

---

## Key Design Decisions

1. **Selective activation** — The 5-gate escalation logic ensures agents only fire for genuinely complex queries, not for simple lookups or moderate questions.
2. **Document-grounded** — All agents are instructed to use ONLY the retrieved document context. This prevents hallucination and keeps answers faithful to uploaded materials.
3. **Cost-tiered models** — Sonnet (expensive, capable) is used only for the Planner's complex reasoning. Haiku (cheap, fast) handles the repeated analysis and composition steps.
4. **Budget-aware** — The shared `TokenBudget` prevents runaway costs. Agents check before every call and stop gracefully if the budget is exhausted.
5. **Graceful fallbacks** — Every stage has a fallback: Planner failure -> single step, Analyst failure -> partial findings noted, Composer failure -> raw findings concatenated.
