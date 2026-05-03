# Acadia Log IQ — AI-Powered Document Intelligence Platform

> **Version 3.0** | Phases 1–6 Complete | Production-Ready  
> Hybrid RAG system for operational document analysis with multi-model routing,  
> multi-agent troubleshooting, and answer validation guardrails.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Technology Stack](#technology-stack)
4. [Project Structure](#project-structure)
5. [Phase-by-Phase Development](#phase-by-phase-development)
6. [How It Works — End to End](#how-it-works--end-to-end)
7. [AWS Services & Model Setup](#aws-services--model-setup)
8. [Environment Variables](#environment-variables)
9. [Local Development Setup](#local-development-setup)
10. [EC2 Deployment](#ec2-deployment)
11. [Docker Reference](#docker-reference)
12. [Database Setup (RDS + pgvector)](#database-setup-rds--pgvector)
13. [S3 Storage Setup](#s3-storage-setup)
14. [Clerk Authentication](#clerk-authentication)
15. [Document Ingestion Pipeline](#document-ingestion-pipeline)
16. [Retrieval Pipeline](#retrieval-pipeline)
17. [Model Routing & Cost Optimization](#model-routing--cost-optimization)
18. [Multi-Agent Troubleshooting](#multi-agent-troubleshooting)
19. [Answer Validation & Confidence](#answer-validation--confidence)
20. [Duplicate & Version Detection](#duplicate--version-detection)
21. [Health Checks & Observability](#health-checks--observability)
22. [Security & Data Privacy](#security--data-privacy)
23. [API Reference](#api-reference)
24. [Testing](#testing)
25. [Troubleshooting](#troubleshooting)
26. [Rollback & Migration](#rollback--migration)
27. [Known Limitations](#known-limitations)
28. [Future Extensions](#future-extensions)

---

## Overview

Acadia Log IQ is an enterprise document intelligence platform that lets operations teams upload large technical documents (runbooks, SOPs, KBs, vendor manuals) and ask natural-language questions. The system retrieves relevant information using a hybrid search pipeline, routes to the most cost-effective AI model, and returns grounded, validated answers.

### Key Capabilities

- **Upload any document format** — PDF, DOCX, TXT, MD, JSON, LOG (up to 100MB)
- **Smart adaptive chunking** — preserves document structure (scenarios, procedures, sections)
- **Hybrid retrieval** — vector search + BM25 + full-text keyword search + metadata filtering
- **Cost-optimized model routing** — Claude Haiku for most queries, Sonnet for complex ones
- **Multi-agent troubleshooting** — Planner → Analyst → Composer pipeline for complex queries
- **Answer validation** — grounding checks, fabrication detection, confidence scoring
- **Version-aware** — detects duplicate/updated documents, supersedes old versions
- **Session-based chat** — conversation history with feedback (like/dislike)
- **Clerk authentication** — optional enterprise SSO integration

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         React Frontend                          │
│              (Ant Design + Tailwind + react-markdown)           │
└──────────────────────────┬──────────────────────────────────────┘
                           │ REST API (JSON)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                             │
│                                                                 │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ Ingestion│  │ Retrieval │  │ Routing  │  │ Validation   │  │
│  │ Pipeline │  │ Pipeline  │  │ Pipeline │  │ Pipeline     │  │
│  │ (Ph 1-2) │  │ (Ph 3)   │  │ (Ph 4)  │  │ (Ph 6)      │  │
│  └────┬─────┘  └────┬──────┘  └────┬─────┘  └──────┬───────┘  │
│       │              │              │               │          │
│       │         ┌────┴──────┐  ┌───┴────┐   ┌──────┴───────┐  │
│       │         │ Agents   │  │ Models │   │ Confidence   │  │
│       │         │ (Ph 5)   │  │ Haiku  │   │ Scorer +     │  │
│       │         │ Planner  │  │ Sonnet │   │ Grounding    │  │
│       │         │ Analyst  │  │ Mistral│   │ Checker      │  │
│       │         │ Composer │  └────────┘   └──────────────┘  │
│       │         └──────────┘                                  │
└───────┼───────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────┐  ┌───────────────────┐  ┌────────────────┐
│ PostgreSQL + pgv  │  │ Amazon Bedrock    │  │ S3 / Local     │
│ (RDS)             │  │ (LLM + Embed)    │  │ (File Storage) │
│ • documents       │  │ • Titan Embed V2 │  │ • Raw uploads  │
│ • chunks          │  │ • Claude Haiku   │  │                │
│ • embeddings      │  │ • Claude Sonnet  │  │                │
│ • chat_sessions   │  │ • Mistral 7B     │  │                │
│ • document_vers   │  │   (reranking)    │  │                │
└───────────────────┘  └───────────────────┘  └────────────────┘
```

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | React 18, Ant Design 5, Tailwind CSS | Chat UI, file upload, session management |
| **Backend** | FastAPI (Python 3.11+) | REST API, background tasks, middleware |
| **Embeddings** | Amazon Titan Embed Text V2 | 1024-dim vectors for semantic search |
| **Answer Gen (default)** | Claude Haiku 4.5 via Bedrock | Grounded answer generation (~$0.001/query) |
| **Answer Gen (complex)** | Claude Sonnet 4.6 via Bedrock | Multi-step reasoning (~$0.01/query) |
| **Reranking** | Mistral 7B via Bedrock | Cheap chunk relevance scoring |
| **Metadata Extraction** | Claude Haiku via Bedrock | Document metadata during ingestion |
| **Database** | PostgreSQL 15 + pgvector | Vectors, chunks, documents, sessions |
| **File Storage** | Local filesystem / S3 | Raw uploaded files |
| **Auth** | Clerk (optional) | JWT-based SSO, user management |
| **Deployment** | Docker, Docker Compose | Containerized local + EC2 |

---

## Project Structure

```
acadia-log-iq/
├── backend/
│   ├── api.py                              # FastAPI app, all endpoints, /ask pipeline
│   ├── config.py                           # Pydantic settings (Phases 2-6)
│   ├── vector_store.py                     # PostgreSQL/pgvector, BM25, CRUD
│   ├── db/
│   │   └── connection.py                   # SQLAlchemy session factory
│   ├── ingestion/
│   │   ├── structured_parser.py            # 3-strategy adaptive document parser
│   │   └── prompt_templates.py             # Haiku metadata extraction prompts
│   ├── metadata/
│   │   └── structure_config.py             # Operational section taxonomy
│   ├── services/
│   │   ├── bedrock_haiku.py                # Claude Haiku client (JSON invoke)
│   │   └── contextual_ingestion_service.py # Chunking + metadata + version detection
│   ├── retrieval/                          # Phase 3: Hybrid retrieval
│   │   ├── __init__.py
│   │   ├── orchestrator.py                 # Main retrieval coordinator
│   │   ├── query_classifier.py             # Query intent (keyword/semantic/mixed)
│   │   ├── keyword_search.py               # PostgreSQL FTS + ILIKE
│   │   ├── fusion.py                       # Weighted RRF result merging
│   │   └── reranker.py                     # Modular reranker (LLM/CrossEncoder/None)
│   ├── routing/                            # Phase 4: Model routing
│   │   ├── __init__.py
│   │   ├── model_router.py                 # Haiku/Sonnet selection + invocation
│   │   ├── complexity_classifier.py        # Query difficulty scoring
│   │   └── context_builder.py              # Enriched prompt assembly
│   ├── agents/                             # Phase 5: Multi-agent troubleshooting
│   │   ├── __init__.py
│   │   ├── orchestrator.py                 # Escalation gate + pipeline
│   │   ├── base.py                         # Token budget, LLM invoker
│   │   ├── planner.py                      # Query decomposition (Sonnet)
│   │   ├── analyst.py                      # Per-step analysis (Haiku)
│   │   └── composer.py                     # Answer synthesis (Haiku)
│   ├── validation/                         # Phase 6: Answer validation
│   │   ├── __init__.py
│   │   ├── validator.py                    # Main validation pipeline
│   │   ├── confidence_scorer.py            # 4-signal confidence blend
│   │   ├── grounding_checker.py            # Fabrication + version checks
│   │   └── eval_harness.py                 # Offline evaluation utilities
│   ├── storage/
│   │   └── local_storage.py                # File storage provider
│   ├── clerk_auth.py                       # Clerk JWT verification
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── components/                     # React components
│   │   ├── hooks/                          # ChatContext, useAuthInterceptor
│   │   ├── services/api.js                 # Axios client
│   │   ├── App.js
│   │   ├── index.js
│   │   └── index.css                       # Tailwind + theme styles
│   ├── package.json
│   ├── nginx.conf
│   └── Dockerfile
├── tests/
│   ├── test_retrieval.py                   # Phase 3 tests
│   ├── test_routing.py                     # Phase 4 tests
│   ├── test_agents.py                      # Phase 5 tests
│   └── test_validation.py                  # Phase 6 tests
├── docker-compose.yml                      # Local development
├── docker-compose.ec2.yml                  # EC2 deployment
└── README.md                               # This file
```

---

## Phase-by-Phase Development

### Phase 1 — Foundation
Basic FastAPI backend, file upload, naive character-based chunking, Titan embeddings, pgvector storage, Mistral 7B for answer generation, React chat UI.

### Phase 2 — Contextual Ingestion
Structured document parser (DOCX/PDF/TXT), heading-aware chunking, Claude Haiku metadata extraction (vendor, product, domain, tags), duplicate/version detection with SHA-256 fingerprinting, concurrent embedding with ThreadPoolExecutor.

### Phase 3 — Hybrid Retrieval
Four-channel search: pgvector cosine similarity + in-memory BM25 + PostgreSQL full-text search (ts_vector) + metadata JSONB filtering. Query intent classifier (keyword/semantic/mixed). Reciprocal Rank Fusion with strategy-aware weight adjustment. Modular reranker (LLM-based by default).

### Phase 4 — Model Routing
Complexity classifier scores queries 0.0-1.0 using 5 signals (multi-step patterns, reasoning patterns, context size, retrieval confidence, multi-document span). Routes to Claude Haiku (default for simple+moderate) or Sonnet (complex). Context builder enriches prompts with session history, metadata hints, and confidence signals.

### Phase 5 — Multi-Agent Troubleshooting
Selective escalation for complex queries matching agent-eligible patterns (troubleshooting, comparison, synthesis). 3-agent pipeline: Planner (Sonnet) decomposes query → Analyst (Haiku) executes per-step → Composer (Haiku) synthesizes. Shared token budget with hard ceiling. Only ~10-15% of queries trigger agents.

### Phase 6 — Validation Guardrails
Post-generation validation: 4-signal confidence scoring (retrieval strength, query coverage, grounding faithfulness, consistency). Grounding checker detects fabricated URLs/emails/phones, flags superseded sources. Failed answers replaced with safe fallbacks. JSONL evaluation logging for offline quality analysis.

### Accuracy Fix — Adaptive Chunking
Root cause of 10% initial accuracy: documents with all-Normal Word styles produced franken-chunks mixing unrelated scenarios. Fix: 3-strategy adaptive parser: (1) Word heading styles, (2) content-based regex patterns, (3) LLM-based section discovery via Haiku for truly unstructured documents. Combined with switching default generation from Mistral 7B to Claude Haiku.

---

## How It Works — End to End

### Document Upload Flow
```
User uploads DOCX/PDF
    │
    ▼
1. File saved to local storage / S3
2. SHA-256 fingerprint computed
3. Duplicate check against existing documents
4. Structured parser runs (3-strategy heading detection)
5. Scenario-aware chunking (each section = 1 chunk)
6. Claude Haiku extracts metadata per chunk (concurrent, batched)
7. Titan Embed V2 generates embeddings (concurrent)
8. Chunks + embeddings + metadata inserted into PostgreSQL
9. BM25 index updated in-memory
10. Version detection: new_document / new_version / exact_duplicate
```

### Question Answering Flow
```
User asks question
    │
    ▼
1. Embed query with Titan
2. Phase 3: 4-channel parallel search → RRF fusion → rerank
3. Grounding gate: reject if insufficient document support
4. Phase 4: Classify complexity → select model (Haiku/Sonnet)
5. Phase 5: Escalate to agents if complex + pattern match
6. Generate answer from retrieved document context
7. Phase 6: Validate grounding, check fabrications, score confidence
8. Return answer with sources, confidence, and diagnostics
```

---

## AWS Services & Model Setup

### Required AWS Services

| Service | Purpose | Required |
|---------|---------|----------|
| **Amazon Bedrock** | LLM inference (Haiku, Sonnet, Mistral, Titan Embed) | Yes |
| **Amazon RDS** | PostgreSQL 15 + pgvector for vectors/chunks | Yes (or self-hosted PG) |
| **Amazon S3** | Raw file storage (optional, local storage works) | Optional |
| **Amazon EC2** | Application hosting | Yes (for cloud deploy) |
| **Amazon SES** | Feedback email notifications | Optional |

### Bedrock Model Access

Enable these models in the AWS Bedrock console (region: us-east-1):

| Model | Bedrock ID | Purpose | Cost |
|-------|-----------|---------|------|
| Titan Embed V2 | `amazon.titan-embed-text-v2:0` | Embeddings | ~$0.0001/query |
| Claude Haiku 4.5 | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Default answers + metadata | ~$0.001/query |
| Claude Sonnet 4.6 | `us.anthropic.claude-sonnet-4-6` | Complex reasoning | ~$0.01/query |
| Mistral 7B | `mistral.mistral-7b-instruct-v0:2` | Reranking only | ~$0.0001/query |

### IAM Policy (Least Privilege)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "BedrockInvoke",
      "Effect": "Allow",
      "Action": "bedrock:InvokeModel",
      "Resource": [
        "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2:0",
        "arn:aws:bedrock:us-east-1::foundation-model/mistral.mistral-7b-instruct-v0:2",
        "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-haiku-20241022-v1:0",
        "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-sonnet-20241022-v2:0"
      ]
    },
    {
      "Sid": "S3Storage",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
      "Resource": "arn:aws:s3:::your-bucket-name/*"
    },
    {
      "Sid": "SESEmail",
      "Effect": "Allow",
      "Action": "ses:SendEmail",
      "Resource": "*",
      "Condition": {
        "StringEquals": { "ses:FromAddress": "noreply@yourdomain.com" }
      }
    }
  ]
}
```

---

## Environment Variables

### Backend (`backend/.env`)

```env
# ============================================================
# AWS / Bedrock
# ============================================================
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=                    # Leave blank on EC2 (use IAM Role)
AWS_SECRET_ACCESS_KEY=                # Leave blank on EC2 (use IAM Role)
AWS_SESSION_TOKEN=

# Bedrock Models
BEDROCK_EMBED_MODEL=amazon.titan-embed-text-v2:0
BEDROCK_LLM_MODEL=mistral.mistral-7b-instruct-v0:2
BEDROCK_HAIKU_MODEL=us.anthropic.claude-haiku-4-5-20251001-v1:0
BEDROCK_SONNET_MODEL=us.anthropic.claude-sonnet-4-6

# ============================================================
# Database (PostgreSQL + pgvector)
# ============================================================
DATABASE_URL=postgresql+psycopg://user:password@host:5432/dbname

# ============================================================
# Storage
# ============================================================
UPLOAD_DIR=/app/uploads
# STORAGE_TYPE=s3                     # Uncomment for S3
# S3_BUCKET=your-bucket-name          # Required if STORAGE_TYPE=s3

# ============================================================
# Chunking & Ingestion
# ============================================================
CHUNK_MAX_CHARS=6000
CHUNK_MIN_CHARS=200
CHUNK_BATCH_SIZE=6
ENABLE_LLM_CHUNK_FALLBACK=true
ENABLE_METADATA_EXTRACTION=true

# ============================================================
# Model Routing
# ============================================================
ENABLE_MODEL_ROUTING=true
ROUTING_DEFAULT_MODEL=haiku

# ============================================================
# Agents
# ============================================================
ENABLE_AGENT_MODE=true
AGENT_COMPLEXITY_THRESHOLD=0.65

# ============================================================
# Validation
# ============================================================
ENABLE_ANSWER_VALIDATION=true
ENABLE_EVAL_LOGGING=true

# ============================================================
# Authentication (Clerk — optional)
# ============================================================
CLERK_ENABLED=false
CLERK_PUBLISHABLE_KEY=
CLERK_SECRET_KEY=                     # NEVER commit this

# ============================================================
# API
# ============================================================
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
API_KEY=                              # Optional API key auth

# ============================================================
# Email (SES — optional)
# ============================================================
SES_ENABLED=false
SES_SENDER_EMAIL=noreply@yourdomain.com
SES_FEEDBACK_RECIPIENT=team@yourdomain.com
```

### Frontend (`frontend/.env`)

```env
REACT_APP_API_BASE=http://localhost:8000
REACT_APP_CLERK_PUBLISHABLE_KEY=      # Leave blank to disable auth
```

---

## Local Development Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- PostgreSQL 15 with pgvector extension
- AWS credentials with Bedrock access

### Step 1: Database
```bash
# Install pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

# Run migrations (or let the app create tables on first start)
```

### Step 2: Backend
```bash
cd backend
cp .env.example .env
# Edit .env: set DATABASE_URL, AWS credentials

pip install -r requirements.txt
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
```

### Step 3: Frontend
```bash
cd frontend
cp .env.example .env
# Edit .env: set REACT_APP_API_BASE=http://localhost:8000

npm install
npm start
```

Open http://localhost:3000

---

## EC2 Deployment

### Instance Requirements
- **AMI:** Ubuntu 22.04 LTS
- **Type:** t3.medium minimum (2 vCPU, 4 GB RAM)
- **Storage:** 30+ GB
- **Security Group:** ports 22, 8000, 8501

### Step 1: Install Docker
```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin git
sudo usermod -aG docker $USER
newgrp docker
```

### Step 2: Clone and Configure
```bash
git clone https://github.com/your-org/acadia-log-iq.git
cd acadia-log-iq

cd backend && cp .env.example .env
nano .env  # Set DATABASE_URL, AWS_REGION, model IDs
cd ..
```

### Step 3: Build and Deploy
```bash
export REACT_APP_CLERK_PUBLISHABLE_KEY=pk_test_xxx  # or leave blank

docker compose -f docker-compose.ec2.yml up --build -d
docker compose -f docker-compose.ec2.yml logs -f  # Watch logs
```

### Step 4: Verify
```bash
curl http://localhost:8000/health
```

### Step 5: Reset Data (After Code Updates)
If you updated the parser/chunking logic, re-index documents:
```bash
# Option A: Delete via UI and re-upload

# Option B: Database reset
docker compose -f docker-compose.ec2.yml exec -T api python -c "
from backend.db.connection import SessionLocal
from sqlalchemy import text
with SessionLocal() as db:
    db.execute(text('UPDATE documents SET current_version_id = NULL'))
    db.execute(text('DELETE FROM chat_messages'))
    db.execute(text('DELETE FROM chat_sessions'))
    db.execute(text('DELETE FROM embeddings'))
    db.execute(text('DELETE FROM chunks'))
    db.execute(text('DELETE FROM document_metadata'))
    db.execute(text('DELETE FROM ingestion_jobs'))
    db.execute(text('DELETE FROM document_versions'))
    db.execute(text('DELETE FROM documents'))
    db.commit()
    print('Reset complete')
"
```

---

## Docker Reference

### Backend Dockerfile
```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential curl libpq-dev poppler-utils
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY backend /app/backend
RUN useradd -m -u 1000 appuser && mkdir -p /app/uploads && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Frontend Dockerfile
```dockerfile
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json ./
RUN npm install --legacy-peer-deps
COPY . .
ARG REACT_APP_API_BASE=http://localhost:8000
ARG REACT_APP_CLERK_PUBLISHABLE_KEY=
RUN npm run build
FROM nginx:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

---

## Database Setup (RDS + pgvector)

### RDS Configuration
- **Engine:** PostgreSQL 15+
- **Instance:** db.t3.micro (dev) / db.r6g.large (prod)
- **Storage:** 20 GB gp3 (auto-scaling)
- **pgvector:** Enable via `CREATE EXTENSION vector;`

### Tables
| Table | Purpose |
|-------|---------|
| `documents` | Logical documents (name, owner, status, version family) |
| `document_versions` | Version history with fingerprints |
| `document_metadata` | Extracted metadata (vendor, product, domain) |
| `chunks` | Text chunks with section headings, chunk types |
| `embeddings` | 1024-dim vectors (pgvector) |
| `ingestion_jobs` | Upload processing status tracking |
| `chat_sessions` | Conversation sessions |
| `chat_messages` | Individual messages with feedback |

---

## S3 Storage Setup

For production, configure S3 as the primary file store:

```env
STORAGE_TYPE=s3
S3_BUCKET=acadia-logiq-uploads
AWS_REGION=us-east-1
```

S3 bucket policy:
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"AWS": "arn:aws:iam::ACCOUNT:role/ec2-role"},
    "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
    "Resource": "arn:aws:s3:::acadia-logiq-uploads/*"
  }]
}
```

---

## Document Ingestion Pipeline

### 3-Strategy Adaptive Parser

```
Document uploaded
    │
    ▼
Strategy 1: Check Word heading styles (Heading 1/2/3)
    │ Found headings? → Use them ✓
    │ No headings found? ↓
    ▼
Strategy 2: Content-based pattern detection
    │ Regex: "Scenario A:", "Chapter 1:", "Step 1:", etc.
    │ Found patterns? → Use them ✓
    │ No patterns found? ↓
    ▼
Strategy 3: LLM section discovery (Haiku, ~$0.002)
    │ Send first 8000 chars → Haiku identifies sections
    │ Re-tag blocks → Use LLM headings ✓
    │ Nothing found? → Fall back to character-based chunking
```

### Supported Document Formats
| Format | Parser | Heading Detection |
|--------|--------|-------------------|
| DOCX | python-docx | Word styles + content patterns + LLM |
| PDF | PyMuPDF (fitz) | Content patterns + LLM |
| TXT/MD | Line-based | Markdown headings + content patterns + LLM |
| JSON/LOG | Line-based | Content patterns + LLM |

---

## Retrieval Pipeline

### 4-Channel Hybrid Search
```
Query → Embed with Titan
    │
    ├─→ Channel 1: pgvector cosine similarity (25 candidates)
    ├─→ Channel 2: BM25 term frequency (20 candidates)
    ├─→ Channel 3: PostgreSQL FTS + ILIKE fallback (15 candidates)
    └─→ Channel 4: Metadata JSONB filter (10 candidates)
         │
         ▼
    RRF Fusion (strategy-aware weights)
         │
         ▼
    Reranker (Mistral 7B scores 0-10)
         │
         ▼
    Top 6 chunks → context assembly
```

### Weight Adjustment by Query Strategy
| Strategy | Vector | BM25 | Keyword |
|----------|--------|------|---------|
| Semantic ("how to troubleshoot...") | 0.55 | 0.25 | 0.20 |
| Keyword ("ORA-00942 error") | 0.25 | 0.35 | 0.40 |
| Mixed (default) | 0.45 | 0.30 | 0.25 |

---

## Model Routing & Cost Optimization

### Routing Policy
| Complexity Tier | Score Range | Model | Cost/Query |
|----------------|-------------|-------|-----------|
| Simple | 0.0 – 0.30 | Claude Haiku | ~$0.001 |
| Moderate | 0.30 – 0.70 | Claude Haiku | ~$0.001 |
| Complex | 0.70 – 1.0 | Claude Sonnet | ~$0.01 |

### Complexity Signals
| Signal | Weight | What It Measures |
|--------|--------|-----------------|
| Multi-step | 0.30 | Comparison, workflow, step-by-step queries |
| Reasoning | 0.25 | Why/analyze/recommend/root-cause queries |
| Context size | 0.15 | Large context harder to reason over |
| Low confidence | 0.20 | Low retrieval confidence = risky |
| Multi-document | 0.10 | Answer spans multiple sources |

### Estimated Monthly Cost (1000 queries/day)
| Component | Queries | Model | Monthly Cost |
|-----------|---------|-------|-------------|
| Embeddings | 30K | Titan | ~$3 |
| Answers (85% simple) | 25.5K | Haiku | ~$25 |
| Answers (15% complex) | 4.5K | Sonnet | ~$45 |
| Reranking | 30K | Mistral | ~$3 |
| **Total** | | | **~$76/month** |

---

## Multi-Agent Troubleshooting

### Escalation Gate (ALL conditions must be true)
1. `ENABLE_AGENT_MODE = true`
2. Complexity tier = "complex"
3. Score > 0.65
4. Query matches pattern (troubleshoot, compare, synthesize, remediate)
5. At least 1 source document available

### Agent Pipeline
```
Planner (Sonnet, ~1K tokens)
    → "Check interface status", "Verify routing", "Test connectivity"
         │
Analyst (Haiku × N steps, ~1.5K tokens each)
    → Per-step findings from document context
         │
Composer (Haiku, ~2K tokens)
    → Coherent bullet-point answer
```

### Cost Controls
- Token budget: 8000 total across all agents
- Timeout: 45 seconds wall-clock
- Max steps: 4 from planner
- Fallback: raw findings if composer fails

---

## Answer Validation & Confidence

### Confidence Scoring (4 signals)
| Signal | Weight | Measures |
|--------|--------|---------|
| Retrieval | 0.30 | Top reranked chunk score |
| Coverage | 0.25 | Query terms found in context |
| Grounding | 0.25 | Answer terms found in context |
| Consistency | 0.20 | Non-empty, no hallucination phrases |

### Grounding Checks
- **Fabricated URLs** — URLs in answer not in source documents
- **Fabricated emails** — email addresses not in sources
- **Fabricated phones** — phone numbers not in sources
- **Sentence grounding** — <40% term overlap → flagged
- **Version awareness** — superseded sources → warning appended

### Decision Matrix
| Condition | Action |
|-----------|--------|
| Confidence ≥ 0.35 AND grounding pass | Return original answer |
| Superseded sources detected | Append version warning |
| Fabricated specifics found | Replace with safe fallback |
| Confidence < 0.35 | Replace with insufficient-evidence fallback |

---

## Duplicate & Version Detection

```
New document uploaded
    │
    ▼
SHA-256 fingerprint matches existing? → exact_duplicate (skip)
    │ No match
    ▼
Normalized filename + title similarity > threshold?
    │ Yes → new_version (supersede old, index new)
    │ No  → new_document (index fresh)
```

- **Active documents** are searched by default
- **Superseded documents** are excluded from retrieval
- **Version families** track related document lineage
- Configure via `INCLUDE_OLD_VERSIONS=true` to search all versions

---

## Health Checks & Observability

### Endpoints
| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Service health, model status, chunk count |
| `GET /upload_status/{job_id}` | Ingestion progress tracking |

### Structured Logging
All logs use the `acadia-log-iq` logger with structured format:
```
2026-03-16 19:43:54 - acadia-log-iq - INFO - Retrieval complete: strategy=mixed vector=25 bm25=20 kw=15 → fused=15 → reranked=6 (2648ms)
2026-03-16 19:43:54 - acadia-log-iq - INFO - Model routing: query='...' → model=haiku | simple query (score=0.092)
2026-03-16 19:43:57 - acadia-log-iq - INFO - Generation complete: model=haiku, 3129ms, 12563 prompt chars
2026-03-16 19:43:57 - acadia-log-iq - INFO - Confidence: 0.652 (ret=0.80 cov=0.71 gnd=0.58 con=1.00) → PASS
```

### Evaluation Logging
Every request logs a JSONL eval record (for offline quality analysis):
```
EVAL_RECORD: {"query":"...", "confidence":0.652, "passed":true, "model_used":"haiku", ...}
```

### Response Diagnostics
The `/ask` response includes `context_stats` with:
- Retrieval: strategy, per-channel candidate counts, timing
- Routing: model used, complexity score/tier, generation time
- Agents: mode, steps, tokens, timing
- Validation: passed, confidence, issues count, version warnings

---

## Security & Data Privacy

### Secret Management
- **Never commit `.env` files** — use `.env.example` templates
- **Clerk secret key** — backend only, never in frontend
- **AWS credentials** — use IAM Roles on EC2, never hardcode
- **Database password** — set via environment variable

### Data Privacy
- **No PII in eval logs** — only query (truncated), confidence scores, source names
- **Document content stays in PostgreSQL** — never sent to external services except Bedrock
- **Bedrock data policy** — AWS does not use your data for model training
- **Session data** — stored in PostgreSQL, deletable via API
- **File uploads** — stored locally or in S3 (your control)

### Code Protection
- **`.dockerignore`** — excludes `.env`, `node_modules`, `.git`
- **Non-root Docker user** — `appuser` with UID 1000
- **CORS whitelist** — only allowed origins can call the API
- **Rate limiting** — 30 requests/minute on `/ask`, 100/minute on `/upload`

---

## API Reference

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/health` | No | Health check |
| GET | `/me` | Yes | Current user info |
| POST | `/upload?file_type=kb` | Yes | Upload document |
| GET | `/upload_status/{job_id}` | Yes | Ingestion progress |
| POST | `/ask` | Yes | Ask question |
| GET | `/files` | Yes | List documents |
| DELETE | `/files/{file_id}` | Yes | Delete document |
| GET | `/chat/sessions` | Yes | List sessions |
| GET | `/chat/sessions/{id}` | Yes | Get session |
| DELETE | `/chat/sessions/{id}` | Yes | Delete session |
| DELETE | `/chat/sessions` | Yes | Clear all sessions |
| POST | `/reset` | Yes | Full data reset |
| POST | `/feedback/submit` | Yes | Submit feedback |
| POST | `/feedback/state` | Yes | Save like/dislike |

---

## Testing

```bash
# All tests (no database or AWS needed)
pytest tests/ -v

# Specific phases
pytest tests/test_retrieval.py -v     # Phase 3: query classifier, fusion
pytest tests/test_routing.py -v       # Phase 4: complexity, routing policy
pytest tests/test_agents.py -v        # Phase 5: escalation gate, pipeline
pytest tests/test_validation.py -v    # Phase 6: confidence, grounding

# Evaluation harness (offline quality benchmarking)
python -c "
from backend.validation.eval_harness import run_eval_suite, EVAL_SUITE
results = run_eval_suite(EVAL_SUITE)
correct = sum(1 for r in results if r.correct)
print(f'{correct}/{len(results)} eval cases passed')
"
```

---

## Troubleshooting

### "Settings object has no attribute X"
Your `config.py` is missing Phase 6 settings. Replace with the latest version and clear `__pycache__`:
```powershell
Get-ChildItem -Directory -Filter "__pycache__" -Recurse | Remove-Item -Recurse -Force
```

### "Python-dotenv could not parse statement at line N"
Your `.env` file has a bare comment without `#` prefix. Check the offending line and add `#`.

### "Foreign key violation on /reset"
The `reset_pg_data()` function needs to null out `current_version_id` before deleting `document_versions`. Run the SQL reset manually (see EC2 Deployment section).

### "Haiku returned invalid JSON" warnings during ingestion
Normal — Haiku sometimes hits `max_tokens` on large chunks. The retry logic handles this. If it persists across ALL chunks, increase `HAIKU_MAX_TOKENS` in config.

### Low accuracy / wrong answers
1. Did you re-index documents after updating the parser? Delete and re-upload.
2. Check that `config.py` has `CHUNK_MAX_CHARS=6000` and `ROUTING_DEFAULT_MODEL=haiku`
3. Verify `structured_parser.py` has the 3-strategy heading detection

### EC2 deployment doesn't reflect code changes
Docker caches layers. Force a fresh build:
```bash
docker compose -f docker-compose.ec2.yml up --build --force-recreate -d
```

---

## Rollback & Migration

### Rolling Back to Phase N
Each phase is additive. To disable later phases:
- **Disable agents:** `ENABLE_AGENT_MODE=false`
- **Disable validation:** `ENABLE_ANSWER_VALIDATION=false`
- **Disable model routing:** `ENABLE_MODEL_ROUTING=false` (uses `ROUTING_DEFAULT_MODEL`)
- **Disable LLM chunking fallback:** `ENABLE_LLM_CHUNK_FALLBACK=false`

### Database Migrations
No formal migration tool (Alembic is in requirements but not wired). Schema changes are handled by the application on first run. For manual schema updates, connect to PostgreSQL directly.

### Data Integrity
- Always `UPDATE documents SET current_version_id = NULL` before deleting `document_versions`
- Delete tables in order: embeddings → chunks → document_metadata → ingestion_jobs → document_versions → documents

---

## Known Limitations

1. **Embedding model** — Titan Embed V2 is general-purpose; telecom-specific terms may have weak embeddings. Keyword search compensates but a domain-specific model would improve further.
2. **No streaming** — answers are returned complete, not streamed token-by-token.
3. **Single-user upload** — concurrent uploads from the same user may conflict.
4. **BM25 in-memory** — rebuilt from PostgreSQL at startup; large datasets (100K+ chunks) may slow startup.
5. **No formal migration tool** — schema changes require manual SQL.
6. **Reranker uses Mistral** — LLM-based reranking is slow (~2-5s). Cross-encoder would be faster.

---

## Future Extensions

1. **Streaming responses** — SSE/WebSocket for token-by-token answer delivery
2. **Cohere Embed V3** — domain-aware embeddings for better telecom/networking retrieval
3. **Cross-encoder reranker** — local model, faster than LLM reranking
4. **Multi-tenant** — per-organization document isolation
5. **Alembic migrations** — formal schema versioning
6. **Prometheus metrics** — request latency, model usage, token consumption dashboards
7. **S3 storage provider** — full S3 integration for file uploads (currently local)
8. **Batch upload** — ZIP file containing multiple documents
9. **Document viewer** — preview uploaded documents in the UI
10. **Admin dashboard** — usage analytics, cost tracking, quality metrics


How to run: backend

 python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload 2>&1 | Tee-Object logs\backend.log  

 Forntend:
 Cd frontend

 npm install

 npm run build

 nmp start