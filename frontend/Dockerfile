# ─── Build stage ──────────────────────────────────
FROM node:18-alpine AS build
WORKDIR /app
COPY package*.json ./
# --legacy-peer-deps preserved: Clerk 5.x + ajv 8.x peer-dep conflict
RUN npm install --legacy-peer-deps && \
    npm install ajv@8.17.1 --legacy-peer-deps
COPY . .

# ─── Build-time env vars (React bakes these into the JS bundle) ─────
# Any REACT_APP_* the UI code reads MUST be declared as ARG here,
# otherwise docker build drops them even if set in the shell.

# API_BASE is set at build time; defaults to relative /api proxied by nginx
ARG REACT_APP_API_BASE=http://localhost:8000
ENV REACT_APP_API_BASE=$REACT_APP_API_BASE

# Clerk publishable key (safe to embed in frontend — it's a PUBLIC key)
ARG REACT_APP_CLERK_PUBLISHABLE_KEY=
ENV REACT_APP_CLERK_PUBLISHABLE_KEY=$REACT_APP_CLERK_PUBLISHABLE_KEY

# Optional API key (empty unless deployment uses it)
ARG REACT_APP_API_KEY=
ENV REACT_APP_API_KEY=$REACT_APP_API_KEY

# Sprint 1 — Guided Workflow landing page + mode state
ARG REACT_APP_GUIDED_WORKFLOW_ENABLED=true
ENV REACT_APP_GUIDED_WORKFLOW_ENABLED=$REACT_APP_GUIDED_WORKFLOW_ENABLED

# Sprint 2 — context-break detection + pattern cards
ARG REACT_APP_LOGIQ_SPRINT2_FRONTEND=true
ENV REACT_APP_LOGIQ_SPRINT2_FRONTEND=$REACT_APP_LOGIQ_SPRINT2_FRONTEND

# Sprint 3A — mode-aware prompts + post-👍 action chips
ARG REACT_APP_LOGIQ_SPRINT3A_FRONTEND=false
ENV REACT_APP_LOGIQ_SPRINT3A_FRONTEND=$REACT_APP_LOGIQ_SPRINT3A_FRONTEND

# Sprint 3B — 👎 KB pivot + confidence banner (backend-driven UI)
ARG REACT_APP_LOGIQ_SPRINT3B_FRONTEND=false
ENV REACT_APP_LOGIQ_SPRINT3B_FRONTEND=$REACT_APP_LOGIQ_SPRINT3B_FRONTEND

# Sprint 3C — Escalation mode contact card chips
ARG REACT_APP_LOGIQ_SPRINT3C_FRONTEND=false
ENV REACT_APP_LOGIQ_SPRINT3C_FRONTEND=$REACT_APP_LOGIQ_SPRINT3C_FRONTEND

# Sprint 3D — Ticket Handling sub-mode chips
ARG REACT_APP_LOGIQ_SPRINT3D_FRONTEND=false
ENV REACT_APP_LOGIQ_SPRINT3D_FRONTEND=$REACT_APP_LOGIQ_SPRINT3D_FRONTEND

# Sprint 3E — Vendor/OEM mode chips
ARG REACT_APP_LOGIQ_SPRINT3E_FRONTEND=false
ENV REACT_APP_LOGIQ_SPRINT3E_FRONTEND=$REACT_APP_LOGIQ_SPRINT3E_FRONTEND

# Sprint 3-PREP-B — bulk ingestion doc_kind dropdown + sidebar badge
ARG REACT_APP_LOGIQ_BULK_INGEST_FRONTEND=false
ENV REACT_APP_LOGIQ_BULK_INGEST_FRONTEND=$REACT_APP_LOGIQ_BULK_INGEST_FRONTEND

# Sprint 4 — Fingerprint-First Expert Copilot landing screen
ARG REACT_APP_LOGIQ_SPRINT4_FRONTEND=false 
ENV REACT_APP_LOGIQ_SPRINT4_FRONTEND=$REACT_APP_LOGIQ_SPRINT4_FRONTEND

# Sprint 6 — Tier-1 Alert Copilot (form-driven intake module)
ARG REACT_APP_LOGIQ_TIER1_COPILOT_FRONTEND=false
ENV REACT_APP_LOGIQ_TIER1_COPILOT_FRONTEND=$REACT_APP_LOGIQ_TIER1_COPILOT_FRONTEND

# Sprint 7 — Tier-1 Progressive Workflow (deeper diagnostics, escalation package, stuck detection)
ARG REACT_APP_LOGIQ_TIER1_PROGRESSIVE_FRONTEND=false
ENV REACT_APP_LOGIQ_TIER1_PROGRESSIVE_FRONTEND=$REACT_APP_LOGIQ_TIER1_PROGRESSIVE_FRONTEND

# Sprint 8 Track A — UX fixes (confidence labels, real arrow pagination, skeleton loaders, intake redesign)
ARG REACT_APP_LOGIQ_TIER1_UX_FIXES_FRONTEND=false
ENV REACT_APP_LOGIQ_TIER1_UX_FIXES_FRONTEND=$REACT_APP_LOGIQ_TIER1_UX_FIXES_FRONTEND

# Sprint 8 Track B — Modern Acadia-navy theme (toggleable via in-app button)
ARG REACT_APP_LOGIQ_TIER1_MODERN_THEME=false
ENV REACT_APP_LOGIQ_TIER1_MODERN_THEME=$REACT_APP_LOGIQ_TIER1_MODERN_THEME

# Sprint 8.1 — Temporary download buttons for demo testing (REMOVE AFTER DEMO)
ARG REACT_APP_LOGIQ_TIER1_DOWNLOAD_DEMO=false
ENV REACT_APP_LOGIQ_TIER1_DOWNLOAD_DEMO=$REACT_APP_LOGIQ_TIER1_DOWNLOAD_DEMO

# Sprint 9 — Universal Intake (single unified intake form for all issue types, replacing the guided workflow)
ARG REACT_APP_LOGIQ_UNIVERSAL_INTAKE_FRONTEND=false
ENV REACT_APP_LOGIQ_UNIVERSAL_INTAKE_FRONTEND=$REACT_APP_LOGIQ_UNIVERSAL_INTAKE_FRONTEND


# Accept build timestamp for cache-busting and version stamping
ARG BUILD_TIMESTAMP=unknown
ENV REACT_APP_BUILD_TIMESTAMP=$BUILD_TIMESTAMP
ENV VITE_BUILD_TIMESTAMP=$BUILD_TIMESTAMP

# Inject timestamp into index.html as meta tag (replaces placeholder)
RUN sed -i "s|<!--BUILD_TIMESTAMP-->|<meta name=\"build-timestamp\" content=\"$BUILD_TIMESTAMP\">|g" public/index.html || true

RUN npm run build

# ─── Serving stage (Nginx) ────────────────────────
FROM nginx:alpine

# Copy custom nginx config (with cache-control headers)
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built React app to Nginx serving directory
# IMPORTANT: This project uses Create React App (CRA), so build output is /app/build
# (Vite would output to /app/dist instead — adjust the COPY path if migrating to Vite)
COPY --from=build /app/build /usr/share/nginx/html

# Write timestamp file inside container for verification
ARG BUILD_TIMESTAMP=unknown
RUN echo "$BUILD_TIMESTAMP" > /usr/share/nginx/html/BUILD_TIMESTAMP

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]