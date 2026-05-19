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

# Accept build timestamp for cache-busting and version stamping
ARG BUILD_TIMESTAMP=unknown
ENV REACT_APP_BUILD_TIMESTAMP=$BUILD_TIMESTAMP
ENV VITE_BUILD_TIMESTAMP=$BUILD_TIMESTAMP

# Sentry — frontend error monitoring. DSN is the only required value;
# when empty the SDK init becomes a no-op so the bundle still ships.
# The DSN is a *public* identifier (designed for browser embedding)
# but rate-limited at Sentry's edge, so embedding it is safe.
ARG REACT_APP_SENTRY_DSN=
ENV REACT_APP_SENTRY_DSN=$REACT_APP_SENTRY_DSN
ARG REACT_APP_SENTRY_ENV=dev
ENV REACT_APP_SENTRY_ENV=$REACT_APP_SENTRY_ENV

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