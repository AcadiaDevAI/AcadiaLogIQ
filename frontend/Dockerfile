# ─── Build stage ──────────────────────────────────
FROM node:18-alpine AS build
WORKDIR /app
COPY package*.json ./
# --legacy-peer-deps preserved: Clerk 5.x + ajv 8.x peer-dep conflict
RUN npm install --legacy-peer-deps && \
    npm install ajv@8.17.1 --legacy-peer-deps
COPY . .

# API_BASE is set at build time; defaults to relative /api proxied by nginx
ARG REACT_APP_API_BASE=http://localhost:8000
ENV REACT_APP_API_BASE=$REACT_APP_API_BASE

# Clerk publishable key (safe to embed in frontend — it's a PUBLIC key)
ARG REACT_APP_CLERK_PUBLISHABLE_KEY=
ENV REACT_APP_CLERK_PUBLISHABLE_KEY=$REACT_APP_CLERK_PUBLISHABLE_KEY

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
