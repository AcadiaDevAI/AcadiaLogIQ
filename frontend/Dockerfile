# ---- Build Stage ----
FROM node:20-alpine AS build

WORKDIR /app

# Copy manifests first so npm install caches on lockfile changes only
COPY package.json package-lock.json* ./
RUN npm install --legacy-peer-deps && \
    npm install ajv@8.17.1 --legacy-peer-deps

COPY . .

# API_BASE is set at build time; defaults to relative /api proxied by nginx
ARG REACT_APP_API_BASE=http://localhost:8000
ENV REACT_APP_API_BASE=$REACT_APP_API_BASE

# Clerk publishable key (safe to embed in frontend — it's a PUBLIC key)
ARG REACT_APP_CLERK_PUBLISHABLE_KEY=
ENV REACT_APP_CLERK_PUBLISHABLE_KEY=$REACT_APP_CLERK_PUBLISHABLE_KEY

# Cache-busting build timestamp. Changing this arg invalidates every
# downstream layer, forcing `npm run build` to re-run and ensuring a
# fresh bundle hash. Also surfaced into the UI so operators can tell
# at a glance whether a deployment is live.
ARG BUILD_TIMESTAMP=unset
ENV REACT_APP_BUILD_TIMESTAMP=$BUILD_TIMESTAMP
RUN echo "BUILD_TIMESTAMP=$BUILD_TIMESTAMP" && npm run build

# ---- Production Stage ----
FROM nginx:alpine

# Custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built app
COPY --from=build /app/build /usr/share/nginx/html

# Stamp the timestamp into a plain file too, so operators can
# `docker exec acadialogiq-ui cat /usr/share/nginx/html/BUILD_TIMESTAMP`
ARG BUILD_TIMESTAMP=unset
RUN echo "$BUILD_TIMESTAMP" > /usr/share/nginx/html/BUILD_TIMESTAMP

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
