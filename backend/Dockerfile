                                                                                                                                                                                                          
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libpq-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY backend /app/backend

# Entrypoint script: picks api vs worker role from APP_ROLE.
# Owned by root so the unprivileged appuser can't tamper with it.
COPY backend/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/uploads && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Health check still points at the API port. The worker container
# (APP_ROLE=worker) doesn't open a port — ECS will use a separate
# health mechanism (target=stopped → restart) for that service. For
# local docker-compose runs with the default role=api, the curl
# check works as before.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Phase 2 — entrypoint picks gunicorn (api role) or
# ``python -m backend.jobs.worker`` (worker role) based on APP_ROLE.
# Defaults to api so existing deployments keep working unchanged.
ENTRYPOINT ["/app/entrypoint.sh"]