#!/usr/bin/env bash
# Backend container entrypoint.
#
# The same image runs two roles based on ``$APP_ROLE``:
#
#   APP_ROLE=api     → gunicorn + uvicorn workers (HTTP server)
#   APP_ROLE=worker  → python -m backend.jobs.worker (queue consumer)
#
# Defaults to ``api`` so the existing docker-compose deployment keeps
# working unchanged. Production ECS task definitions set the env var
# explicitly on each service.
#
# Why one image (instead of two Dockerfiles):
#   * Identical code path — workers import the same modules the API
#     does. Two builds means two image hashes to track for every
#     deploy.
#   * Faster CI — one ``docker build`` per release.
#   * Trivial debugging — `docker exec -it … bash` lands in the same
#     filesystem on both roles.
#
# Worker counts are env-driven so dev / staging / prod can each pick
# a sensible value without rebuilding. See README and Phase 0.4 for
# the RDS connection-budget math.

set -euo pipefail

# Resolve the worker count. ``GUNICORN_WORKERS`` overrides the
# default. When unset we fall back to a conservative ``2`` — the
# real production value comes from the ECS task definition, but if
# someone runs the image bare-metal we don't want it grinding under
# (2*CPU)+1 workers that exhaust the RDS connection pool.
WORKERS="${GUNICORN_WORKERS:-2}"

# Gunicorn timeout — must be LONGER than the slowest in-band call
# (which since Phase 1 is just the cache write, ~milliseconds — but
# we leave headroom for boot-time tasks like the BM25 rebuild that
# can run for ~30 s on a cold start at corpus scale).
TIMEOUT="${GUNICORN_TIMEOUT:-120}"

# How long gunicorn waits for a worker to finish in-flight requests
# on a graceful reload (SIGTERM). Keep this comfortably above any
# realistic request lifetime in the API tier — at Phase 1 there are
# no long-running API requests (those moved to the worker), so 30 s
# is plenty.
GRACEFUL="${GUNICORN_GRACEFUL_TIMEOUT:-30}"

# ALB / nginx idle timeout is typically 60 s. We keep our keep-alive
# slightly higher so connections survive routine ALB → app pings.
KEEPALIVE="${GUNICORN_KEEPALIVE:-75}"

# Slow-leak hedge — recycle each worker after this many requests.
# Jitter prevents all workers recycling at once.
MAX_REQ="${GUNICORN_MAX_REQUESTS:-1000}"
MAX_REQ_JITTER="${GUNICORN_MAX_REQUESTS_JITTER:-100}"

# Bind address. 0.0.0.0:8000 inside the container; the ALB or the
# host port mapping handles outside exposure.
BIND="${GUNICORN_BIND:-0.0.0.0:8000}"

# Stdout / stderr go to the Docker JSON log driver (or the awslogs
# driver in production — see docker-compose.ec2.yml). Setting
# ``--access-logfile -`` directs gunicorn's access log there too so
# CloudWatch ingests one schema instead of two.
ACCESS_LOG="${GUNICORN_ACCESS_LOG:--}"
ERROR_LOG="${GUNICORN_ERROR_LOG:--}"


role="${APP_ROLE:-api}"
echo "[entrypoint] APP_ROLE=${role}"

case "${role}" in
    api)
        echo "[entrypoint] starting gunicorn workers=${WORKERS} timeout=${TIMEOUT}"
        exec gunicorn backend.api:app \
            --bind "${BIND}" \
            --workers "${WORKERS}" \
            --worker-class uvicorn.workers.UvicornWorker \
            --timeout "${TIMEOUT}" \
            --graceful-timeout "${GRACEFUL}" \
            --keep-alive "${KEEPALIVE}" \
            --max-requests "${MAX_REQ}" \
            --max-requests-jitter "${MAX_REQ_JITTER}" \
            --access-logfile "${ACCESS_LOG}" \
            --error-logfile "${ERROR_LOG}" \
            --forwarded-allow-ips='*'
        ;;
    worker)
        echo "[entrypoint] starting backend.jobs.worker"
        exec python -m backend.jobs.worker
        ;;
    *)
        echo "[entrypoint] unknown APP_ROLE=${role} — expected 'api' or 'worker'"
        exit 64  # EX_USAGE
        ;;
esac
