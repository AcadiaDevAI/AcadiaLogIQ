#!/bin/bash
# Acadia LogIQ — Production redeploy script
# Usage: ./redeploy.sh
# Run from project root on EC2

set -e  # Exit immediately on any error

# ── Configuration ─────────────────────────────────
PROJECT_ROOT="/home/ec2-user/AcadiaLogIQ"
COMPOSE_FILE="docker-compose.ec2.yml"
LOG_DIR="$PROJECT_ROOT/deploy_logs"
GIT_BRANCH="AcadiaLogIQ-Dev"

# ── Setup logging ─────────────────────────────────
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "Acadia LogIQ Deployment"
echo "Started: $(date)"
echo "Log: $LOG_FILE"
echo "=========================================="

# ── Step 1: Pull latest code ──────────────────────
cd "$PROJECT_ROOT"
echo ""
echo ">>> Step 1: Pulling latest code from $GIT_BRANCH..."
git pull origin "$GIT_BRANCH"

# ── Step 2: Stop existing containers ──────────────
echo ""
echo ">>> Step 2: Stopping existing containers..."
docker-compose -f "$COMPOSE_FILE" down --remove-orphans

# ── Step 3: Remove old images ─────────────────────
echo ""
echo ">>> Step 3: Removing old images..."
docker rm -f acadialogiq-api acadialogiq-ui 2>/dev/null || true
docker rmi acadialogiq-api acadialogiq-ui 2>/dev/null || true

# ── Step 4: Clean Docker build cache ──────────────
echo ""
echo ">>> Step 4: Cleaning Docker build cache..."
docker builder prune -af

# ── Step 5: Generate unique build timestamp ───────
export BUILD_TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo ""
echo ">>> Step 5: Build timestamp: $BUILD_TIMESTAMP"

# ── Step 6: Build images with timestamp ───────────
echo ""
echo ">>> Step 6: Building images (this takes 3-5 min)..."
docker-compose -f "$COMPOSE_FILE" build --no-cache \
  --build-arg BUILD_TIMESTAMP="$BUILD_TIMESTAMP" || {
    echo "ERROR: Build failed. Existing containers were already stopped."
    exit 1
  }

# ── Step 7: Start fresh containers ────────────────
echo ""
echo ">>> Step 7: Starting containers..."
docker-compose -f "$COMPOSE_FILE" up -d

# ── Step 8: Wait for health checks ────────────────
echo ""
echo ">>> Step 8: Waiting 15 seconds for containers to initialize..."
sleep 15

# ── Step 9: Verify containers are running ─────────
echo ""
echo ">>> Step 9: Container status:"
docker ps --filter "name=acadialogiq" \
  --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# ── Step 10: Verify build timestamp in container ──
echo ""
echo ">>> Step 10: Verifying build timestamp..."
DEPLOYED_TIMESTAMP=$(docker exec acadialogiq-ui \
  cat /usr/share/nginx/html/BUILD_TIMESTAMP 2>/dev/null || echo "NOT FOUND")
echo "Deployed timestamp: $DEPLOYED_TIMESTAMP"

if [ "$DEPLOYED_TIMESTAMP" != "$BUILD_TIMESTAMP" ]; then
  echo "WARNING: Timestamp mismatch! Expected $BUILD_TIMESTAMP got $DEPLOYED_TIMESTAMP"
  echo "Build may have used cached layers despite --no-cache."
fi

# ── Step 11: Verify cache-control headers ─────────
echo ""
echo ">>> Step 11: Cache-control headers on /:"
curl -sI http://localhost:8501/ | grep -i cache-control || \
  echo "WARNING: No cache-control header found. Check nginx.conf."

# ── Done ──────────────────────────────────────────
echo ""
echo "=========================================="
echo "Deployment complete: $(date)"
echo "Frontend: http://<EC2-PUBLIC-IP>:8501"
echo "Backend:  http://<EC2-PUBLIC-IP>:8000"
echo "Build version: $BUILD_TIMESTAMP"
echo "=========================================="
echo ""
echo "TIP: Hard refresh your browser (Ctrl+Shift+R) to see new frontend."
