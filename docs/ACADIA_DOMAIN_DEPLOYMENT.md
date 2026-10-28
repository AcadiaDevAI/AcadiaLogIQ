# Deploying Acadia Log-IQ at `app.acadiaconsultants.com`

End-to-end runbook for putting the existing EC2 deployment behind a host-level
Nginx reverse proxy on a real domain, then layering HTTPS on top.

> **Scope / safety.** Nothing in this repo change deploys itself. The only new
> artifacts are `nginx/acadia.conf` (this proxy config) and this document. No
> application code, no `docker-compose*.yml`, no secrets, and no CORS settings
> were modified — they did not need to be (see
> [Why no code changes were required](#appendix--why-no-code-changes-were-required)).

---

## Architecture after this change

```
                             Internet
                                │
                 app.acadiaconsultants.com  (DNS A → 34.228.87.148)
                                │
                     ┌──────────▼───────────┐
                     │  HOST Nginx (EC2)     │   /etc/nginx/conf.d/acadia.conf
                     │  :80  (:443 later)    │
                     └─────┬───────────┬─────┘
          location /       │           │   location /api/   (strips /api)
                           ▼           ▼
             127.0.0.1:8501            127.0.0.1:8000
            ┌───────────────┐         ┌────────────────┐
            │ ui container  │         │ api container  │
            │ (nginx + SPA) │         │ (FastAPI)      │
            │  logiq-ui     │         │  logiq-api     │
            └───────────────┘         └────────────────┘
```

- `/`            → React SPA served by the `ui` container (port 8501).
- `/api/<path>`  → FastAPI in the `api` container (port 8000), with the `/api`
  prefix **stripped** (`/api/ask` → `/ask`).
- Single origin ⇒ **no CORS** between browser and API.
- Existing direct access on `http://34.228.87.148:8501` keeps working, because
  the container's own nginx (`frontend/nginx.conf`) is untouched.

---

## Target environment (recap)

| Item | Value |
|---|---|
| AWS region | `us-east-1` |
| EC2 instance | `AcadiaIQ-dev` |
| Elastic IP | `34.228.87.148` |
| Domain / subdomain | `acadiaconsultants.com` / `app.acadiaconsultants.com` |
| DNS provider | Network Solutions |
| Compose file | `docker-compose.ec2.yml` |
| Frontend container | `logiq-ui` → host `8501` |
| Backend container | `logiq-api` → host `8000` |
| Host Nginx | installed & running; ports 80/443 free |

---

## Deployment steps

> Run steps 1 locally. Steps 2–12 run on the `AcadiaIQ-dev` EC2 host over SSH.
> `sudo` is required for every Nginx command.

### 1. Commit and push the changes (local machine)

```bash
# From the repo root (…/AICode_Chatbot)
git checkout -b feat/acadia-domain-nginx
git add nginx/acadia.conf docs/ACADIA_DOMAIN_DEPLOYMENT.md
git commit -m "Add host Nginx reverse proxy for app.acadiaconsultants.com"
git push -u origin feat/acadia-domain-nginx
```

Open a PR and merge to your deploy branch (e.g. `main`) per your normal flow.

> These are the **only** two files to commit. No code / compose / env files changed.

### 2. Pull the changes onto `AcadiaIQ-dev`

```bash
ssh ec2-user@34.228.87.148          # or your configured SSH alias / user
cd /path/to/AICode_Chatbot          # the checkout used by docker-compose.ec2.yml
git fetch origin
git checkout main
git pull --ff-only origin main
```

### 3. Confirm the production frontend build variable

No action needed — `docker-compose.ec2.yml` already hardcodes the correct value:

```yaml
ui:
  build:
    args:
      REACT_APP_API_BASE: /api          # ← already present; this is what we want
```

So the rebuilt bundle will call the backend via the same-origin `/api` path.
(If you ever parameterise it, keep `/api` as the default for this host.)

### 4. Rebuild ONLY the frontend container

The host-nginx change does not require rebuilding the backend. Rebuild just the
`ui` service so the bundle is baked with `REACT_APP_API_BASE=/api`:

```bash
docker compose -f docker-compose.ec2.yml build ui
docker compose -f docker-compose.ec2.yml up -d ui
```

- `api` is **not** rebuilt or restarted.
- `--no-deps` is implied for the single service; `api` stays up. If compose
  tries to touch `api`, add `--no-deps`: `up -d --no-deps ui`.

> If the bundle was already built with `/api` previously, you can skip step 4
> entirely — nothing in the image changed. Rebuild only if the current image
> was built with a different `REACT_APP_API_BASE`.

### 5. Install the Nginx config

```bash
sudo cp nginx/acadia.conf /etc/nginx/conf.d/acadia.conf
```

> If a stock catch-all vhost answers on port 80 first (e.g. Ubuntu's
> `/etc/nginx/sites-enabled/default`), disable it so Host-based routing picks
> our block:
> `sudo rm -f /etc/nginx/sites-enabled/default` (Ubuntu) — Amazon Linux's
> `conf.d` has no default server by default.

### 6. Test the Nginx configuration

```bash
sudo nginx -t
```

Expected: `syntax is ok` / `test is successful`.

> If it reports `"connection_upgrade" ... duplicate`, another config already
> defines that `map`. Fix: delete the `map $http_upgrade $connection_upgrade {…}`
> block at the top of `/etc/nginx/conf.d/acadia.conf`, then re-run `nginx -t`.

### 7. Reload Nginx safely

```bash
sudo systemctl reload nginx     # graceful: no dropped connections
# Fallback if not under systemd:  sudo nginx -s reload
```

### 8. Verify routing (before DNS — using the Host header)

You can validate end-to-end on the box itself without waiting for DNS:

```bash
# Backend health through the proxy (/api/health → /health)
curl -s -H "Host: app.acadiaconsultants.com" http://127.0.0.1/api/health ; echo

# SPA index through the proxy
curl -sI -H "Host: app.acadiaconsultants.com" http://127.0.0.1/ | head -n1

# SSE endpoint responds as a stream (expect text/event-stream; Ctrl-C to stop)
curl -N -H "Host: app.acadiaconsultants.com" \
     -H "Content-Type: application/json" \
     -X POST http://127.0.0.1/api/ask/stream \
     -d '{"q":"ping","session_id":null}'
```

Expected: `/api/health` returns the JSON health payload; `/` returns `200`.
Direct `http://34.228.87.148:8501` should **still** work unchanged.

### 9. Configure Network Solutions DNS

In the Network Solutions control panel for `acadiaconsultants.com`, add/confirm:

| Type | Host / Name | Value | TTL |
|---|---|---|---|
| `A` | `app` | `34.228.87.148` | 3600 (or lowest allowed) |

- This creates `app.acadiaconsultants.com → 34.228.87.148`.
- Do **not** proxy/forward — a plain A record to the Elastic IP.
- Verify propagation (can take minutes to a few hours):

```bash
dig +short app.acadiaconsultants.com        # → 34.228.87.148
nslookup app.acadiaconsultants.com
```

Once it resolves, open `http://app.acadiaconsultants.com` in a browser and
confirm the app loads and API calls (Network tab → `/api/...`) return 200.

### 10. Enable HTTPS with Certbot + Let's Encrypt (AFTER DNS resolves)

Only run this once `dig +short app.acadiaconsultants.com` returns the Elastic IP,
and port 443 is open in the EC2 security group.

```bash
# Amazon Linux 2023:
sudo dnf install -y certbot python3-certbot-nginx
# (Amazon Linux 2:  sudo amazon-linux-extras install epel -y && sudo yum install -y certbot python3-certbot-nginx)
# (Ubuntu:          sudo apt-get update && sudo apt-get install -y certbot python3-certbot-nginx)

sudo certbot --nginx -d app.acadiaconsultants.com \
     --redirect -m dev@acadiaconsultants.com --agree-tos --no-eff-email
```

What `--nginx` does automatically:
- Solves the HTTP-01 challenge over port 80.
- Edits `/etc/nginx/conf.d/acadia.conf` in place: adds a `listen 443 ssl`
  server block, the `ssl_certificate` / `ssl_certificate_key` lines, and
  (with `--redirect`) an 80→443 redirect.
- Installs a systemd timer for auto-renewal.

Then re-test and reload:

```bash
sudo nginx -t && sudo systemctl reload nginx
curl -sI https://app.acadiaconsultants.com/ | head -n1          # 200
curl -s  https://app.acadiaconsultants.com/api/health ; echo    # health JSON
sudo certbot renew --dry-run                                    # confirm renewal works
```

Because the proxy already sends `X-Forwarded-Proto $scheme`, the backend sees
`https` automatically once TLS is terminated here — no app change required.

### 11. Update Clerk production-domain settings (if required)

The deployment currently uses a **development** Clerk instance (`pk_test_…`),
which accepts any origin — so the app will work at the new domain with no Clerk
change. For a hardened production setup:

1. In the Clerk Dashboard, either add `app.acadiaconsultants.com` to the
   **development** instance's allowed origins, **or** create a **production**
   instance and complete its DNS (CNAME) records at Network Solutions.
2. If you move to a production instance, update the keys **without committing
   secrets**:
   - Backend: set `CLERK_PUBLISHABLE_KEY` / `CLERK_SECRET_KEY` (`pk_live_…` /
     `sk_live_…`) in `backend/.env` on the EC2 host only.
   - Frontend: pass `REACT_APP_CLERK_PUBLISHABLE_KEY=pk_live_…` as a build arg
     (it is already wired in `docker-compose.ec2.yml`), then rebuild `ui`.
3. The backend does **not** enforce `azp`/origin (verified in
   `backend/clerk_auth.py`), so no allow-list change is needed server-side.

> Security note: a Clerk **secret** key currently lives in `frontend/.env`.
> Secret keys must never ship in a frontend bundle or be committed. Rotate it in
> the Clerk dashboard and keep `sk_…` only in `backend/.env` on the host.

### 12. Restrict public access to ports 8000 and 8501

Once everything works through the domain, stop exposing the raw app ports to the
internet so the proxy is the only entry point.

**Option A — EC2 Security Group (recommended, no downtime):**
In the AWS console, edit the `AcadiaIQ-dev` instance's inbound rules:
- **Keep**: `80` and `443` open to `0.0.0.0/0`.
- **Remove / restrict**: inbound `8000` and `8501` (delete the rules, or scope
  them to your office IP for debugging). The host proxy reaches the containers
  over `127.0.0.1`, which is unaffected by the security group.

**Option B — bind containers to localhost only** (defence-in-depth, needs a
container restart, so do it in a maintenance window — out of scope for this
no-restart task):
In `docker-compose.ec2.yml`, change the published ports to loopback:
```yaml
api: ports: ["127.0.0.1:8000:8000"]
ui:  ports: ["127.0.0.1:8501:80"]
```
then `docker compose -f docker-compose.ec2.yml up -d`. (Option A alone is
sufficient; use B only if you want the extra layer.)

Verify from an external host:
```bash
curl -m 5 http://34.228.87.148:8000/health   # should now TIME OUT / be refused
curl -m 5 http://34.228.87.148:8501/         # should now TIME OUT / be refused
curl -sI http://app.acadiaconsultants.com/    # still 200 via the proxy
```

---

## Rollback

Each layer rolls back independently; start from the smallest blast radius.

**A. Remove the host proxy (fastest — restores pre-change state):**
```bash
sudo rm /etc/nginx/conf.d/acadia.conf
sudo nginx -t && sudo systemctl reload nginx
```
The domain stops serving, but `:8501` / `:8000` direct access (if the security
group still allows it) is exactly as before. Re-open 8000/8501 in the security
group if you tightened them in step 12.

**B. Undo HTTPS but keep HTTP proxy:**
```bash
sudo certbot delete --cert-name app.acadiaconsultants.com
# then restore the HTTP-only version of the file and reload:
sudo cp nginx/acadia.conf /etc/nginx/conf.d/acadia.conf
sudo nginx -t && sudo systemctl reload nginx
```

**C. Revert the frontend image** (only if step 4 rebuilt it and caused trouble):
```bash
# Roll the repo back and rebuild, or re-run with the previous image tag.
git checkout <previous-commit> -- docker-compose.ec2.yml
docker compose -f docker-compose.ec2.yml build ui
docker compose -f docker-compose.ec2.yml up -d ui
```

**D. Revert the repo change entirely:**
```bash
git revert <merge-commit>      # removes nginx/acadia.conf + this doc from the tree
```

**E. DNS rollback:** delete the `app` A record in Network Solutions (or point it
elsewhere). Propagation lag applies.

---

## Verification checklist

- [ ] `sudo nginx -t` → `test is successful`
- [ ] `curl -H "Host: app.acadiaconsultants.com" http://127.0.0.1/api/health` → health JSON
- [ ] `/api/ask/stream` returns `Content-Type: text/event-stream` and streams frames
- [ ] File upload (≤100 MB) succeeds through the domain
- [ ] `http://34.228.87.148:8501` still works (existing EC2 access preserved)
- [ ] Local `npm start` on :3000 still talks to :8000 (local dev preserved)
- [ ] `dig +short app.acadiaconsultants.com` → `34.228.87.148`
- [ ] Browser loads `https://app.acadiaconsultants.com`; Network tab shows `/api/...` 200s
- [ ] `certbot renew --dry-run` passes
- [ ] Ports 8000 / 8501 no longer reachable from the public internet

---

## Appendix — why no code changes were required

| Concern | Finding | Action |
|---|---|---|
| Frontend API base | `frontend/src/services/api.js` → `process.env.REACT_APP_API_BASE \|\| "http://localhost:8000"`. The `window.location.replace(":8501",":8000")` variant is **commented out**. | None. `docker-compose.ec2.yml` already builds with `REACT_APP_API_BASE=/api`. |
| Other API modules | `tier1Api.js`, `journey/journeyApi.js`, `intake/intakeApi.js` all `import { api }` from the shared axios instance. | None — they inherit `/api` + the Clerk interceptor. |
| Non-axios calls | `fetch(\`${API_BASE}/ask/stream\`)` uses the same base ⇒ `/api/ask/stream`; `putToS3()` PUTs directly to an S3 URL from the backend (bypasses nginx). | None — proxy handles `/ask/stream`; S3 PUT needs no proxy. |
| FastAPI prefix | `backend/api.py` mounts all routes at root; no `root_path`. | The proxy strips `/api` so `/api/ask → /ask`. |
| CORS | Same-origin in every real deployment (SPA + API under one host). `allow_origins` does not even list the EC2 IP today, yet :8501 works — proof the model is same-origin. | None. |
| SSE / streaming | `POST /ask/stream` returns `text/event-stream`. | `location = /api/ask/stream` sets `proxy_buffering off` + long timeout. |
| Uploads | `MAX_FILE_SIZE_MB=100` (`backend/config.py`). | `client_max_body_size 100M` in the proxy. |
| WebSockets | None in use (SSE only). | `map $http_upgrade $connection_upgrade` included for future use; harmless today. |
| Clerk | `azp`/origin not enforced; issuer derived from publishable key. | Works at the new domain as-is; prod instance optional (step 11). |
| `X-Forwarded-Proto` | Sent as `$scheme`. | Backend sees `https` automatically after Certbot — no app change. |
