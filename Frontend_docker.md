Frontend changes are not reflecting on EC2 deployment even after running full docker rebuild sequence (down --remove-orphans, docker rm -f, docker rmi, docker builder prune -af, docker-compose build --no-cache, docker-compose up -d). Backend works correctly. The frontend container appears healthy but serves old static files.

Please diagnose and fix in this exact order:

1. INSPECT THE FRONTEND DOCKERFILE at frontend/Dockerfile (or Dockerfile.frontend or similar). Verify:
   - Multi-stage build: stage 1 runs `npm install` and `npm run build` to produce dist/
   - Stage 2 copies dist/ into Nginx (or serving stage)
   - The COPY commands reference the CORRECT source paths from the build stage
   - There is NO volume mount in docker-compose.ec2.yml that would override the built dist/ folder with stale host files

2. INSPECT docker-compose.ec2.yml frontend service. Verify:
   - No `volumes:` mount pointing at frontend/dist or frontend/build that would shadow the container's freshly built files
   - The image is built from the Dockerfile (not pulled from a registry with cached old image)
   - Build context points to the correct frontend/ directory
   - Environment variables like REACT_APP_API_URL or VITE_API_URL are set correctly for EC2 (not localhost)

3. CHECK FOR BROWSER/CDN CACHING. The most common reason "rebuilt frontend doesn't reflect" is browser cache, not Docker:
   - Verify Nginx config inside the frontend container has cache-control headers that prevent aggressive caching of index.html (it should have `Cache-Control: no-cache, no-store, must-revalidate` for index.html, while hashed assets like main.abc123.js can be cached forever)
   - If Nginx config is missing or wrong, fix it

4. VERIFY THE BUILD ACTUALLY PICKED UP NEW CODE. Run inside the running frontend container:
   docker exec acadialogiq-ui sh -c "ls -la /usr/share/nginx/html/ && cat /usr/share/nginx/html/index.html | head -20"
   Check the timestamps of the built files and the JS bundle hash in index.html. If timestamps are old or hash matches a previous build, the build stage cached.

5. CHECK GIT STATE ON EC2. Confirm the latest code is actually pulled:
   cd <project-root> && git log -1 --oneline && git status
   If git is out of date or has uncommitted changes overriding the new code, fix that first.

6. PROVIDE THE COMPLETE FIX as patches to:
   - frontend/Dockerfile (if cache-busting is needed via ARG with build timestamp)
   - docker-compose.ec2.yml (if volume mounts are causing the override)
   - frontend/nginx.conf (if cache headers are wrong)
   
   Plus give me the exact 4-command rebuild sequence I should run on EC2 after applying the fixes.

7. ADD A VERSION STAMP to the frontend so we can verify the new build is actually live. Add a small `<meta name="build-timestamp" content="..." />` tag in index.html that gets injected at build time, OR a small "v1.2.3-buildhash" footer text in App.jsx that updates with each build. This way we can immediately tell from the browser whether the new code is loaded.

Paste the contents of the relevant files (Dockerfile, docker-compose.ec2.yml, nginx.conf) and the diagnostic command outputs in your response so I can see exactly what's happening.