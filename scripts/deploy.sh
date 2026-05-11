#!/bin/bash
# Production deploy script — to be run on the VPS.
# Usage:
#   chmod +x scripts/deploy.sh
#   ./scripts/deploy.sh
#
# Steps:
#   1. Pull latest code from GitHub
#   2. Rebuild backend Docker container
#   3. Build frontend (Vite production build with .env.production)
#   4. Copy dist files to nginx static root
#   5. Reload nginx + smoke test
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> [1/5] Pull from GitHub"
git pull

echo "==> [2/5] Rebuild backend container"
docker compose up -d --build

echo "==> [3/5] Build frontend"
cd frontend
npm ci
npm run build
cd "$ROOT"

echo "==> [4/5] Deploy frontend dist to /var/www/data_ina"
sudo mkdir -p /var/www/data_ina
sudo rm -rf /var/www/data_ina/*
sudo cp -r frontend/dist/* /var/www/data_ina/
sudo chown -R www-data:www-data /var/www/data_ina

echo "==> [5/5] Reload nginx"
sudo systemctl reload nginx

echo
echo "==> Smoke tests"
sleep 2
if curl -fsS http://127.0.0.1:8000/health >/dev/null; then
    echo "    backend /health: OK"
else
    echo "    backend /health: FAIL (check 'docker compose logs api')"
    exit 1
fi

if [ -f /var/www/data_ina/index.html ]; then
    echo "    frontend index.html: OK"
else
    echo "    frontend index.html: MISSING"
    exit 1
fi

echo
echo "Deployment finished."
