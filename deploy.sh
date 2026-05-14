#!/usr/bin/env bash
# One-command production redeploy for the OVH VPS.
#
# Run on the VPS from the repo root:
#   cd ~/data_ina
#   bash deploy.sh
#
# This script never writes secrets. It expects the real HF_TOKEN to already be
# present in the local, gitignored .env file on the VPS.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

STATIC_ROOT="/var/www/data_ina"
DOMAIN="https://projetsignal.fr"

echo "==> [1/8] Check local .env"
if [[ ! -f .env ]]; then
  echo "Missing .env. Create it from .env.example and add HF_TOKEN before deploying." >&2
  exit 1
fi

if ! grep -q '^HF_TOKEN=.\+' .env; then
  echo "HF_TOKEN is empty in .env. Refusing to deploy a backend that cannot read HF." >&2
  exit 1
fi

echo "==> [2/8] Pull latest code"
git pull --ff-only

echo "==> [3/8] Ensure persistent runtime directories"
mkdir -p data/state data/cache logs

if [[ ! -f data/state/dictionaries.json && -f dictionaries.json ]]; then
  cp dictionaries.json data/state/dictionaries.json
fi

# The container runs as uid 1000. Keep these directories writable by that user.
sudo chown -R 1000:1000 data/state data/cache logs

echo "==> [4/8] Ensure VPS-safe runtime env"
grep -q '^INA_DICTIONARY_PATH=' .env \
  || echo 'INA_DICTIONARY_PATH=/app/state/dictionaries.json' >> .env

if grep -q '^INA_DISABLE_PREWARM=' .env; then
  sed -i 's/^INA_DISABLE_PREWARM=.*/INA_DISABLE_PREWARM=1/' .env
else
  echo 'INA_DISABLE_PREWARM=1' >> .env
fi

if grep -q '^INA_TAGGING_CACHE_MAXSIZE=' .env; then
  sed -i 's/^INA_TAGGING_CACHE_MAXSIZE=.*/INA_TAGGING_CACHE_MAXSIZE=2/' .env
else
  echo 'INA_TAGGING_CACHE_MAXSIZE=2' >> .env
fi

echo "==> [5/8] Build/restart FastAPI container"
sudo docker compose up -d --build

echo "==> [6/8] Build React frontend"
cd frontend
if [[ -f package-lock.json ]]; then
  npm ci
else
  npm install
fi
npm run build
cd "$ROOT"

echo "==> [7/8] Publish frontend to nginx"
sudo mkdir -p "$STATIC_ROOT"
sudo rm -rf "$STATIC_ROOT"/*
sudo cp -r frontend/dist/* "$STATIC_ROOT"/
sudo chown -R www-data:www-data "$STATIC_ROOT"
sudo nginx -t
sudo systemctl reload nginx

echo "==> [8/8] Smoke tests"
sleep 5

sudo docker compose ps

curl -fsS http://127.0.0.1:8000/health | python3 -m json.tool >/dev/null
echo "    local backend /health: OK"

curl -fsS "$DOMAIN/api/health" | python3 -m json.tool >/dev/null
echo "    public backend /api/health: OK"

curl -fsSI "$DOMAIN/" >/dev/null
echo "    public frontend: OK"

echo
echo "Deploy finished: $DOMAIN"
