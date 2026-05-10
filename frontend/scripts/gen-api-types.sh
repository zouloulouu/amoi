#!/usr/bin/env sh
set -eu

npx openapi-typescript "${VITE_API_BASE_URL:-http://localhost:8000}/openapi.json" -o src/api/schema.d.ts
