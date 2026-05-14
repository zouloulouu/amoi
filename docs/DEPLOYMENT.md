# Deployment

Production runs on one VPS:

- nginx serves the React build from `/var/www/data_ina`
- nginx proxies `/api/*` to FastAPI on `127.0.0.1:8000`
- Docker Compose runs the FastAPI container
- Certbot manages HTTPS for `projetsignal.fr`

## First-Time VPS Setup

Install system dependencies:

```bash
sudo apt update
sudo apt install -y git curl ca-certificates docker.io docker-compose-v2 nginx certbot python3-certbot-nginx nodejs npm
sudo systemctl enable --now docker
sudo systemctl enable --now nginx
```

Clone the repo:

```bash
cd ~
git clone https://github.com/zouloulouu/amoi.git data_ina
cd ~/data_ina
```

Create `.env` from `.env.example` and fill the real HuggingFace token locally on the VPS:

```bash
cp .env.example .env
nano .env
chmod 600 .env
```

The production `.env` must include:

```bash
HF_TOKEN=<real HuggingFace token>
INA_HF_REPO_ID=zouloulouu/data_ina_clean
INA_DICTIONARY_PATH=/app/state/dictionaries.json
INA_DISABLE_PREWARM=1
INA_TAGGING_CACHE_MAXSIZE=2
```

Never commit `.env`.

## Deploy

From the VPS:

```bash
cd ~/data_ina
bash deploy.sh
```

The root `deploy.sh` does the full deployment:

1. pulls the latest Git commit
2. checks `.env` and runtime directories
3. rebuilds/restarts the FastAPI container
4. builds the React frontend
5. copies `frontend/dist` to `/var/www/data_ina`
6. reloads nginx
7. smoke-tests local backend, public API, and public frontend

## Verify

```bash
sudo docker compose ps
curl -fsS https://projetsignal.fr/api/health | python3 -m json.tool
curl -I https://projetsignal.fr/
```

Expected:

- Docker service `api` is `healthy`
- `/api/health` returns `status: ok`
- frontend returns `HTTP/2 200` or `HTTP/1.1 200`

## Troubleshooting

Container and API:

```bash
sudo docker compose ps
sudo docker compose logs --tail=120 api
curl -i http://127.0.0.1:8000/health
```

nginx:

```bash
sudo nginx -t
sudo tail -80 /var/log/nginx/error.log
sudo systemctl reload nginx
```

Certificate renewal:

```bash
sudo certbot renew --dry-run
```

## Rollback

```bash
cd ~/data_ina
git log --oneline -10
git checkout <commit>
bash deploy.sh
```

Return to `main` afterwards:

```bash
git checkout main
git pull --ff-only
```
